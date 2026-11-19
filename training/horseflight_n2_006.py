"""Run NNUE training sweeps against read-only data caches."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_COUNT, PAD_INDEX, MAX_ACTIVE = 781, 781, 38
S_ACC, S_HIDDEN, S_OUTPUT, ACT_MAX = 256, 128, 256, 127
DEFAULT_SEED = 20261530


def fake_quant(x, scale, qmin, qmax):
    clipped = torch.clamp(x, min=qmin / scale, max=qmax / scale)
    quantized = torch.round(clipped * scale) / scale
    return clipped + (quantized - clipped).detach()


class HorseflightNNUE(nn.Module):
    """781 → 256 → 32 → 1 network for quantization-aware training."""
    def __init__(self, accumulator_width=256, hidden_width=32):
        super().__init__()
        self.accumulator_width = accumulator_width
        self.hidden_width = hidden_width
        self.feature_weights = nn.Parameter(torch.empty(FEATURE_COUNT + 1, accumulator_width))
        self.accumulator_bias = nn.Parameter(torch.zeros(accumulator_width))
        self.hidden = nn.Linear(accumulator_width, hidden_width)
        self.output = nn.Linear(hidden_width, 1)
        nn.init.normal_(self.feature_weights, mean=0., std=0.025)
        nn.init.constant_(self.accumulator_bias, 0.05)
        nn.init.normal_(self.hidden.weight, std=0.05)
        nn.init.zeros_(self.hidden.bias)
        nn.init.normal_(self.output.weight, std=0.05)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.feature_weights[PAD_INDEX].zero_()

    def forward(self, active_indices):
        w1 = fake_quant(self.feature_weights, S_ACC, -32768, 32767)
        b1 = fake_quant(self.accumulator_bias, S_ACC, -(2 ** 31), 2 ** 31 - 1)
        contributions = F.embedding(active_indices, w1, padding_idx=PAD_INDEX)
        accumulator = contributions.sum(dim=1) + b1
        a1 = fake_quant(torch.clamp(accumulator, 0., 1.), ACT_MAX, 0, ACT_MAX)
        w2 = fake_quant(self.hidden.weight, S_HIDDEN, -128, 127)
        b2 = fake_quant(self.hidden.bias, ACT_MAX * S_HIDDEN, -(2 ** 31), 2 ** 31 - 1)
        hidden = F.linear(a1, w2, b2)
        a2 = fake_quant(torch.clamp(hidden, 0., 1.), ACT_MAX, 0, ACT_MAX)
        w3 = fake_quant(self.output.weight, S_OUTPUT, -32768, 32767)
        b3 = fake_quant(self.output.bias, S_OUTPUT, -(2 ** 31), 2 ** 31 - 1)
        return F.linear(a2, w3, b3).squeeze(-1)


@dataclass(frozen=True)
class Trial:
    lr: float = 0.001
    batch_size: int = 2048
    weight_decay: float = 0.0001
    mirror_p: float = 0.5
    scheduler: str = "plateau"
    symmetry_lambda: float = 0.0
    loss: str = "mse"
    cp_scale: float = 400.0
    epochs: int = 30
    symmetry_fraction: float = 0.25
    seed: int = DEFAULT_SEED

    def __post_init__(self):
        if not (0 <= self.mirror_p <= 1 and 0 <= self.symmetry_fraction <= 1):
            raise ValueError("Invalid augmentation probability")
        if self.scheduler not in {"plateau", "cosine"} or self.loss not in {"mse", "huber"}:
            raise ValueError("Unsupported loss/scheduler")
        if min(self.lr, self.batch_size, self.cp_scale, self.epochs) <= 0:
            raise ValueError("Positive lr, batch, cp_scale and epochs required")
        if self.weight_decay < 0 or self.symmetry_lambda < 0:
            raise ValueError("Negative decay/symmetry lambda")


@dataclass(frozen=True)
class Runtime:
    precision: str = "fp32"  # fp32: old numerical behavior; tf32; bf16
    compile_mode: str = "none"  # none | default | reduce-overhead
    dataset_device: str = "cuda"  # cuda | cpu (CPU training: use cpu)
    # Compilation is optional because hardware/PyTorch version may not support graph capture.

    def __post_init__(self):
        if self.precision not in {"fp32", "tf32", "bf16"}:
            raise ValueError("precision must be fp32, tf32 or bf16")
        if self.compile_mode not in {"none", "default", "reduce-overhead"}:
            raise ValueError("Unknown compile mode")
        if self.dataset_device not in {"cuda", "cpu"}:
            raise ValueError("dataset_device must be cuda or cpu")


def configure_runtime(runtime: Runtime, device: torch.device):
    if runtime.precision != "fp32" and device.type != "cuda":
        raise RuntimeError("BF16/TF32 experiments require CUDA here")
    if runtime.compile_mode != "none" and device.type != "cuda":
        raise RuntimeError("compile speed experiments require CUDA here")
    if runtime.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested, but this CUDA device does not support native BF16")
    torch.set_float32_matmul_precision("high" if runtime.precision == "tf32" else "highest")


def autocast_context(runtime: Runtime, device: torch.device):
    return (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and runtime.precision == "bf16"
            else contextlib.nullcontext())


def trial_id(config: Trial):
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def atomic_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    os.replace(temp, path)


def atomic_torch(path: Path, obj: dict):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp)
    os.replace(temp, path)


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cache_paths(root: Path):
    base = root / "research_data/horseflight_n2_v1_1/encoded"
    extra = root / "research_data/horseflight_n2_5m_v1/encoded"
    return {
        "base_features": base / "train_features.npy",
        "base_cp": base / "train_cp.npy",
        "extra_features": extra / "extra_features.npy",
        "extra_cp": extra / "extra_cp.npy",
        "val_features": base / "val_features.npy",
        "val_cp": base / "val_cp.npy",
        "mirror_blocked": base / "n2_004_mirror_leak_mask.npy",
    }


def load_data(root: Path, device: torch.device, runtime: Runtime):
    """Load the cached dataset for one worker."""
    paths = cache_paths(root)
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen caches:\n" + "\n".join(missing))
    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    expected = {
        "base_features": ((1_000_000, 38), np.uint16),
        "base_cp": ((1_000_000,), np.int16),
        "extra_features": ((4_000_000, 38), np.uint16),
        "extra_cp": ((4_000_000,), np.int16),
        "val_features": ((50_000, 38), np.uint16),
        "val_cp": ((50_000,), np.int16),
        "mirror_blocked": ((1_000_000,), np.bool_),
    }
    for name, (shape, dtype) in expected.items():
        if arrays[name].shape != shape or arrays[name].dtype != np.dtype(dtype):
            raise ValueError(f"Invalid {name}: {arrays[name].shape} {arrays[name].dtype}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    dataset_id = "horseflight-n2-5m:" + hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:20]
    target = device if runtime.dataset_device == "cuda" else torch.device("cpu")
    if runtime.dataset_device == "cuda" and target.type != "cuda":
        raise RuntimeError("Requested GPU dataset but CUDA is unavailable")

    # With CUDA placement, keep the indices on the GPU between trials.
    def tensor(a, dtype):
        return torch.from_numpy(np.array(a, copy=True)).to(device=target, dtype=dtype)
    train_x = torch.cat([tensor(arrays["base_features"], torch.long),
                         tensor(arrays["extra_features"], torch.long)])
    train_cp = torch.cat([tensor(arrays["base_cp"], torch.float32),
                          tensor(arrays["extra_cp"], torch.float32)])
    val_x = tensor(arrays["val_features"], torch.long)
    val_cp = tensor(arrays["val_cp"], torch.float32)
    blocked = torch.cat([tensor(arrays["mirror_blocked"], torch.bool),
                         torch.zeros(4_000_000, dtype=torch.bool, device=target)])
    bytes_used = sum(x.numel() * x.element_size() for x in (train_x, train_cp, val_x, val_cp, blocked))
    print(f"Data: {len(train_x):,} train / {len(val_x):,} val; {bytes_used/2**30:.2f} GiB on {target}; {dataset_id}", flush=True)
    return train_x, train_cp, val_x, val_cp, blocked, dataset_id


def build_mirror_lut(device):
    lut = torch.arange(782, dtype=torch.long, device=device)
    ids = torch.arange(768, dtype=torch.long, device=device)
    lut[:768] = (((ids // 64 + 6) % 12) * 64) + ((ids % 64) ^ 56)
    lut[768] = 781
    lut[769:773] = torch.tensor([771, 772, 769, 770], device=device)
    return lut


def fast_mirror(x, lut):
    white_turn = (x == 768).any(dim=1)
    transformed = lut[x].clone()
    transformed[:, -1] = torch.where(~white_turn, 768, transformed[:, -1])
    return transformed.sort(dim=1).values



# CPU reference for checking the GPU mirror transform.
_REFERENCE_LUT = np.arange(782, dtype=np.uint16)
for _i in range(768):
    _REFERENCE_LUT[_i] = (((_i // 64 + 6) % 12) * 64) + ((_i % 64) ^ 56)
_REFERENCE_LUT[769:773] = [771, 772, 769, 770]


def mirror_reference(row):
    active = np.asarray(row, dtype=np.uint16)
    active = active[active != PAD_INDEX]
    mapped = _REFERENCE_LUT[active]
    if 768 in active:
        mapped = mapped[mapped != 768]
    else:
        mapped = np.r_[mapped, np.uint16(768)]
    mapped.sort()
    if len(mapped) > MAX_ACTIVE:
        raise ValueError("Mirrored row exceeds feature width")
    out = np.full(MAX_ACTIVE, PAD_INDEX, dtype=np.uint16)
    out[:len(mapped)] = mapped
    return out


def preflight_mirror(train_x, device, seed=20261028):
    examples = []
    for features in ([324, 764, 768, 769, 770, 771, 772],
                     [324, 436, 764, 771, 776], [12, 324, 436, 764, 768]):
        row = np.full(MAX_ACTIVE, PAD_INDEX, dtype=np.uint16)
        row[:len(features)] = sorted(features)
        examples.append(row)
    generator = np.random.default_rng(seed + 500)
    sample = generator.choice(len(train_x), min(len(train_x), 512), replace=False)
    ids = torch.tensor(sample, dtype=torch.long, device=train_x.device)
    chosen = train_x[ids].to(device=device, dtype=torch.long)
    source = torch.cat([torch.from_numpy(np.stack(examples)).to(device).long(), chosen])
    actual = fast_mirror(source, build_mirror_lut(device))
    reference = np.stack([mirror_reference(row) for row in source.cpu().numpy()])
    if not np.array_equal(actual.cpu().numpy(), reference):
        raise AssertionError("GPU mirror disagrees with independent CPU reference")
    if not torch.equal(fast_mirror(actual, build_mirror_lut(device)), source):
        raise AssertionError("Mirror involution failed")
    print(f"PASS mirror preflight: {len(source)} independent comparisons", flush=True)


def campaign_a(seed=DEFAULT_SEED, epochs=30):
    configs = [Trial(lr=lr, scheduler=sched, batch_size=batch, weight_decay=decay,
                     mirror_p=mirror_p, seed=seed, epochs=epochs)
               for lr, sched, batch, decay, mirror_p in itertools.product(
                   [0.0005, 0.001, 0.002], ["plateau", "cosine"], [1024, 2048, 4096],
                   [0., 0.0001, 0.01], [0.25, 0.5, 0.75])]
    assert len(configs) == len({trial_id(x) for x in configs}) == 162
    return configs


def campaign_b(top_configs, seed=DEFAULT_SEED, epochs=30):
    configs = [replace(top, symmetry_lambda=lam, loss=loss, cp_scale=scale,
                       seed=seed, epochs=epochs)
               for top in top_configs
               for lam, loss, scale in itertools.product([0., 0.01, 0.1], ["mse", "huber"], [400., 600.])]
    return list({trial_id(c): c for c in configs}.values())

class Laboratory:
    def __init__(self, *, train_x, train_cp, val_x, val_cp, mirror_blocked,
                 output_dir, device="cuda:0", dataset_id="synthetic", runtime=None,
                 model_factory=None):
        self.device = torch.device(device)
        self.runtime = runtime or Runtime(dataset_device="cuda" if self.device.type == "cuda" else "cpu")
        configure_runtime(self.runtime, self.device)
        self.model_factory = model_factory or (lambda: HorseflightNNUE(256, 32))
        self.train_x, self.train_cp = train_x, train_cp.reshape(-1).float()
        self.val_x, self.val_cp = val_x, val_cp.reshape(-1).float()
        self.blocked = mirror_blocked.bool()
        self.data_device = self.train_x.device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_id = dataset_id
        self.lut = build_mirror_lut(self.device)
        if not (self.train_x.device == self.train_cp.device == self.val_x.device == self.val_cp.device == self.blocked.device):
            raise ValueError("Dataset tensors must have a common device")
        if self.runtime.dataset_device == "cuda" and self.data_device != self.device:
            raise ValueError("GPU-resident dataset not on worker GPU")
        if self.runtime.dataset_device == "cpu" and self.data_device.type != "cpu":
            raise ValueError("CPU-resident data must be on CPU")
        if self.train_x.dtype != torch.long or self.val_x.dtype != torch.long:
            raise ValueError("Feature indices must be int64")
        if self.train_x.ndim != 2 or self.val_x.ndim != 2 or self.train_x.shape[1] != 38 or self.val_x.shape[1] != 38:
            raise ValueError("Feature width must be 38")
        if len(self.train_x) != len(self.train_cp) or len(self.train_x) != len(self.blocked) or len(self.val_x) != len(self.val_cp):
            raise ValueError("Feature/label/mask length mismatch")
        if not bool(((self.train_x >= 0) & (self.train_x <= 781)).all()):
            raise ValueError("Invalid training features")
        if not bool(((self.val_x >= 0) & (self.val_x <= 781)).all()):
            raise ValueError("Invalid validation features")

    def _to_device(self, tensor):
        return tensor if tensor.device == self.device else tensor.to(self.device, non_blocking=True)

    def _new_model(self, config: Trial):
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        base = self.model_factory().to(self.device)
        optimizer = torch.optim.AdamW(base.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        if config.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs, eta_min=1e-5)
        forward = (torch.compile(base, mode=self.runtime.compile_mode)
                   if self.runtime.compile_mode != "none" else base)
        return base, forward, optimizer, scheduler

    @torch.inference_mode()
    def validate(self, model, cp_scale):
        model.eval()
        sum_sq = sym_sq = sym_abs = 0.
        n = sym_count = 0
        with autocast_context(self.runtime, self.device):
            for start in range(0, len(self.val_x), 4096):
                end = min(start + 4096, len(self.val_x))
                x = self._to_device(self.val_x[start:end])
                logits = model(x).reshape(-1).float()
                pred = torch.sigmoid(logits * (cp_scale / 400.))
                truth = torch.sigmoid(self._to_device(self.val_cp[start:end]) / 400.)
                sum_sq += (pred - truth).square().sum().item()
                n += len(x)
                if start < 8192:
                    mirror_logits = model(fast_mirror(x, self.lut)).reshape(-1).float()
                    delta = pred + torch.sigmoid(mirror_logits * (cp_scale / 400.)) - 1.
                    sym_sq += delta.square().sum().item()
                    sym_abs += delta.abs().sum().item()
                    sym_count += len(x)
        return {"val_mse_canonical": sum_sq / n,
                "sym_mse_canonical": sym_sq / sym_count,
                "sym_mae_canonical": sym_abs / sym_count}

    def _generators(self, config):
        data_gen = torch.Generator(device=self.data_device).manual_seed(config.seed + 10_000)
        aug = torch.Generator(device=self.device).manual_seed(config.seed + 20_000)
        sym = torch.Generator(device=self.device).manual_seed(config.seed + 30_000)
        return data_gen, aug, sym

    def _batch(self, row_ids):
        return (self._to_device(self.train_x[row_ids]),
                self._to_device(self.train_cp[row_ids]),
                self._to_device(self.blocked[row_ids]))

    def _train_step(self, base, model, optimizer, config, original_x, cp, blocked, aug_rng, sym_rng):
        eligible = ~blocked
        x = original_x
        mirrored = torch.zeros((), device=self.device, dtype=torch.long)
        paired = torch.zeros((), device=self.device, dtype=torch.long)
        if config.mirror_p:
            flip = (torch.rand(len(x), generator=aug_rng, device=self.device) < config.mirror_p) & eligible
            x = torch.where(flip[:, None], fast_mirror(original_x, self.lut), original_x)
            cp = torch.where(flip, -cp, cp)
            mirrored = flip.sum()
        target = torch.sigmoid(cp / config.cp_scale)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(self.runtime, self.device):
            prediction = torch.sigmoid(model(x).reshape(-1).float())
            primary = (F.mse_loss(prediction, target) if config.loss == "mse"
                       else F.huber_loss(prediction, target, delta=0.1))
            loss = primary
            if config.symmetry_lambda:
                choose = (torch.rand(len(x), generator=sym_rng, device=self.device)
                          < config.symmetry_fraction) & eligible
                # Extracting these indices synchronizes CUDA only when symmetry loss is enabled.
                indices = torch.where(choose)[0]
                if indices.numel():
                    mirrored_pred = torch.sigmoid(model(fast_mirror(x[indices], self.lut)).reshape(-1).float())
                    sym = (prediction[indices] + mirrored_pred - 1.).square().mean()
                    loss = loss + config.symmetry_lambda * sym
                    paired = indices.numel() * torch.ones((), dtype=torch.long, device=self.device)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            base.feature_weights[PAD_INDEX].zero_()
        return primary.detach(), mirrored.detach(), paired.detach()

    def _summary(self, config, history, best_mse, best_epoch):
        winning = next((r for r in history if r["epoch"] == best_epoch), None)
        return {"id": trial_id(config), "dataset_id": self.dataset_id,
                "config": asdict(config), "runtime": asdict(self.runtime),
                "epochs_done": len(history), "completed": len(history) == config.epochs,
                "last_val_mse": history[-1]["val_mse_canonical"] if history else None,
                "best_epoch": best_epoch, "best_mse": best_mse,
                "best_sym_mae": winning["sym_mae_canonical"] if winning else None,
                "seconds_sum": sum(x["epoch_seconds"] for x in history), "history": history}

    def run(self, config: Trial, *, max_epochs=None, verbose=True):
        """Train to a rung budget without changing trial identity or scheduler horizon."""
        budget = config.epochs if max_epochs is None else int(max_epochs)
        if not 1 <= budget <= config.epochs:
            raise ValueError("Invalid rung budget")
        rid = trial_id(config)
        directory = self.output_dir / rid
        directory.mkdir(parents=True, exist_ok=True)
        config_dict = asdict(config)
        config_path = directory / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config_dict:
            raise ValueError(f"Config collision: {rid}")
        if not config_path.exists():
            atomic_json(config_path, config_dict)
        last = directory / "last.pt"
        best = directory / "best.pt"
        summary_path = directory / "summary.json"
        base, model, optimizer, scheduler = self._new_model(config)
        order_rng, aug_rng, sym_rng = self._generators(config)
        history, best_mse, best_epoch, start_epoch = [], math.inf, 0, 1
        if last.exists():
            ck = torch.load(last, map_location=self.device, weights_only=True)
            if ck["config"] != config_dict or ck["dataset_id"] != self.dataset_id:
                raise ValueError("Checkpoint config/dataset mismatch")
            saved_runtime = ck.get("runtime")
            if saved_runtime is None:
                # Older checkpoints used the default runtime and did not record it.
                if self.runtime != Runtime():
                    raise ValueError("Legacy FP32 checkpoint cannot silently switch runtime")
            elif saved_runtime != asdict(self.runtime):
                raise ValueError(f"Runtime mismatch on resume: saved={saved_runtime}; requested={asdict(self.runtime)}")
            base.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            order_rng.set_state(ck["order_rng"].cpu())
            aug_rng.set_state(ck["aug_rng"].cpu())
            sym_rng.set_state(ck["consistency_rng"].cpu())
            torch.set_rng_state(ck["cpu_rng"].cpu())
            if self.device.type == "cuda" and ck.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device=self.device)
            history = ck["history"]
            best_mse, best_epoch = ck["best_mse"], ck["best_epoch"]
            start_epoch = ck["epoch"] + 1
            if verbose:
                print(f"{rid}: resuming at epoch {start_epoch}; target {budget}", flush=True)
        if start_epoch > budget:
            # Rebuild the summary from checkpoint history.
            return self._summary(config, history, best_mse, best_epoch)
        n = len(self.train_x)
        for epoch in range(start_epoch, budget + 1):
            started = time.perf_counter()
            model.train()
            order = torch.randperm(n, device=self.data_device, generator=order_rng)
            losses = torch.zeros((), device=self.device)
            mirrored = torch.zeros((), device=self.device, dtype=torch.long)
            paired = torch.zeros((), device=self.device, dtype=torch.long)
            lr = optimizer.param_groups[0]["lr"]
            for ids in order.split(config.batch_size):
                original_x, cp, blocked = self._batch(ids)
                primary, count_mirror, count_paired = self._train_step(
                    base, model, optimizer, config, original_x, cp, blocked, aug_rng, sym_rng)
                losses += primary * len(ids)
                mirrored += count_mirror
                paired += count_paired
            metrics = self.validate(model, config.cp_scale)
            score = metrics["val_mse_canonical"]
            if not math.isfinite(score):
                raise RuntimeError("Non-finite validation; previous checkpoint intact")
            improved = score < best_mse - 1e-7
            if improved:
                best_mse, best_epoch = score, epoch
            if config.scheduler == "plateau":
                scheduler.step(score)
            else:
                scheduler.step()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            record = {"epoch": epoch, "supervised_loss": losses.item() / n,
                      **metrics, "lr": lr, "mirrored_rows": mirrored.item(),
                      "paired_rows": paired.item(), "epoch_seconds": time.perf_counter() - started}
            history.append(record)
            ck = {"dataset_id": self.dataset_id, "config": config_dict,
                  "runtime": asdict(self.runtime), "epoch": epoch,
                  "best_mse": best_mse, "best_epoch": best_epoch, "history": history,
                  "model": base.state_dict(), "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler.state_dict(), "order_rng": order_rng.get_state(),
                  "aug_rng": aug_rng.get_state(), "consistency_rng": sym_rng.get_state(),
                  "cpu_rng": torch.get_rng_state(),
                  "cuda_rng": (torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None)}
            if improved:
                atomic_torch(best, ck)
            atomic_torch(last, ck)
            summary = self._summary(config, history, best_mse, best_epoch)
            atomic_json(summary_path, summary)
            if verbose:
                print(f"{rid}: e{epoch:02d}/{config.epochs} mse={score:.7f} "
                      f"best={best_mse:.7f} lr={lr:.6g} {record['epoch_seconds']:.1f}s", flush=True)
        return summary

    def run_many(self, configs: Iterable[Trial], *, max_epochs=None):
        results = []
        for config in configs:
            results.append(self.run(config, max_epochs=max_epochs))
            self.export_leaderboard()
        return results

    def export_leaderboard(self):
        rows = []
        for path in self.output_dir.glob("*/summary.json"):
            try:
                item = json.loads(path.read_text())
                if item.get("dataset_id") != self.dataset_id:
                    continue
                row = {"id": item["id"], "best_mse": item["best_mse"],
                       "last_val_mse": item.get("last_val_mse"),
                       "best_epoch": item["best_epoch"], "best_sym_mae": item.get("best_sym_mae"),
                       "epochs_done": item["epochs_done"], "completed": item.get("completed", False),
                       "seconds_sum": item["seconds_sum"], **item["config"]}
                rows.append(row)
            except (ValueError, KeyError, OSError):
                continue
        rows.sort(key=lambda r: (r["best_mse"], r["id"]))
        if rows:
            path = self.output_dir / "leaderboard.csv"
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            os.replace(tmp, path)
        return rows

    def _benchmark_indices(self, batch_size, step):
        # Captures data gathers and transfers, rather than reusing a pre-gathered batch.
        return (torch.arange(batch_size, device=self.data_device) +
                (step * batch_size)) % len(self.train_x)

    def profile(self, config: Trial, *, steps=20, warmup=5, output_dir=None):
        """Profile batch gathering and training steps without saving checkpoints."""
        if steps <= 0 or warmup < 0:
            raise ValueError("Invalid steps or warmup")
        from torch.profiler import ProfilerActivity, profile, record_function
        base, model, opt, _ = self._new_model(config)
        _, aug, sym = self._generators(config)
        model.train()
        for i in range(warmup):
            x, cp, blocked = self._batch(self._benchmark_indices(config.batch_size, i))
            self._train_step(base, model, opt, config, x, cp, blocked, aug, sym)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        destination = Path(output_dir or (self.output_dir / "profiles"))
        destination.mkdir(parents=True, exist_ok=True)
        activities = [ProfilerActivity.CPU]
        if self.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        start = time.perf_counter()
        with profile(activities=activities, record_shapes=False, profile_memory=True,
                     acc_events=True) as prof:
            for i in range(warmup, warmup + steps):
                with record_function("Horseflight_batch_gather"):
                    x, cp, blocked = self._batch(self._benchmark_indices(config.batch_size, i))
                with record_function("Horseflight_train_step"):
                    self._train_step(base, model, opt, config, x, cp, blocked, aug, sym)
                prof.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - start
        stem = f"{trial_id(config)}_{self.device.type}{self.device.index or 0}_{self.runtime.precision}_{self.runtime.compile_mode}"
        table_path, trace_path = destination / f"{stem}.txt", destination / f"{stem}.json"
        sort_by = "self_cuda_time_total" if self.device.type == "cuda" else "self_cpu_time_total"
        table_path.write_text(prof.key_averages().table(sort_by=sort_by, row_limit=30))
        prof.export_chrome_trace(str(trace_path))
        result = {"steps": steps, "batch_size": config.batch_size,
                  "measured_seconds_including_profiler_overhead": seconds,
                  "profiled_examples_per_second": steps * config.batch_size / seconds,
                  "peak_allocated_gib": (torch.cuda.max_memory_allocated(self.device) / 2**30
                                         if self.device.type == "cuda" else None),
                  "trace": str(trace_path), "table": str(table_path),
                  "runtime": asdict(self.runtime)}
        atomic_json(destination / f"{stem}_summary.json", result)
        return result

    def benchmark(self, config: Trial, *, steps=60, warmup=10):
        """Measure throughput with batch gathering and no profiler."""
        base, model, opt, _ = self._new_model(config)
        _, aug, sym = self._generators(config)
        model.train()
        for i in range(warmup):
            x, cp, blocked = self._batch(self._benchmark_indices(config.batch_size, i))
            self._train_step(base, model, opt, config, x, cp, blocked, aug, sym)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t = time.perf_counter()
        for i in range(warmup, warmup + steps):
            x, cp, blocked = self._batch(self._benchmark_indices(config.batch_size, i))
            self._train_step(base, model, opt, config, x, cp, blocked, aug, sym)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        sec = time.perf_counter() - t
        return {"steps": steps, "seconds": sec,
                "examples_per_second": steps * config.batch_size / sec,
                "includes_batch_gather": True, "precision": self.runtime.precision,
                "compile_mode": self.runtime.compile_mode}



def export_leaderboard_from_disk(output_dir: Path, dataset_id: str | None = None):
    """Write the leaderboard after the workers finish."""
    rows = []
    for path in Path(output_dir).glob("*/summary.json"):
        try:
            item = json.loads(path.read_text())
            if dataset_id is not None and item.get("dataset_id") != dataset_id:
                continue
            rows.append({"id": item["id"], "best_mse": item["best_mse"],
                         "last_val_mse": item.get("last_val_mse"),
                         "best_epoch": item["best_epoch"], "best_sym_mae": item.get("best_sym_mae"),
                         "epochs_done": item["epochs_done"],
                         "completed": item.get("completed", item["epochs_done"] == item["config"]["epochs"]),
                         "seconds_sum": item["seconds_sum"], **item["config"]})
        except (OSError, KeyError, ValueError):
            continue
    rows.sort(key=lambda r: (r["best_mse"], r["id"]))
    if rows:
        path = Path(output_dir) / "leaderboard.csv"
        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    return rows


def top_a_configs(output_dir: Path, *, n=3, seed=DEFAULT_SEED, epochs=30):
    rows = []
    for cfg in campaign_a(seed, epochs):
        summary = Path(output_dir) / trial_id(cfg) / "summary.json"
        if not summary.exists():
            continue
        item = json.loads(summary.read_text())
        if item["epochs_done"] != epochs or item["config"] != asdict(cfg):
            continue
        rows.append((item["best_mse"], cfg))
    rows.sort(key=lambda x: x[0])
    if len(rows) < n:
        raise RuntimeError(f"Need {n} completed Campaign A configurations, found {len(rows)}")
    return [cfg for _, cfg in rows[:n]]


def confirmation_configs(output_dir: Path, *, n=5, seed=DEFAULT_SEED, epochs=30):
    rows = []
    for path in Path(output_dir).glob("*/summary.json"):
        item = json.loads(path.read_text())
        if item.get("epochs_done") != epochs or item["config"]["seed"] != seed:
            continue
        if item["config"]["epochs"] != epochs:
            continue
        rows.append((item["best_mse"], Trial(**item["config"])))
    rows.sort(key=lambda x: x[0])
    if len(rows) < n:
        raise RuntimeError(f"Need {n} completed configurations, found {len(rows)}")
    return [replace(cfg, seed=20261529) for _, cfg in rows[:n]]


def _validate_devices(devices):
    devices = [d.strip() for d in devices if d.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Supply unique worker devices")
    for name in devices:
        dev = torch.device(name)
        if dev.type != "cuda" or dev.index is None:
            raise ValueError("Parallel workers require explicit CUDA indices e.g. cuda:0")
        if not torch.cuda.is_available() or dev.index >= torch.cuda.device_count():
            raise RuntimeError(f"GPU unavailable: {name}")
    return devices


def _launch_workers(configs, *, budget, devices, root, output_dir, runtime, tag):
    """Start one process per assigned GPU and load its dataset once."""
    devices = _validate_devices(devices)
    configs = list(configs)
    if len({trial_id(c) for c in configs}) != len(configs):
        raise ValueError("Duplicate configuration would cause checkpoint race")
    run_dir = Path(output_dir) / "_launcher" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    # Assign trials round-robin so the split is reproducible.
    shards = [configs[i::len(devices)] for i in range(len(devices))]
    processes = []
    for i, (device, shard) in enumerate(zip(devices, shards)):
        if not shard:
            continue
        spec = {"device": device, "root": str(root), "output_dir": str(output_dir),
                "runtime": asdict(runtime), "max_epochs": budget,
                "configs": [asdict(c) for c in shard]}
        spec_file = run_dir / f"worker_{i}.json"
        log_file = run_dir / f"worker_{i}.log"
        atomic_json(spec_file, spec)
        log = log_file.open("w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                 "--action", "worker", "--spec", str(spec_file)],
                                stdout=log, stderr=subprocess.STDOUT, cwd=str(root))
        processes.append((proc, log, log_file, device))
        print(f"Started {device}: {len(shard)} trials -> {log_file}", flush=True)
    failures = []
    try:
        for proc, log, path, device in processes:
            rc = proc.wait()
            log.close()
            if rc:
                tail = "\n".join(path.read_text(errors="replace").splitlines()[-20:])
                failures.append(f"{device} exited {rc}\n{tail}")
    except KeyboardInterrupt:
        for proc, _, _, _ in processes:
            proc.terminate()
        for proc, log, _, _ in processes:
            proc.wait()
            log.close()
        raise
    if failures:
        raise RuntimeError("Worker failures (re-run to resume):\n" + "\n\n".join(failures))
    # Check every trial reached the rung's epoch budget before promoting it.
    dataset_ids = set()
    for cfg in configs:
        path = Path(output_dir) / trial_id(cfg) / "summary.json"
        if not path.exists():
            raise RuntimeError(f"Missing worker result: {trial_id(cfg)}")
        row = json.loads(path.read_text())
        if row["config"] != asdict(cfg) or row["epochs_done"] < budget:
            raise RuntimeError(f"Incomplete result: {trial_id(cfg)}")
        if row.get("runtime") not in (None, asdict(runtime)):
            raise RuntimeError(f"Mixed numerical runtimes: {trial_id(cfg)}")
        dataset_ids.add(row["dataset_id"])
    if len(dataset_ids) != 1:
        raise RuntimeError("Workers saw different frozen datasets")
    dataset_id = next(iter(dataset_ids))
    export_leaderboard_from_disk(Path(output_dir), dataset_id)
    return dataset_id


def run_campaign(configs, *, strategy, devices, root, output_dir, runtime,
                 keep=(40, 10), rungs=(3, 10, 30), campaign_name="a"):
    """Prune after each rung; only runs that reach the final budget are complete."""
    configs = list(configs)
    if strategy == "full":
        _launch_workers(configs, budget=configs[0].epochs, devices=devices, root=root,
                        output_dir=output_dir, runtime=runtime, tag=f"{campaign_name}_full")
        print(f"FULL campaign finished: {len(configs)} runs", flush=True)
        return configs
    if strategy != "halving":
        raise ValueError("Unknown strategy")
    if len(rungs) != 3 or not 0 < rungs[0] < rungs[1] < rungs[2] == configs[0].epochs:
        raise ValueError("Rungs must increase and end at the full horizon")
    if not 0 < keep[1] < keep[0] < len(configs):
        raise ValueError("Invalid reduction counts")
    state_dir = Path(output_dir) / "_launcher"
    state = state_dir / f"{campaign_name}_halving.json"
    initial = [trial_id(c) for c in configs]
    blueprint = {"initial": initial, "runtime": asdict(runtime),
                 "rungs": list(rungs), "keep": list(keep), "strategy": "halving"}
    if state.exists():
        old = json.loads(state.read_text())
        if any(old.get(k) != v for k, v in blueprint.items()):
            raise ValueError("Existing pruning manifest differs; cannot silently change design")
    else:
        old = {**blueprint, "stages": []}
        atomic_json(state, old)
    available = {trial_id(c): c for c in configs}
    survivors = configs
    dataset_id = None
    for j, budget in enumerate(rungs):
        if j < len(old["stages"]):
            stage = old["stages"][j]
            if stage["budget"] != budget or stage["entrants"] != [trial_id(c) for c in survivors]:
                raise ValueError("Pruning manifest inconsistent with candidate progression")
            dataset_id = stage["dataset_id"]
            promoted = stage["promoted"]
            survivors = [available[rid] for rid in promoted]
            print(f"Stage {j+1} already frozen; reusing {len(survivors)} promotions", flush=True)
            continue
        entrants = [trial_id(c) for c in survivors]
        print(f"Stage {j+1}: {len(survivors)} candidates to epoch {budget}", flush=True)
        this_id = _launch_workers(survivors, budget=budget, devices=devices,
                                  root=root, output_dir=output_dir, runtime=runtime,
                                  tag=f"{campaign_name}_rung_{budget}")
        if dataset_id is not None and dataset_id != this_id:
            raise RuntimeError("Frozen dataset changed between stages")
        dataset_id = this_id
        def stage_score(cfg):
            row = json.loads((Path(output_dir) / trial_id(cfg) / "summary.json").read_text())
            history = [h["val_mse_canonical"] for h in row["history"] if h["epoch"] <= budget]
            if len(history) != budget:
                raise RuntimeError("Missing historical rung metrics")
            return min(history)
        ranked = sorted(survivors, key=lambda c: (stage_score(c), trial_id(c)))
        survivors = ranked[:keep[j]] if j < len(keep) else ranked
        stage = {"budget": budget, "dataset_id": dataset_id, "entrants": entrants,
                 "promoted": [trial_id(c) for c in survivors],
                 "scores": {trial_id(c): stage_score(c) for c in ranked}}
        old["stages"].append(stage)
        atomic_json(state, old)
        print(f"Frozen Stage {j+1}: {len(survivors)} surviving candidates", flush=True)
    return survivors


def run_worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    device = torch.device(spec["device"])
    runtime = Runtime(**spec["runtime"])
    root = Path(spec["root"])
    train_x, train_cp, val_x, val_cp, blocked, dataset_id = load_data(root, device, runtime)
    preflight_mirror(train_x, device)
    lab = Laboratory(train_x=train_x, train_cp=train_cp, val_x=val_x, val_cp=val_cp,
                     mirror_blocked=blocked, output_dir=spec["output_dir"],
                     device=device, dataset_id=dataset_id, runtime=runtime)
    for i, c in enumerate(spec["configs"], 1):
        cfg = Trial(**c)
        print(f"{device}: {i}/{len(spec['configs'])} {trial_id(cfg)}", flush=True)
        lab.run(cfg, max_epochs=spec["max_epochs"])
    print(f"{device}: completed {len(spec['configs'])} trials", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="Horseflight N2-006 parallel/halving training")
    p.add_argument("--action", choices=["campaign", "worker", "profile", "benchmark"], default="campaign")
    p.add_argument("--spec", help="Internal worker task JSON")
    p.add_argument("--root", type=Path, default=Path(os.environ.get("HORSEFLIGHT_ROOT", ".")))
    p.add_argument("--output", type=Path)
    p.add_argument("--campaign", choices=["a", "b", "confirmation"], default="a")
    p.add_argument("--strategy", choices=["full", "halving"], default="full")
    p.add_argument("--devices", default="cuda:0,cuda:1,cuda:2")
    p.add_argument("--precision", choices=["fp32", "tf32", "bf16"], default="fp32")
    p.add_argument("--compile", dest="compile_mode", choices=["none", "default", "reduce-overhead"], default="none")
    p.add_argument("--dataset-device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--warmup", type=int, default=8)
    args = p.parse_args(argv)
    if args.action == "worker":
        if not args.spec:
            p.error("worker requires --spec")
        run_worker(args.spec)
        return
    root = args.root.expanduser().resolve()
    output = (args.output or (root / "research_data/horseflight_n2_5m_v1/checkpoints/n2_006")).resolve()
    runtime = Runtime(precision=args.precision, compile_mode=args.compile_mode,
                      dataset_device=args.dataset_device)
    devices = _validate_devices(args.devices.split(","))
    if args.action in ("profile", "benchmark"):
        dev = torch.device(devices[0])
        data = load_data(root, dev, runtime)
        preflight_mirror(data[0], dev)
        lab = Laboratory(train_x=data[0], train_cp=data[1], val_x=data[2], val_cp=data[3],
                         mirror_blocked=data[4], dataset_id=data[5],
                         output_dir=output, device=dev, runtime=runtime)
        trial = Trial(batch_size=2048, epochs=args.epochs)
        result = (lab.profile(trial, steps=args.steps, warmup=args.warmup)
                  if args.action == "profile" else
                  lab.benchmark(trial, steps=args.steps, warmup=args.warmup))
        print(json.dumps(result, indent=2))
        return
    if args.campaign == "a":
        configs = campaign_a(epochs=args.epochs)
    elif args.campaign == "b":
        configs = campaign_b(top_a_configs(output, epochs=args.epochs), epochs=args.epochs)
    else:
        configs = confirmation_configs(output, epochs=args.epochs)
    if args.strategy == "halving" and args.campaign != "a":
        p.error("Halving is currently supported for Campaign A only")
    if args.strategy == "halving" and args.epochs != 30:
        p.error("Default 3/10/30 halving rungs require --epochs 30")
    run_campaign(configs, strategy=args.strategy, devices=devices, root=root, output_dir=output,
                 runtime=runtime, campaign_name=args.campaign)


if __name__ == "__main__":
    main()
