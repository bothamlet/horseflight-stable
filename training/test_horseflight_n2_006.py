"""CPU checks for training, checkpoint resume, and trial promotion."""
import json
import tempfile
from dataclasses import replace, asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import horseflight_n2_006 as hf


def synthetic_laboratory(where):
    rng = np.random.default_rng(12)
    n, m = 64, 32
    x = np.full((n + m, 38), 781, dtype=np.int64)
    for i in range(n + m):
        ids = list(rng.choice(768, size=12, replace=False))
        if i % 2:
            ids.append(768)
        ids.extend([769 + i % 4, 773 + i % 8])
        x[i, :len(ids)] = sorted(ids)
    labels = rng.integers(-200, 200, size=n+m).astype('float32')
    return hf.Laboratory(
        train_x=torch.from_numpy(x[:n]), train_cp=torch.from_numpy(labels[:n]),
        val_x=torch.from_numpy(x[n:]), val_cp=torch.from_numpy(labels[n:]),
        mirror_blocked=torch.zeros(n, dtype=torch.bool), output_dir=where,
        device='cpu', dataset_id='fixture:v1', runtime=hf.Runtime(dataset_device='cpu'))


def test_contract():
    assert len(hf.campaign_a()) == 162
    assert len(hf.campaign_b(hf.campaign_a()[:3])) == 36
    assert len(set(hf.trial_id(c) for c in hf.campaign_a())) == 162
    torch.manual_seed(21)
    model = hf.HorseflightNNUE()
    x = torch.tensor([[4, 20, 768] + [781] * 35], dtype=torch.long)
    y = model(x)
    y.sum().backward()
    assert y.shape == (1,) and torch.isfinite(y).all()
    assert all(p.grad is not None for p in model.parameters())
    mirrored = hf.fast_mirror(x, hf.build_mirror_lut(torch.device('cpu')))
    assert hf.fast_mirror(mirrored, hf.build_mirror_lut(torch.device('cpu'))).equal(x.sort(dim=1).values)
    print('PASS feature/QAT/campaign contract')


def test_resume_and_profiler():
    with tempfile.TemporaryDirectory() as tmp:
        lab = synthetic_laboratory(tmp)
        cfg = hf.Trial(batch_size=16, epochs=2, mirror_p=.5, scheduler='cosine', seed=33)
        stage1 = lab.run(cfg, max_epochs=1, verbose=False)
        assert stage1['epochs_done'] == 1 and not stage1['completed']
        stage2 = lab.run(cfg, max_epochs=2, verbose=False)
        assert stage2['epochs_done'] == 2 and stage2['completed']
        cached = lab.run(cfg, verbose=False)
        assert cached['best_mse'] == stage2['best_mse']
        ck = torch.load(Path(tmp) / hf.trial_id(cfg) / 'last.pt', weights_only=True)
        assert ck['epoch'] == 2 and ck['runtime']['precision'] == 'fp32'
        with tempfile.TemporaryDirectory() as other:
            fresh = synthetic_laboratory(other).run(cfg, verbose=False)
            assert abs(fresh['best_mse'] - stage2['best_mse']) < 1e-9
        bench = lab.benchmark(replace(cfg, batch_size=8), steps=1, warmup=0)
        assert bench['examples_per_second'] > 0
        prof = lab.profile(replace(cfg, batch_size=8), steps=1, warmup=0)
        assert Path(prof['trace']).exists() and Path(prof['table']).exists()
        print('PASS checkpoint resume, repeatability, benchmark, CPU profiler')


def test_halving_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        cfgs = [replace(hf.Trial(epochs=30), seed=i) for i in range(162)]
        calls = []
        def fake_launch(configs, *, budget, output_dir, **kwargs):
            calls.append((len(configs), budget))
            for config in configs:
                rid = hf.trial_id(config)
                d = output_dir / rid
                d.mkdir(parents=True, exist_ok=True)
                # Seed 0 wins at every budget; keep per-epoch scores for rung checks.
                history = [{'epoch': e, 'val_mse_canonical': (config.seed + 1) / 10000 + (30-e)/100000}
                           for e in range(1, budget+1)]
                hf.atomic_json(d / 'summary.json', {'id': rid, 'config': asdict(config),
                                'dataset_id': 'fixture', 'epochs_done': budget,
                                'runtime': asdict(hf.Runtime()), 'history': history})
            return 'fixture'
        with patch.object(hf, '_launch_workers', fake_launch):
            winners = hf.run_campaign(cfgs, strategy='halving', devices=['cuda:0'],
                                      root=output, output_dir=output,
                                      runtime=hf.Runtime(), campaign_name='test')
            assert len(winners) == 10 and len({c.seed for c in winners}) == 10
            assert calls == [(162,3), (40,10), (10,30)], calls
            calls.clear()
            again = hf.run_campaign(cfgs, strategy='halving', devices=['cuda:0'],
                                    root=output, output_dir=output,
                                    runtime=hf.Runtime(), campaign_name='test')
            assert calls == [] and [hf.trial_id(c) for c in again] == [hf.trial_id(c) for c in winners]
            manifest = json.loads((output/'_launcher/test_halving.json').read_text())
            assert len(manifest['stages']) == 3
        print('PASS 162 -> 40 -> 10 pruning, persistent manifest, restart')


if __name__ == '__main__':
    test_contract()
    test_resume_and_profiler()
    test_halving_manifest()
    print('ALL CPU TESTS PASSED')
