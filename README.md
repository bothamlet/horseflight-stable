# Horseflight

Horseflight is a chess-engine project. **Phase 1A — Foundations** implements the original,
reproducible chess-state layer; it does not yet play. There is no search, evaluation, UCI
interface, neural network, tablebase, GPU work or engine-strength claim.

The earlier research portfolio remains in this repository as history and reference; it is
not an architecture commitment and none of its experiments were run.

## What exists now

A dependency-free Rust library plus a small development CLI:

| Area | Implementation |
|---|---|
| Representation | Typed colours, roles, pieces, squares and coordinate moves; hybrid square array + role/colour occupancy bitboards with a single piece-update helper |
| Move generation | Original ray-walking sliding attacks; candidate generation followed by own-king-safety filtering; standard castling, en passant and all promotions |
| State transitions | In-place make/unmake; fixed-size compact undo; exact restoration of pieces, bitboards, side, rights, EP, clocks and hash; crate-private unchecked transition |
| Evaluator seam | Fixed-size before/after piece deltas for ordinary moves, captures, en passant, castling and promotions |
| Hashing | Deterministic SplitMix64-seeded Zobrist; incremental update and independent full recomputation; legal-EP-only repetition identity |
| Rules | FEN parsing/serialization, checkmate, stalemate, conservative insufficient material, threefold/fifty-move claims, automatic fivefold/seventy-five-move draws, mate precedence |
| Verification | Perft/divide, focused unit and integration tests, and an independent development-only `shakmaty` comparison |
| Future boundary | `Position` + `Undo` are ready for a future search/evaluator; `Game` is the history and draw-rule layer |

## Quick start

```sh
cargo test
cargo test --release -- --include-ignored
cargo clippy --all-targets -- -D warnings
cargo fmt --check

# Standard perft
cargo run --release --bin perft -- 5
cargo run --release --bin perft -- --divide 4

# Perft from a FEN
cargo run --release --bin perft -- \
  --fen 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1' 4
```

`perft` exits nonzero with a readable error for an invalid FEN, missing depth or a depth
above the development cap. Depth zero is one node.

## Correctness and scope boundaries

- Move generation is original; no external engine move-generation code is copied.
- Production dependencies: none. `shakmaty` is in `[dev-dependencies]` only, for
  independent reference tests; it is not linked into the library or CLI.
- FEN validates structure and basic invariants, not full reachability. It preserves a
  structurally valid but uncapturable EP target and carries no history.
- Draw adjudication separates claimable threefold/fifty-move rules from automatic
  fivefold/seventy-five-move draws. Checkmate takes precedence. Insufficient material is
  deliberately conservative (K vs K, K+B vs K, K+N vs K, same-colour-bishop classes) and
  is **not** a complete FIDE dead-position proof; K+NN vs K and K+N vs K+N are not treated
  as dead.
- `shakmaty` is GPL-3.0-or-later and development-only. No project license has been chosen.

## Verification

The human-readable checkpoint is
[`docs/phase-1a-foundations.md`](docs/phase-1a-foundations.md). It records the exact
commands and results: format, Clippy, debug/release tests, deep perft, reference
comparison seed/count, research validation, and local release CLI timing/NPS samples for
start position perft 5 and Kiwipete perft 4. Those timings are local perft samples, not
engine strength or a benchmark comparison.

Current verification summary (2026-10-22): `cargo fmt --check`, Clippy `-D warnings`,
`cargo test` (55 passed, 2 labeled deep-perft tests ignored by default),
`cargo test --release -- --include-ignored` (all 57 passed), warning-denied `cargo doc`,
the research validator and its 6 Python tests all pass. The independent seeded reference comparison covers 3,915 random-walk positions across
12 standard/special-rule FENs, plus dedicated white/black EP rank-x-ray and expanded
material fixtures. Only Rust 1.96.0 was installed and tested; `Cargo.toml` declares an
inferred 1.83 minimum.

## Historical research portfolio

The research catalogue is preserved and remains the original research deliverable.
[`research/README.md`](research/README.md) is the historical snapshot of the accepted research
portfolio front page, with working links for its new location.

| File | Contents |
|---|---|
| [`research/README.md`](research/README.md) | Historical research-portfolio front page and framing |
| [`research/grounded.md`](research/grounded.md) | A001–A100 grounded mechanisms |
| [`research/unconventional.md`](research/unconventional.md) | B001–B100 unconventional mechanisms |
| [`research/priorities.md`](research/priorities.md) | Twenty ranked options, E1–E5 experiments |
| [`research/methodology.md`](research/methodology.md) | Evidence, match and statistics methodology |
| [`research/sources.md`](research/sources.md) | 164-source register |
| [`scripts/validate_research.py`](scripts/validate_research.py) | Structural research validator |
| [`scripts/test_validate_research.py`](scripts/test_validate_research.py) | Validator negative-fixture tests |

Historical documents:

- [`docs/implementation.md`](docs/implementation.md) — the accepted research handoff
  (research-only scope; superseded as an implementation description by Phase 1A).
- [`docs/mission-001.md`](docs/mission-001.md) — the approved Phase 1A contract and
  verification checkpoint pointer.
- [`docs/phase-1a-foundations.md`](docs/phase-1a-foundations.md) — Phase 1A architecture,
  limits and evidence.

The research did not choose an architecture, run engine matches, train a model or measure
an Elo gain. Nothing in this repository should be read as a strength claim or as evidence
that a research mechanism works.
