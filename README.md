# Horseflight

Horseflight is a chess-engine project. **Phase 1A — Foundations** delivered the original,
reproducible chess-state layer. **Phase 1B — First Thought** adds an original replaceable
evaluation and a deterministic alpha-beta search with bounded quiescence, plus an `analyze`
CLI. Phase 1B is **accepted** after independent review and executable checks. **Mission 003**
delivers an **accepted functional baseline** for local play and analysis with the real engine.
**Mission 004 — Mock-inspired cockpit** is **accepted** after independent review and GPT's
executable/visual checks: a square blue/cream board, player bars, real captures/PGN and confirmed
Play resignation, with clocks, opening names and Stockfish comparison only as visibly inactive
grey placeholders. The cockpit frontend, server and launcher are no longer part of the tracked
checkout: they are absent from fresh clones, and any retained copy is a git-ignored local file in
a clone that already has one. There is no transposition table, neural network, tablebase, GPU
work, parallel search or absolute or official rating claim. A bounded UCI match baseline and its
relative analysis now exist in [`docs/baseline-001.md`](docs/baseline-001.md); that
relative measurement is not an absolute, official, FIDE or CCRL rating.

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
| Evaluation | Replaceable `Evaluator` trait; original material plus pawn-advancement and minor-centralisation terms in `eval::classical`; material-only baseline; clamped score bands separating evaluation, mate and infinity |
| Neural boundary | Versioned 768-input binary piece-square schema v1 in `eval::features`, allocation-free extraction and delta updates; documented logical artifact contract and N2 integration plan in [`docs/neural-evaluation.md`](docs/neural-evaluation.md). No model or inference is implemented. |
| Search | Iterative-deepening full-window negamax alpha-beta; previous-PV and MVV/LVA ordering; bounded quiescence with all check evasions and quiet promotions; deterministic node budgets and truthful stop/claim/terminal reporting |
| Analysis CLI | `analyze` reports action, claim, score, completed depth, legal PV, main/quiescence nodes, stop reason, time and NPS, with depth/limit/FEN/move-history and ordering/quiescence toggles |
| Verification | Perft/divide, focused unit and integration tests, an independent development-only `shakmaty` comparison, an independent unpruned negamax reference and bounded exhaustive mate proofs |
| Cooperative cancellation | Standard-library `SearchControl`; `search_controlled` polls at node and claim-probe boundaries and reports `SearchStop::Cancelled`; plain `search` is unchanged and deterministic |
| Cockpit (Mission 003; retained local only) | Git-ignored `web/server` Axum adapter and React/TypeScript/Vite presentation that may be retained in existing local checkouts: local-only loopback host, server-authoritative legal moves/SAN/history/claims/search, play/analysis, guarded one-worker engine jobs, FEN/navigation/branching and manual annotations. Not tracked, not built by CI and absent from fresh clones |
| Future boundary | `Position` + `Undo` still expose evaluator deltas; a full interface protocol, persistent database or strength programme is not implemented |

## Quick start

```sh
# Engine workspace (the only tracked crate).
cargo test --workspace
cargo test --workspace --release -- --include-ignored
cargo clippy --workspace --all-targets -- -D warnings
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

```sh
# Search the start position
cargo run --release --bin analyze -- --depth 4

# Search a FEN with a node cap and both ablation toggles
cargo run --release --bin analyze -- \
  --fen 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1' \
  --depth 3 --nodes 200000 --no-ordering --no-quiescence

# Build real repetition history from coordinate moves
cargo run --release --bin analyze -- \
  --fen '7k/Q7/8/8/8/8/8/K7 b - - 0 1' \
  --moves 'h8g8 a1b1 g8h8 b1a1 h8g8 a1b1 g8h8 b1a1' --depth 2
```

`analyze` prints the action (move, draw claim, terminal result or unsearched fallback), the
side-to-move score (centipawns or mate distance), the completed depth, a legal PV, total
quiescence/main node counts, the stop reason, elapsed time and NPS; invalid options, FEN,
moves or limits exit nonzero. Scores are clamped so a custom evaluator cannot forge a mate,
node budgets are hard aggregate limits, and the caller's position/history are never mutated.

## Local cockpit (retained, untracked)

The Mission 003/004 cockpit is not part of the tracked checkout. Its `web/` frontend and server,
plus the `scripts/run-cockpit.sh` launcher, are git-ignored, retained only in checkouts that
already have them, and absent from fresh clones. CI does not install Node or build them. The core
library still has an empty normal dependency list; the retained server keeps its HTTP/UI
dependencies in `web/server/Cargo.toml` and remains an independently buildable crate through an
explicit `--manifest-path`.

Preservation is verified only in this checkout. In any other clone, back up `web/` and
`scripts/run-cockpit.sh` before pulling or checking out this removal: a normal Git update can
delete files that were formerly tracked, and `.gitignore` is not a backup.

```sh
# Only where the ignored cockpit files are retained locally.
# One-time frontend dependency installation (retained local lockfile)
npm --prefix web/frontend ci

# One-command local launch after installation: builds frontend + server, then serves
./scripts/run-cockpit.sh
# open http://127.0.0.1:7331
```

`HF_COCKPIT_PORT` selects another non-privileged loopback port. The server binds only
`127.0.0.1`, serves one same-origin app, validates Host/Origin/JSON body/config inputs, and has
no public bind, authentication, telemetry, remote asset, cloud service or production chess.js /
Stockfish dependency. FEN parsing remains structural rather than a reachability proof.

When present locally, the browser presents server-approved state only. Horseflight is the sole
authority for legal destinations, move application, SAN, timeline/history, captures, PGN outcomes
and claims; React Chessboard is presentation and dnd-kit, not a rules engine. Manual arrows and
square highlights are local presentation and never change the revision or game. The Mission 004
sidebar adds real PGN copy, backend-derived capture indicators and a confirmed, revision-guarded
Play resignation while preserving FEN/history/settings/annotations/search controls in accessible
panels. Clocks, opening names and Stockfish/Difference are inactive grey placeholders, not
integrations.

```sh
# Retained-local frontend checks and real-browser tests, only where the ignored files exist.
# The e2e script rebuilds the frontend and release server from the explicit server manifest,
# then starts the real backend; Chrome is /usr/bin/google-chrome. CI does not run these.
npm --prefix web/frontend test
npm --prefix web/frontend run build
npm --prefix web/frontend run e2e

# Engine/workspace regression gates
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps
cargo tree -e normal -p horseflight
python3 scripts/validate_research.py
python3 -m unittest discover -s scripts -p 'test*.py'
```

Historical evidence, exact commands, the engine-control API and remaining limits are archived in
[`docs/mission-003.md`](docs/mission-003.md); the Mission 004 implementation checkpoint is in
[`docs/mission-004.md`](docs/mission-004.md). Screenshots under `docs/evidence/mission-003/` and
`docs/evidence/mission-004/` are tracked historical evidence, not current verification. The
cockpit is a historical testing baseline, not a strength claim.

## Benchmark records

New benchmark runs are recorded under [`bench/`](bench/) as write-once artifacts:
an ordinary JSON config, a per-game CSV, a JSON summary and an integrity file,
regenerated from the raw Fastchess PGNs. The historical runner
(`bench/match.sh`) and analysis scripts are unchanged.

```sh
python3 -m bench.record verify --config bench/configs/baseline-001.json
python3 -m bench.record verify --config bench/configs/book-001.json
python3 -m bench.record_v2 verify --config bench/configs/baseline-001-v2-binding.json
python3 -m bench.build_manifest verify \
  --manifest bench/builds/n1-0906016-analyze-default.json
python3 -m bench.build_manifest verify \
  --manifest bench/builds/n1-0906016-analyze-depth3.json
python3 -m bench.build_manifest verify \
  --manifest bench/builds/n1-0906016-uci.json
```

Two v1 records exist: `baseline-001` (preserved 100-game historical match, no book) and
`book-001` (100-game book-enabled match whose raw batches and final derived analysis live
outside this checkout under `../book-001/`, with absolute digest-pinned inputs). The v2
`baseline-001-v2-binding` record imports one preserved Baseline001 batch to demonstrate
build/experiment binding; it is not a new match. Version 2 keeps v1 bytes unchanged.
`bench/builds/observations.json` links real analyze/UCI subprocess output to the
manifest build IDs; it is evidence, not a tournament.

`verify` re-reads the preserved Baseline001 games, recomputes the record and
byte-compares it; it does **not** rerun the match and makes no absolute rating
claim. Process, branch and evidence rules are in
[`docs/research-workflow.md`](docs/research-workflow.md); ownership and
deliberately excluded tooling are in
[`docs/technology-boundaries.md`](docs/technology-boundaries.md). Capability
history and lifecycle status are in [`docs/technology-tree.md`](docs/technology-tree.md);
release/build/experiment identities and the contributor workflow are in
[`docs/versioning.md`](docs/versioning.md).

## Correctness and scope boundaries

- Move generation is original; no external engine move-generation code is copied.
- Core engine production dependencies: none. The retained-local cockpit, when present, keeps its
  own web/UI dependencies outside the tracked workspace.
  `shakmaty` is in `[dev-dependencies]` only, for
  independent reference tests; it is not linked into the library or CLI.
- FEN validates structure and basic invariants, not full reachability. It preserves a
  structurally valid but uncapturable EP target and carries no history.
- Draw adjudication separates claimable threefold/fifty-move rules from automatic
  fivefold/seventy-five-move draws. Checkmate takes precedence. Insufficient material is
  deliberately conservative (K vs K, K+B vs K, K+N vs K, same-colour-bishop classes) and
  is **not** a complete FIDE dead-position proof; K+NN vs K and K+N vs K+N are not treated
  as dead.
- Search owns terminal and rule scoring; claimable threefold/fifty-move draws are treated as
  optional zero-score actions (current or intended-move claims) rather than forced ends, while
  automatic outcomes stay terminal with mate first. Quiescence is bounded by a capture-ply
  limit plus an absolute ply cap: a horizon approximation, not an exact tactical solve.
- `shakmaty` is GPL-3.0-or-later and development-only. No project license has been chosen.

## Verification

Human-readable checkpoints:

- [`docs/phase-1a-foundations.md`](docs/phase-1a-foundations.md) — the accepted Phase 1A
  snapshot: format, Clippy, debug/release tests, deep perft, the seeded `shakmaty` reference
  comparison, research validation and local perft CLI samples. Its evidence is unchanged.
- [`docs/neural-evaluation.md`](docs/neural-evaluation.md) — Mission N1 schema v1,
  artifact contract and the exact N2 encoder/inference/push-pop boundary. Interface only;
  no model, weights or inference exist.
- [`docs/technology-tree.md`](docs/technology-tree.md) — stable CORE/SEARCH/EVAL IDs,
  introduction commits, dependencies, tests and honest lifecycle status (integrated,
  measured-not-adopted, experimental-unmerged or planned).
- [`docs/versioning.md`](docs/versioning.md) — source revision vs engine release vs build
  identity vs experiment; SemVer tag policy, build-manifest rules and contributor workflow.
- [`docs/phase-1b-first-thought.md`](docs/phase-1b-first-thought.md) — Phase 1B evaluation,
  search, draw policy, approximations, the current gate results, and the fixed-depth/fixed-node
  ordering and quiescence ablations with actual node counts and timings.
- [`docs/mission-003.md`](docs/mission-003.md) — the historical accepted cockpit baseline and
  archived workspace checks: 148 debug tests passed (2 deep tests ignored), all 150 release tests
  passed, 17 frontend unit tests and 9 real-browser tests passed against the actual engine.
  Formatting, Clippy, docs, research validation, launch and the overflow regression also passed.
  Those counts describe the archived cockpit checkpoint, not the current engine-only checkout.
- [`docs/mission-004.md`](docs/mission-004.md) — the historical accepted mock-inspired cockpit:
  161 debug tests (2 ignored), 163 release tests, 20 frontend unit tests and 15 real-browser tests
  passed in GPT's independent acceptance. Those counts include the now-untracked cockpit and are
  archived, not current. Screenshots are under `docs/evidence/mission-004/`. Browser verification
  was Chrome-only; lower sidebar controls can require vertical scrolling at laptop heights.
  Clocks, opening names and Stockfish comparison remain disabled placeholders.
- [`docs/baseline-001.md`](docs/baseline-001.md) — the completed 100-game Baseline001 match and
  its relative-rating analysis, pending parent acceptance; this is not an absolute or
  official Elo claim.

Accepted Phase 1B verification summary (2026-10-22): `cargo fmt
--check`, Clippy `-D warnings`, `cargo test` (115 passed, 2 labeled deep-perft tests ignored
by default), `cargo test --release -- --include-ignored` (all 117 passed), warning-denied
`cargo doc`, the research validator, its 6 Python tests and `cargo tree -e normal` all pass.
All 57 Phase 1A test functions and the standard/deep perft results are preserved. Phase 1B
adds evaluation, search, draw-claim, quiescence, mate-proof, unpruned-reference and CLI tests;
its figures are local samples, not engine strength or benchmark evidence. Only Rust 1.96.0 was
installed and tested; `Cargo.toml` declares an inferred 1.83 minimum.

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
- [`docs/mission-002.md`](docs/mission-002.md) — the approved Phase 1B contract.
- [`docs/phase-1b-first-thought.md`](docs/phase-1b-first-thought.md) — Phase 1B
  architecture, measurements and limits.

The research did not choose an architecture, run engine matches, train a model or measure
an Elo gain. Nothing in this repository should be read as a strength claim or as evidence
that a research mechanism works.
