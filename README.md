# Horseflight

Horseflight is a chess-engine project. **Phase 1A — Foundations** delivered the original,
reproducible chess-state layer. **Phase 1B — First Thought** adds an original replaceable
evaluation and a deterministic alpha-beta search with bounded quiescence, plus an `analyze`
CLI. Phase 1B is **accepted** after independent review and executable checks. **Mission 003**
delivers an **accepted functional baseline** for local play and analysis with the real engine.
**Mission 004 — Mock-inspired cockpit** is **accepted** after independent review and GPT's
executable/visual checks: a square blue/cream board, player bars, real captures/PGN and confirmed
Play resignation, with clocks, opening names and Stockfish comparison only as visibly inactive
grey placeholders. There is no transposition table, UCI interface, neural network, tablebase, GPU
work, parallel search, engine match or engine-strength/Elo claim.

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
| Evaluation | Replaceable `Evaluator` trait; original material plus pawn-advancement and minor-centralisation terms; material-only baseline; clamped score bands separating evaluation, mate and infinity |
| Search | Iterative-deepening full-window negamax alpha-beta; previous-PV and MVV/LVA ordering; bounded quiescence with all check evasions and quiet promotions; deterministic node budgets and truthful stop/claim/terminal reporting |
| Analysis CLI | `analyze` reports action, claim, score, completed depth, legal PV, main/quiescence nodes, stop reason, time and NPS, with depth/limit/FEN/move-history and ordering/quiescence toggles |
| Verification | Perft/divide, focused unit and integration tests, an independent development-only `shakmaty` comparison, an independent unpruned negamax reference and bounded exhaustive mate proofs |
| Cooperative cancellation | Standard-library `SearchControl`; `search_controlled` polls at node and claim-probe boundaries and reports `SearchStop::Cancelled`; plain `search` is unchanged and deterministic |
| Cockpit (Mission 003) | `web/server` Axum adapter and React/TypeScript/Vite presentation: local-only loopback host, server-authoritative legal moves/SAN/history/claims/search, play/analysis, guarded one-worker engine jobs, FEN/navigation/branching and manual annotations |
| Future boundary | `Position` + `Undo` still expose evaluator deltas; a full interface protocol, persistent database or strength programme is not implemented |

## Quick start

```sh
# Workspace-wide: core library plus the cockpit server crate.
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

## Local cockpit (Mission 003)

The cockpit is a separate workspace member. The core library still has an empty normal
dependency list; `horseflight-cockpit` adds the local HTTP/UI dependencies.

```sh
# One-time frontend dependency installation (committed lockfile)
npm --prefix web/frontend ci

# One-command local launch after installation: builds frontend + server, then serves
./scripts/run-cockpit.sh
# open http://127.0.0.1:7331
```

`HF_COCKPIT_PORT` selects another non-privileged loopback port. The server binds only
`127.0.0.1`, serves one same-origin app, validates Host/Origin/JSON body/config inputs, and has
no public bind, authentication, telemetry, remote asset, cloud service or production chess.js /
Stockfish dependency. FEN parsing remains structural rather than a reachability proof.

The browser presents server-approved state only. Horseflight is the sole authority for legal
destinations, move application, SAN, timeline/history, captures, PGN outcomes and claims; React
Chessboard is presentation and dnd-kit, not a rules engine. Manual arrows and square highlights are
local presentation and never change the revision or game. The Mission 004 sidebar adds real PGN
copy, backend-derived capture indicators and a confirmed, revision-guarded Play resignation while
preserving FEN/history/settings/annotations/search controls in accessible panels. Clocks, opening
names and Stockfish/Difference are inactive grey placeholders, not integrations.

```sh
# Frontend checks and real-browser tests. The e2e script rebuilds the frontend and release
# server first, then starts the real backend; Chrome is /usr/bin/google-chrome.
npm --prefix web/frontend test
npm --prefix web/frontend run build
npm --prefix web/frontend run e2e

# Engine/workspace regression gates
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps
cargo tree -e normal -p horseflight
python3 scripts/validate_research.py
python3 -m unittest discover -s scripts -p 'test*.py'
```

Evidence, exact commands, the engine-control API and remaining limits are in
[`docs/mission-003.md`](docs/mission-003.md); the Mission 004 implementation checkpoint is in
[`docs/mission-004.md`](docs/mission-004.md). Screenshots are local files under
`docs/evidence/mission-003/` and `docs/evidence/mission-004/`; nothing is published. The cockpit is
a testing baseline, not a strength claim.

## Correctness and scope boundaries

- Move generation is original; no external engine move-generation code is copied.
- Core engine production dependencies: none; the separate cockpit has web/UI dependencies.
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
- [`docs/phase-1b-first-thought.md`](docs/phase-1b-first-thought.md) — Phase 1B evaluation,
  search, draw policy, approximations, the current gate results, and the fixed-depth/fixed-node
  ordering and quiescence ablations with actual node counts and timings.
- [`docs/mission-003.md`](docs/mission-003.md) — the accepted cockpit baseline and latest
  workspace checks: 148 debug tests passed (2 deep tests ignored), all 150 release tests passed,
  17 frontend unit tests and 9 real-browser tests passed against the actual engine. Formatting,
  Clippy, docs, research validation, launch and the overflow regression also passed. At that
  checkpoint, visual polish and richer cancelled-result presentation were deferred.
- [`docs/mission-004.md`](docs/mission-004.md) — the accepted mock-inspired cockpit: 161 debug
  tests (2 ignored), 163 release tests, 20 frontend unit tests and 15 real-browser tests passed
  in GPT's independent acceptance. Screenshots are under `docs/evidence/mission-004/`.
  Browser verification is Chrome-only; lower sidebar controls can require vertical scrolling
  at laptop heights. Clocks, opening names and Stockfish comparison remain disabled placeholders.

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
