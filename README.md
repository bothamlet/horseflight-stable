# Horseflight

Horseflight is a Rust chess engine with a UCI interface and command-line tools for analysis and move-generation testing. It uses classical evaluation by default and can switch to an embedded, quantized neural evaluator (NNUE).

## Run

With Rust 1.85 or newer:

```sh
cargo build --release --locked
./target/release/analyze --depth 5 --moves 'e2e4 e7e5'
./target/release/perft --divide 4
```

Add `target/release/uci` to a UCI-compatible chess GUI. Classical evaluation is the default; select `nnue` under the engine’s `Eval` option to try the neural evaluator. You can also run the binary directly:

```text
uci
setoption name Eval value nnue
isready
position startpos moves e2e4 e7e5
go depth 5
```

Wait for `bestmove`, then send `quit` to exit.

## Engine

- **Board and rules:** Square array and occupancy bitboards, reversible moves, incremental Zobrist hashing, and history-aware draw claims.
- **Search:** Iterative-deepening negamax with alpha-beta pruning, move ordering, bounded quiescence, and depth, node, and clock limits.
- **Evaluation:** A handcrafted evaluator and an opt-in 781 → 256 → 32 → 1 integer NNUE. The neural accumulator updates on make/unmake; AVX2 is detected at runtime with a scalar fallback. The Rust engine has no runtime crate dependencies.

The bundled weights were trained on the [Lichess Open Database](https://database.lichess.org/). The [training notebook](training/nnue-training.ipynb) documents a separate experiment on GPU throughput, hyperparameter tuning, and mirror augmentation. It does not reproduce the bundled weights.

## Verify

```sh
cargo fmt --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --release --all-features
cargo test --locked --release --test perft -- --ignored
```

Tests cover move generation against perft and an independent rules library, search and draw handling, the UCI protocol, and exact scalar/AVX2 agreement.
