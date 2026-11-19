//! Exact agreement between scalar and AVX2 inference.
//!
//! The scalar [`forward_trace`] is the independent arithmetic oracle: it keeps
//! the original `i32` loops and the asset's `i8`/`i16` tensor types. Every test
//! here compares [`forward_trace_optimized`] (the AVX2/scalar dispatched path
//! that production [`forward`] uses) against it, never against another copy of
//! the same kernel.
//!
//! Coverage:
//!
//! - all 144 supplied golden fixtures, every tensor: `a1`, `h`, `a2`, output
//!   numerator, and the final centipawn score;
//! - a deterministic legal-move walk (500 plies from several starts) so the
//!   comparison sees incremental/session-shaped accumulators as well as the
//!   fixtures' rebuilds;
//! - the incremental [`Session`] at every node of that walk, so the optimized
//!   head is compared exactly where search calls it.

use std::fs;
use std::path::Path;

use horseflight::eval::n2::session::Session;
use horseflight::eval::n2::{
    self, forward, forward_numerator, forward_trace, forward_trace_optimized, Accumulator,
    N2Evaluator, ACCUMULATOR_SIZE, HIDDEN_SIZE,
};
use horseflight::eval::{EvaluationSession, Evaluator};
use horseflight::{Position, STARTPOS_FEN};

/// Extract the `"fen":"..."` value without a JSON dependency.
fn fen_field(line: &str) -> &str {
    let marker = "\"fen\":";
    let start = line.find(marker).expect("golden line has a fen") + marker.len();
    let rest = line[start..].trim_start();
    let rest = rest.strip_prefix('"').expect("fen opens with a quote");
    let end = rest.find('"').expect("fen closes");
    &rest[..end]
}

fn golden_positions() -> Vec<Position> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/n2-002/goldens.jsonl");
    let text = fs::read_to_string(&path).expect("read goldens");
    let positions: Vec<Position> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| Position::from_fen(fen_field(line)).expect("valid golden fen"))
        .collect();
    assert_eq!(positions.len(), 144, "expected all 144 supplied goldens");
    positions
}

fn assert_trace_matches(label: &str, position: &Position) {
    let accumulator = Accumulator::from_position(position);
    let scalar = forward_trace(&accumulator);
    let optimized = forward_trace_optimized(&accumulator);
    assert_eq!(
        scalar.a1.len(),
        ACCUMULATOR_SIZE,
        "{label}: a1 length contract"
    );
    assert_eq!(scalar.h.len(), HIDDEN_SIZE, "{label}: h length contract");
    assert_eq!(scalar.a2.len(), HIDDEN_SIZE, "{label}: a2 length contract");
    assert_eq!(
        scalar, optimized,
        "{label}: optimized trace differs from scalar oracle"
    );
    assert_eq!(
        forward_numerator(&accumulator),
        scalar.output_numerator,
        "{label}: forward_numerator differs from oracle"
    );
    assert_eq!(
        forward(&accumulator),
        n2::div_ties_even(scalar.output_numerator * 400, 32_512),
        "{label}: forward centipawn conversion differs from the oracle numerator"
    );
}

#[test]
fn all_144_goldens_match_through_the_dispatched_head() {
    for (row, position) in golden_positions().iter().enumerate() {
        assert_trace_matches(&format!("golden row {row}"), position);
    }
}

#[test]
fn stateless_evaluator_matches_the_scalar_oracle_on_all_goldens() {
    let evaluator = N2Evaluator;
    for (row, position) in golden_positions().iter().enumerate() {
        let accumulator = Accumulator::from_position(position);
        let expected = {
            let white_relative = n2::forward(&accumulator);
            if position.side_to_move().is_white() {
                white_relative
            } else {
                -white_relative
            }
        };
        assert_eq!(
            evaluator.evaluate(position),
            horseflight::eval::clamp_evaluation(expected),
            "golden row {row}: stateless evaluator differs"
        );
    }
}

/// Deterministic SplitMix64 stream for reproducible walks.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

#[test]
fn deterministic_legal_walk_matches_at_every_position() {
    let starts = [
        STARTPOS_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ];
    for (seed, fen) in starts.iter().enumerate() {
        let mut rng = Rng(0xA2_0000 + seed as u64);
        let mut position = Position::from_fen(fen).expect("walk start");
        let mut session = Session::new(&position);
        for ply in 0..500 {
            let accumulator = Accumulator::from_position(&position);
            assert_eq!(
                forward_trace(&accumulator),
                forward_trace_optimized(&accumulator),
                "seed {seed} ply {ply}: dispatched trace differs"
            );
            assert!(
                session.matches_rebuild(&position),
                "session drift at ply {ply}"
            );
            let white_relative = forward(&accumulator);
            let expected = if position.side_to_move().is_white() {
                white_relative
            } else {
                -white_relative
            };
            assert_eq!(
                session.evaluate(&position),
                horseflight::eval::clamp_evaluation(expected),
                "seed {seed} ply {ply}: session evaluation differs"
            );

            let legal = position.legal_moves();
            if legal.is_empty() {
                position = Position::from_fen(fen).expect("restart");
                session = Session::new(&position);
                continue;
            }
            let movement = legal[rng.below(legal.len())];
            let before = position;
            let undo = position.play(movement).expect("legal move");
            session.on_make(undo.piece_changes(), &before, &position);
            if session.depth() + 4 >= horseflight::eval::n2::session::MAX_SESSION_DEPTH {
                // Restart the incremental state at the current position; the
                // walk is long enough to exceed the engine's undo stack.
                session = Session::new(&position);
            }
        }
    }
}
