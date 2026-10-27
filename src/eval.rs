//! Original, replaceable position evaluation.
//!
//! Search only ever sees a bounded integer score through the [`Evaluator`]
//! trait, so the handcrafted terms in [`classical`] can be replaced
//! (material-only, a test double, a full-rebuild neural evaluator) without
//! touching search code.
//! Those terms are original and deliberately small; no engine piece-square
//! tables or tuned constants are copied.
//!
//! This module is the evaluation boundary. It owns the [`Evaluator`] trait,
//! score bands, mate/score predicates and the clamp search applies to every
//! evaluator result. The concrete handcrafted and material evaluators live in
//! [`classical`] and are re-exported here for compatibility, so existing
//! `eval::HandcraftedEvaluator` and crate-root paths keep working.
//!
//! [`features`] defines the versioned binary piece-square input schema a future
//! neural evaluator will consume. It is intentionally independent of the
//! classical terms and of game state not represented by piece placement.
//!
//! # Score bands and sign
//!
//! Every score is integer centipawns from the **side to move's** perspective.
//! Evaluation scores are clamped to [`EVAL_MIN`]..=[`EVAL_MAX`] (±30 000 cp)
//! by [`clamp_evaluation`], which search applies to every evaluator result. A
//! custom evaluator therefore cannot overflow window arithmetic or impersonate
//! a proven mate. Mate scores are **root-relative**: a terminal node at ply
//! `p` half-moves from the root is scored `-MATE_SCORE + p` for checkmate
//! against the side to move, and the final root score encodes the root's
//! distance to mate rather than a distance from every scored node. Mate scores
//! are always strictly above the evaluation band while `p <= MATE_MAX_PLY`.
//! [`INFINITY`] brackets the search window outside both bands.
//!
//! # Terms
//!
//! The current classical evaluator is documented in [`classical`]: material,
//! pawn advancement and minor centralisation. The white-relative terms are
//! mirror symmetric: reflecting the board vertically and swapping colours
//! negates the white-relative score. Because the returned score is from the
//! side to move's perspective, swapping both the board and the side to move
//! leaves that returned score unchanged.

pub mod classical;
pub mod features;

pub use classical::{
    role_value, HandcraftedEvaluator, MaterialEvaluator, BISHOP_VALUE, KING_VALUE, KNIGHT_VALUE,
    PAWN_VALUE, QUEEN_VALUE, ROOK_VALUE,
};

use crate::position::Position;

/// A replaceable position evaluator.
///
/// Implementors must return centipawns from the side to move's perspective and
/// should stay near the evaluation band; search clamps the result with
/// [`clamp_evaluation`] either way.
pub trait Evaluator {
    fn evaluate(&self, position: &Position) -> i32;
}

/// Largest score a proven mate can produce at the root.
pub const MATE_SCORE: i32 = 32_000;
/// Mate scores are only produced within this many plies of the root.
pub const MATE_MAX_PLY: u32 = 128;
/// Lowest positive mate score: a mate at the [`MATE_MAX_PLY`] ply cap.
pub const MATE_BAND_FLOOR: i32 = MATE_SCORE - MATE_MAX_PLY as i32;
/// Search-window bound outside the evaluation and mate bands.
pub const INFINITY: i32 = 32_500;
/// Upper bound for a static evaluation score.
pub const EVAL_MAX: i32 = 30_000;
/// Lower bound for a static evaluation score.
pub const EVAL_MIN: i32 = -EVAL_MAX;

/// Clamp a raw evaluation into [`EVAL_MIN`]..=[`EVAL_MAX`].
pub fn clamp_evaluation(raw: i32) -> i32 {
    raw.clamp(EVAL_MIN, EVAL_MAX)
}

/// Whether `score` is a proven-mate score rather than a static evaluation.
///
/// Only the two bounded mate intervals are admitted:
/// [`MATE_BAND_FLOOR`]..=[`MATE_SCORE`] for a mate for the side to move and
/// `-MATE_SCORE`..=`-MATE_BAND_FLOOR` for a mate against it. [`INFINITY`],
/// evaluation scores, out-of-band integers and the `i32` extremes are all
/// rejected, so this predicate is total and never a wrapper for "very large".
pub const fn is_mate_score(score: i32) -> bool {
    (score >= MATE_BAND_FLOOR && score <= MATE_SCORE)
        || (score <= -MATE_BAND_FLOOR && score >= -MATE_SCORE)
}

/// Plies to the forced mate encoded by `score`, if it is a mate score.
///
/// Returns `None` outside the two mate intervals, so the `abs` here can never
/// overflow and the distance is always `0..=MATE_MAX_PLY`.
pub const fn mate_distance_plies(score: i32) -> Option<u32> {
    if is_mate_score(score) {
        Some((MATE_SCORE - score.abs()) as u32)
    } else {
        None
    }
}
