//! Evaluation interface and score ranges shared by search.
//!
//! [`classical`] provides the handcrafted and material-only evaluators.
//! [`n2`] contains the embedded NNUE. [`features`] defines a separate,
//! piece-placement-only feature schema.
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
pub mod n2;

pub use classical::{
    role_value, HandcraftedEvaluator, MaterialEvaluator, BISHOP_VALUE, KING_VALUE, KNIGHT_VALUE,
    PAWN_VALUE, QUEEN_VALUE, ROOK_VALUE,
};

use crate::position::{PieceChange, Position};

/// A replaceable position evaluator.
///
/// Implementors must return centipawns from the side to move's perspective and
/// should stay near the evaluation band; search clamps the result with
/// [`clamp_evaluation`] either way.
pub trait Evaluator {
    fn evaluate(&self, position: &Position) -> i32;
}

/// An evaluator that also maintains reversible state across make/unmake.
///
/// Search only calls this boundary when a caller explicitly supplies a
/// session; [`Evaluator`] stays the stateless contract, so every existing
/// evaluator and caller is unchanged. Implementations must hold an evaluation
/// for the *current* position after [`EvaluationSession::reset`], keep it
/// exactly in step through [`EvaluationSession::on_make`] and
/// [`EvaluationSession::on_unmake`], and return to the reset state when every
/// make has been matched by an unmake. [`EvaluationSession::evaluate`] returns
/// side-to-move centipawns, like [`Evaluator::evaluate`]; search clamps it with
/// [`clamp_evaluation`].
pub trait EvaluationSession {
    /// Side-to-move centipawns for the current position of the session.
    fn evaluate(&self, position: &Position) -> i32;
    /// Advance the session over one legal move.
    fn on_make(&mut self, changes: &[PieceChange], before: &Position, after: &Position);
    /// Reverse the most recent [`EvaluationSession::on_make`].
    fn on_unmake(&mut self);
    /// Rebuild the session from `root` and discard any recorded history.
    fn reset(&mut self, root: &Position);
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
