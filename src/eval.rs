//! Original, replaceable position evaluation.
//!
//! Search only ever sees a bounded integer score through the [`Evaluator`]
//! trait, so the handcrafted terms below can be replaced (material-only, a
//! test double, a future incremental evaluator) without touching search code.
//! Those terms are original and deliberately small; no engine piece-square
//! tables or tuned constants are copied.
//!
//! # Score bands and sign
//!
//! Every score is integer centipawns from the **side to move's** perspective.
//! Evaluation scores are clamped to [`EVAL_MIN`]..=[`EVAL_MAX`] (±30 000 cp)
//! by [`clamp_evaluation`], which search applies to every evaluator result. A
//! custom evaluator therefore cannot overflow window arithmetic or impersonate
//! a proven mate. Mate scores occupy `MATE_SCORE - ply` for a forced mate
//! delivered in `ply` half-moves from the scored node, so they are always
//! strictly above the evaluation band while `ply <= MATE_MAX_PLY`. [`INFINITY`]
//! brackets the search window outside both bands.
//!
//! # Terms
//!
//! - Material: pawn 100, knight 320, bishop 330, rook 500, queen 900, king 0.
//! - Pawn advancement: 6 cp per rank advanced from the pawn's own start rank.
//! - Minor centralisation: a knight or bishop earns 4 cp per step closer than
//!   three to the nearest of d4/e4/d5/e5.
//!
//! Every term is colour- and rank-mirror symmetric: reflecting the board
//! vertically and swapping colours negates the evaluation.

use crate::board::Board;
use crate::position::Position;
use crate::types::{Color, Role, Square};

/// Centipawn value of a pawn.
pub const PAWN_VALUE: i32 = 100;
/// Centipawn value of a knight.
pub const KNIGHT_VALUE: i32 = 320;
/// Centipawn value of a bishop.
pub const BISHOP_VALUE: i32 = 330;
/// Centipawn value of a rook.
pub const ROOK_VALUE: i32 = 500;
/// Centipawn value of a queen.
pub const QUEEN_VALUE: i32 = 900;
/// Centipawn value of a king; kings are never material.
pub const KING_VALUE: i32 = 0;

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

/// A replaceable position evaluator.
///
/// Implementors must return centipawns from the side to move's perspective and
/// should stay near the evaluation band; search clamps the result with
/// [`clamp_evaluation`] either way.
pub trait Evaluator {
    fn evaluate(&self, position: &Position) -> i32;
}

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

/// Material and modest positional terms.
#[derive(Clone, Copy, Debug, Default)]
pub struct HandcraftedEvaluator;

impl Evaluator for HandcraftedEvaluator {
    fn evaluate(&self, position: &Position) -> i32 {
        let board = position.board();
        let score = colour_score(board, Color::White) - colour_score(board, Color::Black);
        signed_for_side_to_move(score, position)
    }
}

/// Material only: a controlled baseline for tests and ablations.
#[derive(Clone, Copy, Debug, Default)]
pub struct MaterialEvaluator;

impl Evaluator for MaterialEvaluator {
    fn evaluate(&self, position: &Position) -> i32 {
        let board = position.board();
        let score = material(board, Color::White) - material(board, Color::Black);
        signed_for_side_to_move(score, position)
    }
}

const PAWN_ADVANCEMENT_BONUS: i32 = 6;
const MINOR_CENTRE_BONUS: i32 = 4;

fn signed_for_side_to_move(white_perspective: i32, position: &Position) -> i32 {
    match position.side_to_move() {
        Color::White => white_perspective,
        Color::Black => -white_perspective,
    }
}

fn colour_score(board: Board, color: Color) -> i32 {
    material(board, color) + positional(board, color)
}

fn material(board: Board, color: Color) -> i32 {
    let mut total = 0;
    for role in Role::ALL {
        total += role_value(role) * board.pieces(color, role).count() as i32;
    }
    total
}

fn positional(board: Board, color: Color) -> i32 {
    let mut total = 0;
    for square in board.pieces(color, Role::Pawn).iter() {
        total += PAWN_ADVANCEMENT_BONUS * pawn_advancement(color, square);
    }
    for role in [Role::Knight, Role::Bishop] {
        for square in board.pieces(color, role).iter() {
            total += MINOR_CENTRE_BONUS * (3 - centre_distance(square));
        }
    }
    total
}

/// Centipawn value of one piece of `role`.
pub const fn role_value(role: Role) -> i32 {
    match role {
        Role::Pawn => PAWN_VALUE,
        Role::Knight => KNIGHT_VALUE,
        Role::Bishop => BISHOP_VALUE,
        Role::Rook => ROOK_VALUE,
        Role::Queen => QUEEN_VALUE,
        Role::King => KING_VALUE,
    }
}

/// Ranks advanced from the pawn's own start rank (0..=5 for a legal pawn).
fn pawn_advancement(color: Color, square: Square) -> i32 {
    match color {
        Color::White => square.rank() as i32 - 1,
        Color::Black => 6 - square.rank() as i32,
    }
}

/// Chebyshev distance to the nearest of d4, e4, d5, e5 (0..=3).
fn centre_distance(square: Square) -> i32 {
    let file_gap = gap_from_centre(square.file());
    let rank_gap = gap_from_centre(square.rank());
    file_gap.max(rank_gap)
}

fn gap_from_centre(axis: u8) -> i32 {
    let axis = axis as i32;
    if axis < 3 {
        3 - axis
    } else if axis > 4 {
        axis - 4
    } else {
        0
    }
}
