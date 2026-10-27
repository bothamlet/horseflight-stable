//! Handcrafted classical evaluation terms.
//!
//! This module owns the current playing evaluation: material plus the original
//! pawn-advancement and minor-centralisation terms. The arithmetic here is
//! deliberately preserved exactly from the pre-split implementation; it is a
//! behavioural fixture, not a tuning surface.
//!
//! # Terms
//!
//! - Material: pawn 100, knight 320, bishop 330, rook 500, queen 900, king 0.
//! - Pawn advancement: 6 cp per rank advanced from the pawn's own start rank.
//! - Minor centralisation: a knight or bishop earns 4 cp per step closer than
//!   three to the nearest of d4/e4/d5/e5.
//!
//! Every white-relative term is colour- and rank-mirror symmetric: reflecting
//! the board vertically and swapping colours negates the white-relative score.
//! Since the evaluator returns a side-to-move score, swapping the side to move
//! together with the board leaves the returned score unchanged.

use super::Evaluator;
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
