//! Versioned binary piece-square feature schema for a future neural evaluator.
//!
//! Schema v1 is a fixed **768-input** encoding for one colour perspective:
//! two relative colours × six roles × 64 squares. Each input is binary (`0.0`
//! or `1.0`). It describes only piece placement, not whose turn it is,
//! castling rights, en passant, clocks or history. See [`SCHEMA_ID`] and
//! [`SCHEMA_VERSION`] for the exact identity a model artifact must record.
//!
//! # Index layout
//!
//! ```text
//! index = (relative_color * 6 + role) * 64 + oriented_square
//! relative_color = 0 for a piece of the perspective's colour, 1 for the enemy
//! role = pawn 0, knight 1, bishop 2, rook 3, queen 4, king 5
//! oriented_square = square.index() for the white perspective,
//!                   square.index() ^ 56 for the black perspective
//! ```
//!
//! `Square` is already `a1 = 0 .. h8 = 63`, so the black perspective vertically
//! flips the rank: `a1` maps to index `0 ^ 56 = 56`, the `a8` square index,
//! without changing files.
//! Both kings are ordinary inputs in the role-5 block; no king-relative
//! encoding exists in v1.
//!
//! # Iteration and determinism
//!
//! [`extract_into`] clears the output and scans squares in ascending index order
//! `a1 .. h8`; for each occupied square it sets exactly one input. A structurally
//! valid [`Position`] cannot place two pieces on one square, so setting inputs is
//! order-independent in practice. The scan visits every square and never
//! assumes a maximum piece count: promoted-material positions with far more than
//! 32 pieces are encoded exactly.
//!
//! # Dual perspective
//!
//! v1 is one perspective. A dual-perspective evaluator extracts the same
//! position once with [`Color::White`] and once with [`Color::Black`]. These are
//! not two halves of one vector; they are two full 768-input vectors whose
//! friendly/enemy blocks are swapped and whose board coordinates are oriented
//! per perspective.
//!
//! # Versioning
//!
//! Any change to layout, dimension, role order, perspective handling or
//! inclusion of state features requires a new [`SCHEMA_VERSION`] and a new
//! artifact contract. King-relative features, side-to-move inputs, castling, en
//! passant, clocks or history are deliberately absent from v1; adding any of
//! them is a version change, not an in-place extension.

use crate::position::{PieceChange, Position};
use crate::types::{Color, Role, Square};

/// Stable schema identifier recorded by a model artifact.
pub const SCHEMA_ID: &str = "horseflight-piece-square";
/// Schema version. Increment for any layout or content change.
pub const SCHEMA_VERSION: u32 = 1;
/// Colour perspectives, in artifact tensor order: white then black.
pub const PERSPECTIVES: [Color; 2] = [Color::White, Color::Black];
/// Role order inside each relative-colour block.
pub const ROLES: [Role; 6] = Role::ALL;
/// Number of binary inputs per perspective.
pub const FEATURE_DIMENSION: usize = PERSPECTIVES.len() * ROLES.len() * 64;
/// One perspective's binary feature vector.
pub type FeatureVector = [f32; FEATURE_DIMENSION];

/// Index of one piece input for `perspective`.
///
/// Squares are oriented by perspective: white uses `a1 = 0 .. h8 = 63`; black
/// uses `square ^ 56`, which flips the rank. The perspective's own pieces use
/// `relative_color = 0` ("friendly") and the opponent's use `1` ("enemy").
pub const fn piece_feature_index(
    piece_color: Color,
    role: Role,
    square: Square,
    perspective: Color,
) -> usize {
    let relative_color = if piece_color.index() == perspective.index() {
        0
    } else {
        1
    };
    let oriented_square = if perspective.is_white() {
        square.index() as usize
    } else {
        (square.index() ^ 56) as usize
    };
    (relative_color * ROLES.len() + role.index()) * 64 + oriented_square
}

/// Extract one perspective's binary vector without allocating.
pub fn extract(position: &Position, perspective: Color) -> FeatureVector {
    let mut features = [0.0; FEATURE_DIMENSION];
    extract_into(position, perspective, &mut features);
    features
}

/// Fill `features` with one perspective's binary vector.
///
/// `features` is cleared first. Squares are scanned in ascending `a1 .. h8`
/// order; every occupied square sets exactly one input to `1.0`.
pub fn extract_into(position: &Position, perspective: Color, features: &mut FeatureVector) {
    features.fill(0.0);
    let board = position.board();
    for index in 0..64 {
        let square = Square::from_index(index);
        if let Some(piece) = board.piece_at(square) {
            features[piece_feature_index(piece.color, piece.role, square, perspective)] = 1.0;
        }
    }
}

/// Apply one make/unmake piece delta to an extracted vector.
///
/// Clear the `before` input, then set the `after` input. Castling produces
/// several deltas, en passant changes three squares (origin, destination and the
/// captured pawn square), and promotions change the moving piece's role;
/// applying every delta forward reproduces a full rebuild.
pub fn apply_change(features: &mut FeatureVector, perspective: Color, change: &PieceChange) {
    if let Some(before) = change.before {
        features[piece_feature_index(before.color, before.role, change.square, perspective)] = 0.0;
    }
    if let Some(after) = change.after {
        features[piece_feature_index(after.color, after.role, change.square, perspective)] = 1.0;
    }
}

/// Reverse one piece delta: clear the `after` input, then set the `before`.
///
/// Apply deltas in reverse order to walk an extracted vector back to the
/// pre-move board exactly.
pub fn revert_change(features: &mut FeatureVector, perspective: Color, change: &PieceChange) {
    if let Some(after) = change.after {
        features[piece_feature_index(after.color, after.role, change.square, perspective)] = 0.0;
    }
    if let Some(before) = change.before {
        features[piece_feature_index(before.color, before.role, change.square, perspective)] = 1.0;
    }
}
