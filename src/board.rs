use core::fmt;

use crate::bitboard::Bitboard;
use crate::types::{Color, Piece, Role, Square};

/// Hybrid board storage: a square array for fast point queries and
/// piece/colour occupancy bitboards for set operations.
///
/// The three representations are kept consistent only through
/// `Board::set_piece`; no caller outside this module mutates a field directly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Board {
    squares: [Option<Piece>; 64],
    by_role: [Bitboard; 6],
    by_color: [Bitboard; 2],
}

impl Board {
    pub const fn empty() -> Board {
        Board {
            squares: [None; 64],
            by_role: [Bitboard::EMPTY; 6],
            by_color: [Bitboard::EMPTY; 2],
        }
    }

    pub fn piece_at(self, square: Square) -> Option<Piece> {
        self.squares[square.index() as usize]
    }

    pub const fn by_role(self, role: Role) -> Bitboard {
        self.by_role[role.index()]
    }

    pub const fn by_color(self, color: Color) -> Bitboard {
        self.by_color[color.index()]
    }

    pub const fn pieces(self, color: Color, role: Role) -> Bitboard {
        self.by_color(color).intersect(self.by_role(role))
    }

    pub const fn occupied(self) -> Bitboard {
        self.by_color(Color::White)
            .union(self.by_color(Color::Black))
    }

    pub fn king_square(self, color: Color) -> Option<Square> {
        self.pieces(color, Role::King).iter().next()
    }

    /// Read-only access to the square array, useful for invariant checks and
    /// evaluator tests.
    pub fn squares(&self) -> &[Option<Piece>; 64] {
        &self.squares
    }

    /// The single mutation point for the hybrid representation.
    pub(crate) fn set_piece(&mut self, square: Square, piece: Option<Piece>) {
        let index = square.index() as usize;
        if let Some(old) = self.squares[index] {
            self.by_role[old.role.index()] = self.by_role[old.role.index()].without(square);
            self.by_color[old.color.index()] = self.by_color[old.color.index()].without(square);
        }
        if let Some(new) = piece {
            self.by_role[new.role.index()] = self.by_role[new.role.index()].with(square);
            self.by_color[new.color.index()] = self.by_color[new.color.index()].with(square);
        }
        self.squares[index] = piece;
    }

    pub fn validate(&self) -> Result<(), BoardInvariantError> {
        let mut occupied_from_squares = Bitboard::EMPTY;
        for index in 0..64 {
            let square = Square::from_index(index);
            match self.squares[index as usize] {
                Some(piece) => {
                    occupied_from_squares = occupied_from_squares.with(square);
                    if !self.by_color(piece.color).contains(square) {
                        return Err(BoardInvariantError::ColorBitMissing(square));
                    }
                    if self.by_color(piece.color.opposite()).contains(square) {
                        return Err(BoardInvariantError::ColorOverlap(square));
                    }
                    let mut role_count = 0;
                    for role in Role::ALL {
                        if self.by_role(role).contains(square) {
                            role_count += 1;
                        }
                    }
                    if role_count != 1 || !self.by_role(piece.role).contains(square) {
                        return Err(BoardInvariantError::RoleBitMismatch(square));
                    }
                }
                None => {
                    if self.occupied().contains(square) {
                        return Err(BoardInvariantError::OccupancyWithoutPiece(square));
                    }
                }
            }
        }

        if occupied_from_squares != self.occupied() {
            return Err(BoardInvariantError::OccupancyMismatch);
        }

        let mut role_union = Bitboard::EMPTY;
        for role in Role::ALL {
            if self.by_role(role).intersect(role_union).any() {
                return Err(BoardInvariantError::OverlappingRoleBoards);
            }
            role_union = role_union.union(self.by_role(role));
        }
        if role_union != occupied_from_squares {
            return Err(BoardInvariantError::RoleUnionMismatch);
        }
        if self
            .by_color(Color::White)
            .intersect(self.by_color(Color::Black))
            .any()
        {
            return Err(BoardInvariantError::OverlappingColorBoards);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoardInvariantError {
    ColorBitMissing(Square),
    ColorOverlap(Square),
    RoleBitMismatch(Square),
    OccupancyWithoutPiece(Square),
    OccupancyMismatch,
    OverlappingRoleBoards,
    RoleUnionMismatch,
    OverlappingColorBoards,
}

impl fmt::Display for BoardInvariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            BoardInvariantError::ColorBitMissing(square) => {
                write!(f, "colour occupancy bit missing at {square}")
            }
            BoardInvariantError::ColorOverlap(square) => {
                write!(f, "both colour occupancy bits set at {square}")
            }
            BoardInvariantError::RoleBitMismatch(square) => {
                write!(f, "piece-role occupancy mismatch at {square}")
            }
            BoardInvariantError::OccupancyWithoutPiece(square) => {
                write!(f, "occupancy bit set without a piece at {square}")
            }
            BoardInvariantError::OccupancyMismatch => {
                f.write_str("square array and colour occupancy disagree")
            }
            BoardInvariantError::OverlappingRoleBoards => f.write_str("role bitboards overlap"),
            BoardInvariantError::RoleUnionMismatch => {
                f.write_str("role bitboards do not cover exactly the occupied squares")
            }
            BoardInvariantError::OverlappingColorBoards => f.write_str("colour bitboards overlap"),
        }
    }
}

impl std::error::Error for BoardInvariantError {}
