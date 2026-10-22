use core::fmt;

use crate::attacks;
use crate::board::{Board, BoardInvariantError};
use crate::fen::FenError;
use crate::hash;
use crate::movegen::{self, MoveList};
use crate::types::{CastlingRights, CastlingSide, Color, Move, MoveKind, Piece, Role, Square};

pub const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// One reversible board-state change, recorded at a single square.
/// Replaying `before` in reverse order inverts the change; replaying `after`
/// forward applies it. This is the interface a future incremental evaluator
/// can consume without knowing move semantics.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PieceChange {
    pub square: Square,
    pub before: Option<Piece>,
    pub after: Option<Piece>,
}

impl PieceChange {
    const fn placeholder() -> PieceChange {
        PieceChange {
            square: Square::A1,
            before: None,
            after: None,
        }
    }
}

const MAX_PIECE_CHANGES: usize = 4;

#[derive(Clone, Copy, Debug)]
struct ChangeLog {
    changes: [PieceChange; MAX_PIECE_CHANGES],
    len: usize,
}

impl ChangeLog {
    fn new() -> ChangeLog {
        ChangeLog {
            changes: [PieceChange::placeholder(); MAX_PIECE_CHANGES],
            len: 0,
        }
    }

    fn push(&mut self, change: PieceChange) {
        assert!(
            self.len < MAX_PIECE_CHANGES,
            "a chess move changes at most four squares"
        );
        self.changes[self.len] = change;
        self.len += 1;
    }

    fn as_slice(&self) -> &[PieceChange] {
        &self.changes[..self.len]
    }
}

/// Fixed-size undo record. It contains everything needed to restore the
/// position exactly, plus the board-square deltas for future evaluators.
///
/// Records are LIFO: pass the most recent outstanding record to
/// [`Position::unmake`] before an older one.
#[derive(Clone, Copy, Debug)]
pub struct Undo {
    movement: Move,
    change_log: ChangeLog,
    castling: CastlingRights,
    ep_square: Option<Square>,
    halfmove_clock: u32,
    fullmove_number: u32,
    captured: Option<Piece>,
    post_hash: u64,
}

impl Undo {
    pub const fn movement(&self) -> Move {
        self.movement
    }

    pub fn piece_changes(&self) -> &[PieceChange] {
        self.change_log.as_slice()
    }

    pub const fn captured(&self) -> Option<Piece> {
        self.captured
    }

    pub const fn castling_rights_before(&self) -> CastlingRights {
        self.castling
    }

    pub const fn ep_square_before(&self) -> Option<Square> {
        self.ep_square
    }

    pub const fn halfmove_clock_before(&self) -> u32 {
        self.halfmove_clock
    }

    pub const fn fullmove_number_before(&self) -> u32 {
        self.fullmove_number
    }

    fn moving_piece(&self) -> Option<Piece> {
        self.change_log
            .as_slice()
            .iter()
            .find(|change| change.square == self.movement.from())
            .and_then(|change| change.before)
    }

    fn expected_post_side(&self) -> Option<Color> {
        self.moving_piece().map(|piece| piece.color.opposite())
    }

    fn is_zeroing(&self) -> bool {
        self.captured.is_some()
            || self.movement.kind() == MoveKind::EnPassant
            || self
                .moving_piece()
                .is_some_and(|piece| piece.role == Role::Pawn)
    }

    fn expected_post_halfmove_clock(&self) -> Option<u32> {
        if self.is_zeroing() {
            Some(0)
        } else {
            self.halfmove_clock.checked_add(1)
        }
    }

    fn expected_post_fullmove_number(&self, post_side: Color) -> Option<u32> {
        if post_side == Color::White {
            self.fullmove_number.checked_add(1)
        } else {
            Some(self.fullmove_number)
        }
    }

    fn expected_post_ep_square(&self) -> Option<Square> {
        match self.movement.kind() {
            MoveKind::DoublePawnPush => Some(Square::from_file_rank(
                self.movement.from().file(),
                (self.movement.from().rank() + self.movement.to().rank()) / 2,
            )),
            _ => None,
        }
    }
}

/// A complete chess position without game history.
///
/// Equality compares every stored state field, including the incremental hash.
/// Repetition identity is a different, narrower relation exposed by
/// [`Position::same_repetition_identity`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Position {
    board: Board,
    side_to_move: Color,
    castling: CastlingRights,
    ep_square: Option<Square>,
    halfmove_clock: u32,
    fullmove_number: u32,
    hash: u64,
}

impl Position {
    pub fn startpos() -> Position {
        Position::from_fen(STARTPOS_FEN).expect("built-in start position is valid")
    }

    /// Parse a FEN position with no game history attached.
    pub fn from_fen(fen: &str) -> Result<Position, FenError> {
        crate::fen::parse(fen)
    }

    pub fn to_fen(&self) -> String {
        crate::fen::serialize(self)
    }

    pub const fn board(&self) -> Board {
        self.board
    }

    pub const fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    pub const fn castling_rights(&self) -> CastlingRights {
        self.castling
    }

    /// Raw FEN en-passant target. It can be structurally valid even when no
    /// legal en-passant capture exists; use [`Position::legal_ep_square`] for
    /// the repetition-relevant definition.
    pub const fn ep_square(&self) -> Option<Square> {
        self.ep_square
    }

    pub const fn halfmove_clock(&self) -> u32 {
        self.halfmove_clock
    }

    pub const fn fullmove_number(&self) -> u32 {
        self.fullmove_number
    }

    /// Incremental Zobrist key. Includes piece placement, side to move,
    /// castling rights and an en-passant file only when a *legal* en-passant
    /// capture exists. Excludes the halfmove clock, fullmove number and all
    /// history, so it is the hash half of repetition identity.
    pub const fn hash(&self) -> u64 {
        self.hash
    }

    /// Independent full recomputation using the same table, for validation.
    pub fn recompute_hash(&self) -> u64 {
        let mut hash = 0u64;
        for index in 0..64 {
            let square = Square::from_index(index);
            if let Some(piece) = self.board.piece_at(square) {
                hash ^= hash::piece_square_key(piece, square);
            }
        }
        hash ^= hash::castling_key(self.castling);
        hash ^= hash::side_key(self.side_to_move);
        hash ^= self.ep_hash_key();
        hash
    }

    fn ep_hash_key(&self) -> u64 {
        movegen::legal_ep_square(self).map_or(0, |square| hash::ep_file_key(square.file()))
    }

    pub fn legal_moves(&self) -> MoveList {
        movegen::legal_moves(self)
    }

    pub fn is_legal(&self, movement: Move) -> bool {
        movegen::resolve_legal_move(self, movement).is_some()
    }

    /// En-passant target only when at least one fully legal en-passant capture
    /// exists. This includes pin and x-ray cases.
    pub fn legal_ep_square(&self) -> Option<Square> {
        movegen::legal_ep_square(self)
    }

    pub fn in_check(&self) -> bool {
        self.board
            .king_square(self.side_to_move)
            .is_some_and(|king| {
                attacks::is_square_attacked(self.board, king, self.side_to_move.opposite())
            })
    }

    pub fn is_checkmate(&self) -> bool {
        self.in_check() && self.legal_moves().is_empty()
    }

    pub fn is_stalemate(&self) -> bool {
        !self.in_check() && self.legal_moves().is_empty()
    }

    /// Conservative proven-dead-material detector for the documented global
    /// classes. True only when there are no pawns, rooks or queens anywhere and
    /// either there are zero or one total minors, or there are at least two
    /// minors with no knights and all bishops on one square colour. K+NN vs K
    /// and K+N vs K+N are deliberately not classified as dead. This is not a
    /// complete FIDE dead-position proof.
    pub fn has_insufficient_material(&self) -> bool {
        let board = self.board;
        let pawns_rooks_queens = board
            .by_role(Role::Pawn)
            .union(board.by_role(Role::Rook))
            .union(board.by_role(Role::Queen));
        if pawns_rooks_queens.any() {
            return false;
        }

        let knights = board.by_role(Role::Knight);
        let mut bishops = board.by_role(Role::Bishop);
        if knights.count() + bishops.count() <= 1 {
            return true;
        }
        if knights.any() {
            return false;
        }

        let Some(first) = bishops.pop_lsb() else {
            return false;
        };
        let first_is_dark = (first.file() + first.rank()) % 2 == 0;
        bishops
            .iter()
            .all(|square| ((square.file() + square.rank()) % 2 == 0) == first_is_dark)
    }

    /// Safely play a move. The position is unchanged when the move is
    /// rejected. A coordinate move is resolved to the canonical generated move,
    /// so the returned undo's [`Undo::movement`] can differ in kind from the
    /// caller's value (for example `normal(e2, e4)` becomes a double pawn push).
    /// Undo records must be consumed in LIFO order.
    pub fn play(&mut self, movement: Move) -> Result<Undo, MoveError> {
        let Some(canonical) = movegen::resolve_legal_move(self, movement) else {
            return Err(MoveError::Illegal { movement });
        };
        self.make_unchecked(canonical)
    }

    /// Apply an already-validated legal move without regenerating move lists.
    ///
    /// `movegen` calls this only for moves it has produced (or, indirectly,
    /// through `perft`). It is not public: arbitrary callers cannot apply
    /// pseudo-legal or malformed moves by mistake.
    pub(crate) fn make_unchecked(&mut self, movement: Move) -> Result<Undo, MoveError> {
        let Some(moving_piece) = self.board.piece_at(movement.from()) else {
            return Err(MoveError::Illegal { movement });
        };
        if moving_piece.color != self.side_to_move {
            return Err(MoveError::Illegal { movement });
        }

        let captured_square = match movement.kind() {
            MoveKind::EnPassant => Some(Square::from_file_rank(
                movement.to().file(),
                movement.from().rank(),
            )),
            MoveKind::Normal | MoveKind::DoublePawnPush => {
                let target = self.board.piece_at(movement.to());
                target
                    .filter(|piece| piece.color == self.side_to_move.opposite())
                    .map(|_| movement.to())
            }
            MoveKind::Castle => None,
        };
        let captured = captured_square.and_then(|square| self.board.piece_at(square));
        if movement.kind() == MoveKind::EnPassant
            && captured != Some(Piece::new(self.side_to_move.opposite(), Role::Pawn))
        {
            return Err(MoveError::Illegal { movement });
        }

        let zeroing = moving_piece.role == Role::Pawn || captured.is_some();
        if !zeroing && self.halfmove_clock == u32::MAX {
            return Err(MoveError::CounterOverflow);
        }
        if self.side_to_move == Color::Black && self.fullmove_number == u32::MAX {
            return Err(MoveError::CounterOverflow);
        }

        let previous_castling = self.castling;
        let previous_ep_square = self.ep_square;
        let previous_halfmove_clock = self.halfmove_clock;
        let previous_fullmove_number = self.fullmove_number;

        let pre_castling_key = hash::castling_key(self.castling);
        let pre_ep_key = self.ep_hash_key();

        let mut changes = ChangeLog::new();
        match movement.kind() {
            MoveKind::Normal | MoveKind::DoublePawnPush => {
                let placed = movement
                    .promotion_role()
                    .map_or(moving_piece, |role| Piece::new(moving_piece.color, role));
                self.set_piece_recorded(movement.from(), None, &mut changes);
                self.set_piece_recorded(movement.to(), Some(placed), &mut changes);
            }
            MoveKind::EnPassant => {
                let Some(captured_square) = captured_square else {
                    return Err(MoveError::Illegal { movement });
                };
                self.set_piece_recorded(movement.from(), None, &mut changes);
                self.set_piece_recorded(captured_square, None, &mut changes);
                self.set_piece_recorded(movement.to(), Some(moving_piece), &mut changes);
            }
            MoveKind::Castle => {
                let side = if movement.to().file() > movement.from().file() {
                    CastlingSide::KingSide
                } else {
                    CastlingSide::QueenSide
                };
                let rook_from = side.rook_square(self.side_to_move);
                let rook_to = side.rook_destination(self.side_to_move);
                self.set_piece_recorded(movement.from(), None, &mut changes);
                self.set_piece_recorded(rook_from, None, &mut changes);
                self.set_piece_recorded(
                    movement.to(),
                    Some(Piece::new(self.side_to_move, Role::King)),
                    &mut changes,
                );
                self.set_piece_recorded(
                    rook_to,
                    Some(Piece::new(self.side_to_move, Role::Rook)),
                    &mut changes,
                );
            }
        }

        let mut new_castling = self.castling;
        if moving_piece.role == Role::King {
            new_castling = new_castling.without_color(self.side_to_move);
        } else if moving_piece.role == Role::Rook {
            new_castling = new_castling.without_rook_square(movement.from());
        }
        if let (Some(captured_piece), Some(square)) = (captured, captured_square) {
            if captured_piece.role == Role::Rook {
                new_castling = new_castling.without_rook_square(square);
            }
        }
        self.castling = new_castling;
        self.hash ^= pre_castling_key ^ hash::castling_key(self.castling);

        self.ep_square = match movement.kind() {
            MoveKind::DoublePawnPush => Some(Square::from_file_rank(
                movement.from().file(),
                (movement.from().rank() + movement.to().rank()) / 2,
            )),
            _ => None,
        };

        self.halfmove_clock = if zeroing { 0 } else { self.halfmove_clock + 1 };
        if self.side_to_move == Color::Black {
            self.fullmove_number += 1;
        }

        let old_side = self.side_to_move;
        self.side_to_move = self.side_to_move.opposite();
        self.hash ^= hash::side_key(old_side) ^ hash::side_key(self.side_to_move);

        let new_ep_key = self.ep_hash_key();
        self.hash ^= pre_ep_key ^ new_ep_key;

        Ok(Undo {
            movement,
            change_log: changes,
            castling: previous_castling,
            ep_square: previous_ep_square,
            halfmove_clock: previous_halfmove_clock,
            fullmove_number: previous_fullmove_number,
            captured,
            post_hash: self.hash,
        })
    }

    fn set_piece_recorded(
        &mut self,
        square: Square,
        after: Option<Piece>,
        changes: &mut ChangeLog,
    ) {
        let before = self.board.piece_at(square);
        if before == after {
            return;
        }
        self.board.set_piece(square, after);
        self.hash ^= before.map_or(0, |piece| hash::piece_square_key(piece, square))
            ^ after.map_or(0, |piece| hash::piece_square_key(piece, square));
        changes.push(PieceChange {
            square,
            before,
            after,
        });
    }

    /// Unmake the most recent move. LIFO is explicit: the record is rejected
    /// without mutation unless the current position has the recorded post-move
    /// hash, exact side, clock counters, raw en-passant field and recorded
    /// touched-square pieces. This detects
    /// stale records for genuinely repeated positions, whose hashes and boards
    /// match but whose clocks differ. It is not a provenance proof: a record
    /// from a different `Position` copy with identical post-move state is
    /// indistinguishable, and callers still own the LIFO discipline.
    pub fn unmake(&mut self, undo: &Undo) -> Result<(), UnmakeError> {
        let Some(post_side) = undo.expected_post_side() else {
            return Err(UnmakeError::NotLastMove);
        };
        let Some(post_halfmove_clock) = undo.expected_post_halfmove_clock() else {
            return Err(UnmakeError::NotLastMove);
        };
        let Some(post_fullmove_number) = undo.expected_post_fullmove_number(post_side) else {
            return Err(UnmakeError::NotLastMove);
        };
        if self.hash != undo.post_hash
            || self.side_to_move != post_side
            || self.halfmove_clock != post_halfmove_clock
            || self.fullmove_number != post_fullmove_number
            || self.ep_square != undo.expected_post_ep_square()
            || undo
                .piece_changes()
                .iter()
                .any(|change| self.board.piece_at(change.square) != change.after)
        {
            return Err(UnmakeError::NotLastMove);
        }

        let post_castling_key = hash::castling_key(self.castling);
        let post_ep_key = self.ep_hash_key();

        for change in undo.piece_changes().iter().rev() {
            self.hash ^= change
                .before
                .map_or(0, |piece| hash::piece_square_key(piece, change.square))
                ^ change
                    .after
                    .map_or(0, |piece| hash::piece_square_key(piece, change.square));
            self.board.set_piece(change.square, change.before);
        }

        let old_side = self.side_to_move;
        self.side_to_move = self.side_to_move.opposite();
        self.hash ^= hash::side_key(old_side) ^ hash::side_key(self.side_to_move);

        self.castling = undo.castling;
        self.ep_square = undo.ep_square;
        self.halfmove_clock = undo.halfmove_clock;
        self.fullmove_number = undo.fullmove_number;

        let pre_castling_key = hash::castling_key(self.castling);
        let pre_ep_key = self.ep_hash_key();
        self.hash ^= post_castling_key ^ pre_castling_key;
        self.hash ^= post_ep_key ^ pre_ep_key;

        Ok(())
    }

    /// Exact repetition identity: side, board, castling rights and an
    /// en-passant square only when a legal en-passant capture exists. Clock
    /// fields and raw-but-uncapturable EP fields are excluded.
    pub fn same_repetition_identity(&self, other: &Position) -> bool {
        self.hash == other.hash
            && self.side_to_move == other.side_to_move
            && self.castling == other.castling
            && self.legal_ep_square() == other.legal_ep_square()
            && self.board == other.board
    }

    pub fn validate_invariants(&self) -> Result<(), PositionInvariantError> {
        self.board.validate()?;

        for color in Color::ALL {
            match self.board.pieces(color, Role::King).count() {
                1 => {}
                0 => return Err(PositionInvariantError::MissingKing(color)),
                _ => return Err(PositionInvariantError::ExtraKing(color)),
            }
        }

        for color in Color::ALL {
            let king_square = CastlingSide::king_square(color);
            if (self.castling.has(color, CastlingSide::KingSide)
                || self.castling.has(color, CastlingSide::QueenSide))
                && self.board.piece_at(king_square) != Some(Piece::new(color, Role::King))
            {
                return Err(PositionInvariantError::CastlingKingNotHome(color));
            }
            for side in [CastlingSide::KingSide, CastlingSide::QueenSide] {
                if self.castling.has(color, side)
                    && self.board.piece_at(side.rook_square(color))
                        != Some(Piece::new(color, Role::Rook))
                {
                    return Err(PositionInvariantError::CastlingRookNotHome(color, side));
                }
            }
        }

        if let Some(ep) = self.ep_square {
            if !movegen::en_passant_is_structural(self.board, ep, self.side_to_move) {
                return Err(PositionInvariantError::InvalidEnPassant(ep));
            }
        }

        let recomputed = self.recompute_hash();
        if recomputed != self.hash {
            return Err(PositionInvariantError::HashMismatch {
                stored: self.hash,
                recomputed,
            });
        }
        Ok(())
    }

    pub(crate) fn from_parts(
        board: Board,
        side_to_move: Color,
        castling: CastlingRights,
        ep_square: Option<Square>,
        halfmove_clock: u32,
        fullmove_number: u32,
    ) -> Position {
        let mut position = Position {
            board,
            side_to_move,
            castling,
            ep_square,
            halfmove_clock,
            fullmove_number,
            hash: 0,
        };
        position.hash = position.recompute_hash();
        position
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoveError {
    Illegal { movement: Move },
    CounterOverflow,
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MoveError::Illegal { movement } => write!(f, "illegal move: {movement}"),
            MoveError::CounterOverflow => f.write_str("move would overflow a FEN counter (u32)"),
        }
    }
}

impl std::error::Error for MoveError {}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnmakeError {
    NotLastMove,
}

impl fmt::Display for UnmakeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("undo record does not match the current post-move state")
    }
}

impl std::error::Error for UnmakeError {}

#[derive(Clone, Copy, Debug)]
pub enum PositionInvariantError {
    Board(BoardInvariantError),
    MissingKing(Color),
    ExtraKing(Color),
    CastlingKingNotHome(Color),
    CastlingRookNotHome(Color, CastlingSide),
    InvalidEnPassant(Square),
    HashMismatch { stored: u64, recomputed: u64 },
}

impl From<BoardInvariantError> for PositionInvariantError {
    fn from(error: BoardInvariantError) -> Self {
        PositionInvariantError::Board(error)
    }
}

impl fmt::Display for PositionInvariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PositionInvariantError::Board(error) => write!(f, "{error}"),
            PositionInvariantError::MissingKing(color) => {
                write!(f, "missing {color} king")
            }
            PositionInvariantError::ExtraKing(color) => {
                write!(f, "more than one {color} king")
            }
            PositionInvariantError::CastlingKingNotHome(color) => {
                write!(f, "{color} castling right without king on home square")
            }
            PositionInvariantError::CastlingRookNotHome(color, side) => {
                write!(
                    f,
                    "{color} {side:?} castling right without rook on home square"
                )
            }
            PositionInvariantError::InvalidEnPassant(square) => {
                write!(f, "structurally invalid en-passant target {square}")
            }
            PositionInvariantError::HashMismatch { stored, recomputed } => {
                write!(
                    f,
                    "hash mismatch: stored {stored:#018x}, recomputed {recomputed:#018x}"
                )
            }
        }
    }
}

impl std::error::Error for PositionInvariantError {}
