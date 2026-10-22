use std::ops::Deref;

use crate::attacks;
use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::position::Position;
use crate::types::{CastlingSide, Color, Move, MoveKind, Piece, Role, Square};

/// All legal moves in deterministic generation order.
#[derive(Clone, Debug, Default)]
pub struct MoveList(Vec<Move>);

impl MoveList {
    pub(crate) fn with_capacity(capacity: usize) -> MoveList {
        MoveList(Vec::with_capacity(capacity))
    }

    pub(crate) fn push(&mut self, movement: Move) {
        self.0.push(movement);
    }
}

impl Deref for MoveList {
    type Target = [Move];

    fn deref(&self) -> &[Move] {
        &self.0
    }
}

impl AsRef<[Move]> for MoveList {
    fn as_ref(&self) -> &[Move] {
        &self.0
    }
}

impl IntoIterator for MoveList {
    type Item = Move;
    type IntoIter = std::vec::IntoIter<Move>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = &'a Move;
    type IntoIter = std::slice::Iter<'a, Move>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Generate all legal moves: pseudo-legal candidates first, then an
/// own-king-safety filter that also covers pins, discovered checks and
/// en-passant discoveries.
pub fn legal_moves(position: &Position) -> MoveList {
    let mut candidates = MoveList::with_capacity(64);
    generate_pseudo_legal(position, &mut candidates);

    let mut legal = MoveList::with_capacity(candidates.len());
    for movement in candidates.iter() {
        if is_move_safe(position, *movement) {
            legal.push(*movement);
        }
    }
    legal
}

/// Resolve a coordinate move (from, to, optional promotion) to the canonical
/// legal move generated for the position. The move kind is derived from the
/// position: e2-e4 becomes a double pawn push, e1-g1 becomes castling and a
/// pawn capture onto an en-passant target becomes en passant. This lets a
/// public caller provide a plain coordinate move without guessing the kind.
pub fn resolve_legal_move(position: &Position, movement: Move) -> Option<Move> {
    legal_moves(position).into_iter().find(|legal| {
        legal.from() == movement.from()
            && legal.to() == movement.to()
            && legal.promotion_role() == movement.promotion_role()
    })
}

/// The en-passant target square only when at least one fully legal en-passant
/// capture exists. This is the rule used by repetition identity and hashing.
pub(crate) fn legal_ep_square(position: &Position) -> Option<Square> {
    let ep = position.ep_square()?;
    let us = position.side_to_move();
    let board = position.board();

    let captured_square = ep.offset(0, -us.forward())?;
    match board.piece_at(captured_square) {
        Some(piece) if piece.color == us.opposite() && piece.role == Role::Pawn => {}
        _ => return None,
    }

    let mut candidate_pawns =
        attacks::pawn_attacks(us.opposite(), ep).intersect(board.pieces(us, Role::Pawn));
    while let Some(from) = candidate_pawns.pop_lsb() {
        let movement = Move::en_passant(from, ep);
        if is_move_safe(position, movement) {
            return Some(ep);
        }
    }
    None
}

fn generate_pseudo_legal(position: &Position, moves: &mut MoveList) {
    let board = position.board();
    let us = position.side_to_move();
    let us_occupied = board.by_color(us);
    let occupied = board.occupied();
    let empty = occupied.complement();

    generate_pawn_moves(position, moves, board, us, empty);
    generate_knight_moves(moves, board, us, us_occupied);
    generate_slider_moves(moves, board, us, us_occupied, occupied);
    generate_king_moves(position, moves, board, us, us_occupied);
}

fn generate_pawn_moves(
    position: &Position,
    moves: &mut MoveList,
    board: Board,
    us: Color,
    empty: Bitboard,
) {
    let forward = us.forward();
    let promotion_rank = us.promotion_rank();
    let them = us.opposite();
    let ep_square = position.ep_square();

    let mut pawns = board.pieces(us, Role::Pawn);
    while let Some(from) = pawns.pop_lsb() {
        if let Some(to) = from.offset(0, forward) {
            if empty.contains(to) {
                if to.rank() == promotion_rank {
                    add_promotions(moves, from, to);
                } else {
                    moves.push(Move::normal(from, to));
                    if from.rank() == us.pawn_start_rank() {
                        if let Some(double_to) = from.offset(0, forward * 2) {
                            if empty.contains(double_to) {
                                moves.push(Move::double_pawn_push(from, double_to));
                            }
                        }
                    }
                }
            }
        }

        for file_delta in [-1, 1] {
            let Some(to) = from.offset(file_delta, forward) else {
                continue;
            };
            if let Some(target) = board.piece_at(to) {
                if target.color == them && target.role != Role::King {
                    if to.rank() == promotion_rank {
                        add_promotions(moves, from, to);
                    } else {
                        moves.push(Move::normal(from, to));
                    }
                }
            } else if Some(to) == ep_square {
                let Some(captured_square) = to.offset(0, -forward) else {
                    continue;
                };
                if let Some(captured) = board.piece_at(captured_square) {
                    if captured.color == them && captured.role == Role::Pawn {
                        moves.push(Move::en_passant(from, to));
                    }
                }
            }
        }
    }
}

fn add_promotions(moves: &mut MoveList, from: Square, to: Square) {
    for role in Role::PROMOTION_TARGETS {
        moves.push(Move::promotion(from, to, role));
    }
}

fn generate_knight_moves(moves: &mut MoveList, board: Board, us: Color, us_occupied: Bitboard) {
    let mut knights = board.pieces(us, Role::Knight);
    while let Some(from) = knights.pop_lsb() {
        let mut targets = attacks::knight_attacks(from).intersect(us_occupied.complement());
        while let Some(to) = targets.pop_lsb() {
            if can_capture_square(board, to, us) {
                moves.push(Move::normal(from, to));
            }
        }
    }
}

fn generate_slider_moves(
    moves: &mut MoveList,
    board: Board,
    us: Color,
    us_occupied: Bitboard,
    occupied: Bitboard,
) {
    let mut bishops = board.pieces(us, Role::Bishop);
    while let Some(from) = bishops.pop_lsb() {
        push_slider_targets(
            moves,
            board,
            us,
            from,
            attacks::bishop_attacks(from, occupied),
            us_occupied,
        );
    }

    let mut rooks = board.pieces(us, Role::Rook);
    while let Some(from) = rooks.pop_lsb() {
        push_slider_targets(
            moves,
            board,
            us,
            from,
            attacks::rook_attacks(from, occupied),
            us_occupied,
        );
    }

    let mut queens = board.pieces(us, Role::Queen);
    while let Some(from) = queens.pop_lsb() {
        push_slider_targets(
            moves,
            board,
            us,
            from,
            attacks::queen_attacks(from, occupied),
            us_occupied,
        );
    }
}

fn push_slider_targets(
    moves: &mut MoveList,
    board: Board,
    us: Color,
    from: Square,
    attacks: Bitboard,
    us_occupied: Bitboard,
) {
    let mut targets = attacks.intersect(us_occupied.complement());
    while let Some(to) = targets.pop_lsb() {
        if can_capture_square(board, to, us) {
            moves.push(Move::normal(from, to));
        }
    }
}

fn generate_king_moves(
    position: &Position,
    moves: &mut MoveList,
    board: Board,
    us: Color,
    us_occupied: Bitboard,
) {
    let Some(king_from) = board.king_square(us) else {
        return;
    };

    let mut targets = attacks::king_attacks(king_from).intersect(us_occupied.complement());
    while let Some(to) = targets.pop_lsb() {
        if can_capture_square(board, to, us) {
            moves.push(Move::normal(king_from, to));
        }
    }

    if king_from != CastlingSide::king_square(us) {
        return;
    }

    let them = us.opposite();
    for side in [CastlingSide::KingSide, CastlingSide::QueenSide] {
        if !position.castling_rights().has(us, side) {
            continue;
        }
        let rook_from = side.rook_square(us);
        if board.piece_at(rook_from) != Some(Piece::new(us, Role::Rook)) {
            continue;
        }

        let empty_squares: &[Square] = match side {
            CastlingSide::KingSide => {
                if us.is_white() {
                    &[Square::F1, Square::G1]
                } else {
                    &[Square::F8, Square::G8]
                }
            }
            CastlingSide::QueenSide => {
                if us.is_white() {
                    &[Square::B1, Square::C1, Square::D1]
                } else {
                    &[Square::B8, Square::C8, Square::D8]
                }
            }
        };
        if empty_squares
            .iter()
            .any(|square| board.piece_at(*square).is_some())
        {
            continue;
        }

        let path = [
            king_from,
            match side {
                CastlingSide::KingSide => {
                    if us.is_white() {
                        Square::F1
                    } else {
                        Square::F8
                    }
                }
                CastlingSide::QueenSide => {
                    if us.is_white() {
                        Square::D1
                    } else {
                        Square::D8
                    }
                }
            },
            side.king_destination(us),
        ];
        if path
            .iter()
            .any(|square| attacks::is_square_attacked(board, *square, them))
        {
            continue;
        }

        moves.push(Move::castle(king_from, side.king_destination(us)));
    }
}

fn can_capture_square(board: Board, square: Square, us: Color) -> bool {
    match board.piece_at(square) {
        None => true,
        Some(piece) => piece.color != us && piece.role != Role::King,
    }
}

fn is_move_safe(position: &Position, movement: Move) -> bool {
    let us = position.side_to_move();
    let mut board = position.board();

    match movement.kind() {
        MoveKind::Normal | MoveKind::DoublePawnPush => {
            let Some(piece) = board.piece_at(movement.from()) else {
                return false;
            };
            let placed = movement
                .promotion_role()
                .map_or(piece, |role| Piece::new(piece.color, role));
            board.set_piece(movement.from(), None);
            board.set_piece(movement.to(), Some(placed));
        }
        MoveKind::EnPassant => {
            let Some(piece) = board.piece_at(movement.from()) else {
                return false;
            };
            let captured_square =
                Square::from_file_rank(movement.to().file(), movement.from().rank());
            board.set_piece(movement.from(), None);
            board.set_piece(captured_square, None);
            board.set_piece(movement.to(), Some(piece));
        }
        MoveKind::Castle => {
            let side = if movement.to().file() > movement.from().file() {
                CastlingSide::KingSide
            } else {
                CastlingSide::QueenSide
            };
            let king = Piece::new(us, Role::King);
            let rook = Piece::new(us, Role::Rook);
            board.set_piece(movement.from(), None);
            board.set_piece(side.rook_square(us), None);
            board.set_piece(movement.to(), Some(king));
            board.set_piece(side.rook_destination(us), Some(rook));
        }
    }

    let Some(king_square) = board.king_square(us) else {
        return false;
    };
    !attacks::is_square_attacked(board, king_square, us.opposite())
}

/// Check whether a raw en-passant field is structurally possible. This does
/// not require a capturer: uncapturable but valid fields are preserved by FEN.
pub(crate) fn en_passant_is_structural(board: Board, ep: Square, side_to_move: Color) -> bool {
    let mover = side_to_move.opposite();
    let expected_rank = match side_to_move {
        Color::White => 5,
        Color::Black => 2,
    };
    if ep.rank() != expected_rank {
        return false;
    }
    if board.piece_at(ep).is_some() {
        return false;
    }
    let Some(pushed_pawn_square) = ep.offset(0, mover.forward()) else {
        return false;
    };
    let Some(origin_square) = ep.offset(0, -mover.forward()) else {
        return false;
    };
    board.piece_at(pushed_pawn_square) == Some(Piece::new(mover, Role::Pawn))
        && board.piece_at(origin_square).is_none()
}
