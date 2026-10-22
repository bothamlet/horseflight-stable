use crate::bitboard::Bitboard;
use crate::board::Board;
use crate::types::{Color, Role, Square};

const KNIGHT_DELTAS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];

const KING_DELTAS: [(i8, i8); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

const BISHOP_DELTAS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ROOK_DELTAS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

const fn build_knight_attacks() -> [Bitboard; 64] {
    let mut table = [Bitboard::EMPTY; 64];
    let mut square = 0u8;
    while square < 64 {
        let file = (square % 8) as i8;
        let rank = (square / 8) as i8;
        let mut i = 0usize;
        while i < KNIGHT_DELTAS.len() {
            let target_file = file + KNIGHT_DELTAS[i].0;
            let target_rank = rank + KNIGHT_DELTAS[i].1;
            if target_file >= 0 && target_file < 8 && target_rank >= 0 && target_rank < 8 {
                table[square as usize] = table[square as usize]
                    .with(Square::from_file_rank(target_file as u8, target_rank as u8));
            }
            i += 1;
        }
        square += 1;
    }
    table
}

const fn build_king_attacks() -> [Bitboard; 64] {
    let mut table = [Bitboard::EMPTY; 64];
    let mut square = 0u8;
    while square < 64 {
        let file = (square % 8) as i8;
        let rank = (square / 8) as i8;
        let mut i = 0usize;
        while i < KING_DELTAS.len() {
            let target_file = file + KING_DELTAS[i].0;
            let target_rank = rank + KING_DELTAS[i].1;
            if target_file >= 0 && target_file < 8 && target_rank >= 0 && target_rank < 8 {
                table[square as usize] = table[square as usize]
                    .with(Square::from_file_rank(target_file as u8, target_rank as u8));
            }
            i += 1;
        }
        square += 1;
    }
    table
}

const fn build_pawn_attacks() -> [[Bitboard; 64]; 2] {
    let mut table = [[Bitboard::EMPTY; 64]; 2];
    let mut square = 0u8;
    while square < 64 {
        let file = (square % 8) as i8;
        let rank = (square / 8) as i8;

        let white_rank = rank + 1;
        if white_rank < 8 {
            if file > 0 {
                table[0][square as usize] = table[0][square as usize]
                    .with(Square::from_file_rank((file - 1) as u8, white_rank as u8));
            }
            if file < 7 {
                table[0][square as usize] = table[0][square as usize]
                    .with(Square::from_file_rank((file + 1) as u8, white_rank as u8));
            }
        }

        let black_rank = rank - 1;
        if black_rank >= 0 {
            if file > 0 {
                table[1][square as usize] = table[1][square as usize]
                    .with(Square::from_file_rank((file - 1) as u8, black_rank as u8));
            }
            if file < 7 {
                table[1][square as usize] = table[1][square as usize]
                    .with(Square::from_file_rank((file + 1) as u8, black_rank as u8));
            }
        }

        square += 1;
    }
    table
}

static KNIGHT_ATTACKS: [Bitboard; 64] = build_knight_attacks();
static KING_ATTACKS: [Bitboard; 64] = build_king_attacks();
static PAWN_ATTACKS: [[Bitboard; 64]; 2] = build_pawn_attacks();

pub(crate) fn pawn_attacks(color: Color, square: Square) -> Bitboard {
    PAWN_ATTACKS[color.index()][square.index() as usize]
}

pub(crate) fn knight_attacks(square: Square) -> Bitboard {
    KNIGHT_ATTACKS[square.index() as usize]
}

pub(crate) fn king_attacks(square: Square) -> Bitboard {
    KING_ATTACKS[square.index() as usize]
}

/// Walk each diagonal from `square`, including the first occupied square.
pub(crate) fn bishop_attacks(square: Square, occupied: Bitboard) -> Bitboard {
    ray_attacks(square, occupied, &BISHOP_DELTAS)
}

/// Walk each rank/file from `square`, including the first occupied square.
pub(crate) fn rook_attacks(square: Square, occupied: Bitboard) -> Bitboard {
    ray_attacks(square, occupied, &ROOK_DELTAS)
}

pub(crate) fn queen_attacks(square: Square, occupied: Bitboard) -> Bitboard {
    bishop_attacks(square, occupied).union(rook_attacks(square, occupied))
}

fn ray_attacks(square: Square, occupied: Bitboard, deltas: &[(i8, i8)]) -> Bitboard {
    let mut attacks = Bitboard::EMPTY;
    let file = square.file() as i8;
    let rank = square.rank() as i8;
    for &(file_delta, rank_delta) in deltas {
        let mut target_file = file + file_delta;
        let mut target_rank = rank + rank_delta;
        while (0..8).contains(&target_file) && (0..8).contains(&target_rank) {
            let target = Square::from_file_rank(target_file as u8, target_rank as u8);
            attacks = attacks.with(target);
            if occupied.contains(target) {
                break;
            }
            target_file += file_delta;
            target_rank += rank_delta;
        }
    }
    attacks
}

/// Whether `by` attacks `square` in the given board.
pub(crate) fn is_square_attacked(board: Board, square: Square, by: Color) -> bool {
    let pawns = board.pieces(by, Role::Pawn);
    if pawn_attacks(by.opposite(), square).intersect(pawns).any() {
        return true;
    }
    if knight_attacks(square)
        .intersect(board.pieces(by, Role::Knight))
        .any()
    {
        return true;
    }
    if king_attacks(square)
        .intersect(board.pieces(by, Role::King))
        .any()
    {
        return true;
    }

    let occupied = board.occupied();
    let diagonal_attackers = board
        .pieces(by, Role::Bishop)
        .union(board.pieces(by, Role::Queen));
    if bishop_attacks(square, occupied)
        .intersect(diagonal_attackers)
        .any()
    {
        return true;
    }

    let straight_attackers = board
        .pieces(by, Role::Rook)
        .union(board.pieces(by, Role::Queen));
    rook_attacks(square, occupied)
        .intersect(straight_attackers)
        .any()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pawn_attacks_are_directional() {
        let e4 = Square::from_ascii("e4").unwrap();
        assert_eq!(
            pawn_attacks(Color::White, e4),
            Bitboard::from_square(Square::from_ascii("d5").unwrap())
                .with(Square::from_ascii("f5").unwrap())
        );
        assert_eq!(
            pawn_attacks(Color::Black, e4),
            Bitboard::from_square(Square::from_ascii("d3").unwrap())
                .with(Square::from_ascii("f3").unwrap())
        );
    }

    #[test]
    fn edge_squares_do_not_wrap() {
        let a1 = Square::A1;
        for target in pawn_attacks(Color::Black, a1).iter() {
            assert_eq!(target.file(), 1);
        }
        let h8 = Square::H8;
        for target in pawn_attacks(Color::White, h8).iter() {
            assert_eq!(target.file(), 6);
        }
        assert!(knight_attacks(a1).contains(Square::from_ascii("b3").unwrap()));
        assert!(!knight_attacks(a1).contains(Square::from_ascii("h2").unwrap()));
    }

    #[test]
    fn rook_ray_stops_at_blocker() {
        let a1 = Square::A1;
        let a4 = Square::from_ascii("a4").unwrap();
        let occupied = Bitboard::from_square(a4);
        let attacks = rook_attacks(a1, occupied);
        assert!(attacks.contains(Square::from_ascii("a2").unwrap()));
        assert!(attacks.contains(a4));
        assert!(!attacks.contains(Square::from_ascii("a5").unwrap()));
    }
}
