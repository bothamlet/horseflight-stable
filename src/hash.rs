use crate::types::{CastlingRights, CastlingSide, Color, Piece, Square};

/// Fixed seed for the deterministic SplitMix64 Zobrist stream.
pub const ZOBRIST_SEED: u64 = 0x484F_5253_4546_4C54; // "HORSEFLT"

const SPLITMIX_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_MULTIPLIER_1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_MULTIPLIER_2: u64 = 0x94D0_49BB_1331_11EB;

const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_INCREMENT);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_MULTIPLIER_1);
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_MULTIPLIER_2);
    value ^ (value >> 31)
}

/// Zobrist tables are generated in a fixed order:
/// 12 piece slots (colour-major, then role) x 64 squares, four castling
/// rights in `KQkq` order, eight en-passant files, then the black-to-move key.
pub(crate) struct Zobrist {
    pieces: [[u64; 64]; 12],
    castling: [u64; 4],
    ep_files: [u64; 8],
    black_to_move: u64,
}

impl Zobrist {
    const fn generate(seed: u64) -> Zobrist {
        let mut state = seed;
        let mut pieces = [[0u64; 64]; 12];
        let mut piece = 0usize;
        while piece < 12 {
            let mut square = 0usize;
            while square < 64 {
                pieces[piece][square] = splitmix64(&mut state);
                square += 1;
            }
            piece += 1;
        }

        let mut castling = [0u64; 4];
        let mut index = 0usize;
        while index < 4 {
            castling[index] = splitmix64(&mut state);
            index += 1;
        }

        let mut ep_files = [0u64; 8];
        index = 0;
        while index < 8 {
            ep_files[index] = splitmix64(&mut state);
            index += 1;
        }

        let black_to_move = splitmix64(&mut state);

        Zobrist {
            pieces,
            castling,
            ep_files,
            black_to_move,
        }
    }
}

pub(crate) const ZOBRIST: Zobrist = Zobrist::generate(ZOBRIST_SEED);

pub(crate) const fn piece_square_key(piece: Piece, square: Square) -> u64 {
    ZOBRIST.pieces[piece.index()][square.index() as usize]
}

pub(crate) fn castling_key(rights: CastlingRights) -> u64 {
    let mut key = 0u64;
    for color in Color::ALL {
        for side in [CastlingSide::KingSide, CastlingSide::QueenSide] {
            if rights.has(color, side) {
                key ^= ZOBRIST.castling[CastlingRights::index_of(color, side)];
            }
        }
    }
    key
}

pub(crate) const fn ep_file_key(file: u8) -> u64 {
    ZOBRIST.ep_files[file as usize]
}

pub(crate) const fn side_key(color: Color) -> u64 {
    match color {
        Color::White => 0,
        Color::Black => ZOBRIST.black_to_move,
    }
}
