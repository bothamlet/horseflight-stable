use core::fmt;

use crate::board::Board;
use crate::movegen;
use crate::position::Position;
use crate::types::{CastlingRights, CastlingSide, Color, Piece, Role, Square};

/// Errors from FEN parsing. Parsing performs structural/basic validation, not
/// full reachability validation: it does not prove that a position can arise in
/// a game, and it does not reject positions with a side-not-to-move in check.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum FenError {
    NonAscii,
    WrongFieldCount { found: usize },
    WrongRankCount { found: usize },
    RankWidth { rank: u8, width: u8 },
    InvalidPieceChar { ch: char },
    InvalidSide { ch: char },
    InvalidCastling { text: String },
    InvalidEnPassant { text: String },
    InvalidCounter { field: &'static str, text: String },
    CounterTooLarge { field: &'static str },
    FullmoveZero,
    MissingKing(Color),
    ExtraKing(Color),
    PawnOnBackRank(Square),
    CastlingKingNotHome(Color),
    CastlingRookNotHome(Color, CastlingSide),
}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FenError::NonAscii => f.write_str("FEN must be ASCII"),
            FenError::WrongFieldCount { found } => {
                write!(f, "FEN must have 6 fields, found {found}")
            }
            FenError::WrongRankCount { found } => {
                write!(f, "FEN board must have 8 ranks, found {found}")
            }
            FenError::RankWidth { rank, width } => {
                write!(f, "FEN rank {rank} has width {width}, expected 8")
            }
            FenError::InvalidPieceChar { ch } => write!(f, "invalid FEN piece character {ch:?}"),
            FenError::InvalidSide { ch } => write!(f, "invalid FEN side-to-move {ch:?}"),
            FenError::InvalidCastling { text } => {
                write!(f, "invalid FEN castling field {text:?}")
            }
            FenError::InvalidEnPassant { text } => {
                write!(f, "invalid FEN en-passant field {text:?}")
            }
            FenError::InvalidCounter { field, text } => {
                write!(f, "invalid {field} counter {text:?}")
            }
            FenError::CounterTooLarge { field } => {
                write!(f, "{field} counter exceeds u32::MAX")
            }
            FenError::FullmoveZero => f.write_str("fullmove number must be at least 1"),
            FenError::MissingKing(color) => write!(f, "missing {color} king"),
            FenError::ExtraKing(color) => write!(f, "more than one {color} king"),
            FenError::PawnOnBackRank(square) => {
                write!(f, "pawn on back rank at {square}")
            }
            FenError::CastlingKingNotHome(color) => {
                write!(f, "{color} castling right without king on home square")
            }
            FenError::CastlingRookNotHome(color, side) => {
                write!(
                    f,
                    "{color} {side:?} castling right without rook on home square"
                )
            }
        }
    }
}

impl std::error::Error for FenError {}

pub(crate) fn parse(fen: &str) -> Result<Position, FenError> {
    if !fen.is_ascii() {
        return Err(FenError::NonAscii);
    }

    let fields: Vec<&str> = fen.split_ascii_whitespace().collect();
    if fields.len() != 6 {
        return Err(FenError::WrongFieldCount {
            found: fields.len(),
        });
    }

    let board = parse_board(fields[0])?;
    validate_kings_and_pawns(board)?;
    let side_to_move = parse_side(fields[1])?;
    let castling = parse_castling(fields[2], board)?;
    let ep_square = parse_ep_square(fields[3], board, side_to_move)?;
    let halfmove_clock = parse_counter(fields[4], "halfmove")?;
    let fullmove_number = parse_counter(fields[5], "fullmove")?;
    if fullmove_number == 0 {
        return Err(FenError::FullmoveZero);
    }

    Ok(Position::from_parts(
        board,
        side_to_move,
        castling,
        ep_square,
        halfmove_clock,
        fullmove_number,
    ))
}

fn parse_board(text: &str) -> Result<Board, FenError> {
    let ranks: Vec<&str> = text.split('/').collect();
    if ranks.len() != 8 {
        return Err(FenError::WrongRankCount { found: ranks.len() });
    }

    let mut board = Board::empty();
    for (index, rank_text) in ranks.iter().enumerate() {
        let rank = 7 - index as u8;
        let mut file = 0u8;
        for ch in rank_text.chars() {
            if let Some(digit) = ch.to_digit(10) {
                if digit == 0 || digit > 8 {
                    return Err(FenError::RankWidth {
                        rank: rank + 1,
                        width: file + digit as u8,
                    });
                }
                file += digit as u8;
            } else {
                let Some(piece) = Piece::from_fen_char(ch) else {
                    return Err(FenError::InvalidPieceChar { ch });
                };
                if file >= 8 {
                    return Err(FenError::RankWidth {
                        rank: rank + 1,
                        width: file + 1,
                    });
                }
                board.set_piece(Square::from_file_rank(file, rank), Some(piece));
                file += 1;
            }
            if file > 8 {
                return Err(FenError::RankWidth {
                    rank: rank + 1,
                    width: file,
                });
            }
        }
        if file != 8 {
            return Err(FenError::RankWidth {
                rank: rank + 1,
                width: file,
            });
        }
    }
    Ok(board)
}

fn validate_kings_and_pawns(board: Board) -> Result<(), FenError> {
    for color in Color::ALL {
        match board.pieces(color, Role::King).count() {
            1 => {}
            0 => return Err(FenError::MissingKing(color)),
            _ => return Err(FenError::ExtraKing(color)),
        }
    }

    for color in Color::ALL {
        let mut pawns = board.pieces(color, Role::Pawn);
        while let Some(square) = pawns.pop_lsb() {
            if square.rank() == 0 || square.rank() == 7 {
                return Err(FenError::PawnOnBackRank(square));
            }
        }
    }
    Ok(())
}

fn parse_side(text: &str) -> Result<Color, FenError> {
    let bytes = text.as_bytes();
    if bytes.len() == 1 {
        match bytes[0] {
            b'w' => return Ok(Color::White),
            b'b' => return Ok(Color::Black),
            _ => {}
        }
    }
    let ch = text.chars().next().unwrap_or('\0');
    Err(FenError::InvalidSide { ch })
}

fn parse_castling(text: &str, board: Board) -> Result<CastlingRights, FenError> {
    if text == "-" {
        return Ok(CastlingRights::NONE);
    }
    if text.is_empty() {
        return Err(FenError::InvalidCastling {
            text: text.to_string(),
        });
    }

    let mut rights = CastlingRights::NONE;
    for ch in text.chars() {
        let (color, side) = match ch {
            'K' => (Color::White, CastlingSide::KingSide),
            'Q' => (Color::White, CastlingSide::QueenSide),
            'k' => (Color::Black, CastlingSide::KingSide),
            'q' => (Color::Black, CastlingSide::QueenSide),
            _ => {
                return Err(FenError::InvalidCastling {
                    text: text.to_string(),
                })
            }
        };
        if rights.has(color, side) {
            return Err(FenError::InvalidCastling {
                text: text.to_string(),
            });
        }
        rights = rights.with(color, side);
    }

    for color in Color::ALL {
        if (rights.has(color, CastlingSide::KingSide) || rights.has(color, CastlingSide::QueenSide))
            && board.piece_at(CastlingSide::king_square(color))
                != Some(Piece::new(color, Role::King))
        {
            return Err(FenError::CastlingKingNotHome(color));
        }
        for side in [CastlingSide::KingSide, CastlingSide::QueenSide] {
            if rights.has(color, side)
                && board.piece_at(side.rook_square(color)) != Some(Piece::new(color, Role::Rook))
            {
                return Err(FenError::CastlingRookNotHome(color, side));
            }
        }
    }
    Ok(rights)
}

fn parse_ep_square(
    text: &str,
    board: Board,
    side_to_move: Color,
) -> Result<Option<Square>, FenError> {
    if text == "-" {
        return Ok(None);
    }
    let Some(square) = Square::from_ascii(text) else {
        return Err(FenError::InvalidEnPassant {
            text: text.to_string(),
        });
    };
    if !movegen::en_passant_is_structural(board, square, side_to_move) {
        return Err(FenError::InvalidEnPassant {
            text: text.to_string(),
        });
    }
    Ok(Some(square))
}

fn parse_counter(text: &str, field: &'static str) -> Result<u32, FenError> {
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(FenError::InvalidCounter {
            field,
            text: text.to_string(),
        });
    }
    let mut value = 0u32;
    for byte in text.bytes() {
        value = value
            .checked_mul(10)
            .and_then(|value| value.checked_add((byte - b'0') as u32))
            .ok_or(FenError::CounterTooLarge { field })?;
    }
    Ok(value)
}

pub(crate) fn serialize(position: &Position) -> String {
    let mut text = String::with_capacity(90);
    for rank in (0..8).rev() {
        let mut empty = 0u8;
        for file in 0..8 {
            match position
                .board()
                .piece_at(Square::from_file_rank(file, rank))
            {
                Some(piece) => {
                    if empty != 0 {
                        text.push((b'0' + empty) as char);
                        empty = 0;
                    }
                    text.push(piece.fen_char());
                }
                None => empty += 1,
            }
        }
        if empty != 0 {
            text.push((b'0' + empty) as char);
        }
        if rank != 0 {
            text.push('/');
        }
    }
    text.push(' ');
    text.push(position.side_to_move().fen_char());
    text.push(' ');
    text.push_str(&position.castling_rights().to_fen());
    text.push(' ');
    match position.ep_square() {
        Some(square) => text.push_str(&square.to_string()),
        None => text.push('-'),
    }
    text.push(' ');
    text.push_str(&position.halfmove_clock().to_string());
    text.push(' ');
    text.push_str(&position.fullmove_number().to_string());
    text
}
