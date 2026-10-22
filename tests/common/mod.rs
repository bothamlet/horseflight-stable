#![allow(dead_code)]

use std::collections::BTreeSet;

use horseflight::{Game, Move, Position, Square};

pub const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub const KIWIPETE_FEN: &str =
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

pub fn sq(text: &str) -> Square {
    Square::from_ascii(text).unwrap_or_else(|| panic!("invalid test square {text:?}"))
}

pub fn position(fen: &str) -> Position {
    Position::from_fen(fen).unwrap_or_else(|error| panic!("invalid test FEN {fen:?}: {error}"))
}

pub fn game(fen: &str) -> Game {
    Game::from_fen(fen).unwrap_or_else(|error| panic!("invalid test FEN {fen:?}: {error}"))
}

pub fn normal(from: &str, to: &str) -> Move {
    Move::normal(sq(from), sq(to))
}

pub fn promotion(from: &str, to: &str, role: horseflight::Role) -> Move {
    Move::promotion(sq(from), sq(to), role)
}

pub fn uci_move(position: &Position, text: &str) -> Move {
    position
        .legal_moves()
        .iter()
        .copied()
        .find(|movement| movement.to_uci() == text)
        .unwrap_or_else(|| panic!("{text} is not legal in {}", position.to_fen()))
}

pub fn play_uci(game: &mut Game, text: &str) {
    let movement = uci_move(game.position(), text);
    game.play(movement)
        .unwrap_or_else(|error| panic!("failed to play {text}: {error}"));
}

pub fn legal_uci(position: &Position) -> BTreeSet<String> {
    position
        .legal_moves()
        .iter()
        .map(|movement| movement.to_uci())
        .collect()
}

pub fn assert_valid(position: &Position) {
    if let Err(error) = position.validate_invariants() {
        panic!(
            "position invariant failure: {error}\nFEN: {}",
            position.to_fen()
        );
    }
    assert_eq!(
        position.hash(),
        position.recompute_hash(),
        "incremental hash disagrees with recomputation in {}",
        position.to_fen()
    );
}

/// Vertically mirror `fen`: reverse the rank order and swap piece colours,
/// producing the colour-swapped equivalent position used by symmetry tests.
pub fn mirror_fen(fen: &str) -> String {
    let fields: Vec<&str> = fen.split_whitespace().collect();
    assert_eq!(fields.len(), 6, "FEN must have six fields: {fen}");
    let placement: Vec<String> = fields[0]
        .split('/')
        .rev()
        .map(|rank| {
            rank.chars()
                .map(|ch| {
                    if ch.is_ascii_uppercase() {
                        ch.to_ascii_lowercase()
                    } else if ch.is_ascii_lowercase() {
                        ch.to_ascii_uppercase()
                    } else {
                        ch
                    }
                })
                .collect()
        })
        .collect();
    let side = if fields[1] == "w" { "b" } else { "w" };
    // White kingside becomes black kingside and vice versa.
    let mut rights = [false; 4];
    for ch in fields[2].chars() {
        match ch {
            'K' => rights[2] = true,
            'Q' => rights[3] = true,
            'k' => rights[0] = true,
            'q' => rights[1] = true,
            '-' => {}
            other => panic!("unexpected castling character {other:?} in {fen}"),
        }
    }
    let castling: String = ['K', 'Q', 'k', 'q']
        .into_iter()
        .enumerate()
        .filter(|(index, _)| rights[*index])
        .map(|(_, ch)| ch)
        .collect();
    let castling = if castling.is_empty() {
        "-".to_string()
    } else {
        castling
    };
    let ep = match fields[3] {
        "-" => "-".to_string(),
        square => {
            let bytes = square.as_bytes();
            let file = bytes[0] as char;
            let rank = 9 - i32::from(bytes[1] - b'0');
            format!("{file}{rank}")
        }
    };
    format!(
        "{} {side} {castling} {ep} {} {}",
        placement.join("/"),
        fields[4],
        fields[5]
    )
}

/// Play a coordinate move sequence from a FEN to build real game history.
pub fn walk(fen: &str, moves: &[&str]) -> Game {
    let mut game = game(fen);
    for text in moves {
        play_uci(&mut game, text);
    }
    game
}
