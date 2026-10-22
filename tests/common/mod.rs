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
