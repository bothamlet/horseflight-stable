mod common;

use common::{assert_valid, normal, position, sq, STARTPOS_FEN};
use horseflight::{Game, MoveError, Position};

const ROUND_TRIP_FENS: &[&str] = &[
    STARTPOS_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "n1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
];

#[test]
fn fen_round_trips_and_canonical_startpos() {
    for fen in ROUND_TRIP_FENS {
        let parsed = position(fen);
        assert_valid(&parsed);
        assert_eq!(&parsed.to_fen(), fen, "FEN round trip changed the position");
    }
    assert_eq!(Position::startpos().to_fen(), STARTPOS_FEN);
}

#[test]
fn uncapturable_but_structural_ep_is_preserved() {
    let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    let parsed = position(fen);
    assert_eq!(parsed.ep_square(), Some(sq("e3")));
    assert_eq!(parsed.legal_ep_square(), None);
    assert_eq!(parsed.to_fen(), fen);

    let without_ep = position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    assert_eq!(parsed.hash(), without_ep.hash());
    assert!(parsed.same_repetition_identity(&without_ep));
}

#[test]
fn legal_and_illegal_ep_fields_are_distinguished() {
    let legal = position("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    assert_eq!(legal.legal_ep_square(), Some(sq("d6")));
    let no_ep = position("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1");
    assert_ne!(legal.hash(), no_ep.hash());
    assert!(!legal.same_repetition_identity(&no_ep));

    let pinned = position("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    assert_eq!(pinned.legal_ep_square(), None);
    let pinned_no_ep = position("k3r3/8/8/3pP3/8/8/8/4K3 w - - 0 1");
    assert_eq!(pinned.hash(), pinned_no_ep.hash());
    assert!(pinned.same_repetition_identity(&pinned_no_ep));
}

#[test]
fn clocks_are_excluded_from_the_hash() {
    let a = position("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    let b = position("4k3/8/8/8/8/8/8/4K3 w - - 99 77");
    assert_eq!(a.hash(), b.hash());
    assert_eq!(a.to_fen(), "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    assert_eq!(b.to_fen(), "4k3/8/8/8/8/8/8/4K3 w - - 99 77");
}

#[test]
fn side_to_move_and_castling_rights_are_hashed() {
    let white = position("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    let black = position("4k3/8/8/8/8/8/8/4K3 b - - 0 1");
    assert_ne!(white.hash(), black.hash());

    let with_rights = position("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    let without = position("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1");
    assert_ne!(with_rights.hash(), without.hash());
    assert!(!with_rights.same_repetition_identity(&without));
}

#[test]
fn malformed_fens_are_errors_not_panics() {
    let cases = [
        "",
        "    ",
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "9/8/8/8/8/8/8/8 w - - 0 1",
        "8/8/8/8/8/8/8 w - - 0 1",
        "8/8/8/8/8/8/8/8/8 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KK - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - -1 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 4294967296 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 extra",
        "♔7/8/8/8/8/8/8/7k w - - 0 1",
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "1k6/8/8/8/8/8/8/7K w KQkq - 0 1",
    ];
    for fen in cases {
        assert!(
            Position::from_fen(fen).is_err(),
            "expected FEN error for {fen:?}"
        );
        if let Err(error) = Position::from_fen(fen) {
            assert!(!error.to_string().is_empty());
        }
    }
}

#[test]
fn truncated_and_mutated_fens_never_panic() {
    let valid = STARTPOS_FEN;
    for end in 0..=valid.len() {
        let _ = Position::from_fen(&valid[..end]);
    }
    for (index, _) in valid.char_indices() {
        for replacement in ['x', '9', '0', '/', ' ', '-', '♔'] {
            let mut mutated = valid.to_string();
            mutated.replace_range(index..index + 1, &replacement.to_string());
            let _ = Position::from_fen(&mutated);
        }
    }
}

#[test]
fn ep_structural_validation_rejects_bad_targets() {
    let cases = [
        // Wrong rank for the side to move.
        "4k3/8/8/3pP3/8/8/8/4K3 w - d5 0 1",
        // Target occupied.
        "4k3/8/3N4/3pP3/8/8/8/4K3 w - d6 0 1",
        // Pushed pawn missing.
        "4k3/8/8/4P3/8/8/8/4K3 w - d6 0 1",
        // Origin square occupied.
        "4k3/3r4/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ];
    for fen in cases {
        assert!(
            Position::from_fen(fen).is_err(),
            "accepted bad EP FEN {fen:?}"
        );
    }
}

#[test]
fn counters_accept_u32_limits_and_reject_overflow_without_mutation() {
    let high = position("4k3/8/8/8/8/8/8/4K3 w - - 4294967295 4294967295");
    assert_eq!(high.halfmove_clock(), u32::MAX);
    assert_eq!(high.fullmove_number(), u32::MAX);

    let mut quiet = position("4k3/8/8/8/8/8/8/4K3 w - - 4294967295 1");
    let before = quiet;
    let error = quiet.play(normal("e1", "e2")).unwrap_err();
    assert_eq!(error, MoveError::CounterOverflow);
    assert_eq!(quiet, before);

    let mut black_to_move = position("4k3/8/8/8/8/8/8/4K3 b - - 0 4294967295");
    let before = black_to_move;
    let error = black_to_move.play(normal("e8", "e7")).unwrap_err();
    assert_eq!(error, MoveError::CounterOverflow);
    assert_eq!(black_to_move, before);

    let mut zeroing = position("4k3/8/8/8/8/8/4P3/4K3 w - - 4294967295 1");
    let movement = common::uci_move(&zeroing, "e2e3");
    zeroing
        .play(movement)
        .expect("zeroing move resets the counter");
    assert_eq!(zeroing.halfmove_clock(), 0);
}

#[test]
fn fen_history_resets_on_parse() {
    let mut game = Game::from_fen(STARTPOS_FEN).unwrap();
    for text in [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ] {
        common::play_uci(&mut game, text);
    }
    assert!(game.threefold_claim_available());

    let fresh = Game::from_fen(STARTPOS_FEN).unwrap();
    assert_eq!(fresh.history_len(), 0);
    assert!(!fresh.threefold_claim_available());
    assert_eq!(fresh.positions().len(), 1);
}
