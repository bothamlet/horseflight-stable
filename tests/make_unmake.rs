mod common;

use common::{assert_valid, normal, position, sq, uci_move, KIWIPETE_FEN, STARTPOS_FEN};
use horseflight::{Move, Piece, Position, Role, Undo};

const FIXTURES: &[&str] = &[
    STARTPOS_FEN,
    KIWIPETE_FEN,
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "n1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/1p6/N1R1K3 b - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
];

fn assert_delta_inversion(before: &Position, after: &Position, undo: &Undo) {
    let changes = undo.piece_changes();
    assert!(
        !changes.is_empty() && changes.len() <= 4,
        "unexpected piece delta count {}",
        changes.len()
    );

    let mut restored = *after.board().squares();
    for change in changes.iter().rev() {
        assert_eq!(
            after.board().piece_at(change.square),
            change.after,
            "delta does not describe the post-make board at {}",
            change.square
        );
        restored[change.square.index() as usize] = change.before;
    }
    assert_eq!(
        &restored,
        before.board().squares(),
        "reverse replay of piece deltas does not restore the board"
    );

    let mut forward = *before.board().squares();
    for change in changes {
        assert_eq!(
            before.board().piece_at(change.square),
            change.before,
            "delta does not describe the pre-make board at {}",
            change.square
        );
        forward[change.square.index() as usize] = change.after;
    }
    assert_eq!(
        &forward,
        after.board().squares(),
        "forward replay of piece deltas does not match the post-make board"
    );
}

#[test]
fn every_legal_move_round_trips_exactly_with_hash_and_deltas() {
    for fen in FIXTURES {
        let original = position(fen);
        assert_valid(&original);
        let moves: Vec<Move> = original.legal_moves().iter().copied().collect();
        assert!(!moves.is_empty(), "fixture has no legal moves: {fen}");
        for movement in moves {
            let mut trial = original;
            let before = trial;
            let undo = trial
                .play(movement)
                .unwrap_or_else(|error| panic!("{} failed in {fen}: {error}", movement.to_uci()));
            assert_valid(&trial);
            assert_delta_inversion(&before, &trial, &undo);
            assert_eq!(undo.movement(), movement);
            trial
                .unmake(&undo)
                .unwrap_or_else(|error| panic!("unmake {} failed: {error}", movement.to_uci()));
            assert_eq!(trial, before, "unmake changed state after {movement}");
            assert_valid(&trial);
        }
    }
}

#[test]
fn whole_sequences_unwind_to_the_exact_start() {
    for fen in FIXTURES {
        let start = position(fen);
        let mut trial = start;
        let mut undos = Vec::new();
        for ply in 0..24 {
            let moves: Vec<Move> = trial.legal_moves().iter().copied().collect();
            if moves.is_empty() {
                break;
            }
            let movement = moves[(ply * 7 + 3) % moves.len()];
            let undo = trial.play(movement).expect("generated move is legal");
            assert_valid(&trial);
            undos.push(undo);
        }
        while let Some(undo) = undos.pop() {
            trial.unmake(&undo).unwrap();
            assert_valid(&trial);
        }
        assert_eq!(trial, start, "sequence did not unwind exactly for {fen}");
    }
}

#[test]
fn special_moves_record_complete_piece_deltas() {
    let castle = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";

    for (text, king_to, rook_to) in [("e1g1", "g1", "f1"), ("e1c1", "c1", "d1")] {
        let mut position = position(castle);
        let before = position;
        let movement = uci_move(&position, text);
        let undo = position.play(movement).unwrap();
        assert!(undo.movement().is_castle());
        assert_eq!(undo.piece_changes().len(), 4);
        assert_eq!(
            position.board().piece_at(sq(king_to)),
            Some(Piece::new(horseflight::Color::White, Role::King))
        );
        assert_eq!(
            position.board().piece_at(sq(rook_to)),
            Some(Piece::new(horseflight::Color::White, Role::Rook))
        );
        assert_delta_inversion(&before, &position, &undo);
        position.unmake(&undo).unwrap();
        assert_eq!(position, before);
    }

    let mut ep = position("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    let before = ep;
    let undo = ep.play(uci_move(&ep, "e5d6")).unwrap();
    assert!(undo.movement().is_en_passant());
    assert_eq!(undo.piece_changes().len(), 3);
    assert_eq!(
        undo.captured(),
        Some(Piece::new(horseflight::Color::Black, Role::Pawn))
    );
    assert_eq!(ep.board().piece_at(sq("d5")), None);
    assert_eq!(
        ep.board().piece_at(sq("d6")),
        Some(Piece::new(horseflight::Color::White, Role::Pawn))
    );
    assert_delta_inversion(&before, &ep, &undo);
    ep.unmake(&undo).unwrap();
    assert_eq!(ep, before);

    let mut promotion = position("n1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1");
    let before = promotion;
    let undo = promotion.play(uci_move(&promotion, "b7c8q")).unwrap();
    assert_eq!(undo.piece_changes().len(), 2);
    assert_eq!(
        undo.captured(),
        Some(Piece::new(horseflight::Color::Black, Role::Rook))
    );
    assert_eq!(
        promotion.board().piece_at(sq("c8")),
        Some(Piece::new(horseflight::Color::White, Role::Queen))
    );
    assert_delta_inversion(&before, &promotion, &undo);
    promotion.unmake(&undo).unwrap();
    assert_eq!(promotion, before);

    let mut quiet_promotion = position("n1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1");
    let undo = quiet_promotion
        .play(uci_move(&quiet_promotion, "b7b8n"))
        .unwrap();
    assert_eq!(
        quiet_promotion.board().piece_at(sq("b8")),
        Some(Piece::new(horseflight::Color::White, Role::Knight))
    );
    assert_delta_inversion(&before, &quiet_promotion, &undo);
    quiet_promotion.unmake(&undo).unwrap();
    assert_eq!(quiet_promotion.board().piece_at(sq("b8")), None);
}

#[test]
fn undo_layout_is_fixed_and_compact() {
    let position_size = std::mem::size_of::<Position>();
    let undo_size = std::mem::size_of::<Undo>();
    let change_size = std::mem::size_of::<horseflight::PieceChange>();
    eprintln!("Position={position_size} Undo={undo_size} PieceChange={change_size}");
    assert!(
        undo_size <= 96,
        "undo record grew unexpectedly: {undo_size}"
    );
    assert!(
        change_size <= 8,
        "piece change record is not compact: {change_size}"
    );
}

#[test]
fn stale_undo_for_repeated_key_with_different_clocks_is_rejected() {
    let start = Position::startpos();
    let mut position = start;
    let first_undo = position.play(normal("g1", "f3")).unwrap();
    let after_first = position;
    let mut undos = vec![first_undo];
    for text in ["g8f6", "f3g1", "f6g8", "g1f3"] {
        undos.push(position.play(common::uci_move(&position, text)).unwrap());
    }

    let snapshot = position;
    assert_eq!(position.hash(), after_first.hash());
    assert_eq!(position.side_to_move(), after_first.side_to_move());
    assert_ne!(position, after_first, "clock fields should differ");
    assert_eq!(position.halfmove_clock(), 5);
    assert_eq!(position.fullmove_number(), 3);

    assert!(position.unmake(&first_undo).is_err());
    assert_eq!(position, snapshot);
    assert_eq!(position.hash(), position.recompute_hash());

    for undo in undos.iter().rev() {
        position.unmake(undo).unwrap();
        assert_valid(&position);
        assert_eq!(position.hash(), position.recompute_hash());
    }
    assert_eq!(position, start);
}

#[test]
fn unmake_rejects_records_out_of_lifo_order_without_mutation() {
    let mut position = Position::startpos();
    let first = position.play(normal("g1", "f3")).unwrap();
    let second = position.play(normal("g8", "f6")).unwrap();
    let snapshot = position;

    assert!(position.unmake(&first).is_err());
    assert_eq!(position, snapshot);

    position.unmake(&second).unwrap();
    position.unmake(&first).unwrap();
    assert_eq!(position, Position::startpos());
}

#[test]
fn public_play_rejects_illegal_moves_atomically() {
    let cases: &[(&str, Move)] = &[
        (STARTPOS_FEN, normal("e2", "e5")),
        (STARTPOS_FEN, normal("e7", "e5")),
        (STARTPOS_FEN, normal("a1", "a8")),
        ("4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1", normal("e2", "c3")),
        (
            "4k3/8/8/8/8/8/4r3/4K2R w K - 0 1",
            Move::castle(sq("e1"), sq("g1")),
        ),
    ];
    for (fen, movement) in cases {
        let mut position = position(fen);
        let before = position;
        assert!(!position.is_legal(*movement));
        assert!(
            position.play(*movement).is_err(),
            "accepted illegal move {movement} in {fen}"
        );
        assert_eq!(position, before, "illegal move mutated {fen}");
    }
}
