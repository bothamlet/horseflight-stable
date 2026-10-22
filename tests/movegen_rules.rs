mod common;

use common::{assert_valid, legal_uci, normal, position, sq, uci_move, KIWIPETE_FEN};
use horseflight::{Color, Piece, Position, Role};

const CASTLE_FEN: &str = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";

fn contains(position: &Position, text: &str) -> bool {
    legal_uci(position).contains(text)
}

#[test]
fn castling_is_legal_only_when_in_check_transit_and_destination_are_safe() {
    let legal = position(CASTLE_FEN);
    assert!(contains(&legal, "e1g1"));
    assert!(contains(&legal, "e1c1"));

    let in_check = position("4k3/8/8/8/8/8/4r3/4K2R w K - 0 1");
    assert!(!contains(&in_check, "e1g1"));

    let through_check = position("4kr2/8/8/8/8/8/8/4K2R w K - 0 1");
    assert!(!contains(&through_check, "e1g1"));

    let into_check = position("4k1r1/8/8/8/8/8/8/4K2R w K - 0 1");
    assert!(!contains(&into_check, "e1g1"));

    let queenside_clear = position("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1");
    assert!(contains(&queenside_clear, "e1c1"));

    let blocked_by_knight = position("4k3/8/8/8/8/8/8/RN2K3 w Q - 0 1");
    assert!(!contains(&blocked_by_knight, "e1c1"));

    let transit_attacked = position("3rk3/8/8/8/8/8/8/R3K3 w Q - 0 1");
    assert!(!contains(&transit_attacked, "e1c1"));

    let destination_attacked = position("2r1k3/8/8/8/8/8/8/R3K3 w Q - 0 1");
    assert!(!contains(&destination_attacked, "e1c1"));

    // b1 need only be empty, not safe: the king never crosses it.
    let b_file_attacked = position("1r2k3/8/8/8/8/8/8/R3K3 w Q - 0 1");
    assert!(contains(&b_file_attacked, "e1c1"));
}

#[test]
fn castling_applies_and_unmakes_both_sides() {
    for (text, king_to, rook_to, rook_from) in [
        ("e1g1", "g1", "f1", "h1"),
        ("e1c1", "c1", "d1", "a1"),
        ("e8g8", "g8", "f8", "h8"),
        ("e8c8", "c8", "d8", "a8"),
    ] {
        let mut black_to_move = position("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
        if text.starts_with("e1") {
            let mut white_to_move = position(CASTLE_FEN);
            let before = white_to_move;
            let movement = uci_move(&white_to_move, text);
            let undo = white_to_move.play(movement).unwrap();
            let color = Color::White;
            assert_eq!(
                white_to_move.board().piece_at(sq(king_to)),
                Some(Piece::new(color, Role::King))
            );
            assert_eq!(
                white_to_move.board().piece_at(sq(rook_to)),
                Some(Piece::new(color, Role::Rook))
            );
            assert_eq!(white_to_move.board().piece_at(sq(rook_from)), None);
            assert!(!white_to_move
                .castling_rights()
                .has(color, horseflight::CastlingSide::KingSide));
            assert!(!white_to_move
                .castling_rights()
                .has(color, horseflight::CastlingSide::QueenSide));
            assert_valid(&white_to_move);
            white_to_move.unmake(&undo).unwrap();
            assert_eq!(white_to_move, before);
        } else {
            let before = black_to_move;
            let movement = uci_move(&black_to_move, text);
            let undo = black_to_move.play(movement).unwrap();
            let color = Color::Black;
            assert_eq!(
                black_to_move.board().piece_at(sq(king_to)),
                Some(Piece::new(color, Role::King))
            );
            assert_eq!(
                black_to_move.board().piece_at(sq(rook_to)),
                Some(Piece::new(color, Role::Rook))
            );
            assert_eq!(black_to_move.board().piece_at(sq(rook_from)), None);
            assert!(!black_to_move
                .castling_rights()
                .has(color, horseflight::CastlingSide::KingSide));
            assert!(!black_to_move
                .castling_rights()
                .has(color, horseflight::CastlingSide::QueenSide));
            black_to_move.unmake(&undo).unwrap();
            assert_eq!(black_to_move, before);
        }
    }
}

#[test]
fn rights_are_lost_on_king_moves_rook_moves_and_rook_captures() {
    let mut king_move = position(CASTLE_FEN);
    let undo = king_move.play(normal("e1", "e2")).unwrap();
    assert!(!king_move
        .castling_rights()
        .has(Color::White, horseflight::CastlingSide::KingSide));
    assert!(!king_move
        .castling_rights()
        .has(Color::White, horseflight::CastlingSide::QueenSide));
    assert!(king_move
        .castling_rights()
        .has(Color::Black, horseflight::CastlingSide::KingSide));
    assert!(king_move
        .castling_rights()
        .has(Color::Black, horseflight::CastlingSide::QueenSide));
    king_move.unmake(&undo).unwrap();
    assert_eq!(king_move.castling_rights().to_fen(), "KQkq");

    let mut queenside_rook = position(CASTLE_FEN);
    queenside_rook.play(normal("a1", "a2")).unwrap();
    assert_eq!(queenside_rook.castling_rights().to_fen(), "Kkq");

    let mut kingside_rook = position(CASTLE_FEN);
    kingside_rook.play(normal("h1", "h2")).unwrap();
    assert_eq!(kingside_rook.castling_rights().to_fen(), "Qkq");

    let mut black_rook_captures_a1 = position("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
    let undo = black_rook_captures_a1.play(normal("a8", "a1")).unwrap();
    assert_eq!(black_rook_captures_a1.castling_rights().to_fen(), "Kk");
    black_rook_captures_a1.unmake(&undo).unwrap();
    assert_eq!(black_rook_captures_a1.castling_rights().to_fen(), "KQkq");

    let mut black_rook_captures_h1 = position("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
    black_rook_captures_h1.play(normal("h8", "h1")).unwrap();
    assert_eq!(black_rook_captures_h1.castling_rights().to_fen(), "Qq");
}

#[test]
fn en_passant_legal_pin_expiry_and_apply() {
    let legal = position("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    assert_eq!(legal.legal_ep_square(), Some(sq("d6")));
    assert!(contains(&legal, "e5d6"));
    assert!(uci_move(&legal, "e5d6").is_en_passant());

    let pinned = position("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    assert_eq!(pinned.legal_ep_square(), None);
    assert!(!contains(&pinned, "e5d6"));

    let mut applied = position("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    let before = applied;
    let undo = applied.play(uci_move(&applied, "e5d6")).unwrap();
    assert_eq!(applied.board().piece_at(sq("d5")), None);
    assert_eq!(
        applied.board().piece_at(sq("d6")),
        Some(Piece::new(Color::White, Role::Pawn))
    );
    assert_eq!(applied.ep_square(), None);
    applied.unmake(&undo).unwrap();
    assert_eq!(applied, before);

    let mut expiry = position("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
    assert_eq!(expiry.legal_ep_square(), Some(sq("d6")));
    let undo = expiry.play(normal("a2", "a3")).unwrap();
    assert_eq!(expiry.ep_square(), None);
    assert_eq!(expiry.legal_ep_square(), None);
    assert!(!contains(&expiry, "e5d6"));
    expiry.unmake(&undo).unwrap();
    assert_eq!(expiry.legal_ep_square(), Some(sq("d6")));

    let mut double_push = Position::startpos();
    double_push.play(normal("e2", "e4")).unwrap();
    assert_eq!(double_push.ep_square(), Some(sq("e3")));
    assert_eq!(double_push.legal_ep_square(), None);
}

#[test]
fn en_passant_rank_xray_exposes_rook_attack_for_both_colours() {
    let cases = [
        (
            "7k/8/8/r2pP2K/8/8/8/8 w - d6 0 1",
            "7k/8/8/r2pP2K/8/8/8/8 w - - 0 1",
            "e5d6",
        ),
        (
            "7K/8/8/8/R2Pp2k/8/8/8 b - d3 0 1",
            "7K/8/8/8/R2Pp2k/8/8/8 b - - 0 1",
            "e4d3",
        ),
    ];
    for (with_ep_fen, without_ep_fen, ep_move) in cases {
        let with_ep = position(with_ep_fen);
        assert_eq!(with_ep.ep_square(), Some(sq(&ep_move[2..4])));
        assert_eq!(with_ep.legal_ep_square(), None);
        assert!(
            !contains(&with_ep, ep_move),
            "{ep_move} should expose the king to a rank rook attack"
        );
        assert!(!with_ep.in_check());

        let without_ep = position(without_ep_fen);
        assert_eq!(with_ep.hash(), without_ep.hash());
        assert!(with_ep.same_repetition_identity(&without_ep));
        assert_valid(&with_ep);
        assert_valid(&without_ep);
    }
}

#[test]
fn all_capture_and_quiet_underpromotions_exist_for_both_colours() {
    for (fen, from, targets) in [
        (
            "n1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
            "b7",
            ["b8", "a8", "c8"],
        ),
        (
            "4k3/8/8/8/8/8/1p6/N1R1K3 b - - 0 1",
            "b2",
            ["b1", "a1", "c1"],
        ),
    ] {
        let legal = position(fen);
        let mut promotions = 0;
        for target in targets {
            for suffix in ["n", "b", "r", "q"] {
                let text = format!("{from}{target}{suffix}");
                assert!(contains(&legal, &text), "missing promotion {text} in {fen}");
                promotions += 1;
            }
        }
        assert_eq!(
            legal
                .legal_moves()
                .iter()
                .filter(|movement| movement.is_promotion())
                .count(),
            promotions,
            "unexpected promotion count in {fen}"
        );

        for target in targets {
            for (role, suffix) in [
                (Role::Knight, "n"),
                (Role::Bishop, "b"),
                (Role::Rook, "r"),
                (Role::Queen, "q"),
            ] {
                let text = format!("{from}{target}{suffix}");
                let mut trial = legal;
                let before = trial;
                let movement = uci_move(&trial, &text);
                let undo = trial.play(movement).unwrap();
                let expected_color = if from == "b7" {
                    Color::White
                } else {
                    Color::Black
                };
                assert_eq!(
                    trial.board().piece_at(sq(target)),
                    Some(Piece::new(expected_color, role)),
                    "promotion {text} placed the wrong piece"
                );
                assert_eq!(trial.board().piece_at(sq(from)), None);
                assert_valid(&trial);
                trial.unmake(&undo).unwrap();
                assert_eq!(trial, before);
            }
        }
    }
}

#[test]
fn pins_and_double_check_are_filtered_by_own_king_safety() {
    let pinned = position("4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1");
    assert_eq!(
        pinned
            .legal_moves()
            .iter()
            .filter(|movement| movement.from() == sq("e2"))
            .count(),
        0,
        "pinned knight escaped the pin"
    );

    let double_check = position("4r1k1/8/8/8/1b6/8/8/4K3 w - - 0 1");
    assert!(double_check.in_check());
    let moves: Vec<_> = double_check.legal_moves().iter().copied().collect();
    assert!(!moves.is_empty());
    assert!(
        moves.iter().all(|movement| movement.from() == sq("e1")),
        "non-king move generated in double check"
    );

    for fen in [
        KIWIPETE_FEN,
        CASTLE_FEN,
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    ] {
        let start = position(fen);
        for movement in start.legal_moves().iter().copied() {
            let mut trial = start;
            trial.play(movement).unwrap();
            let white_king = trial.board().king_square(Color::White).unwrap();
            let black_king = trial.board().king_square(Color::Black).unwrap();
            let file_distance = white_king.file().abs_diff(black_king.file());
            let rank_distance = white_king.rank().abs_diff(black_king.rank());
            assert!(
                file_distance > 1 || rank_distance > 1,
                "kings adjacent after {} in {fen}",
                movement.to_uci()
            );
        }
    }
}

#[test]
fn checkmate_and_stalemate_are_distinguished() {
    let mate = position("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
    assert!(mate.is_checkmate());
    assert!(!mate.is_stalemate());
    assert!(mate.legal_moves().is_empty());

    let stalemate = position("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    assert!(!stalemate.in_check());
    assert!(stalemate.is_stalemate());
    assert!(!stalemate.is_checkmate());
    assert!(stalemate.legal_moves().is_empty());
}
