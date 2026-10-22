mod common;

use common::{mirror_fen, position};

use horseflight::eval::{
    clamp_evaluation, is_mate_score, mate_distance_plies, Evaluator, HandcraftedEvaluator,
    MaterialEvaluator, BISHOP_VALUE, EVAL_MAX, EVAL_MIN, INFINITY, KNIGHT_VALUE, MATE_BAND_FLOOR,
    MATE_MAX_PLY, MATE_SCORE, PAWN_VALUE, QUEEN_VALUE, ROOK_VALUE,
};
use horseflight::{search, Game, Position, SearchAction, SearchConfig};

fn config(depth: u32) -> SearchConfig {
    SearchConfig {
        max_depth: depth,
        node_limit: 200_000,
        ..SearchConfig::default()
    }
}

#[test]
fn material_evaluator_accounts_for_every_piece_value() {
    let evaluator = MaterialEvaluator;
    let cases = [
        ("4k3/8/8/8/8/8/8/4K1N1 w - - 0 1", KNIGHT_VALUE),
        ("4k3/8/8/8/8/8/8/4K1B1 w - - 0 1", BISHOP_VALUE),
        ("4k3/8/8/8/8/8/8/4K2R w - - 0 1", ROOK_VALUE),
        ("4k3/8/8/8/8/8/8/4K1Q1 w - - 0 1", QUEEN_VALUE),
        ("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", PAWN_VALUE),
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0),
        ("4k3/8/8/8/8/8/8/R3K2R w - - 0 1", 2 * ROOK_VALUE),
        (
            "4k3/8/8/8/8/8/8/R3K1NR b - - 0 1",
            -(2 * ROOK_VALUE + KNIGHT_VALUE),
        ),
        (
            "4k3/8/8/8/8/8/8/R3K1N1 w - - 0 1",
            ROOK_VALUE + KNIGHT_VALUE,
        ),
        ("4k3/8/8/8/8/8/8/R3K2Q w - - 0 1", ROOK_VALUE + QUEEN_VALUE),
    ];
    for (fen, expected) in cases {
        assert_eq!(
            evaluator.evaluate(&position(fen)),
            expected,
            "material score of {fen}"
        );
    }
}

#[test]
fn evaluation_is_signed_from_the_side_to_move_perspective() {
    let handcrafted = HandcraftedEvaluator;
    for (placement, castling, ep) in [
        ("4k3/8/8/8/8/8/8/R3K3", "-", "-"),
        ("r3k3/8/8/8/8/8/8/4K3", "-", "-"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", "KQkq", "-"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR", "KQkq", "-"),
    ] {
        let fen = format!("{placement} w {castling} {ep} 0 1");
        let white_to_move = position(&fen);
        let black_to_move = position(&format!("{placement} b {castling} {ep} 0 1"));
        assert_eq!(
            handcrafted.evaluate(&white_to_move),
            -handcrafted.evaluate(&black_to_move),
            "side-to-move sign for {fen}"
        );
    }
}

#[test]
fn handcrafted_evaluation_is_invariant_under_colour_and_rank_mirroring() {
    let evaluator = HandcraftedEvaluator;
    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/8/8/8/8/2n1b3/8/4K3 b - - 3 12",
        "8/8/8/8/8/5k2/6q1/7K w - - 0 1",
        "r1bq1r2/ppp3pp/2n1k3/3pP3/3P4/2P2N2/PP3PPP/R2QKB1R b KQ - 2 11",
    ] {
        // Mirroring swaps colours, ranks and the side to move together, so the
        // advantage of the side to move is unchanged.
        let original = position(fen);
        let mirrored_fen = mirror_fen(fen);
        let mirrored = position(&mirrored_fen);
        assert_eq!(
            evaluator.evaluate(&original),
            evaluator.evaluate(&mirrored),
            "mirror invariance for {fen} -> {mirrored_fen}"
        );
        assert_eq!(original.side_to_move(), mirrored.side_to_move().opposite());
        assert_eq!(
            MaterialEvaluator.evaluate(&original),
            MaterialEvaluator.evaluate(&mirrored)
        );
    }
}

#[test]
fn positional_terms_reward_pawn_advancement_and_minor_centralisation() {
    let evaluator = HandcraftedEvaluator;
    let back_rank_pawn = evaluator.evaluate(&position("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"));
    let advanced_pawn = evaluator.evaluate(&position("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"));
    assert!(
        advanced_pawn > back_rank_pawn,
        "advanced {advanced_pawn} vs {back_rank_pawn}"
    );
    assert_eq!(
        evaluator.evaluate(&position("4k3/8/8/8/8/8/8/4K3 w - - 0 1")),
        0,
        "kings carry no material and no positional term"
    );

    let corner_knight = evaluator.evaluate(&position("4k3/8/8/8/8/8/8/N3K3 w - - 0 1"));
    let central_knight = evaluator.evaluate(&position("4k3/8/8/8/4N3/8/8/4K3 w - - 0 1"));
    assert!(
        central_knight > corner_knight,
        "central {central_knight} vs corner {corner_knight}"
    );

    // Material dominates every positional term.
    let queen_down = evaluator.evaluate(&position("4k3/8/8/8/8/8/8/4K1q1 w - - 0 1"));
    assert!(queen_down < -QUEEN_VALUE / 2);
}

#[test]
fn evaluation_scores_are_clamped_into_the_finite_band() {
    assert_eq!(clamp_evaluation(i32::MAX), EVAL_MAX);
    assert_eq!(clamp_evaluation(i32::MIN), EVAL_MIN);
    assert_eq!(clamp_evaluation(42), 42);
    assert!(!is_mate_score(EVAL_MAX));
    assert!(!is_mate_score(EVAL_MIN));
    assert_eq!(mate_distance_plies(EVAL_MAX), None);
    assert_eq!(mate_distance_plies(MATE_SCORE - 3), Some(3));
    assert_eq!(mate_distance_plies(-(MATE_SCORE - 5)), Some(5));
}

#[test]
fn mate_classification_is_limited_to_the_two_bounded_intervals() {
    assert_eq!(MATE_BAND_FLOOR, MATE_SCORE - MATE_MAX_PLY as i32);
    // (score, is a mate, distance in plies).
    let cases: [(i32, bool, Option<u32>); 14] = [
        // The two admitted closed intervals and their exact endpoints.
        (MATE_SCORE, true, Some(0)),
        (-MATE_SCORE, true, Some(0)),
        (MATE_BAND_FLOOR, true, Some(MATE_MAX_PLY)),
        (-MATE_BAND_FLOOR, true, Some(MATE_MAX_PLY)),
        (MATE_SCORE - 1, true, Some(1)),
        (-(MATE_SCORE - 1), true, Some(1)),
        // Just outside each interval on the finite side.
        (MATE_BAND_FLOOR - 1, false, None),
        (-MATE_BAND_FLOOR + 1, false, None),
        // Beyond a mate score at the root, including the window bound.
        (MATE_SCORE + 1, false, None),
        (-(MATE_SCORE + 1), false, None),
        (INFINITY, false, None),
        (-INFINITY, false, None),
        (i32::MAX, false, None),
        (i32::MIN, false, None),
    ];
    for (score, expected_mate, expected_distance) in cases {
        assert_eq!(
            is_mate_score(score),
            expected_mate,
            "is_mate_score({score})"
        );
        assert_eq!(
            mate_distance_plies(score),
            expected_distance,
            "mate_distance_plies({score})"
        );
    }
    // A delivered mate is distance zero, not a fabricated ply.
    assert_eq!(mate_distance_plies(MATE_SCORE), Some(0));
    assert_eq!(mate_distance_plies(-MATE_SCORE), Some(0));
}

#[test]
fn a_wild_custom_evaluator_cannot_forge_a_mate_or_overflow() {
    struct HugeEvaluator;
    impl Evaluator for HugeEvaluator {
        fn evaluate(&self, _position: &Position) -> i32 {
            i32::MAX
        }
    }

    let game = Game::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
    let result = search(&game, &HugeEvaluator, &config(2)).unwrap();
    assert_eq!(result.stop, horseflight::SearchStop::DepthReached);
    let score = result.score.expect("a completed search scores");
    assert_eq!(score, EVAL_MAX, "clamped to the evaluation band");
    assert!(!is_mate_score(score));
    assert!(matches!(result.action, SearchAction::Move(_)));
}

#[test]
fn search_scores_come_only_from_the_supplied_evaluator() {
    struct WhiteBias;
    impl Evaluator for WhiteBias {
        fn evaluate(&self, position: &Position) -> i32 {
            match position.side_to_move() {
                horseflight::Color::White => 1_000,
                horseflight::Color::Black => -1_000,
            }
        }
    }

    let game = Game::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
    let biased = search(&game, &WhiteBias, &config(1)).unwrap();
    assert_eq!(biased.completed_depth, 1);
    assert_eq!(biased.score, Some(1_000));

    let material = search(&game, &MaterialEvaluator, &config(1)).unwrap();
    assert_eq!(material.score, Some(PAWN_VALUE));

    let handcrafted = search(&game, &HandcraftedEvaluator, &config(1)).unwrap();
    assert!(handcrafted.score.unwrap() > PAWN_VALUE);
}

#[test]
fn evaluator_choice_changes_the_reported_score_without_changing_legality() {
    let game = Game::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
    let material = search(&game, &MaterialEvaluator, &config(2)).unwrap();
    let handcrafted = search(&game, &HandcraftedEvaluator, &config(2)).unwrap();
    assert_ne!(material.score, handcrafted.score);
    for result in [&material, &handcrafted] {
        let SearchAction::Move(movement) = result.action else {
            panic!("expected a searched move, got {:?}", result.action);
        };
        assert!(game.position().is_legal(movement));
    }
}
