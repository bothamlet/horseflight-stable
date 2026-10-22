mod common;

use common::{game, legal_uci};

use horseflight::{
    search, Evaluator, Game, HandcraftedEvaluator, Move, SearchAction, SearchConfig, SearchStop,
    STARTPOS_FEN,
};

fn config(depth: u32, node_limit: u64) -> SearchConfig {
    SearchConfig {
        max_depth: depth,
        node_limit,
        ..SearchConfig::default()
    }
}

const SEARCH_POSITIONS: [(&str, &str); 6] = [
    ("start", STARTPOS_FEN),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ),
    (
        "middlegame",
        "r1bq1r2/ppp3pp/2n1k3/3pP3/3P4/2P2N2/PP3PPP/R2QKB1R b KQ - 2 11",
    ),
    ("pawn endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("rook endgame", "8/8/8/8/8/2k5/2p5/R3K3 w - - 0 1"),
    (
        "open tactical",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    ),
];

#[test]
fn selected_moves_are_legal_and_the_principal_variation_replays() {
    let evaluator = HandcraftedEvaluator;
    for (label, fen) in SEARCH_POSITIONS {
        let game = common::game(fen);
        for depth in 1..=3 {
            let result = search(&game, &evaluator, &config(depth, 2_000_000)).unwrap();
            assert_eq!(
                result.stop,
                SearchStop::DepthReached,
                "{label} depth {depth} should complete"
            );
            assert_eq!(result.completed_depth, depth, "{label} depth {depth}");
            let SearchAction::Move(movement) = result.action else {
                panic!(
                    "{label} depth {depth}: expected a move, got {:?}",
                    result.action
                );
            };
            assert!(
                game.position().is_legal(movement),
                "{label}: {} is illegal",
                movement.to_uci()
            );
            assert_eq!(
                result.pv.first(),
                Some(&movement),
                "{label}: PV starts with the move"
            );
            assert!(
                result.pv.len() <= depth as usize,
                "{label}: PV longer than depth"
            );

            // Replay the reported line on a fresh game built from the same FEN.
            let mut replay = common::game(fen);
            for step in &result.pv {
                replay.play(*step).unwrap_or_else(|error| {
                    panic!(
                        "{label} depth {depth}: PV move {} is not legal after {}: {error}",
                        step.to_uci(),
                        replay.position().to_fen()
                    )
                });
            }
            assert_eq!(result.score.is_some(), result.completed_depth > 0);
        }
    }
}

#[test]
fn pvs_stay_legal_with_ordering_disabled_and_early_terminals() {
    // Generation order searches many non-best moves before a late mate, so a
    // stale sibling PV would surface as an illegal replay here.
    let evaluator = HandcraftedEvaluator;
    for (label, fen) in [
        ("mate in one", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
        ("mate in two", "k7/8/2K5/8/8/3Q4/8/8 w - - 0 1"),
        ("terminal child", "4k3/8/8/2p5/3p4/8/8/3RK3 w - - 0 1"),
    ] {
        let game = common::game(fen);
        for depth in 1..=3 {
            for ordering in [true, false] {
                let result = search(
                    &game,
                    &evaluator,
                    &SearchConfig {
                        max_depth: depth,
                        node_limit: 2_000_000,
                        ordering,
                        quiescence: true,
                        q_depth_limit: 4,
                    },
                )
                .unwrap();
                assert_eq!(result.stop, SearchStop::DepthReached, "{label} {depth}");
                let mut replay = common::game(fen);
                for (index, step) in result.pv.iter().enumerate() {
                    replay.play(*step).unwrap_or_else(|error| {
                        panic!(
                            "{label} depth {depth} ordering {ordering}: PV move {index} {} is illegal after {}: {error}",
                            step.to_uci(),
                            replay.position().to_fen()
                        )
                    });
                }
                assert!(result.pv.len() <= depth as usize, "{label} {depth}");
            }
        }
    }
}

#[test]
fn caller_position_history_and_hash_are_unchanged() {
    let evaluator = HandcraftedEvaluator;
    let mut original = game(STARTPOS_FEN);
    for movement in ["g1f3", "g8f6", "f3g1", "f6g8", "b1c3"] {
        let parsed = common::uci_move(original.position(), movement);
        original.play(parsed).unwrap();
    }
    let snapshot = original.clone();
    let result = search(&original, &evaluator, &config(3, 500_000)).unwrap();
    assert_eq!(result.completed_depth, 3);

    assert_eq!(original.position(), snapshot.position());
    assert_eq!(original.position().hash(), snapshot.position().hash());
    assert_eq!(original.position().to_fen(), snapshot.position().to_fen());
    assert_eq!(original.positions().len(), snapshot.positions().len());
    assert_eq!(
        original.moves().collect::<Vec<Move>>(),
        snapshot.moves().collect::<Vec<Move>>()
    );
    assert_eq!(original.history_len(), snapshot.history_len());
    assert_eq!(original.current_repetition_count(), 1);
}

#[test]
fn fixed_configuration_is_deterministic() {
    let evaluator = HandcraftedEvaluator;
    for (label, fen) in SEARCH_POSITIONS {
        let game = common::game(fen);
        for quiescence in [true, false] {
            for ordering in [true, false] {
                let config = SearchConfig {
                    max_depth: 2,
                    node_limit: 300_000,
                    ordering,
                    quiescence,
                    q_depth_limit: 4,
                };
                let first = search(&game, &evaluator, &config).unwrap();
                let second = search(&game, &evaluator, &config).unwrap();
                assert_eq!(first.action, second.action, "{label}");
                assert_eq!(first.score, second.score, "{label}");
                assert_eq!(first.completed_depth, second.completed_depth, "{label}");
                assert_eq!(first.pv, second.pv, "{label}");
                assert_eq!(first.nodes, second.nodes, "{label}");
                assert_eq!(first.q_nodes, second.q_nodes, "{label}");
                assert_eq!(first.stop, second.stop, "{label}");
                assert_eq!(first.claim, second.claim, "{label}");
            }
        }
    }
}

#[test]
fn zero_node_budget_returns_an_unsearched_fallback() {
    let evaluator = HandcraftedEvaluator;
    let game = game(STARTPOS_FEN);
    let result = search(&game, &evaluator, &config(2, 0)).unwrap();
    assert_eq!(result.nodes, 0);
    assert_eq!(result.q_nodes, 0);
    assert_eq!(result.completed_depth, 0);
    assert_eq!(result.score, None, "a fallback must not report a score");
    assert!(result.pv.is_empty());
    assert_eq!(result.stop, SearchStop::NodeLimit);
    let SearchAction::Fallback(movement) = result.action else {
        panic!("expected a fallback, got {:?}", result.action);
    };
    assert!(game.position().is_legal(movement));
    // Deterministic: the first legal move in generation order.
    assert_eq!(movement, game.position().legal_moves()[0]);
}

#[test]
fn the_node_budget_is_never_exceeded_and_completion_is_truthful() {
    let evaluator = HandcraftedEvaluator;
    let game = game("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    for limit in 0..=400u64 {
        let result = search(&game, &evaluator, &config(3, limit)).unwrap();
        assert!(result.nodes <= limit, "limit {limit} used {}", result.nodes);
        assert!(result.q_nodes <= result.nodes);
        match result.completed_depth {
            0 => {
                assert_eq!(result.score, None, "limit {limit}");
                assert!(result.pv.is_empty(), "limit {limit}");
                assert!(matches!(result.action, SearchAction::Fallback(_)));
                assert_ne!(result.stop, SearchStop::DepthReached, "limit {limit}");
            }
            depth => {
                assert!(result.score.is_some(), "limit {limit}");
                assert!(result.pv.len() <= depth as usize, "limit {limit}");
                assert!(matches!(result.action, SearchAction::Move(_)));
                if depth < 3 {
                    assert_eq!(result.stop, SearchStop::NodeLimit, "limit {limit}");
                }
            }
        }
    }
}

#[test]
fn an_exact_node_budget_completes_depth_one_and_one_less_does_not() {
    let evaluator = HandcraftedEvaluator;
    let game = game("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    let completed = search(&game, &evaluator, &config(1, 100_000)).unwrap();
    assert_eq!(completed.stop, SearchStop::DepthReached);
    assert_eq!(completed.completed_depth, 1);

    let exact = search(&game, &evaluator, &config(1, completed.nodes)).unwrap();
    assert_eq!(exact.stop, SearchStop::DepthReached);
    assert_eq!(exact.completed_depth, 1);
    assert_eq!(exact.action, completed.action);
    assert_eq!(exact.score, completed.score);
    assert_eq!(exact.pv, completed.pv);

    let short = search(&game, &evaluator, &config(1, completed.nodes - 1)).unwrap();
    assert_eq!(short.stop, SearchStop::NodeLimit);
    assert_eq!(short.completed_depth, 0);
    assert_eq!(short.score, None);
    assert!(matches!(short.action, SearchAction::Fallback(_)));
}

#[test]
fn an_interrupted_iteration_keeps_only_the_last_completed_result() {
    let evaluator = HandcraftedEvaluator;
    let game = game("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    let depth_two = search(&game, &evaluator, &config(2, 2_000_000)).unwrap();
    assert_eq!(depth_two.completed_depth, 2);

    let interrupted = search(&game, &evaluator, &config(4, depth_two.nodes + 1)).unwrap();
    assert_eq!(interrupted.stop, SearchStop::NodeLimit);
    assert_eq!(interrupted.completed_depth, 2);
    assert_eq!(interrupted.score, depth_two.score);
    assert_eq!(interrupted.action, depth_two.action);
    assert_eq!(interrupted.pv, depth_two.pv);
    assert!(interrupted.pv.len() <= 2);
    assert!(!interrupted.pv.is_empty());
}

#[test]
fn invalid_configurations_are_rejected_before_searching() {
    let evaluator = HandcraftedEvaluator;
    let game = Game::startpos();
    for config in [
        SearchConfig {
            max_depth: 0,
            ..SearchConfig::default()
        },
        SearchConfig {
            max_depth: horseflight::search::MAX_MAIN_DEPTH + 1,
            ..SearchConfig::default()
        },
        SearchConfig {
            q_depth_limit: horseflight::search::MAX_Q_DEPTH + 1,
            ..SearchConfig::default()
        },
        SearchConfig {
            max_depth: horseflight::search::MAX_MAIN_DEPTH,
            q_depth_limit: horseflight::search::MAX_Q_DEPTH,
            ..SearchConfig::default()
        },
    ] {
        let error = search(&game, &evaluator, &config).unwrap_err();
        assert!(error.to_string().contains("depth"), "{error}");
    }
    // The absolute ply cap is what ties the two individual ceilings together.
    let boundary = SearchConfig {
        max_depth: horseflight::search::MAX_MAIN_DEPTH,
        q_depth_limit: horseflight::search::MAX_PLY - horseflight::search::MAX_MAIN_DEPTH,
        ..SearchConfig::default()
    };
    assert!(boundary.validate().is_ok());
    let over_cap = SearchConfig {
        q_depth_limit: horseflight::search::MAX_PLY - horseflight::search::MAX_MAIN_DEPTH + 1,
        ..boundary
    };
    assert_eq!(
        over_cap.validate(),
        Err(horseflight::search::ConfigError::PlyCapExceeded {
            needed: horseflight::search::MAX_PLY + 1,
            limit: horseflight::search::MAX_PLY,
        })
    );
}

#[test]
fn counter_overflow_is_reported_as_a_transition_stop_without_mutation() {
    // Black to move at the largest fullmove number: every legal move overflows
    // the u32 counter, so no iteration can complete.
    let evaluator = HandcraftedEvaluator;
    let game = game("7k/8/8/8/8/8/8/KQ6 b - - 0 4294967295");
    let snapshot = game.clone();
    let result = search(&game, &evaluator, &config(2, 100_000)).unwrap();
    assert_eq!(result.stop, SearchStop::Transition);
    assert_eq!(result.completed_depth, 0);
    assert_eq!(result.score, None);
    let SearchAction::Fallback(movement) = result.action else {
        panic!("expected a fallback, got {:?}", result.action);
    };
    assert!(game.position().is_legal(movement));
    assert_eq!(game.position(), snapshot.position());
    assert_eq!(game.history_len(), snapshot.history_len());
}

#[test]
fn every_searched_move_is_legal_and_scored_for_the_side_to_move() {
    struct CountingEvaluator;
    impl Evaluator for CountingEvaluator {
        fn evaluate(&self, position: &horseflight::Position) -> i32 {
            let side = position.side_to_move();
            let board = position.board();
            let mut score = 0;
            for role in horseflight::Role::ALL {
                let value = horseflight::eval::role_value(role);
                score += value * board.pieces(side, role).count() as i32;
                score -= value * board.pieces(side.opposite(), role).count() as i32;
            }
            score
        }
    }

    let game = game("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    let result = search(&game, &CountingEvaluator, &config(3, 200_000)).unwrap();
    let SearchAction::Move(movement) = result.action else {
        panic!("expected a move");
    };
    assert!(legal_uci(game.position()).contains(&movement.to_uci()));
    let score = result.score.expect("material-only search scores");
    assert!(score.abs() <= horseflight::eval::EVAL_MAX);
    assert!(!horseflight::eval::is_mate_score(score));
}
