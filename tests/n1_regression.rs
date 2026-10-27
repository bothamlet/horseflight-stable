//! Mission N1 regression snapshots captured from the unchanged original
//! evaluation/search implementation at `master` HEAD 1446a74.
//!
//! These are behavioural fixtures, not tuning targets: every value here was
//! produced by the pre-refactor code and must remain byte-identical after the
//! evaluator module split and the generic-dispatch change. `nodes`/`q_nodes`,
//! action/claim/score/completed depth, PV and stop are deterministic; wall time
//! and NPS are deliberately not part of the snapshot.

mod common;

use common::position;

use horseflight::eval::{Evaluator, HandcraftedEvaluator, MaterialEvaluator};
use horseflight::{
    search, search_controlled, DrawClaim, Game, Outcome, SearchAction, SearchConfig, SearchControl,
    SearchResult, SearchStop, STARTPOS_FEN,
};

const EVAL_CASES: [(&str, &str, i32, i32); 9] = [
    ("start", STARTPOS_FEN, 0, 0),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        -12,
        0,
    ),
    (
        "middlegame",
        "r1bq1r2/ppp3pp/2n1k3/3pP3/3P4/2P2N2/PP3PPP/R2QKB1R b KQ - 2 11",
        -224,
        -200,
    ),
    ("en-passant", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", 6, 0),
    (
        "promotion",
        "n1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
        -690,
        -720,
    ),
    ("checkmate", "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", -900, -900),
    ("stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", -900, -900),
    ("insufficient", "8/8/8/8/8/8/8/K1B4k w - - 0 1", 330, 330),
    ("fifty-move", "7k/8/8/8/q7/8/8/7K w - - 100 60", -900, -900),
];

const LEGAL_CASES: [(&str, &str, &str); 7] = [
    (
        "start",
        STARTPOS_FEN,
        "a2a3 a2a4 b1a3 b1c3 b2b3 b2b4 c2c3 c2c4 d2d3 d2d4 e2e3 e2e4 f2f3 f2f4 g1f3 g1h3 g2g3 g2g4 h2h3 h2h4",
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "a1b1 a1c1 a1d1 a2a3 a2a4 b2b3 c3a4 c3b1 c3b5 c3d1 d2c1 d2e3 d2f4 d2g5 d2h6 d5d6 d5e6 e1c1 e1d1 e1f1 e1g1 e2a6 e2b5 e2c4 e2d1 e2d3 e2f1 e5c4 e5c6 e5d3 e5d7 e5f7 e5g4 e5g6 f3d3 f3e3 f3f4 f3f5 f3f6 f3g3 f3g4 f3h3 f3h5 g2g3 g2g4 g2h3 h1f1 h1g1",
    ),
    (
        "castling",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "a1a2 a1a3 a1a4 a1a5 a1a6 a1a7 a1a8 a1b1 a1c1 a1d1 e1c1 e1d1 e1d2 e1e2 e1f1 e1f2 e1g1 h1f1 h1g1 h1h2 h1h3 h1h4 h1h5 h1h6 h1h7 h1h8",
    ),
    (
        "en-passant",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "e1d1 e1d2 e1e2 e1f1 e1f2 e5d6 e5e6",
    ),
    (
        "promotion",
        "n1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
        "b7a8b b7a8n b7a8q b7a8r b7b8b b7b8n b7b8q b7b8r b7c8b b7c8n b7c8q b7c8r e1d1 e1d2 e1e2 e1f1 e1f2",
    ),
    (
        "checkmate",
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
        "",
    ),
    (
        "stalemate",
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        "",
    ),
];

struct SearchSnapshot {
    label: &'static str,
    fen: &'static str,
    depth: u32,
    node_limit: u64,
    action: &'static str,
    claim: Option<&'static str>,
    score: Option<i32>,
    completed_depth: u32,
    pv: &'static [&'static str],
    nodes: u64,
    q_nodes: u64,
    stop: SearchStop,
}

const SEARCH_SNAPSHOTS: [SearchSnapshot; 10] = [
    SearchSnapshot {
        label: "start-d1",
        fen: STARTPOS_FEN,
        depth: 1,
        node_limit: 200_000,
        action: "move a2a4",
        claim: None,
        score: Some(12),
        completed_depth: 1,
        pv: &["a2a4"],
        nodes: 21,
        q_nodes: 20,
        stop: SearchStop::DepthReached,
    },
    SearchSnapshot {
        label: "start-d2",
        fen: STARTPOS_FEN,
        depth: 2,
        node_limit: 200_000,
        action: "move a2a4",
        claim: None,
        score: Some(0),
        completed_depth: 2,
        pv: &["a2a4", "a7a5"],
        nodes: 96,
        q_nodes: 74,
        stop: SearchStop::DepthReached,
    },
    SearchSnapshot {
        label: "kiwipete-d2",
        fen: "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        depth: 2,
        node_limit: 2_000_000,
        action: "move e2a6",
        claim: None,
        score: Some(-14),
        completed_depth: 2,
        pv: &["e2a6", "b4c3"],
        nodes: 2219,
        q_nodes: 2169,
        stop: SearchStop::DepthReached,
    },
    SearchSnapshot {
        label: "middlegame-d2",
        fen: "r1bq1r2/ppp3pp/2n1k3/3pP3/3P4/2P2N2/PP3PPP/R2QKB1R b KQ - 2 11",
        depth: 2,
        node_limit: 2_000_000,
        action: "move a7a5",
        claim: None,
        score: Some(-224),
        completed_depth: 2,
        pv: &["a7a5", "a2a4"],
        nodes: 492,
        q_nodes: 455,
        stop: SearchStop::DepthReached,
    },
    SearchSnapshot {
        label: "promotion-d1",
        fen: "n1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
        depth: 1,
        node_limit: 200_000,
        action: "move b7c8q",
        claim: None,
        score: Some(900),
        completed_depth: 1,
        pv: &["b7c8q"],
        nodes: 24,
        q_nodes: 23,
        stop: SearchStop::DepthReached,
    },
    SearchSnapshot {
        label: "en-passant-d1",
        fen: "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        depth: 1,
        node_limit: 200_000,
        action: "move e5d6",
        claim: None,
        score: Some(124),
        completed_depth: 1,
        pv: &["e5d6"],
        nodes: 8,
        q_nodes: 7,
        stop: SearchStop::DepthReached,
    },
    SearchSnapshot {
        label: "checkmate",
        fen: "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
        depth: 2,
        node_limit: 200_000,
        action: "terminal checkmate",
        claim: None,
        score: Some(-32_000),
        completed_depth: 0,
        pv: &[],
        nodes: 0,
        q_nodes: 0,
        stop: SearchStop::TerminalRoot,
    },
    SearchSnapshot {
        label: "stalemate",
        fen: "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        depth: 2,
        node_limit: 200_000,
        action: "terminal stalemate",
        claim: None,
        score: Some(0),
        completed_depth: 0,
        pv: &[],
        nodes: 0,
        q_nodes: 0,
        stop: SearchStop::TerminalRoot,
    },
    SearchSnapshot {
        label: "insufficient",
        fen: "8/8/8/8/8/8/8/K1B4k w - - 0 1",
        depth: 2,
        node_limit: 200_000,
        action: "terminal insufficient material",
        claim: None,
        score: Some(0),
        completed_depth: 0,
        pv: &[],
        nodes: 0,
        q_nodes: 0,
        stop: SearchStop::TerminalRoot,
    },
    SearchSnapshot {
        label: "fifty-move-claim",
        fen: "7k/8/8/8/q7/8/8/7K w - - 100 60",
        depth: 2,
        node_limit: 200_000,
        action: "claim current",
        claim: Some("current"),
        score: Some(0),
        completed_depth: 2,
        pv: &[],
        nodes: 8,
        q_nodes: 3,
        stop: SearchStop::DepthReached,
    },
];

fn action_text(action: SearchAction) -> String {
    match action {
        SearchAction::Move(movement) => format!("move {}", movement.to_uci()),
        SearchAction::Claim(DrawClaim::Current) => "claim current".to_string(),
        SearchAction::Claim(DrawClaim::Intended(movement)) => {
            format!("claim intended {}", movement.to_uci())
        }
        SearchAction::Terminal(outcome) => match outcome {
            Outcome::Checkmate { .. } => "terminal checkmate".to_string(),
            Outcome::Stalemate => "terminal stalemate".to_string(),
            Outcome::InsufficientMaterial => "terminal insufficient material".to_string(),
            Outcome::FivefoldRepetition => "terminal fivefold repetition".to_string(),
            Outcome::SeventyFiveMoveDraw => "terminal seventy-five-move draw".to_string(),
            Outcome::Ongoing => "terminal ongoing".to_string(),
        },
        SearchAction::Fallback(movement) => format!("fallback {}", movement.to_uci()),
    }
}

fn claim_text(claim: Option<DrawClaim>) -> Option<String> {
    claim.map(|claim| match claim {
        DrawClaim::Current => "current".to_string(),
        DrawClaim::Intended(movement) => format!("intended {}", movement.to_uci()),
    })
}

fn pv_text(result: &SearchResult) -> Vec<String> {
    result.pv.iter().map(|m| m.to_uci()).collect()
}

fn assert_search_snapshot(snapshot: &SearchSnapshot) {
    let game = Game::from_fen(snapshot.fen).expect("valid fixture FEN");
    let config = SearchConfig {
        max_depth: snapshot.depth,
        node_limit: snapshot.node_limit,
        ..SearchConfig::default()
    };
    let result = search(&game, &HandcraftedEvaluator, &config).expect("valid search config");
    assert_eq!(
        action_text(result.action),
        snapshot.action,
        "{} action",
        snapshot.label
    );
    assert_eq!(
        claim_text(result.claim).as_deref(),
        snapshot.claim,
        "{} claim",
        snapshot.label
    );
    assert_eq!(result.score, snapshot.score, "{} score", snapshot.label);
    assert_eq!(
        result.completed_depth, snapshot.completed_depth,
        "{} completed depth",
        snapshot.label
    );
    assert_eq!(
        pv_text(&result),
        snapshot.pv,
        "{} principal variation",
        snapshot.label
    );
    assert_eq!(result.nodes, snapshot.nodes, "{} nodes", snapshot.label);
    assert_eq!(
        result.q_nodes, snapshot.q_nodes,
        "{} q_nodes",
        snapshot.label
    );
    assert_eq!(result.stop, snapshot.stop, "{} stop", snapshot.label);
}

#[test]
fn classical_and_material_scores_match_the_original_implementation() {
    for (label, fen, handcrafted, material) in EVAL_CASES {
        let position = position(fen);
        assert_eq!(
            HandcraftedEvaluator.evaluate(&position),
            handcrafted,
            "{label} handcrafted score"
        );
        assert_eq!(
            MaterialEvaluator.evaluate(&position),
            material,
            "{label} material score"
        );
    }
}

#[test]
fn legal_move_lists_match_the_original_implementation() {
    for (label, fen, expected) in LEGAL_CASES {
        let position = position(fen);
        let mut moves = position
            .legal_moves()
            .iter()
            .map(|movement| movement.to_uci())
            .collect::<Vec<_>>();
        moves.sort();
        assert_eq!(moves.join(" "), expected, "{label} legal moves");
    }
}

#[test]
fn fixed_search_reports_match_the_original_implementation() {
    for snapshot in &SEARCH_SNAPSHOTS {
        assert_search_snapshot(snapshot);
    }
}

#[test]
fn a_node_budget_fallback_matches_the_original_implementation() {
    let game = Game::from_fen(STARTPOS_FEN).unwrap();
    let config = SearchConfig {
        max_depth: 2,
        node_limit: 1,
        ..SearchConfig::default()
    };
    let result = search(&game, &HandcraftedEvaluator, &config).unwrap();
    assert_eq!(action_text(result.action), "fallback a2a3");
    assert_eq!(claim_text(result.claim), None);
    assert_eq!(result.score, None);
    assert_eq!(result.completed_depth, 0);
    assert!(result.pv.is_empty());
    assert_eq!(result.nodes, 1);
    assert_eq!(result.q_nodes, 0);
    assert_eq!(result.stop, SearchStop::NodeLimit);
}

#[test]
fn concrete_and_trait_object_evaluators_report_identically() {
    for snapshot in &SEARCH_SNAPSHOTS {
        let game = Game::from_fen(snapshot.fen).expect("valid fixture FEN");
        let config = SearchConfig {
            max_depth: snapshot.depth,
            node_limit: snapshot.node_limit,
            ..SearchConfig::default()
        };
        let concrete = search(&game, &HandcraftedEvaluator, &config).unwrap();
        let object: &dyn Evaluator = &HandcraftedEvaluator;
        let dynamic = search(&game, object, &config).unwrap();
        assert_eq!(action_text(concrete.action), action_text(dynamic.action));
        assert_eq!(claim_text(concrete.claim), claim_text(dynamic.claim));
        assert_eq!(concrete.score, dynamic.score);
        assert_eq!(concrete.completed_depth, dynamic.completed_depth);
        assert_eq!(pv_text(&concrete), pv_text(&dynamic));
        assert_eq!(concrete.nodes, dynamic.nodes);
        assert_eq!(concrete.q_nodes, dynamic.q_nodes);
        assert_eq!(concrete.stop, dynamic.stop);
    }
}

#[test]
fn controlled_and_plain_searches_report_identically_when_not_cancelled() {
    let control = SearchControl::new();
    for snapshot in &SEARCH_SNAPSHOTS {
        let game = Game::from_fen(snapshot.fen).expect("valid fixture FEN");
        let config = SearchConfig {
            max_depth: snapshot.depth,
            node_limit: snapshot.node_limit,
            ..SearchConfig::default()
        };
        let plain = search(&game, &HandcraftedEvaluator, &config).unwrap();
        let controlled =
            search_controlled(&game, &HandcraftedEvaluator, &config, &control).unwrap();
        assert_eq!(action_text(plain.action), action_text(controlled.action));
        assert_eq!(claim_text(plain.claim), claim_text(controlled.claim));
        assert_eq!(plain.score, controlled.score);
        assert_eq!(plain.completed_depth, controlled.completed_depth);
        assert_eq!(pv_text(&plain), pv_text(&controlled));
        assert_eq!(plain.nodes, controlled.nodes);
        assert_eq!(plain.q_nodes, controlled.q_nodes);
        assert_eq!(plain.stop, controlled.stop);
    }
}
