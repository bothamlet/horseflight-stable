mod common;

use common::{game, uci_move};

use horseflight::eval::{is_mate_score, MATE_SCORE};
use horseflight::{
    search, Evaluator, HandcraftedEvaluator, MaterialEvaluator, SearchAction, SearchConfig,
    SearchStop,
};

fn config(depth: u32, quiescence: bool, q_depth_limit: u32) -> SearchConfig {
    SearchConfig {
        max_depth: depth,
        node_limit: 2_000_000,
        ordering: true,
        quiescence,
        q_depth_limit,
    }
}

/// Run with quiescence on and off and return `(on, off)` results.
fn compare(
    fen: &str,
    depth: u32,
    q_depth_limit: u32,
    evaluator: &dyn Evaluator,
) -> (horseflight::SearchResult, horseflight::SearchResult) {
    let position = game(fen);
    let on = search(&position, evaluator, &config(depth, true, q_depth_limit)).unwrap();
    let off = search(&position, evaluator, &config(depth, false, q_depth_limit)).unwrap();
    for (label, result) in [("on", &on), ("off", &off)] {
        assert_eq!(
            result.stop,
            SearchStop::DepthReached,
            "{fen} quiescence {label} must complete, not fall back"
        );
        assert_eq!(result.completed_depth, depth, "{fen} quiescence {label}");
    }
    (on, off)
}

#[test]
fn a_poisoned_capture_is_rejected_only_with_quiescence_and_enough_q_depth() {
    // Rxd4?? is answered by cxd4, trading a rook for a pawn.
    let fen = "4k3/8/8/2p5/3p4/8/8/3RK3 w - - 0 1";
    let evaluator = MaterialEvaluator;
    let (on, off) = compare(fen, 1, 4, &evaluator);
    assert_eq!(on.score, Some(300), "quiescence must keep the rook safe");
    assert_ne!(
        on.action,
        SearchAction::Move(uci_move(game(fen).position(), "d1d4"))
    );
    assert_eq!(off.score, Some(400), "the static frontier takes the pawn");
    assert_eq!(
        off.action,
        SearchAction::Move(uci_move(game(fen).position(), "d1d4"))
    );

    // One q ply resolves the capture but not the recapture, so the q-depth
    // limit is what turns the poisoned capture away.
    let shallow = search(&game(fen), &evaluator, &config(1, true, 0)).unwrap();
    assert_eq!(shallow.stop, SearchStop::DepthReached);
    assert_eq!(shallow.score, Some(400));
}

#[test]
fn an_en_passant_capture_is_searched_at_a_quiet_frontier() {
    // After e2e4 the frontier's only capture is the en-passant capture d4xe3,
    // so the frontier must see the pawn fall; a skipped e.p. filter would leave
    // the double push looking best and the score at the static -6.
    let fen = "4k3/8/8/8/3p4/8/4P3/7K w - - 0 1";
    let evaluator = HandcraftedEvaluator;
    let (on, off) = compare(fen, 1, 4, &evaluator);
    let mut frontier = game(fen);
    let push = uci_move(frontier.position(), "e2e4");
    frontier.play(push).unwrap();
    let captures: Vec<_> = frontier
        .position()
        .legal_moves()
        .iter()
        .copied()
        .filter(|movement| {
            movement.is_en_passant()
                || frontier
                    .position()
                    .board()
                    .piece_at(movement.to())
                    .is_some()
        })
        .collect();
    assert_eq!(captures.len(), 1, "one frontier capture: {captures:?}");
    assert!(captures[0].is_en_passant());
    assert_eq!(captures[0].to_uci(), "d4e3");

    assert_eq!(off.score, Some(-6));
    assert_eq!(off.action, SearchAction::Move(push));
    assert_eq!(on.score, Some(-18), "the en-passant capture costs a pawn");
    assert_ne!(on.action, SearchAction::Move(push));

    let shallow = search(&game(fen), &evaluator, &config(1, true, 0)).unwrap();
    assert_eq!(shallow.stop, SearchStop::DepthReached);
    assert_eq!(shallow.score, Some(-6), "a zero q budget cannot resolve it");
}

#[test]
fn a_quiet_promotion_is_searched_at_a_quiet_frontier() {
    // Black's g2 pawn promotes quietly on g1; the frontier must score the
    // queen, not the pawn.
    let fen = "8/8/8/8/8/8/1N4p1/K6k w - - 0 1";
    let (on, off) = compare(fen, 1, 4, &MaterialEvaluator);
    assert_eq!(off.score, Some(220), "white knight against a pawn");
    assert_eq!(on.score, Some(-580), "black promotes to a queen");
    assert_eq!(
        off.score.unwrap() - on.score.unwrap(),
        800,
        "the difference is exactly the quiet queen promotion"
    );

    // Quiet underpromotions are generated and ordered too; the queen is simply
    // the best of them here.
    let frontier = game("8/8/8/8/8/8/6p1/K6k b - - 0 1");
    let promotions: Vec<String> = frontier
        .position()
        .legal_moves()
        .iter()
        .filter(|movement| movement.is_promotion())
        .map(|movement| movement.to_uci())
        .collect();
    assert_eq!(
        promotions,
        vec!["g2g1n", "g2g1b", "g2g1r", "g2g1q"],
        "all four quiet promotions exist at the frontier"
    );
}

#[test]
fn quiet_check_evasions_are_searched_at_the_frontier() {
    // Nc7+ forks the king and queen; at the frontier black has only quiet king
    // evasions, so a static stand pat would miss the queen.
    let fen = "q3k3/7p/8/1N6/8/8/P7/3K4 w - - 0 1";
    let (on, off) = compare(fen, 1, 4, &MaterialEvaluator);
    assert_eq!(off.score, Some(-580), "static material without the fork");
    assert_eq!(
        on.score,
        Some(320),
        "quiescence resolves the check and wins the queen"
    );
    assert_eq!(
        on.action,
        SearchAction::Move(uci_move(game(fen).position(), "b5c7"))
    );
}

#[test]
fn a_static_cutoff_in_check_is_never_used_even_when_the_evaluation_is_positive() {
    // After Rb8+ black is materially ahead, so a checked frontier that stood
    // pat on the static score would report a black advantage. Quiescence must
    // search the evasions and find white's forced mate.
    let fen = "6k1/5ppp/q7/8/8/8/5PPP/1R4K1 w - - 0 1";
    let (on, off) = compare(fen, 1, 4, &HandcraftedEvaluator);
    assert!(
        is_mate_score(on.score.unwrap()),
        "quiescence must find the mate, got {:?}",
        on.score
    );
    assert!(on.score.unwrap() > 0);
    assert!(
        off.score.unwrap() < -300,
        "the static frontier sees black up a queen, got {:?}",
        off.score
    );
    let mate_in_plies = horseflight::eval::mate_distance_plies(on.score.unwrap()).unwrap();
    assert!(mate_in_plies <= 4, "mate distance {mate_in_plies}");
    assert!(on.score.unwrap() < MATE_SCORE);
}

#[test]
fn checkmate_and_dead_positions_are_terminal_at_the_frontier() {
    // Mate in one is detected with and without quiescence.
    let fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1";
    let (on, off) = compare(fen, 1, 4, &HandcraftedEvaluator);
    assert_eq!(on.score, Some(MATE_SCORE - 1));
    assert_eq!(off.score, Some(MATE_SCORE - 1));

    // Winning the last enemy piece leaves king and knight against king, which
    // the frontier must score as an automatic draw rather than material.
    let fen = "r3k3/8/8/1N6/8/8/8/3K4 w - - 0 1";
    let (on, off) = compare(fen, 1, 4, &MaterialEvaluator);
    assert_eq!(off.score, Some(-180), "static material favours the rook");
    assert_eq!(on.score, Some(0), "the fork forces a dead position");
}
