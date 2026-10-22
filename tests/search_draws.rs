mod common;

use common::{game, uci_move, walk};

use horseflight::eval::MATE_SCORE;
use horseflight::{
    search, DrawClaim, HandcraftedEvaluator, MaterialEvaluator, Outcome, SearchAction,
    SearchConfig, SearchStop,
};

fn config(depth: u32, node_limit: u64) -> SearchConfig {
    SearchConfig {
        max_depth: depth,
        node_limit,
        ..SearchConfig::default()
    }
}

const FIVEFOLD_CYCLE: [&str; 16] = [
    "h8g8", "a1b1", "g8h8", "b1a1", "h8g8", "a1b1", "g8h8", "b1a1", "h8g8", "a1b1", "g8h8", "b1a1",
    "h8g8", "a1b1", "g8h8", "b1a1",
];

#[test]
fn automatic_terminal_roots_are_explicit_and_scored() {
    let evaluator = HandcraftedEvaluator;
    let cases = [
        (
            "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
            Outcome::Checkmate {
                winner: horseflight::Color::White,
            },
            -MATE_SCORE,
        ),
        ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", Outcome::Stalemate, 0),
        (
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            Outcome::InsufficientMaterial,
            0,
        ),
        (
            "8/8/4k3/8/8/4K3/8/6R1 w - - 150 100",
            Outcome::SeventyFiveMoveDraw,
            0,
        ),
    ];
    for (fen, outcome, score) in cases {
        let position = game(fen);
        let result = search(&position, &evaluator, &config(3, 100_000)).unwrap();
        assert_eq!(position.outcome(), outcome, "{fen}");
        assert_eq!(result.action, SearchAction::Terminal(outcome), "{fen}");
        assert_eq!(result.score, Some(score), "{fen}");
        assert_eq!(result.completed_depth, 0, "{fen}");
        assert_eq!(result.nodes, 0, "{fen}");
        assert_eq!(result.stop, SearchStop::TerminalRoot, "{fen}");
        assert!(result.pv.is_empty(), "{fen}");
        assert_eq!(result.claim, None, "{fen}");
    }
}

#[test]
fn fivefold_repetition_is_terminal_and_twofold_is_not_a_draw() {
    let evaluator = HandcraftedEvaluator;
    let start = "7k/Q7/8/8/8/8/8/K7 b - - 0 1";
    let fivefold = walk(start, &FIVEFOLD_CYCLE);
    assert_eq!(fivefold.outcome(), Outcome::FivefoldRepetition);
    let result = search(&fivefold, &evaluator, &config(2, 200_000)).unwrap();
    assert_eq!(
        result.action,
        SearchAction::Terminal(Outcome::FivefoldRepetition)
    );
    assert_eq!(result.score, Some(0));

    let twofold = walk(start, &FIVEFOLD_CYCLE[..4]);
    assert_eq!(twofold.current_repetition_count(), 2);
    assert_eq!(twofold.outcome(), Outcome::Ongoing);
    let result = search(&twofold, &evaluator, &config(2, 200_000)).unwrap();
    assert_eq!(result.claim, None, "twofold repetition is not a claim");
    assert_eq!(result.stop, SearchStop::DepthReached);
    assert!(result.score.unwrap() < 0, "black is still a queen down");
}

#[test]
fn fifty_move_thresholds_are_exact_at_the_root() {
    let evaluator = HandcraftedEvaluator;
    let template = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - {clock} 60";
    for clock in [99u32, 100, 149] {
        let fen = template.replace("{clock}", &clock.to_string());
        let position = game(&fen);
        let mate_in_one = uci_move(position.position(), "a1a8");
        let result = search(&position, &evaluator, &config(2, 200_000)).unwrap();
        if clock < 100 {
            // Any quiet move reaches 100 halfmoves, so an intended-move claim
            // exists; the reported example is the first such legal move.
            let Some(DrawClaim::Intended(intended)) = result.claim else {
                panic!(
                    "clock {clock}: expected an intended claim, got {:?}",
                    result.claim
                );
            };
            assert!(
                position.claimable_after(intended).unwrap().fifty_move,
                "clock {clock}: the reported intended move really creates the claim"
            );
        } else {
            assert_eq!(result.claim, Some(DrawClaim::Current), "clock {clock}");
        }
        // Mate takes precedence over the optional claim in every case.
        assert_eq!(result.score, Some(MATE_SCORE - 1), "clock {clock}");
        assert_eq!(
            result.action,
            SearchAction::Move(mate_in_one),
            "clock {clock}"
        );
    }

    let fen = template.replace("{clock}", "150");
    let result = search(&game(&fen), &evaluator, &config(2, 200_000)).unwrap();
    assert_eq!(
        result.action,
        SearchAction::Terminal(Outcome::SeventyFiveMoveDraw)
    );
    assert_eq!(result.claim, None, "automatic outcomes suppress claims");
}

#[test]
fn a_current_threefold_claim_is_recommended_when_the_side_to_move_is_losing() {
    let evaluator = HandcraftedEvaluator;
    let position = walk("7k/Q7/8/8/8/8/8/K7 b - - 0 1", &FIVEFOLD_CYCLE[..8]);
    assert_eq!(position.current_repetition_count(), 3);
    let result = search(&position, &evaluator, &config(3, 500_000)).unwrap();
    assert_eq!(result.claim, Some(DrawClaim::Current));
    assert_eq!(result.action, SearchAction::Claim(DrawClaim::Current));
    assert_eq!(result.score, Some(0));
    assert_eq!(result.completed_depth, 3);
    assert!(result.pv.is_empty(), "a claim is not a move and has no PV");
    assert_eq!(result.stop, SearchStop::DepthReached);
}

#[test]
fn an_intended_threefold_claim_is_recommended_and_never_executed() {
    let evaluator = HandcraftedEvaluator;
    // White is a queen up and shuffles; black is to move at clock seven and
    // g8h8 writes the third occurrence of the starting position.
    let position = walk(
        "7k/8/8/8/8/8/8/K1Q5 w - - 0 1",
        &["a1a2", "h8g8", "a2a1", "g8h8", "a1a2", "h8g8", "a2a1"],
    );
    assert_eq!(position.position().halfmove_clock(), 7);
    let claim_move = uci_move(position.position(), "g8h8");
    let result = search(&position, &evaluator, &config(3, 500_000)).unwrap();
    assert_eq!(result.claim, Some(DrawClaim::Intended(claim_move)));
    assert_eq!(
        result.action,
        SearchAction::Claim(DrawClaim::Intended(claim_move))
    );
    assert_eq!(result.score, Some(0));
    assert!(
        result.pv.is_empty(),
        "the intended claim move is not played"
    );
    assert_ne!(result.stop, SearchStop::TerminalRoot);

    // The same board without history offers no claim and black simply loses.
    let fresh = game("6k1/8/8/8/8/8/8/K1Q5 b - - 7 4");
    let fresh_result = search(&fresh, &evaluator, &config(3, 500_000)).unwrap();
    assert_eq!(fresh_result.claim, None);
    assert_eq!(fresh_result.stop, SearchStop::DepthReached);
    assert!(fresh_result.score.unwrap() < -500);
}

#[test]
fn a_winning_continuation_is_preferred_over_an_optional_draw() {
    let evaluator = HandcraftedEvaluator;
    // At 100 halfmoves a fifty-move claim is available to white, but white can
    // still mate in one and search must prefer the win.
    let fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 100 60";
    let result = search(&game(fen), &evaluator, &config(2, 200_000)).unwrap();
    assert_eq!(result.claim, Some(DrawClaim::Current));
    assert_eq!(
        result.action,
        SearchAction::Move(uci_move(game(fen).position(), "a1a8"))
    );
    assert_eq!(result.score, Some(MATE_SCORE - 1));
    assert_eq!(result.pv.first().unwrap().to_uci(), "a1a8");
}

#[test]
fn a_root_claim_is_reported_even_when_no_search_completes() {
    let evaluator = HandcraftedEvaluator;
    let fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 100 60";
    let result = search(&game(fen), &evaluator, &config(3, 0)).unwrap();
    assert_eq!(result.nodes, 0);
    assert_eq!(result.completed_depth, 0);
    assert_eq!(result.stop, SearchStop::NodeLimit);
    assert_eq!(result.action, SearchAction::Claim(DrawClaim::Current));
    assert_eq!(result.score, Some(0));
    assert!(result.pv.is_empty());
}

#[test]
fn the_frontier_honours_current_and_quiet_intended_fifty_move_claims() {
    let evaluator = MaterialEvaluator;
    // Black's only legal move is a quiet king move. With a two-ply search the
    // quiescence frontier is black to move at 100 halfmoves and can claim the
    // draw; at 99 it can claim one by writing a quiet move. Without the frontier
    // claim check the score would be a large loss instead of a draw.
    let fen = "k7/8/3N4/8/8/8/8/R6K b - - {clock} 60";
    for (clock, expectation, frontier_clock) in [(98u32, "current", 100u32), (97, "intended", 99)] {
        let fen = fen.replace("{clock}", &clock.to_string());
        let position = game(&fen);
        assert_eq!(position.outcome(), Outcome::Ongoing, "clock {clock}");
        assert_eq!(position.position().halfmove_clock(), clock);
        let result = search(&position, &evaluator, &config(2, 500_000)).unwrap();
        assert_eq!(result.stop, SearchStop::DepthReached, "clock {clock}");
        assert_eq!(result.completed_depth, 2, "clock {clock}");
        assert_eq!(result.claim, None, "clock {clock}: no root claim yet");
        assert_eq!(
            result.score,
            Some(0),
            "clock {clock}: the {expectation} claim at {frontier_clock} halfmoves draws"
        );
        assert_eq!(
            result.action,
            SearchAction::Move(uci_move(position.position(), "a8b8")),
            "clock {clock}"
        );
    }

    // The same board without the clock pressure is simply lost for black.
    let plain = game("k7/8/3N4/8/8/8/8/R6K b - - 0 60");
    let result = search(&plain, &evaluator, &config(2, 500_000)).unwrap();
    assert_eq!(result.stop, SearchStop::DepthReached);
    assert!(result.score.unwrap() < -500, "got {:?}", result.score);
}

#[test]
fn a_repeated_position_is_still_lost_when_the_claim_is_declined() {
    let evaluator = HandcraftedEvaluator;
    // Black can claim a draw here, but a three-ply search must not lose the
    // repetition history of the reported line: the claim is the action and the
    // PV stays empty even though the position was reached through real play.
    let position = walk("7k/Q7/8/8/8/8/8/K7 b - - 0 1", &FIVEFOLD_CYCLE[..8]);
    assert_eq!(position.current_repetition_count(), 3);
    let result = search(&position, &evaluator, &config(2, 100_000)).unwrap();
    assert_eq!(result.stop, SearchStop::DepthReached);
    assert_eq!(result.action, SearchAction::Claim(DrawClaim::Current));
    assert!(result.pv.is_empty());

    // With no history the identical board is an ordinary loss for black.
    let fresh = game("7k/Q7/8/8/8/8/8/K7 b - - 8 5");
    let lost = search(&fresh, &evaluator, &config(2, 100_000)).unwrap();
    assert_eq!(lost.claim, None);
    assert!(lost.score.unwrap() < -500);
}
