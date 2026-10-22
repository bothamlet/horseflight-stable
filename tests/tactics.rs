//! Small tactical and mate-distance fixtures.
//!
//! Every mate-distance expectation here is produced by [`mates_within`] and
//! [`mated_within`]: a bounded exhaustive proof over legal moves that never
//! calls the production search. The search is only asked to agree with that
//! proof. Winning mates occur at odd ply distances (`2 * N - 1` for mate in `N`
//! full moves) and the losing side's mate distance is even, because the
//! defender moves first at a losing root.

mod common;

use common::game;

use horseflight::eval::{is_mate_score, mate_distance_plies, MATE_SCORE};
use horseflight::{
    search, Evaluator, HandcraftedEvaluator, MaterialEvaluator, Position, SearchAction,
    SearchConfig, SearchStop,
};

/// Proof horizon: enough for the fixtures below and small enough to stay cheap.
const PROOF_LIMIT: u32 = 5;

fn config(depth: u32) -> SearchConfig {
    SearchConfig {
        max_depth: depth,
        node_limit: 2_000_000,
        ..SearchConfig::default()
    }
}

/// Independent bounded proof: can the side to move force checkmate within
/// `plies` half-moves?
fn mates_within(position: &Position, plies: u32) -> bool {
    if plies == 0 || position.has_insufficient_material() {
        return false;
    }
    let mut next = *position;
    for movement in position.legal_moves().iter().copied() {
        let undo = next.play(movement).expect("generated moves are legal");
        let forced = mated_within(&next, plies - 1);
        next.unmake(&undo).unwrap();
        if forced {
            return true;
        }
    }
    false
}

/// Independent bounded proof: is every legal move followed by a forced mate
/// within `plies` half-moves? Stalemate and dead positions are draws.
fn mated_within(position: &Position, plies: u32) -> bool {
    if position.is_checkmate() {
        return true;
    }
    if plies == 0 || position.has_insufficient_material() {
        return false;
    }
    let legal = position.legal_moves();
    if legal.is_empty() {
        return false;
    }
    let mut next = *position;
    for movement in legal.iter().copied() {
        let undo = next.play(movement).expect("generated moves are legal");
        let forced = mates_within(&next, plies - 1);
        next.unmake(&undo).unwrap();
        if !forced {
            return false;
        }
    }
    true
}

fn shortest_forced_mate(position: &Position) -> Option<u32> {
    (1..=PROOF_LIMIT).find(|&plies| mates_within(position, plies))
}

fn quickest_loss(position: &Position) -> Option<u32> {
    (0..=PROOF_LIMIT).find(|&plies| mated_within(position, plies))
}

fn after(position: &Position, movement: horseflight::Move) -> Position {
    let mut child = *position;
    child.play(movement).expect("generated moves are legal");
    child
}

/// Winning fixtures: the proven shortest mate is `expected_plies` and search
/// must score exactly that distance, choose one of the equally fastest moves,
/// and a genuinely slower winning alternative must exist.
fn assert_winning_fixture(fen: &str, expected_plies: u32, require_slower: bool) {
    let position = game(fen);
    assert_eq!(
        shortest_forced_mate(position.position()),
        Some(expected_plies),
        "{fen}: independent proof of the shortest forced mate"
    );
    let result = search(&position, &HandcraftedEvaluator, &config(expected_plies)).unwrap();
    assert_eq!(result.stop, SearchStop::DepthReached, "{fen}");
    assert_eq!(result.completed_depth, expected_plies, "{fen}");
    assert_eq!(
        result.score,
        Some(MATE_SCORE - expected_plies as i32),
        "{fen}: shorter wins must score higher"
    );
    let SearchAction::Move(chosen) = result.action else {
        panic!("{fen}: expected a move, got {:?}", result.action);
    };

    let root = *position.position();
    let mut optimal = 0;
    let mut slower = 0;
    for movement in root.legal_moves().iter().copied() {
        let child = after(&root, movement);
        if mated_within(&child, expected_plies - 1) {
            optimal += 1;
        } else if mated_within(&child, PROOF_LIMIT) {
            slower += 1;
        }
    }
    assert!(optimal >= 1, "{fen}: proof finds no optimal move");
    if require_slower {
        assert!(
            slower >= 1,
            "{fen}: no genuinely slower winning alternative exists"
        );
    }
    let chosen_child = after(&root, chosen);
    assert!(
        mated_within(&chosen_child, expected_plies - 1),
        "{fen}: search chose the slower {}",
        chosen.to_uci()
    );
}

/// Losing fixtures: the mover cannot avoid mate and must delay it as long as
/// the proof allows. Each defence is measured by the *attacker's* shortest
/// forced mate after that defence.
fn assert_delayed_loss_fixture(fen: &str, expected_plies: u32) {
    let position = game(fen);
    assert_eq!(
        quickest_loss(position.position()),
        Some(expected_plies),
        "{fen}: independent proof of the loss"
    );
    let result = search(&position, &HandcraftedEvaluator, &config(expected_plies)).unwrap();
    assert_eq!(result.stop, SearchStop::DepthReached, "{fen}");
    assert_eq!(result.completed_depth, expected_plies, "{fen}");
    assert_eq!(
        result.score,
        Some(-(MATE_SCORE - expected_plies as i32)),
        "{fen}: a longer defence must score higher"
    );
    let SearchAction::Move(chosen) = result.action else {
        panic!("{fen}: expected a move, got {:?}", result.action);
    };

    let root = *position.position();
    let best_delay = |movement: horseflight::Move| -> Option<u32> {
        let child = after(&root, movement);
        shortest_forced_mate(&child).map(|attacker_plies| attacker_plies + 1)
    };
    let mut maximal = 0;
    let mut shorter = 0;
    for movement in root.legal_moves().iter().copied() {
        match best_delay(movement) {
            Some(delay) if delay == expected_plies => maximal += 1,
            Some(delay) if delay < expected_plies => shorter += 1,
            _ => {}
        }
    }
    assert!(maximal >= 1, "{fen}: proof finds no maximal delay");
    assert!(shorter >= 1, "{fen}: no shorter-loss alternative exists");
    assert_eq!(
        best_delay(chosen),
        Some(expected_plies),
        "{fen}: search chose {} which is mated sooner",
        chosen.to_uci()
    );
}

#[test]
fn mate_in_one_fixtures_hold_for_both_colours() {
    assert_winning_fixture("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 1, false);
    assert_winning_fixture("r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", 1, false);
}

#[test]
fn mate_in_two_fixtures_are_proven_for_both_colours() {
    // Qa6+ Kb8 Qb7#; the proved shortest mate is three half-moves.
    assert_winning_fixture("k7/8/2K5/8/8/3Q4/8/8 w - - 0 1", 3, true);
    // The colour-mirrored position: Qd1+ Ka2 Qa4#.
    assert_winning_fixture("8/8/3q4/8/8/2k5/8/K7 b - - 0 1", 3, true);
    assert_eq!(
        horseflight::eval::mate_distance_plies(MATE_SCORE - 3),
        Some(3)
    );
}

#[test]
fn a_faster_win_is_preferred_over_a_slower_one() {
    // Many queen moves mate in two full moves; a few only mate in three. The
    // proof above checks that a slower alternative really exists, so the
    // preference is not vacuous.
    assert_winning_fixture("6k1/8/6K1/8/8/8/8/6Q1 w - - 0 1", 3, true);
}

#[test]
fn a_longer_defence_is_preferred_over_a_shorter_one() {
    // Black is mated in four half-moves either way; Kc8 delays to white's
    // three-ply mate while Ke8 walks into a one-ply mate, so the defender must
    // prefer Kc8.
    assert_delayed_loss_fixture("3k4/8/3K4/8/8/8/8/6Q1 b - - 0 1", 4);
}

#[test]
fn a_checked_parent_position_is_a_losing_fixture_for_both_colours() {
    // Checked child of the white mate-in-two: black's only reply loses in two.
    let black_lost = game("k7/8/Q1K5/8/8/8/8/8 b - - 1 1");
    assert!(black_lost.position().in_check());
    assert_eq!(quickest_loss(black_lost.position()), Some(2));
    let result = search(&black_lost, &HandcraftedEvaluator, &config(2)).unwrap();
    assert_eq!(result.stop, SearchStop::DepthReached);
    assert_eq!(result.score, Some(-(MATE_SCORE - 2)));

    let white_lost = game("8/8/8/8/8/q1k5/8/K7 w - - 1 1");
    assert!(white_lost.position().in_check());
    assert_eq!(quickest_loss(white_lost.position()), Some(2));
    let result = search(&white_lost, &HandcraftedEvaluator, &config(2)).unwrap();
    assert_eq!(result.stop, SearchStop::DepthReached);
    assert_eq!(result.score, Some(-(MATE_SCORE - 2)));
}

#[test]
fn the_proof_rejects_positions_that_are_not_mates() {
    let drawn = game("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    assert_eq!(shortest_forced_mate(drawn.position()), None);
    assert_eq!(quickest_loss(drawn.position()), None);
    // K+NN vs K is not a dead position and has no short forced mate.
    let two_knights = game("4k3/8/8/8/8/8/8/1N2K1N1 w - - 0 1");
    assert!(!two_knights.position().has_insufficient_material());
    assert_eq!(shortest_forced_mate(two_knights.position()), None);
    // A terminal checkmate is a proven loss at zero plies.
    let mated = game("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    assert_eq!(quickest_loss(mated.position()), Some(0));
}

#[test]
fn terminal_scoring_does_not_depend_on_the_evaluator() {
    let fixtures = [
        ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 1),
        ("k7/8/2K5/8/8/3Q4/8/8 w - - 0 1", 3),
        ("3k4/8/3K4/8/8/8/8/6Q1 b - - 0 1", 4),
    ];
    for (fen, plies) in fixtures {
        let search_position = game(fen);
        let evaluators: [&dyn Evaluator; 2] = [&MaterialEvaluator, &HandcraftedEvaluator];
        for evaluator in evaluators {
            let result = search(&search_position, evaluator, &config(plies)).unwrap();
            assert_eq!(result.stop, SearchStop::DepthReached, "{fen}");
            let score = result.score.expect("a completed search scores");
            assert!(is_mate_score(score), "{fen}: {score}");
            assert_eq!(mate_distance_plies(score), Some(plies), "{fen}");
        }
    }
}
