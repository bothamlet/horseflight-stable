//! Independent unpruned reference negamax.
//!
//! This oracle deliberately shares no recursion with the production search: it
//! is a full-width, no-pruning negamax with its own quiescence frontier, and it
//! adjudicates rules through the public API (`Game::outcome`,
//! `Game::claimable_now`, `Game::claimable_after`) rather than the search's
//! fast paths. Production alpha-beta with ordering on and off must agree with
//! it on the exact root score and on optimal-move membership.

mod common;

use common::{game, uci_move, walk};

use horseflight::eval::{clamp_evaluation, Evaluator, HandcraftedEvaluator, INFINITY, MATE_SCORE};
use horseflight::{search, DrawClaim, Game, Move, SearchAction, SearchConfig, SearchStop};

const MAX_REFERENCE_PLY: u32 = 32;

struct Reference<'a> {
    evaluator: &'a dyn Evaluator,
    quiescence: bool,
    q_depth_limit: u32,
}

impl Reference<'_> {
    /// True when the side to move may claim a draw now or with an intended move.
    fn claim_available(&self, game: &Game, legal: &[Move]) -> bool {
        if game.claimable_now().any() {
            return true;
        }
        legal.iter().copied().any(|movement| {
            game.claimable_after(movement)
                .is_ok_and(|claims| claims.any())
        })
    }

    /// Unpruned negamax value from the side to move's perspective.
    fn value(&self, game: &mut Game, depth: u32, ply: u32) -> i32 {
        assert!(ply < MAX_REFERENCE_PLY, "reference recursion bound");
        let position = *game.position();
        let legal = position.legal_moves();
        let in_check = position.in_check();
        if legal.is_empty() {
            return if in_check {
                -MATE_SCORE + ply as i32
            } else {
                0
            };
        }
        if game.outcome().is_over() {
            // Checkmate and stalemate are handled above, so this is a draw rule.
            return 0;
        }
        let claim = self.claim_available(game, &legal);
        if depth == 0 {
            return self.frontier(game, ply, self.q_depth_limit, claim);
        }
        let mut best = if claim { 0 } else { -INFINITY };
        for movement in legal.iter().copied() {
            game.play(movement).expect("generated moves are legal");
            let score = -self.value(game, depth - 1, ply + 1);
            game.undo().expect("LIFO undo").expect("a move was played");
            if score > best {
                best = score;
            }
        }
        best
    }

    /// Unpruned quiescence: stand pat, captures, promotions and every evasion.
    fn frontier(&self, game: &mut Game, ply: u32, q_left: u32, claim: bool) -> i32 {
        assert!(ply < MAX_REFERENCE_PLY, "reference recursion bound");
        let position = *game.position();
        let legal = position.legal_moves();
        let in_check = position.in_check();
        if legal.is_empty() {
            return if in_check {
                -MATE_SCORE + ply as i32
            } else {
                0
            };
        }
        if game.outcome().is_over() {
            return 0;
        }
        let claim = claim || self.claim_available(game, &legal);
        let mut best = if claim { 0 } else { -INFINITY };
        if !self.quiescence {
            let static_score = clamp_evaluation(self.evaluator.evaluate(&position));
            if static_score > best {
                best = static_score;
            }
            return best;
        }
        if !in_check {
            let stand_pat = clamp_evaluation(self.evaluator.evaluate(&position));
            if stand_pat > best {
                best = stand_pat;
            }
            if q_left == 0 {
                return best;
            }
        }
        for movement in legal.iter().copied() {
            let tactical = in_check
                || movement.is_en_passant()
                || movement.is_promotion()
                || position.board().piece_at(movement.to()).is_some();
            if !tactical {
                continue;
            }
            let next_q = if in_check { q_left } else { q_left - 1 };
            game.play(movement).expect("generated moves are legal");
            let score = -self.frontier(game, ply + 1, next_q, false);
            game.undo().expect("LIFO undo").expect("a move was played");
            if score > best {
                best = score;
            }
        }
        best
    }

    /// Exact unpruned root value plus every root move that achieves it.
    fn root(&self, game: &mut Game, depth: u32) -> (i32, Vec<Move>) {
        let position = *game.position();
        let legal = position.legal_moves();
        let claim = self.claim_available(game, &legal);
        let mut best = if claim { 0 } else { -INFINITY };
        let mut scores = Vec::with_capacity(legal.len());
        for movement in legal.iter().copied() {
            game.play(movement).expect("generated moves are legal");
            let score = -self.value(game, depth - 1, 1);
            game.undo().expect("LIFO undo").expect("a move was played");
            scores.push((movement, score));
            if score > best {
                best = score;
            }
        }
        let optimal = scores
            .into_iter()
            .filter(|(_, score)| *score == best)
            .map(|(movement, _)| movement)
            .collect();
        (best, optimal)
    }
}

fn production_config(
    depth: u32,
    ordering: bool,
    quiescence: bool,
    q_depth_limit: u32,
) -> SearchConfig {
    SearchConfig {
        max_depth: depth,
        node_limit: 2_000_000,
        ordering,
        quiescence,
        q_depth_limit,
    }
}

/// Compare alpha-beta (ordering on and off) against the unpruned reference.
fn assert_matches_reference(
    label: &str,
    position: &Game,
    evaluator: &dyn Evaluator,
    reference: &Reference<'_>,
    depth: u32,
) {
    let mut reference_game = position.clone();
    let (expected_score, optimal) = reference.root(&mut reference_game, depth);
    assert_eq!(
        position.position(),
        reference_game.position(),
        "{label}: the reference left its game mutated"
    );
    for ordering in [true, false] {
        let config = production_config(
            depth,
            ordering,
            reference.quiescence,
            reference.q_depth_limit,
        );
        let result = search(position, evaluator, &config).unwrap();
        assert_eq!(
            result.stop,
            SearchStop::DepthReached,
            "{label} depth {depth} ordering {ordering}: unexpected stop {:?}",
            result.stop
        );
        assert_eq!(
            result.score,
            Some(expected_score),
            "{label} depth {depth} ordering {ordering}: score disagrees with the reference"
        );
        match result.action {
            SearchAction::Move(chosen) => assert!(
                optimal.contains(&chosen),
                "{label} depth {depth} ordering {ordering}: {} is not optimal {:?}",
                chosen.to_uci(),
                optimal
                    .iter()
                    .map(|movement| movement.to_uci())
                    .collect::<Vec<_>>()
            ),
            SearchAction::Claim(_) => {
                assert_eq!(
                    expected_score, 0,
                    "{label}: claimed without a zero reference"
                );
                assert!(optimal.iter().all(|_| expected_score <= 0));
            }
            other => panic!("{label}: unexpected action {other:?}"),
        }
        assert!(
            result.pv.len() <= depth as usize,
            "{label}: PV longer than depth"
        );
        assert!(result.nodes <= config.node_limit);
    }
}

#[test]
fn small_controlled_trees_agree_with_the_unpruned_reference() {
    let handcrafted = HandcraftedEvaluator;
    let reference = Reference {
        evaluator: &handcrafted,
        quiescence: true,
        q_depth_limit: 3,
    };
    let fixtures = [
        ("mate in two", "k7/8/2K5/8/8/3Q4/8/8 w - - 0 1", 2),
        ("losing king", "3k4/8/3K4/8/8/8/8/6Q1 b - - 0 1", 3),
        ("pawn front", "4k3/8/8/8/3p4/8/4P3/7K w - - 0 1", 3),
        ("poisoned pawn", "4k3/8/8/2p5/3p4/8/8/3RK3 w - - 0 1", 2),
        ("mate in one", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 2),
    ];
    for (label, fen, depth) in fixtures {
        let position = game(fen);
        assert_matches_reference(label, &position, &handcrafted, &reference, depth);
    }
}

#[test]
fn the_reference_agrees_with_quiescence_disabled_too() {
    let material = horseflight::MaterialEvaluator;
    let reference = Reference {
        evaluator: &material,
        quiescence: false,
        q_depth_limit: 0,
    };
    for (label, fen, depth) in [
        ("poisoned pawn", "4k3/8/8/2p5/3p4/8/8/3RK3 w - - 0 1", 2),
        ("pawn front", "4k3/8/8/8/3p4/8/4P3/7K w - - 0 1", 3),
        (
            "middle game",
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            1,
        ),
    ] {
        let position = game(fen);
        assert_matches_reference(label, &position, &material, &reference, depth);
    }
}

#[test]
fn repetition_and_clock_claims_match_the_phase_1a_rule_api() {
    let handcrafted = HandcraftedEvaluator;
    let reference = Reference {
        evaluator: &handcrafted,
        quiescence: true,
        q_depth_limit: 2,
    };
    let intended = walk(
        "7k/8/8/8/8/8/8/K1Q5 w - - 0 1",
        &["a1a2", "h8g8", "a2a1", "g8h8", "a1a2", "h8g8", "a2a1"],
    );
    let claim_move = uci_move(intended.position(), "g8h8");
    assert_matches_reference("intended threefold", &intended, &handcrafted, &reference, 2);
    let result = search(
        &intended,
        &handcrafted,
        &production_config(2, true, true, 2),
    )
    .unwrap();
    assert_eq!(
        result.action,
        SearchAction::Claim(DrawClaim::Intended(claim_move))
    );

    let current = walk(
        "7k/Q7/8/8/8/8/8/K7 b - - 0 1",
        &[
            "h8g8", "a1b1", "g8h8", "b1a1", "h8g8", "a1b1", "g8h8", "b1a1",
        ],
    );
    assert_eq!(current.current_repetition_count(), 3);
    assert_matches_reference("current threefold", &current, &handcrafted, &reference, 2);
    let result = search(&current, &handcrafted, &production_config(2, true, true, 2)).unwrap();
    assert_eq!(result.action, SearchAction::Claim(DrawClaim::Current));

    let fifty = game("k7/8/3N4/8/8/8/8/R6K b - - 99 60");
    assert_matches_reference("fifty-move intent", &fifty, &handcrafted, &reference, 2);

    let terminal = game("8/8/4k3/8/8/4K3/8/6R1 w - - 150 100");
    let result = search(
        &terminal,
        &handcrafted,
        &production_config(2, true, true, 2),
    )
    .unwrap();
    assert_eq!(result.stop, SearchStop::TerminalRoot);
}

#[test]
fn the_reference_itself_is_unpruned_and_exact() {
    // Sanity: the reference reproduces the independently proven mate distances.
    let handcrafted = HandcraftedEvaluator;
    let reference = Reference {
        evaluator: &handcrafted,
        quiescence: true,
        q_depth_limit: 4,
    };
    let mut position = game("k7/8/2K5/8/8/3Q4/8/8 w - - 0 1");
    let (score, optimal) = reference.root(&mut position, 3);
    assert_eq!(score, MATE_SCORE - 3);
    assert!(!optimal.is_empty());
    assert!(score > 0);
}
