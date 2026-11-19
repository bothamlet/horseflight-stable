//! Reversible NNUE accumulation during search.
//!
//! These tests prove the incremental NNUE session is exactly interchangeable
//! with the stateless reference evaluator, that it is balanced after every
//! search termination reason (completed, node limit, cancellation, claim
//! probe), that the caller's game is untouched, and that nested make/unmake
//! cycles restore the accumulator bit-for-bit.

use horseflight::eval::n2::session::Session;
use horseflight::eval::n2::{forward, Accumulator, N2Evaluator};
use horseflight::eval::{clamp_evaluation, EvaluationSession as _, Evaluator};
use horseflight::search::{
    search, search_controlled, search_controlled_with_session, search_with_session, SearchConfig,
    SearchControl, SearchResult, SearchStop,
};
use horseflight::{Color, Game, Move, Position, Role, SearchAction, Square, Undo};

fn config(max_depth: u32, node_limit: u64) -> SearchConfig {
    SearchConfig {
        max_depth,
        node_limit,
        ordering: true,
        quiescence: true,
        q_depth_limit: 4,
    }
}

fn games() -> Vec<Game> {
    const FENS: &[&str] = &[
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
        "8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        "2kr3r/pp1q1ppp/2n1pn2/2bp4/8/1PN1PN2/PBPPQPPP/2KR3R w - - 0 10",
        "8/8/8/8/5k2/8/4P3/4K3 w - - 0 1",
        "r7/8/8/8/8/8/8/R3K2k w Q - 0 1",
    ];
    FENS.iter()
        .map(|fen| Game::from_fen(fen).unwrap_or_else(|error| panic!("{fen}: {error}")))
        .collect()
}

fn assert_same_result(expected: &SearchResult, actual: &SearchResult, label: &str) {
    assert_eq!(expected.action, actual.action, "{label}: action differs");
    assert_eq!(expected.claim, actual.claim, "{label}: claim differs");
    assert_eq!(expected.score, actual.score, "{label}: score differs");
    assert_eq!(
        expected.completed_depth, actual.completed_depth,
        "{label}: completed depth differs"
    );
    assert_eq!(expected.pv, actual.pv, "{label}: pv differs");
    assert_eq!(expected.nodes, actual.nodes, "{label}: nodes differ");
    assert_eq!(expected.q_nodes, actual.q_nodes, "{label}: q_nodes differ");
    assert_eq!(expected.stop, actual.stop, "{label}: stop differs");
}

fn assert_accumulator_matches_rebuild(session: &Session, position: &Position, label: &str) {
    let rebuilt = Accumulator::from_position(position);
    assert_eq!(
        session.accumulator().values(),
        rebuilt.values(),
        "{label}: incremental accumulator differs from a full rebuild"
    );
}

#[test]
fn session_search_is_exactly_the_stateless_nnue_search() {
    // A bounded budget keeps this fast in a debug build; both paths stop at the
    // same node, which is what makes the comparison meaningful. Four positions
    // cover startpos, castle/EP-heavy middlegames and a bare endgame; the
    // release-only benchmark suite exercises the same equivalence at depth.
    let config = config(4, 5_000);
    for (index, game) in games().into_iter().take(4).enumerate() {
        let label = format!("game {index}");
        let before = game.clone();

        let stateless = search(&game, &N2Evaluator, &config).expect("stateless search");
        let mut session = Session::new(game.position());
        let incremental = search_with_session(&game, &N2Evaluator, &mut session, &config)
            .expect("session search");

        assert_same_result(&stateless, &incremental, &label);
        assert_eq!(session.depth(), 0, "{label}: session left unbalanced");
        assert!(
            session.matches_rebuild(game.position()),
            "{label}: session accumulator differs from a rebuild at the root"
        );

        // The caller's game is never mutated by either path.
        assert_eq!(
            game.position().to_fen(),
            before.position().to_fen(),
            "{label}: caller FEN changed"
        );
        assert_eq!(game.history_len(), before.history_len(), "{label}: history");
        assert_eq!(
            game.current_repetition_count(),
            before.current_repetition_count(),
            "{label}: repetition context"
        );
    }
}

#[test]
fn node_limited_search_leaves_the_session_balanced() {
    let config = config(12, 5_000);
    for (index, game) in games().into_iter().enumerate() {
        let label = format!("game {index}");
        let mut session = Session::new(game.position());
        let result = search_with_session(&game, &N2Evaluator, &mut session, &config)
            .expect("session search");
        assert_eq!(result.stop, SearchStop::NodeLimit, "{label}: stop reason");
        assert_eq!(session.depth(), 0, "{label}: session left unbalanced");
        assert!(
            session.matches_rebuild(game.position()),
            "{label}: accumulator differs after a node-limit stop"
        );
    }
}

#[test]
fn mid_search_cancellation_leaves_the_session_balanced() {
    // Cancel from a watchdog thread while the search is inside the tree, which
    // is the path `cancelled_without_iteration` does not cover.
    for (index, game) in games().into_iter().take(4).enumerate() {
        let label = format!("game {index}");
        let control = SearchControl::new();
        let watcher = control.clone();
        let (finished, wait) = std::sync::mpsc::channel::<()>();
        let watchdog = std::thread::spawn(move || {
            if matches!(
                wait.recv_timeout(std::time::Duration::from_millis(150)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ) {
                watcher.cancel();
            }
        });
        let mut session = Session::new(game.position());
        let result = search_controlled_with_session(
            &game,
            &N2Evaluator,
            &mut session,
            &config(96, 1_000_000_000),
            &control,
        )
        .expect("session search");
        let _ = finished.send(());
        let _ = watchdog.join();
        assert_eq!(result.stop, SearchStop::Cancelled, "{label}: stop reason");
        assert_eq!(session.depth(), 0, "{label}: session left unbalanced");
        assert!(
            session.matches_rebuild(game.position()),
            "{label}: accumulator differs after mid-search cancellation"
        );
    }
}

#[test]
fn pre_cancelled_search_leaves_the_session_balanced() {
    let control = SearchControl::new();
    control.cancel();
    for (index, game) in games().into_iter().enumerate() {
        let label = format!("game {index}");
        let mut session = Session::new(game.position());
        let result = search_controlled_with_session(
            &game,
            &N2Evaluator,
            &mut session,
            &config(12, 10_000_000),
            &control,
        )
        .expect("session search");
        assert_eq!(result.stop, SearchStop::Cancelled, "{label}: stop reason");
        assert_eq!(session.depth(), 0, "{label}: session left unbalanced");
        assert!(
            session.matches_rebuild(game.position()),
            "{label}: accumulator differs after cancellation"
        );
    }
}

/// Kings-only shuffle that builds a real repetition history, so the search's
/// *intended-claim* probe performs make/unmake pairs through the session.
fn repetition_game(plies: usize) -> Game {
    let mut game = Game::from_fen("4k3/7p/8/8/8/8/7P/4K3 w - - 0 1").unwrap();
    let moves = ["e1d1", "e8d8", "d1e1", "d8e8"];
    for (index, text) in moves.iter().cycle().take(plies).enumerate() {
        let movement = game
            .position()
            .legal_moves()
            .iter()
            .copied()
            .find(|candidate| candidate.to_uci() == *text)
            .unwrap_or_else(|| panic!("ply {index}: {text} is not legal"));
        game.play(movement).expect("legal shuffle");
    }
    game
}

#[test]
fn claim_probe_make_unmake_keeps_the_session_balanced() {
    // Seven quiet plies: the position has occurred twice at halfmove clock 7,
    // so a quiet move that repeats it is an intended threefold claim. The probe
    // plays and undoes moves inside the search, which is exactly the path that
    // could desynchronise an incremental session.
    let game = repetition_game(7);
    assert_eq!(game.position().halfmove_clock(), 7);
    assert!(game.current_repetition_count() >= 2);

    let mut session = Session::new(game.position());
    let result = search_with_session(&game, &N2Evaluator, &mut session, &config(4, 20_000))
        .expect("session search");
    assert!(
        matches!(result.action, SearchAction::Claim(_)),
        "expected a claim, got {:?}",
        result.action
    );
    assert_eq!(
        session.depth(),
        0,
        "session left unbalanced by a claim probe"
    );
    assert!(session.matches_rebuild(game.position()));

    // The same search without a session must agree, and both leave the game
    // untouched.
    let before = game.clone();
    let stateless = search(&game, &N2Evaluator, &config(4, 20_000)).expect("stateless");
    assert_same_result(&stateless, &result, "claim search");
    assert_eq!(game.position().to_fen(), before.position().to_fen());
    assert_eq!(game.history_len(), before.history_len());
}

#[test]
fn fifty_move_probe_keeps_the_session_balanced() {
    // Clock 99 with legal quiet moves: every frontier probes a fifty-move claim.
    let game = Game::from_fen("8/8/8/8/8/8/6k1/4K2R w K - 99 100").unwrap();
    let mut session = Session::new(game.position());
    let result = search_with_session(&game, &N2Evaluator, &mut session, &config(3, 10_000_000))
        .expect("session search");
    assert_eq!(session.depth(), 0);
    assert!(session.matches_rebuild(game.position()));
    let stateless = search(&game, &N2Evaluator, &config(3, 10_000_000)).expect("stateless");
    assert_same_result(&stateless, &result, "fifty-move search");
}

/// Deterministic tiny PRNG: no new dependency, fully reproducible.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Lcg {
        Lcg(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

/// Depth-first tree walk: make/unmake through the session API while branching,
/// including repeated return visits to the same node.
fn walk(
    session: &mut Session,
    position: &mut Position,
    undo_stack: &mut Vec<Undo>,
    rng: &mut Lcg,
    remaining: u32,
    label: &str,
) {
    assert_eq!(
        session.depth(),
        undo_stack.len(),
        "{label}: depth bookkeeping"
    );
    assert_accumulator_matches_rebuild(session, position, label);
    if remaining == 0 {
        return;
    }
    let legal = position.legal_moves();
    if legal.is_empty() {
        return;
    }
    let branches = legal.len().min(3);
    for _ in 0..branches {
        let movement: Move = legal[rng.below(legal.len())];
        let before = *position;
        let undo = position.play(movement).expect("legal move");
        session.on_make(undo.piece_changes(), &before, position);
        undo_stack.push(undo);
        walk(
            session,
            position,
            undo_stack,
            rng,
            remaining - 1,
            &format!("{label}/{movement:?}"),
        );
        let undo = undo_stack.pop().expect("balanced stack");
        position.unmake(&undo).expect("unmake");
        session.on_unmake();
        assert_eq!(
            session.depth(),
            undo_stack.len(),
            "{label}: unbalanced after unmake"
        );
        assert_accumulator_matches_rebuild(session, position, label);
        assert_eq!(*position, before, "{label}: board/hash not restored");
        assert_eq!(position.hash(), before.hash(), "{label}: hash mismatch");
    }
}

#[test]
fn branching_tree_walks_keep_the_session_exact() {
    let mut rng = Lcg::new(20_261_028);
    for (index, game) in games().into_iter().enumerate() {
        let mut position = *game.position();
        let mut session = Session::new(&position);
        let mut undo_stack = Vec::new();
        walk(
            &mut session,
            &mut position,
            &mut undo_stack,
            &mut rng,
            4,
            &format!("game {index}"),
        );
        assert_eq!(session.depth(), 0);
        assert_eq!(position, *game.position());
    }
}

#[test]
fn special_moves_keep_the_session_exact() {
    // Castling (both sides), legal/pinned/x-ray en passant, capture promotions
    // and quiet shuffles, driven through the session API and unwound in reverse.
    let cases: [&str; 6] = [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3",
        "8/8/8/KPp4r/8/8/8/7k w - c6 0 2",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ];
    for (case, fen) in cases.iter().enumerate() {
        let mut position = Position::from_fen(fen).unwrap_or_else(|error| panic!("{fen}: {error}"));
        let start = position;
        let mut session = Session::new(&position);
        let mut stack: Vec<Undo> = Vec::new();
        for ply in 0..40 {
            let legal = position.legal_moves();
            if legal.is_empty() {
                break;
            }
            let movement = legal[ply % legal.len()];
            let before = position;
            let undo = position.play(movement).expect("legal move");
            session.on_make(undo.piece_changes(), &before, &position);
            stack.push(undo);
            assert_accumulator_matches_rebuild(&session, &position, &format!("case {case}"));
        }
        while let Some(undo) = stack.pop() {
            position.unmake(&undo).expect("unmake");
            session.on_unmake();
            assert_accumulator_matches_rebuild(&session, &position, &format!("case {case} back"));
        }
        assert_eq!(session.depth(), 0);
        assert_eq!(position, start, "case {case}: board/hash not restored");
        assert_eq!(position.hash(), start.hash(), "case {case}: hash mismatch");
    }
}

#[test]
fn promotion_underpromotions_keep_the_session_exact() {
    let mut position = Position::from_fen("4k3/1P6/8/8/8/8/6p1/4K3 w - - 0 1").unwrap();
    let mut session = Session::new(&position);
    let start = position;
    for role in [Role::Knight, Role::Bishop, Role::Rook, Role::Queen] {
        let before = position;
        let movement = position
            .legal_moves()
            .iter()
            .copied()
            .find(|movement| {
                movement.from() == Square::from_index(49) && movement.promotion_role() == Some(role)
            })
            .unwrap_or_else(|| panic!("missing b7 promotion to {role:?}"));
        let undo = position.play(movement).expect("legal promotion");
        session.on_make(undo.piece_changes(), &before, &position);
        assert_accumulator_matches_rebuild(&session, &position, &format!("{role:?}"));
        position.unmake(&undo).expect("unmake promotion");
        session.on_unmake();
        assert_accumulator_matches_rebuild(&session, &position, &format!("{role:?} undone"));
    }
    assert_eq!(session.depth(), 0);
    assert_eq!(position, start);
}

#[test]
fn a_dirty_session_is_reset_by_the_next_search() {
    let game = games()[1].clone();
    let clean = Session::new(game.position());

    // Walk a session several moves away from the root and leave it dirty, then
    // hand that *same* session to a search.
    let mut dirty = Session::new(game.position());
    let mut position = *game.position();
    for _ in 0..3 {
        let movement = position.legal_moves()[0];
        let before = position;
        let undo = position.play(movement).expect("legal move");
        dirty.on_make(undo.piece_changes(), &before, &position);
    }
    assert_eq!(
        dirty.depth(),
        3,
        "the session must be dirty before the search"
    );
    assert_ne!(
        dirty.accumulator().values(),
        clean.accumulator().values(),
        "the dirty session must differ from the root build"
    );

    let stateless = search_controlled(
        &game,
        &N2Evaluator,
        &config(3, 100_000),
        &SearchControl::new(),
    )
    .unwrap();
    let incremental = search_with_session(&game, &N2Evaluator, &mut dirty, &config(3, 100_000))
        .expect("session search");
    assert_same_result(&stateless, &incremental, "dirty session");
    // The search reset the session from its own root, ran, and unwound itself.
    assert_eq!(dirty.depth(), 0, "session left unbalanced");
    assert!(
        dirty.matches_rebuild(game.position()),
        "the search must reset the dirty session to the root before evaluating"
    );
}

#[test]
fn the_session_evaluator_path_matches_the_stateless_path() {
    // Guard against a regression where the session path silently became the
    // only correct one.
    let position =
        Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
            .unwrap();
    let stateless = N2Evaluator.evaluate(&position);
    let accumulator = Accumulator::from_position(&position);
    let white_relative = forward(&accumulator);
    let expected = clamp_evaluation(if position.side_to_move() == Color::White {
        white_relative
    } else {
        -white_relative
    });
    assert_eq!(stateless, expected);
}
