mod common;

use horseflight::{
    search, search::search_controlled_restricted, HandcraftedEvaluator, SearchAction, SearchConfig,
    SearchControl, SearchStop, STARTPOS_FEN,
};

#[test]
fn all_legal_and_reversed_match_unrestricted() {
    let game = common::game(STARTPOS_FEN);
    let moves = game.position().legal_moves();
    let mut reversed = moves.to_vec();
    reversed.reverse();
    let config = SearchConfig {
        max_depth: 2,
        ..SearchConfig::default()
    };
    let expected = search(&game, &HandcraftedEvaluator, &config).unwrap();
    for allowed in [moves.as_ref(), reversed.as_slice()] {
        let actual = search_controlled_restricted(
            &game,
            &HandcraftedEvaluator,
            &config,
            &SearchControl::new(),
            allowed,
        )
        .unwrap();
        assert_eq!(
            (
                actual.action,
                actual.claim,
                actual.score,
                actual.completed_depth,
                actual.nodes,
                actual.q_nodes,
                actual.stop,
                actual.pv
            ),
            (
                expected.action,
                expected.claim,
                expected.score,
                expected.completed_depth,
                expected.nodes,
                expected.q_nodes,
                expected.stop,
                expected.pv.clone()
            )
        );
    }
}

#[test]
fn singleton_empty_illegal_terminal_and_precancel() {
    let game = common::game(STARTPOS_FEN);
    let movement = common::uci_move(game.position(), "a2a3");
    let config = SearchConfig {
        max_depth: 1,
        ..SearchConfig::default()
    };
    let control = SearchControl::new();
    let run = |moves: &[horseflight::Move]| {
        search_controlled_restricted(&game, &HandcraftedEvaluator, &config, &control, moves)
    };
    assert_eq!(
        run(&[movement]).unwrap().action,
        SearchAction::Move(movement)
    );
    assert!(run(&[]).is_err());
    assert!(run(&[common::normal("a2", "a5")]).is_err());
    control.cancel();
    let result = run(&[movement]).unwrap();
    assert_eq!(result.action, SearchAction::Fallback(movement));
    assert_eq!(result.stop, SearchStop::Cancelled);
    let terminal = common::game("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
    assert!(matches!(
        search_controlled_restricted(
            &terminal,
            &HandcraftedEvaluator,
            &config,
            &SearchControl::new(),
            &[]
        )
        .unwrap()
        .action,
        SearchAction::Terminal(_)
    ));
}

#[test]
fn canonical_special_singleton_restrictions() {
    use horseflight::eval::n2::{session::Session, N2Evaluator};
    use horseflight::search::search_controlled_with_session_restricted;
    for (fen, text) in [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
        ("8/P7/8/8/8/8/8/K6k w - - 0 1", "a7a8q"),
        ("8/P7/8/8/8/8/8/K6k w - - 0 1", "a7a8n"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
    ] {
        let game = common::game(fen);
        assert_eq!(game.outcome(), horseflight::Outcome::Ongoing);
        let movement = common::uci_move(game.position(), text);
        let config = SearchConfig {
            max_depth: 1,
            ..SearchConfig::default()
        };
        let result = search_controlled_restricted(
            &game,
            &HandcraftedEvaluator,
            &config,
            &SearchControl::new(),
            &[movement],
        )
        .unwrap();
        assert_eq!(result.action, SearchAction::Move(movement), "{text}");
        let mut session = Session::new(game.position());
        let result = search_controlled_with_session_restricted(
            &game,
            &N2Evaluator,
            &mut session,
            &config,
            &SearchControl::new(),
            &[movement],
        )
        .unwrap();
        assert_eq!(result.action, SearchAction::Move(movement), "{text}");
        assert!(session.matches_rebuild(game.position()));
    }
}

struct TrackingSession {
    inner: horseflight::eval::n2::session::Session,
    control: SearchControl,
    cancel: bool,
    resets: usize,
    makes: usize,
    unmakes: usize,
    before: Vec<horseflight::Position>,
}

impl horseflight::eval::EvaluationSession for TrackingSession {
    fn evaluate(&self, position: &horseflight::Position) -> i32 {
        self.inner.evaluate(position)
    }
    fn on_make(
        &mut self,
        changes: &[horseflight::position::PieceChange],
        before: &horseflight::Position,
        after: &horseflight::Position,
    ) {
        self.before.push(*before);
        self.inner.on_make(changes, before, after);
        self.makes += 1;
        assert!(self.inner.matches_rebuild(after));
        if self.cancel && self.inner.depth() >= 2 {
            self.control.cancel();
        }
    }
    fn on_unmake(&mut self) {
        self.inner.on_unmake();
        self.unmakes += 1;
        assert!(self.inner.matches_rebuild(&self.before.pop().unwrap()));
    }
    fn reset(&mut self, root: &horseflight::Position) {
        self.resets += 1;
        self.before.clear();
        self.inner.reset(root);
    }
}

#[test]
fn restricted_real_session_lifecycle() {
    use horseflight::eval::n2::{session::Session, N2Evaluator};
    use horseflight::search::search_controlled_with_session_restricted;
    let game = common::game(STARTPOS_FEN);
    let original = game.clone();
    let other = common::game("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    let allowed = [
        common::uci_move(game.position(), "a2a3"),
        common::uci_move(game.position(), "b2b3"),
    ];
    for mode in 0..4 {
        let control = SearchControl::new();
        if mode == 3 {
            control.cancel();
        }
        let mut session = TrackingSession {
            inner: Session::new(other.position()),
            control: control.clone(),
            cancel: mode == 0,
            resets: 0,
            makes: 0,
            unmakes: 0,
            before: vec![],
        };
        let config = SearchConfig {
            max_depth: if mode == 1 { 1 } else { 4 },
            node_limit: if mode == 2 { 1 } else { 1_000_000 },
            ..SearchConfig::default()
        };
        let result = search_controlled_with_session_restricted(
            &game,
            &N2Evaluator,
            &mut session,
            &config,
            &control,
            &allowed,
        )
        .unwrap();
        match mode {
            0 | 3 => assert_eq!(result.stop, SearchStop::Cancelled),
            2 => assert_eq!(result.stop, SearchStop::NodeLimit),
            _ => {}
        }
        if mode == 0 {
            assert!(result.nodes > 0 && session.makes > 0);
        }
        assert_eq!(session.resets, 1);
        assert_eq!(session.makes, session.unmakes);
        assert!(session.before.is_empty());
        assert_eq!(session.inner.depth(), 0);
        assert!(session.inner.matches_rebuild(game.position()));
        assert_eq!(game.position().to_fen(), original.position().to_fen());
        assert_eq!(game.position().hash(), original.position().hash());
        assert_eq!(game.history_len(), original.history_len());
    }
}

struct CancellingEvaluator(SearchControl);
impl horseflight::eval::Evaluator for CancellingEvaluator {
    fn evaluate(&self, _: &horseflight::Position) -> i32 {
        self.0.cancel();
        0
    }
}

#[test]
fn restricted_live_cancel_fallback_is_allowed() {
    let game = common::game(STARTPOS_FEN);
    let allowed = [
        common::uci_move(game.position(), "a2a3"),
        common::uci_move(game.position(), "b2b3"),
    ];
    let control = SearchControl::new();
    let result = search_controlled_restricted(
        &game,
        &CancellingEvaluator(control.clone()),
        &SearchConfig {
            max_depth: 4,
            ..SearchConfig::default()
        },
        &control,
        &allowed,
    )
    .unwrap();
    assert_eq!(result.stop, SearchStop::Cancelled);
    assert!(result.nodes > 0);
    match result.action {
        SearchAction::Fallback(m) | SearchAction::Move(m) => assert!(allowed.contains(&m)),
        other => panic!("{other:?}"),
    }
    if let Some(first) = result.pv.first() {
        assert!(allowed.contains(first));
    }
}

#[test]
fn f06_restricted_intended_claim() {
    let mut game = common::game("7k/8/8/8/8/8/8/K1Q5 w - - 0 1");
    for text in ["a1a2", "h8g8", "a2a1", "g8h8", "a1a2", "h8g8", "a2a1"] {
        let movement = common::uci_move(game.position(), text);
        game.play(movement).unwrap();
    }
    let claim = common::uci_move(game.position(), "g8h8");
    let excluded = common::uci_move(game.position(), "g8h7");
    assert!(!game.claimable_now().any());
    assert!(game.claimable_after(claim).unwrap().threefold);
    assert!(!game.claimable_after(excluded).unwrap().any());
    let config = SearchConfig {
        max_depth: 2,
        ..SearchConfig::default()
    };
    let run = |m| {
        search_controlled_restricted(
            &game,
            &HandcraftedEvaluator,
            &config,
            &SearchControl::new(),
            &[m],
        )
        .unwrap()
    };
    let result = run(claim);
    assert!(matches!(result.action, SearchAction::Claim(_)));
    assert!(result.claim.is_some());
    let result = run(excluded);
    assert_eq!(result.action, SearchAction::Move(excluded));
    assert!(result.claim.is_none());
}

#[test]
fn f06_restricted_current_claim_is_preserved() {
    let mut game = common::game("7k/8/8/8/8/8/8/K1Q5 b - - 0 1");
    for text in [
        "h8g8", "a1a2", "g8h8", "a2a1", "h8g8", "a1a2", "g8h8", "a2a1",
    ] {
        let movement = common::uci_move(game.position(), text);
        game.play(movement).unwrap();
    }
    assert!(game.claimable_now().any());
    let movement = common::uci_move(game.position(), "h8h7");
    let result = search_controlled_restricted(
        &game,
        &HandcraftedEvaluator,
        &SearchConfig {
            max_depth: 2,
            ..SearchConfig::default()
        },
        &SearchControl::new(),
        &[movement],
    )
    .unwrap();
    assert!(matches!(result.action, SearchAction::Claim(_)));
    assert!(result.claim.is_some());
}
