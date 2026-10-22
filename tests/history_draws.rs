mod common;

use common::{game, play_uci, position, STARTPOS_FEN};
use horseflight::{DrawClaims, Game, Outcome};

fn knight_cycle(game: &mut Game) -> usize {
    for text in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        play_uci(game, text);
    }
    game.current_repetition_count()
}

#[test]
fn threefold_is_a_claim_and_fivefold_is_automatic() {
    let mut game = Game::from_fen(STARTPOS_FEN).unwrap();
    assert_eq!(knight_cycle(&mut game), 2);
    assert!(!game.threefold_claim_available());
    assert_eq!(game.outcome(), Outcome::Ongoing);

    assert_eq!(knight_cycle(&mut game), 3);
    assert!(game.threefold_claim_available());
    assert!(!game.fivefold_repetition());
    assert_eq!(
        game.claimable_now(),
        DrawClaims {
            threefold: true,
            fifty_move: false
        }
    );
    assert_eq!(
        game.outcome(),
        Outcome::Ongoing,
        "threefold is a claim, not automatic"
    );

    assert_eq!(knight_cycle(&mut game), 4);
    assert_eq!(game.outcome(), Outcome::Ongoing);

    assert_eq!(knight_cycle(&mut game), 5);
    assert!(game.fivefold_repetition());
    assert_eq!(game.outcome(), Outcome::FivefoldRepetition);
    assert!(!game.threefold_claim_available());
    assert!(!game.fifty_move_claim_available());
    assert_eq!(game.claimable_now(), DrawClaims::none());
}

#[test]
fn intended_move_claims_cover_the_third_occurrence() {
    let mut game = Game::from_fen(STARTPOS_FEN).unwrap();
    for text in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"] {
        play_uci(&mut game, text);
    }
    assert!(!game.claimable_now().threefold);
    let intended = common::uci_move(game.position(), "f6g8");
    let snapshot = *game.position();
    let history_len = game.positions().len();
    let claims = game.claimable_after(intended).unwrap();
    assert!(claims.threefold);
    assert!(!claims.fifty_move);
    assert!(claims.any());
    assert_eq!(
        *game.position(),
        snapshot,
        "claim query mutated the position"
    );
    assert_eq!(
        game.positions().len(),
        history_len,
        "claim query mutated history"
    );

    game.play(intended).unwrap();
    assert!(game.claimable_now().threefold);
}

#[test]
fn intended_fifth_repetition_remains_claimable_before_execution() {
    let mut game = Game::from_fen(STARTPOS_FEN).unwrap();
    for _ in 0..3 {
        knight_cycle(&mut game);
    }
    for text in ["g1f3", "g8f6", "f3g1"] {
        play_uci(&mut game, text);
    }
    assert_eq!(game.current_repetition_count(), 4);
    assert_eq!(game.outcome(), Outcome::Ongoing);
    assert!(!game.fivefold_repetition());

    let intended = common::uci_move(game.position(), "f6g8");
    let snapshot = *game.position();
    let claims = game.claimable_after(intended).unwrap();
    assert!(claims.threefold);
    assert!(!claims.fifty_move);
    assert_eq!(*game.position(), snapshot);

    game.play(intended).unwrap();
    assert!(game.fivefold_repetition());
    assert_eq!(game.outcome(), Outcome::FivefoldRepetition);
    assert_eq!(game.claimable_now(), DrawClaims::none());
    assert!(!game.threefold_claim_available());
}

#[test]
fn fifty_move_is_a_claim_and_seventy_five_moves_is_automatic() {
    let fen_99 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 99 1";
    let mut game = Game::from_fen(fen_99).unwrap();
    assert!(!game.claimable_now().fifty_move);
    assert_eq!(game.outcome(), Outcome::Ongoing);

    let quiet = common::uci_move(game.position(), "g1f3");
    assert!(game.claimable_after(quiet).unwrap().fifty_move);
    let pawn = common::uci_move(game.position(), "e2e4");
    assert!(!game.claimable_after(pawn).unwrap().fifty_move);

    game.play(quiet).unwrap();
    assert_eq!(game.position().halfmove_clock(), 100);
    assert!(game.claimable_now().fifty_move);
    assert_eq!(game.outcome(), Outcome::Ongoing);

    let mut at_149 =
        Game::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 149 1").unwrap();
    assert!(!at_149.seventy_five_move_draw());
    assert!(at_149.fifty_move_claim_available());
    let quiet = common::uci_move(at_149.position(), "g1f3");
    let snapshot = *at_149.position();
    let claims = at_149.claimable_after(quiet).unwrap();
    assert!(claims.fifty_move, "149->150 is a valid intended-move claim");
    assert_eq!(*at_149.position(), snapshot);
    play_uci(&mut at_149, "g1f3");
    assert_eq!(at_149.position().halfmove_clock(), 150);
    assert!(at_149.seventy_five_move_draw());
    assert_eq!(at_149.outcome(), Outcome::SeventyFiveMoveDraw);
    assert_eq!(at_149.claimable_now(), DrawClaims::none());
    assert!(!at_149.fifty_move_claim_available());

    let at_150 =
        Game::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 150 1").unwrap();
    assert_eq!(at_150.outcome(), Outcome::SeventyFiveMoveDraw);
    assert_eq!(at_150.claimable_now(), DrawClaims::none());
    assert!(!at_150.fifty_move_claim_available());
    assert!(!at_150.threefold_claim_available());
}

#[test]
fn pawn_moves_and_captures_reset_the_halfmove_clock() {
    let mut game =
        Game::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 99 1").unwrap();
    play_uci(&mut game, "e2e4");
    assert_eq!(game.position().halfmove_clock(), 0);
    assert!(!game.claimable_now().fifty_move);

    let mut capture =
        Game::from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 99").unwrap();
    play_uci(&mut capture, "e4d5");
    assert_eq!(capture.position().halfmove_clock(), 0);
}

#[test]
fn checkmate_takes_precedence_over_automatic_draws_and_claims() {
    let mate = game("7k/6Q1/5K2/8/8/8/8/8 b - - 150 100");
    assert_eq!(
        mate.outcome(),
        Outcome::Checkmate {
            winner: horseflight::Color::White
        }
    );
    assert_eq!(mate.claimable_now(), DrawClaims::none());
    assert!(!mate.threefold_claim_available());
    assert!(!mate.fifty_move_claim_available());

    let stalemate = game("7k/5Q2/6K1/8/8/8/8/8 b - - 150 100");
    assert_eq!(stalemate.outcome(), Outcome::Stalemate);
    assert_eq!(stalemate.claimable_now(), DrawClaims::none());

    // A quiet move reaching halfmove 100 but mating: the intended-move claim is
    // available before execution; playing it instead produces checkmate.
    let mut mating_claim = game("7k/6pp/4K3/8/8/8/8/R7 w - - 99 100");
    let mate_in_one = common::uci_move(mating_claim.position(), "a1a8");
    let snapshot = *mating_claim.position();
    let claims = mating_claim.claimable_after(mate_in_one).unwrap();
    assert!(
        claims.fifty_move,
        "99->100 is a valid intended-move claim even if the move would mate"
    );
    assert!(!claims.threefold);
    assert_eq!(
        *mating_claim.position(),
        snapshot,
        "claim query mutated position"
    );
    mating_claim.play(mate_in_one).unwrap();
    assert_eq!(mating_claim.position().halfmove_clock(), 100);
    assert_eq!(
        mating_claim.outcome(),
        Outcome::Checkmate {
            winner: horseflight::Color::White
        }
    );
    assert_eq!(mating_claim.claimable_now(), DrawClaims::none());
    assert!(!mating_claim.fifty_move_claim_available());
}

#[test]
fn ended_current_positions_suppress_intended_move_claims() {
    let seventy_five = game("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 150 1");
    assert_eq!(seventy_five.outcome(), Outcome::SeventyFiveMoveDraw);
    assert_eq!(seventy_five.claimable_now(), DrawClaims::none());
    assert!(!seventy_five.fifty_move_claim_available());
    let quiet = common::uci_move(seventy_five.position(), "g1f3");
    let snapshot = *seventy_five.position();
    let history_len = seventy_five.positions().len();
    assert_eq!(
        seventy_five.claimable_after(quiet).unwrap(),
        DrawClaims::none()
    );
    assert_eq!(*seventy_five.position(), snapshot);
    assert_eq!(seventy_five.positions().len(), history_len);

    let dead_material = game("8/8/8/8/8/8/6k1/1B2K3 w - - 0 1");
    assert_eq!(dead_material.outcome(), Outcome::InsufficientMaterial);
    assert_eq!(dead_material.claimable_now(), DrawClaims::none());
    let movement = dead_material.position().legal_moves()[0];
    let snapshot = *dead_material.position();
    assert_eq!(
        dead_material.claimable_after(movement).unwrap(),
        DrawClaims::none()
    );
    assert_eq!(*dead_material.position(), snapshot);

    let mut fivefold = Game::from_fen(STARTPOS_FEN).unwrap();
    for _ in 0..5 {
        knight_cycle(&mut fivefold);
    }
    assert_eq!(fivefold.outcome(), Outcome::FivefoldRepetition);
    let movement = common::uci_move(fivefold.position(), "g1f3");
    let snapshot = *fivefold.position();
    assert_eq!(
        fivefold.claimable_after(movement).unwrap(),
        DrawClaims::none()
    );
    assert_eq!(*fivefold.position(), snapshot);
}

#[test]
fn rights_difference_is_part_of_repetition_identity() {
    let mut game = game("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    for text in ["e1e2", "e8e7", "e2e1", "e7e8"] {
        play_uci(&mut game, text);
    }
    assert_eq!(
        game.position().to_fen(),
        "r3k2r/8/8/8/8/8/8/R3K2R w - - 4 3"
    );
    assert_eq!(game.current_repetition_count(), 1);
    assert!(!game.threefold_claim_available());

    for _ in 0..2 {
        for text in ["e1e2", "e8e7", "e2e1", "e7e8"] {
            play_uci(&mut game, text);
        }
    }
    assert_eq!(game.current_repetition_count(), 3);
    assert!(game.threefold_claim_available());
}

#[test]
fn raw_uncapturable_ep_does_not_split_repetition_identity() {
    let with_ep = position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    let without_ep = position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    assert!(with_ep.same_repetition_identity(&without_ep));
    assert_eq!(with_ep.hash(), without_ep.hash());

    let legal_ep = position("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
    let none = position("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1");
    assert!(!legal_ep.same_repetition_identity(&none));
    assert_ne!(legal_ep.hash(), none.hash());

    let fresh =
        Game::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
    assert_eq!(fresh.position().legal_ep_square(), None);
    assert_eq!(fresh.history_len(), 0);
}

#[test]
fn conservative_dead_material_positives_and_non_dead_controls() {
    let positives = [
        "8/8/8/8/8/8/6k1/4K3 w - - 0 1",
        "8/8/8/8/8/8/6k1/1B2K3 w - - 0 1",
        "8/8/8/8/8/8/6k1/1N2K3 w - - 0 1",
        "5k2/7b/8/8/8/8/8/1B2K3 w - - 0 1",
        "5k2/8/8/8/8/8/8/1B1B1B1K w - - 0 1",
        "5k2/7b/8/8/8/8/8/1B1B3K w - - 0 1",
    ];
    for fen in positives {
        let position = position(fen);
        assert!(
            position.has_insufficient_material(),
            "expected conservative dead material in {fen}"
        );
        assert_eq!(
            Game::from_fen(fen).unwrap().outcome(),
            Outcome::InsufficientMaterial
        );
    }

    let non_dead = [
        "6k1/8/8/8/8/8/8/1NN1K3 w - - 0 1",
        "6k1/8/8/8/8/8/8/1N2K2n w - - 0 1",
        "5k2/6b1/8/8/8/8/8/1B2K3 w - - 0 1",
        "5k2/8/8/8/8/7n/8/1B2K3 w - - 0 1",
        "5k2/8/8/8/8/7n/8/1B1B3K w - - 0 1",
        "5k2/8/8/8/8/8/8/1BBB1B1K w - - 0 1",
        "5k2/8/8/8/8/8/4p3/1B2K3 w - - 0 1",
        "5k2/8/8/8/8/8/8/1B2K2r w - - 0 1",
        "5k2/8/8/8/8/8/8/1B2K1q1 w - - 0 1",
    ];
    for fen in non_dead {
        let position = position(fen);
        assert!(
            !position.has_insufficient_material(),
            "unexpected automatic draw for {fen}"
        );
        assert_eq!(Game::from_fen(fen).unwrap().outcome(), Outcome::Ongoing);
    }
}

#[test]
fn reversible_game_history_restores_the_initial_position() {
    let mut game = Game::from_fen(STARTPOS_FEN).unwrap();
    let start = game.start_position();
    for text in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"] {
        play_uci(&mut game, text);
    }
    assert_eq!(game.history_len(), 6);
    assert_eq!(
        game.moves()
            .map(|movement| movement.to_uci())
            .collect::<Vec<_>>(),
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]
    );
    while game.undo().unwrap().is_some() {}
    assert_eq!(*game.position(), start);
    assert_eq!(game.history_len(), 0);
    assert!(game.undo().unwrap().is_none());
}
