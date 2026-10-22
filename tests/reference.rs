mod common;

use std::collections::BTreeSet;

use common::{assert_valid, position};
use horseflight::{Move, Position};
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, EnPassantMode, Position as ShakmatyPosition};

const FIXTURES: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "n1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/1p6/N1R1K3 b - - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
];

const SEEDS: [u64; 8] = [
    0x0000_0000_0000_0001,
    0x1234_5678_9abc_def0,
    0xdead_beef_cafe_f00d,
    0x0f1e_2d3c_4b5a_6978,
    0x5555_aaaa_3333_cccc,
    0x8000_0000_0000_0001,
    0xfeed_face_0000_0001,
    0x3141_5926_5358_9793,
];

fn shakmaty_from_fen(fen: &str) -> Chess {
    Fen::from_ascii(fen.as_bytes())
        .unwrap_or_else(|error| panic!("shakmaty rejected fixture {fen}: {error}"))
        .into_position::<Chess>(CastlingMode::Standard)
        .unwrap_or_else(|error| panic!("shakmaty rejected position {fen}: {error:?}"))
}

fn shakmaty_uci(movement: &shakmaty::Move) -> String {
    movement.to_uci(CastlingMode::Standard).to_string()
}

fn compare_position(ours: &Position, theirs: &Chess, context: &str) -> Vec<Move> {
    let our_moves: Vec<Move> = ours.legal_moves().iter().copied().collect();
    let our_uci: BTreeSet<String> = our_moves.iter().map(|movement| movement.to_uci()).collect();
    let their_uci: BTreeSet<String> = theirs.legal_moves().iter().map(shakmaty_uci).collect();
    assert_eq!(
        our_uci,
        their_uci,
        "legal move-set mismatch at {context}: our FEN {}",
        ours.to_fen()
    );

    assert_eq!(
        ours.in_check(),
        theirs.is_check(),
        "check mismatch at {context}: {}",
        ours.to_fen()
    );
    assert_eq!(
        ours.is_checkmate(),
        theirs.is_checkmate(),
        "checkmate mismatch at {context}: {}",
        ours.to_fen()
    );
    assert_eq!(
        ours.is_stalemate(),
        theirs.is_stalemate(),
        "stalemate mismatch at {context}: {}",
        ours.to_fen()
    );
    assert_eq!(
        ours.has_insufficient_material(),
        theirs.is_insufficient_material(),
        "insufficient-material mismatch at {context}: {}",
        ours.to_fen()
    );
    assert_eq!(
        ours.legal_ep_square().map(|square| square.to_string()),
        theirs.legal_ep_square().map(|square| square.to_string()),
        "legal en-passant mismatch at {context}: {}",
        ours.to_fen()
    );

    let our_fen = ours.to_fen();
    let their_fen = Fen::from_position(theirs.clone(), EnPassantMode::Always).to_string();
    assert_eq!(
        our_fen, their_fen,
        "applied state/FEN mismatch at {context}"
    );
    our_moves
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    value
}

#[test]
fn reference_legal_moves_and_applied_state_across_seeded_walks() {
    let mut compared = 0usize;
    for (fixture_index, fen) in FIXTURES.iter().enumerate() {
        for seed in SEEDS {
            let mut ours = position(fen);
            let mut theirs = shakmaty_from_fen(fen);
            let mut rng = seed ^ (fixture_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);

            let initial_context = format!("fixture {fixture_index}, seed {seed:#018x}, initial");
            let mut our_moves = compare_position(&ours, &theirs, &initial_context);
            compared += 1;

            for ply in 0..40 {
                if our_moves.is_empty() {
                    break;
                }
                let movement = our_moves[(xorshift64(&mut rng) as usize) % our_moves.len()];
                let uci = movement.to_uci();
                let shakmaty_move = theirs
                    .legal_moves()
                    .iter()
                    .find(|candidate| shakmaty_uci(candidate) == uci)
                    .cloned()
                    .unwrap_or_else(|| panic!("missing shakmaty move {uci} at ply {ply}"));
                let before_fen = ours.to_fen();

                let undo = ours.play(movement).expect("reference move is legal");
                assert_valid(&ours);
                let mut rewind = ours;
                rewind.unmake(&undo).expect("fresh undo");
                assert_eq!(rewind.to_fen(), before_fen, "rewind mismatch at ply {ply}");

                theirs.play_unchecked(&shakmaty_move);
                let context = format!(
                    "fixture {fixture_index}, seed {seed:#018x}, ply {ply} after {uci}, fen {}",
                    ours.to_fen()
                );
                our_moves = compare_position(&ours, &theirs, &context);
                compared += 1;
            }
        }
    }
    eprintln!(
        "reference comparison: {compared} positions, {} fixtures, seeds {:?}",
        FIXTURES.len(),
        SEEDS
    );
    assert!(
        compared >= 3000,
        "expected at least 3000 reference positions, got {compared}"
    );
}

#[test]
fn reference_non_ongoing_fixtures_are_explicit() {
    let fixtures = [
        "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",     // checkmate
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",     // stalemate
        "8/8/8/8/8/8/6k1/4K3 w - - 0 1",      // K vs K
        "8/8/8/8/8/8/6k1/1B2K3 w - - 0 1",    // K+B vs K
        "8/8/8/8/8/8/6k1/1N2K3 w - - 0 1",    // K+N vs K
        "5k2/7b/8/8/8/8/8/1B2K3 w - - 0 1",   // same-colour bishops
        "5k2/6b1/8/8/8/8/8/1B2K3 w - - 0 1",  // opposite-colour bishops
        "6k1/8/8/8/8/8/8/1NN1K3 w - - 0 1",   // K+NN vs K is not dead
        "6k1/8/8/8/8/8/8/1N2K2n w - - 0 1",   // K+N vs K+N is not dead
        "5k2/8/8/8/8/8/8/1B1B1B1K w - - 0 1", // same-colour promoted bishops
        "5k2/7b/8/8/8/8/8/1B1B3K w - - 0 1",  // same-colour bishops both sides
        "5k2/8/8/8/8/7n/8/1B2K3 w - - 0 1",   // K+B vs K+N mixed minors
        "5k2/8/8/8/8/7n/8/1B1B3K w - - 0 1",  // K+BB vs K+N mixed minors
        "5k2/8/8/8/8/8/8/1BBB1B1K w - - 0 1", // mixed-colour multiple bishops
        "5k2/8/8/8/8/8/4p3/1B2K3 w - - 0 1",  // pawn prevents dead material
        "r4k2/8/8/8/8/8/8/1B2K3 w - - 0 1",   // rook prevents dead material
        "3q1k2/8/8/8/8/8/8/1B2K3 w - - 0 1",  // queen prevents dead material
    ];
    for fen in fixtures {
        let ours = position(fen);
        let theirs = shakmaty_from_fen(fen);
        compare_position(&ours, &theirs, fen);
        match (ours.is_checkmate(), ours.is_stalemate()) {
            (true, false) => assert!(theirs.is_checkmate()),
            (false, true) => assert!(theirs.is_stalemate()),
            _ => {}
        }
    }
}

#[test]
fn reference_ep_rank_xray_fixtures() {
    let cases = [
        (
            "7k/8/8/r2pP2K/8/8/8/8 w - d6 0 1",
            "7k/8/8/r2pP2K/8/8/8/8 w - - 0 1",
            "e5d6",
        ),
        (
            "7K/8/8/8/R2Pp2k/8/8/8 b - d3 0 1",
            "7K/8/8/8/R2Pp2k/8/8/8 b - - 0 1",
            "e4d3",
        ),
    ];
    for (with_ep_fen, without_ep_fen, ep_move) in cases {
        let ours = position(with_ep_fen);
        let theirs = shakmaty_from_fen(with_ep_fen);
        let moves = compare_position(&ours, &theirs, with_ep_fen);
        assert_eq!(ours.legal_ep_square(), None);
        assert!(
            !moves.iter().any(|movement| movement.to_uci() == ep_move),
            "reference move set contained the illegal x-ray EP {ep_move}"
        );

        let no_ep = position(without_ep_fen);
        let no_ep_theirs = shakmaty_from_fen(without_ep_fen);
        compare_position(&no_ep, &no_ep_theirs, without_ep_fen);
        assert_eq!(ours.hash(), no_ep.hash());
        assert!(ours.same_repetition_identity(&no_ep));
    }
}

#[test]
fn reference_ep_fen_convention_after_double_push() {
    let mut ours = Position::startpos();
    let mut theirs = shakmaty_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    let e4 = common::uci_move(&ours, "e2e4");
    let e4_shak = theirs
        .legal_moves()
        .iter()
        .find(|candidate| shakmaty_uci(candidate) == "e2e4")
        .cloned()
        .unwrap();
    ours.play(e4).unwrap();
    theirs.play_unchecked(&e4_shak);

    assert_eq!(
        ours.to_fen(),
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    );
    assert_eq!(
        ours.to_fen(),
        Fen::from_position(theirs, EnPassantMode::Always).to_string()
    );
    assert_eq!(ours.legal_ep_square(), None);

    let nc6 = common::uci_move(&ours, "b8c6");
    ours.play(nc6).unwrap();
    assert_eq!(ours.ep_square(), None);
}
