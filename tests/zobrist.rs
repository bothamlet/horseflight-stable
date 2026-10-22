mod common;

use common::{assert_valid, position, STARTPOS_FEN};
use horseflight::{Position, ZOBRIST_SEED};

#[test]
fn zobrist_tables_are_deterministic_and_pinned() {
    assert_eq!(ZOBRIST_SEED, 0x484F_5253_4546_4C54);
    assert_eq!(Position::startpos().hash(), 0x9215_f654_4b9a_a1c8);
    assert_eq!(
        position("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").hash(),
        0x5c82_a18c_ac8f_9f26
    );
    assert_eq!(
        position("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").hash(),
        0xa840_9ea0_2d4f_efaa
    );
    assert_eq!(
        position("4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1").hash(),
        0xe4fa_3d32_1dbf_81b1
    );
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
fn incremental_hash_equals_recomputation_at_every_make_and_unmake() {
    let fixtures = [
        STARTPOS_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "n1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    ];

    for (fixture_index, fen) in fixtures.iter().enumerate() {
        for seed in [1u64, 0xdead_beef, 0x1234_5678_9abc_def0] {
            let start = position(fen);
            let mut position = start;
            if *fen == STARTPOS_FEN {
                assert!(common::uci_move(&position, "e2e4").is_double_pawn_push());
            }
            let mut undos = Vec::new();
            let mut state = seed ^ (fixture_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            for _ in 0..32 {
                let moves: Vec<_> = position.legal_moves().iter().copied().collect();
                if moves.is_empty() {
                    break;
                }
                let movement = moves[(xorshift64(&mut state) as usize) % moves.len()];
                let undo = position.play(movement).unwrap();
                assert_eq!(position.hash(), position.recompute_hash());
                assert_valid(&position);
                undos.push(undo);
            }
            while let Some(undo) = undos.pop() {
                position.unmake(&undo).unwrap();
                assert_eq!(position.hash(), position.recompute_hash());
                assert_valid(&position);
            }
            assert_eq!(position, start);
        }
    }
}
