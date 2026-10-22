mod common;

use common::position;
use horseflight::{perft, perft_divide};

const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
const POSITION_3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
const POSITION_4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
const POSITION_4_MIRRORED: &str =
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1";
const POSITION_5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
const POSITION_6: &str = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

#[test]
fn startpos_perft_depths_zero_to_four() {
    let start = horseflight::Position::startpos();
    for (depth, expected) in [(0, 1), (1, 20), (2, 400), (3, 8902), (4, 197_281)] {
        assert_eq!(perft(&start, depth).unwrap(), expected, "depth {depth}");
    }
}

#[test]
fn cpw_positions_through_depth_three() {
    let cases = [
        (KIWIPETE, [(1, 48), (2, 2039), (3, 97_862)]),
        (POSITION_3, [(1, 14), (2, 191), (3, 2812)]),
        (POSITION_4, [(1, 6), (2, 264), (3, 9467)]),
        (POSITION_4_MIRRORED, [(1, 6), (2, 264), (3, 9467)]),
        (POSITION_5, [(1, 44), (2, 1486), (3, 62_379)]),
        (POSITION_6, [(1, 46), (2, 2079), (3, 89_890)]),
    ];
    for (fen, depths) in cases {
        let position = position(fen);
        for (depth, expected) in depths {
            assert_eq!(
                perft(&position, depth).unwrap(),
                expected,
                "{fen} depth {depth}"
            );
        }
    }
}

#[test]
#[ignore = "depth-5 start position is exercised by the release CLI acceptance gate"]
fn startpos_perft_depth_five() {
    assert_eq!(
        perft(&horseflight::Position::startpos(), 5).unwrap(),
        4_865_609
    );
}

#[test]
#[ignore = "CPW depth-4 suite is slow in debug; run with cargo test --release -- --ignored"]
fn cpw_positions_through_depth_four() {
    let cases = [
        (KIWIPETE, [(4, 4_085_603)]),
        (POSITION_3, [(4, 43_238)]),
        (POSITION_4, [(4, 422_333)]),
        (POSITION_4_MIRRORED, [(4, 422_333)]),
        (POSITION_5, [(4, 2_103_487)]),
        (POSITION_6, [(4, 3_894_594)]),
    ];
    for (fen, depths) in cases {
        let position = position(fen);
        for (depth, expected) in depths {
            assert_eq!(
                perft(&position, depth).unwrap(),
                expected,
                "{fen} depth {depth}"
            );
        }
    }
}

#[test]
fn divide_sums_match_perft_and_zero_depth_is_one() {
    let start = horseflight::Position::startpos();
    assert!(perft_divide(&start, 0).unwrap().is_empty());
    assert_eq!(perft(&start, 0).unwrap(), 1);

    for depth in 1..=3 {
        let rows = perft_divide(&start, depth).unwrap();
        assert_eq!(rows.len(), 20, "start position has 20 root moves");
        let sum: u64 = rows.iter().map(|(_, count)| count).sum();
        assert_eq!(
            sum,
            perft(&start, depth).unwrap(),
            "divide sum depth {depth}"
        );
        if depth == 1 {
            assert!(rows.iter().all(|(_, count)| *count == 1));
        }
    }

    let mate = position("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1");
    assert!(perft(&mate, 0).unwrap() == 1);
    assert!(perft(&mate, 1).unwrap() == 0);
    assert!(perft_divide(&mate, 1).unwrap().is_empty());
}
