//! Milestone A acceptance tests for the embedded N2-002 integer model.
//!
//! Covers exact golden parity (features, acc, a1, h, a2, output numerator),
//! asset validation failures, ties-even/clamp boundaries, incremental
//! accumulator equality for special and seeded random moves, and the stateless
//! `Evaluator` reference path. Golden data is read as plain JSON lines with
//! `std` only; there is no new dependency.

use std::fs;
use std::path::Path;

use horseflight::eval::n2::{
    self, active_features, asset_bytes, asset_sha256, dense_feature_vector, forward,
    forward_numerator, forward_trace, validate_bytes, Accumulator, AssetError, Model, N2Evaluator,
    ACCUMULATOR_SIZE, DERIVED_SCORE_PERSPECTIVE, EP_FEATURE_BASE, METADATA_SCORE_PERSPECTIVE,
};
use horseflight::eval::{clamp_evaluation, Evaluator};
use horseflight::{Move, Position, Role, Undo};

#[derive(Debug)]
struct Golden {
    fen: String,
    features: Vec<usize>,
    acc: Vec<i32>,
    a1: Vec<i32>,
    h: Vec<i32>,
    a2: Vec<i32>,
    output_numerator: i32,
}

fn string_field(line: &str, key: &str) -> String {
    let marker = format!("\"{key}\":");
    let start = line
        .find(&marker)
        .unwrap_or_else(|| panic!("missing key {key:?} in {line}"));
    let rest = &line[start + marker.len()..];
    let first = rest.find('"').expect("opening quote");
    let rest = &rest[first + 1..];
    let end = rest.find('"').expect("closing quote");
    rest[..end].to_string()
}

fn int_field(line: &str, key: &str) -> i32 {
    let marker = format!("\"{key}\":");
    let start = line
        .find(&marker)
        .unwrap_or_else(|| panic!("missing key {key:?} in {line}"));
    let rest = line[start + marker.len()..].trim_start();
    let end = rest
        .find(|ch: char| !(ch == '-' || ch.is_ascii_digit()))
        .unwrap_or(rest.len());
    rest[..end]
        .parse()
        .unwrap_or_else(|_| panic!("bad integer for {key:?}: {:?}", &rest[..end]))
}

fn int_array(line: &str, key: &str) -> Vec<i32> {
    let marker = format!("\"{key}\":");
    let start = line
        .find(&marker)
        .unwrap_or_else(|| panic!("missing key {key:?} in {line}"));
    let rest = &line[start + marker.len()..];
    let open = rest.find('[').expect("array open");
    let close = rest[open..].find(']').expect("array close") + open;
    let body = &rest[open + 1..close];
    if body.trim().is_empty() {
        return Vec::new();
    }
    body.split(',')
        .map(|part| {
            part.trim()
                .parse()
                .unwrap_or_else(|_| panic!("bad integer array entry {part:?}"))
        })
        .collect()
}

fn load_goldens() -> Vec<Golden> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/n2-002/goldens.jsonl");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| Golden {
            fen: string_field(line, "fen"),
            features: int_array(line, "features")
                .into_iter()
                .map(|value| value as usize)
                .collect(),
            acc: int_array(line, "acc"),
            a1: int_array(line, "a1"),
            h: int_array(line, "h"),
            a2: int_array(line, "a2"),
            output_numerator: int_field(line, "output_numerator"),
        })
        .collect()
}

fn assert_lanes(layer: &str, row: usize, actual: &[i32], expected: &[i32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "golden row {row} {layer}: length {} != expected {}",
        actual.len(),
        expected.len()
    );
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual, expected,
            "golden row {row} {layer} index {index}: actual {actual}, expected {expected}"
        );
    }
}

#[test]
fn all_144_goldens_match_through_every_tensor() {
    let goldens = load_goldens();
    assert_eq!(goldens.len(), 144, "expected 144 supplied goldens");

    for (row, golden) in goldens.iter().enumerate() {
        let position = Position::from_fen(&golden.fen)
            .unwrap_or_else(|error| panic!("golden row {row} FEN invalid: {error}"));

        let features = active_features(&position);
        assert_eq!(
            features, golden.features,
            "golden row {row} features: actual {features:?}, expected {:?}",
            golden.features
        );

        let accumulator = Accumulator::from_position(&position);
        assert_lanes("acc", row, accumulator.values(), &golden.acc);

        let trace = forward_trace(&accumulator);
        assert_lanes("a1", row, &trace.a1, &golden.a1);
        assert_lanes("h", row, &trace.h, &golden.h);
        assert_lanes("a2", row, &trace.a2, &golden.a2);
        assert_eq!(
            trace.output_numerator, golden.output_numerator,
            "golden row {row} output_numerator: actual {}, expected {}",
            trace.output_numerator, golden.output_numerator
        );
        assert_eq!(
            forward_numerator(&accumulator),
            golden.output_numerator,
            "golden row {row} forward_numerator disagrees with trace"
        );
    }
}

#[test]
fn dense_vector_marks_exactly_the_active_inputs() {
    let start = Position::startpos();
    let dense = dense_feature_vector(&start);
    assert_eq!(dense.iter().filter(|value| **value != 0).count(), 37);
    for index in [768, 769, 770, 771, 772] {
        assert_eq!(dense[index], 1, "input {index} should be active");
    }

    let ep = Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
    let dense_ep = dense_feature_vector(&ep);
    assert_eq!(dense_ep[EP_FEATURE_BASE + 3], 1, "d-file EP input");
    assert_eq!(
        active_features(&ep).last().copied(),
        Some(EP_FEATURE_BASE + 3)
    );
}

#[test]
fn asset_validation_distinguishes_failure_modes() {
    let bytes = asset_bytes();
    assert!(validate_bytes(bytes).is_ok());
    assert_eq!(asset_sha256(), n2::sha256::hex(&n2::sha256::digest(bytes)));

    match validate_bytes(&bytes[..100]) {
        Err(AssetError::TruncatedBytes) => {}
        other => panic!("truncated header should be TruncatedBytes, got {other:?}"),
    }

    let mut bad_magic = bytes.to_vec();
    bad_magic[0] = b'X';
    match validate_bytes(&bad_magic) {
        Err(AssetError::BadMagic) => {}
        other => panic!("bad magic should be BadMagic, got {other:?}"),
    }

    let mut bad_checksum = bytes.to_vec();
    let last = bad_checksum.len() - 1;
    bad_checksum[last] ^= 0xff;
    match validate_bytes(&bad_checksum) {
        Err(AssetError::ChecksumMismatch) => {}
        other => panic!("flipped checksum should be ChecksumMismatch, got {other:?}"),
    }

    // Remove bytes from inside the tensor region but keep the trailing checksum,
    // so the fixed tensor length check fails before the checksum is inspected.
    let checksum_at = bytes.len() - 32;
    let mut wrong_length = bytes[..checksum_at - 64].to_vec();
    wrong_length.extend_from_slice(&bytes[checksum_at..]);
    match validate_bytes(&wrong_length) {
        Err(AssetError::TensorLength {
            expected, actual, ..
        }) => {
            assert!(
                actual < expected,
                "{actual} should be shorter than {expected}"
            );
        }
        other => panic!("short tensor region should be TensorLength, got {other:?}"),
    }

    let mut trailing = bytes.to_vec();
    trailing.push(0xaa);
    match validate_bytes(&trailing) {
        Err(AssetError::TrailingBytes) => {}
        other => panic!("appended byte should be TrailingBytes, got {other:?}"),
    }

    let mut bad_version = bytes.to_vec();
    bad_version[8..12].copy_from_slice(&2u32.to_le_bytes());
    match validate_bytes(&bad_version) {
        Err(AssetError::UnsupportedContainerVersion(2)) => {}
        other => panic!("version 2 should be unsupported, got {other:?}"),
    }
}

#[test]
fn embedded_metadata_preserves_supplied_perspective_string() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/n2-002.metadata.json");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    assert!(
        text.contains(METADATA_SCORE_PERSPECTIVE),
        "metadata.json must still carry the supplied perspective string verbatim"
    );
    assert!(DERIVED_SCORE_PERSPECTIVE.contains("white-relative"));
    assert_eq!(n2::provenance_flags(), 0);
    assert_eq!(n2::asset_sha256().len(), 64);
}

#[test]
fn ties_even_cp_conversion_and_black_negation_are_exact() {
    // Known values from an independent Python integer reference.
    let white = Position::startpos();
    let black =
        Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1").unwrap();
    assert_eq!(forward(&Accumulator::from_position(&white)), 14);
    assert_eq!(forward(&Accumulator::from_position(&black)), -28);
    assert_eq!(N2Evaluator.evaluate(&white), 14);
    assert_eq!(N2Evaluator.evaluate(&black), 28);
    assert_eq!(
        N2Evaluator.evaluate(&white),
        clamp_evaluation(forward(&Accumulator::from_position(&white)))
    );
}

#[test]
fn n2_evaluator_is_stateless() {
    let positions = [
        Position::startpos(),
        Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
            .unwrap(),
        Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap(),
    ];
    for position in positions {
        let before = position;
        let score = N2Evaluator.evaluate(&position);
        assert_eq!(position, before, "evaluator must not change the position");
        assert_eq!(
            score,
            clamp_evaluation(forward(&Accumulator::from_position(&position)))
        );
    }
}

const SPECIAL_FENS: &[&str] = &[
    // Start position.
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    // Open board with castling rights for both sides.
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    // Legal en passant, pinned en passant, rank x-ray en passant.
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "7k/8/8/r2pP2K/8/8/8/8 w - d6 0 1",
    // Raw EP target that is not capturable.
    "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
    // Promotions: white and black, with and without capture.
    "n1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
    "4k2n/8/8/8/8/8/1p6/4K3 b - - 0 1",
    "1r2k3/1P6/8/8/8/8/6p1/4K3 w - - 0 1",
];

#[derive(Clone, Copy)]
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Lcg {
        Lcg(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1))
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() >> 33) as usize % bound
    }
}

fn pick_promotion(moves: &[Move], role: Role) -> Option<Move> {
    moves
        .iter()
        .copied()
        .find(|movement| movement.promotion_role() == Some(role))
}

#[test]
fn special_moves_are_incremental_exact() {
    let mut saw_kingside_castle = false;
    let mut saw_queenside_castle = false;
    let mut saw_en_passant = false;
    let mut promotions = Vec::new();

    for fen in SPECIAL_FENS {
        let position = Position::from_fen(fen).unwrap();
        let moves = position.legal_moves();
        for role in Role::PROMOTION_TARGETS {
            if let Some(movement) = pick_promotion(&moves, role) {
                promotions.push((movement, role));
            }
        }
        for movement in moves.iter().copied() {
            if movement.is_castle() {
                if movement.to().file() > movement.from().file() {
                    saw_kingside_castle = true;
                } else {
                    saw_queenside_castle = true;
                }
            }
            if movement.is_en_passant() {
                saw_en_passant = true;
            }

            let mut played = position;
            let mut accumulator = Accumulator::from_position(&position);
            let before = played;
            let undo = played.play(movement).unwrap();
            let after = played;
            accumulator.apply_move(undo.piece_changes(), &before, &after);

            let rebuilt = Accumulator::from_position(&after);
            assert_lanes("incremental acc", 0, accumulator.values(), rebuilt.values());

            let mut reverted = after;
            reverted.unmake(&undo).unwrap();
            accumulator.revert_move(undo.piece_changes(), &before, &after);
            let rebuilt_before = Accumulator::from_position(&before);
            assert_lanes(
                "reverted acc",
                0,
                accumulator.values(),
                rebuilt_before.values(),
            );
            assert_eq!(
                reverted, before,
                "unmake must restore board, hash and state"
            );
            assert_eq!(reverted.hash(), before.hash());
        }
    }

    assert!(saw_kingside_castle, "test fixtures must include O-O");
    assert!(saw_queenside_castle, "test fixtures must include O-O-O");
    assert!(saw_en_passant, "test fixtures must include en passant");
    let roles: Vec<Role> = promotions.iter().map(|(_, role)| *role).collect();
    for role in Role::PROMOTION_TARGETS {
        assert!(
            roles.contains(&role),
            "test fixtures must include promotion to {role:?}"
        );
    }
}

#[test]
fn pinned_en_passant_is_reported_illegal_and_excluded() {
    let position = Position::from_fen("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
    assert_eq!(position.legal_ep_square(), None);
    let features = active_features(&position);
    assert!(
        !features.iter().any(|index| *index >= EP_FEATURE_BASE),
        "pinned EP must not activate an EP input: {features:?}"
    );
    let moves = position.legal_moves();
    assert!(
        !moves.iter().any(|movement| movement.is_en_passant()),
        "pinned EP capture must not be generated"
    );
}

#[test]
fn seeded_random_moves_match_full_rebuild_after_make_and_unmake() {
    let goldens = load_goldens();
    let mut fixtures: Vec<Position> = goldens
        .iter()
        .take(12)
        .map(|golden| Position::from_fen(&golden.fen).unwrap())
        .collect();
    fixtures.push(Position::startpos());

    let mut total_moves = 0usize;
    for seed in 0..120u64 {
        let mut position = fixtures[seed as usize % fixtures.len()];
        let mut accumulator = Accumulator::from_position(&position);
        let mut rng = Lcg::new(seed + 1);
        let mut stack: Vec<(Undo, Position, Position)> = Vec::new();

        for _ply in 0..25 {
            let moves = position.legal_moves();
            if moves.is_empty() {
                break;
            }
            let movement = moves[rng.below(moves.len())];
            let before = position;
            let undo = position.play(movement).unwrap();
            let after = position;

            accumulator.apply_move(undo.piece_changes(), &before, &after);
            let rebuilt = Accumulator::from_position(&after);
            assert_lanes(
                "random incremental acc",
                seed as usize,
                accumulator.values(),
                rebuilt.values(),
            );
            total_moves += 1;
            stack.push((undo, before, after));
        }

        while let Some((undo, before, after)) = stack.pop() {
            let mut reverted = position;
            reverted.unmake(&undo).unwrap();
            accumulator.revert_move(undo.piece_changes(), &before, &after);
            let rebuilt = Accumulator::from_position(&before);
            assert_lanes(
                "random reverted acc",
                seed as usize,
                accumulator.values(),
                rebuilt.values(),
            );
            assert_eq!(
                reverted, before,
                "seed {seed}: board/hash/state not restored by unmake"
            );
            assert_eq!(reverted.hash(), before.hash(), "seed {seed}: hash mismatch");
            position = reverted;
        }
        assert!(stack.is_empty());
    }

    assert!(
        total_moves >= 2000,
        "expected at least 2000 randomized make/unmake moves, ran {total_moves}"
    );
}

#[test]
fn model_meta_exposes_fixed_dimensions_and_schema() {
    let meta = n2::model().meta();
    assert_eq!(meta.in_features, 781);
    assert_eq!(meta.accumulator, 256);
    assert_eq!(meta.hidden, 32);
    assert_eq!(meta.out, 1);
    assert_eq!(meta.acc_scale, 256);
    assert_eq!(meta.act_max, 127);
    assert_eq!(meta.hidden_scale, 128);
    assert_eq!(meta.out_scale, 256);
    assert_eq!(meta.cp_multiplier, 400);
    assert_eq!(meta.source_hashes[0].len(), 32);
    assert_eq!(ACCUMULATOR_SIZE, 256);
    let _: &Model = n2::model();
}
