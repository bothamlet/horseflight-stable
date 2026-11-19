//! Phase A1 stage timings for the N2-002 evaluation path.
//!
//! Requires the profile-only scaffolding, so it only builds with the feature
//! and is `#[ignore]`d as a measurement rather than an assertion:
//!
//! ```text
//! taskset -c 6 cargo test --release --features profiling \
//!     --test n2_opt_a_bench -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Every line is one JSON object with the raw samples in seconds at full
//! `f64` precision, plus min/median/max. Nothing here is a strength claim and
//! no record is a production search time: the stages are timed one at a time
//! and their sum is *not* the cost of an evaluation.

#![cfg(feature = "profiling")]

use std::hint::black_box;
use std::time::Instant;

use horseflight::eval::n2::session::Session;
use horseflight::eval::n2::stages;
use horseflight::eval::n2::{self, Accumulator, N2Evaluator, ACCUMULATOR_SIZE, HIDDEN_SIZE};
use horseflight::eval::{EvaluationSession, Evaluator, HandcraftedEvaluator};
use horseflight::{Game, Position};

/// The historical four-position suite, unchanged and in the original order.
const CORE_FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "2kr3r/pp1q1ppp/2n1pn2/2bp4/8/1PN1PN2/PBPPQPPP/2KR3R w - - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
];

/// Deterministic extra coverage: promotion-rich, sharp tactic, legal en passant,
/// and a sparse king-and-pawn endgame.
const EXTRA_FENS: &[&str] = &[
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3",
    "8/8/8/8/5k2/8/4P3/4K3 w - - 0 1",
];

fn parse_all(fens: &[&str]) -> Vec<Position> {
    fens.iter()
        .map(|fen| Position::from_fen(fen).unwrap_or_else(|error| panic!("{fen}: {error}")))
        .collect()
}

/// Repeated wall-clock samples for one fixed amount of work.
struct Timings {
    units_per_run: usize,
    samples: Vec<f64>,
}

impl Timings {
    fn min(&self) -> f64 {
        self.samples.iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn max(&self) -> f64 {
        self.samples.iter().copied().fold(0.0, f64::max)
    }

    fn median(&self) -> f64 {
        let mut ordered = self.samples.clone();
        ordered.sort_by(|left, right| left.partial_cmp(right).unwrap());
        ordered[ordered.len() / 2]
    }

    fn per_unit_ns(&self, seconds: f64) -> f64 {
        1e9 * seconds / self.units_per_run as f64
    }

    /// `seconds` fields plus the derived per-unit and per-second rates.
    fn fields(&self, unit: &str) -> Vec<(&'static str, String)> {
        vec![
            ("units_per_run", self.units_per_run.to_string()),
            ("repeats", self.samples.len().to_string()),
            ("unit", format!("{unit:?}")),
            ("min_seconds", f(self.min())),
            ("median_seconds", f(self.median())),
            ("max_seconds", f(self.max())),
            ("min_ns_per_unit", f(self.per_unit_ns(self.min()))),
            ("median_ns_per_unit", f(self.per_unit_ns(self.median()))),
            ("max_ns_per_unit", f(self.per_unit_ns(self.max()))),
            (
                "median_units_per_second",
                f(self.units_per_run as f64 / self.median()),
            ),
            ("raw_samples_seconds", format!("{:?}", self.samples)),
        ]
    }
}

fn f(value: f64) -> String {
    format!("{value:.9e}")
}

fn json(kind: &str, fields: &[(&str, String)]) {
    let body = fields
        .iter()
        .map(|(key, value)| format!("\"{key}\":{value}"))
        .collect::<Vec<_>>()
        .join(",");
    println!("{{\"kind\":\"{kind}\",{body}}}");
}

/// One warm-up pass followed by `repeats` timed passes of identical work.
fn measure(repeats: usize, units_per_run: usize, mut run: impl FnMut()) -> Timings {
    run();
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        run();
        samples.push(started.elapsed().as_secs_f64());
    }
    Timings {
        units_per_run,
        samples,
    }
}

const REPEATS: usize = 15;

/// Fixed inputs shared by the head-stage records.
struct Fixtures {
    accumulator: Accumulator,
    a1: [i32; ACCUMULATOR_SIZE],
    h: [i32; HIDDEN_SIZE],
    a2: [i32; HIDDEN_SIZE],
    output_numerator: i32,
}

fn fixtures(position: &Position) -> Fixtures {
    let accumulator = Accumulator::from_position(position);
    let trace = n2::forward_trace(black_box(&accumulator));
    Fixtures {
        a1: trace.a1,
        h: trace.h,
        a2: trace.a2,
        output_numerator: trace.output_numerator,
        accumulator,
    }
}

#[test]
#[ignore = "performance measurement, run in release with --features profiling --ignored"]
fn forward_pass_stages() {
    let fens: Vec<&str> = CORE_FENS.iter().chain(EXTRA_FENS.iter()).copied().collect();
    let all = parse_all(&fens);
    let middlegame = all[1];
    let fixed = fixtures(&middlegame);

    let tensors = stages::tensors();
    let scales = stages::scales();
    json(
        "stage_contract",
        &[
            ("acc_scale", scales.0.to_string()),
            ("act_max", scales.1.to_string()),
            ("hidden_scale", scales.2.to_string()),
            ("w1_len", tensors.w1.len().to_string()),
            ("b1_len", tensors.b1.len().to_string()),
            ("w2_len", tensors.w2.len().to_string()),
            ("b2_len", tensors.b2.len().to_string()),
            ("w3_len", tensors.w3.len().to_string()),
            ("b3_len", tensors.b3.len().to_string()),
            (
                "w2_element_bytes",
                std::mem::size_of_val(&tensors.w2[0]).to_string(),
            ),
            (
                "w1_row_bytes",
                std::mem::size_of_val(&tensors.w1[0]).to_string(),
            ),
        ],
    );

    // Active-feature extraction without allocation: the production visitor.
    let mut extracted = 0usize;
    let timings = measure(REPEATS, 1_000_000, || {
        let mut count = 0usize;
        for index in 0..1_000_000usize {
            let position = &all[index % all.len()];
            n2::for_each_active_feature(black_box(position), |_| count += 1);
        }
        extracted = count;
    });
    let mut fields = vec![
        ("positions", format!("{:?}", all.len())),
        ("features_visited", extracted.to_string()),
    ];
    fields.extend(timings.fields("visits"));
    json("active_feature_extraction", &fields);

    // The allocating collector, for the cost of the `Vec` the visitor avoids.
    let timings = measure(REPEATS, 200_000, || {
        for index in 0..200_000usize {
            black_box(n2::active_features(black_box(&all[index % all.len()])));
        }
    });
    json("active_feature_collect_alloc", &timings.fields("collects"));

    // Accumulator rebuild, both the shipped entry point and the raw bias copy.
    let timings = measure(REPEATS, 50_000, || {
        for index in 0..50_000usize {
            black_box(Accumulator::from_position(black_box(
                &all[index % all.len()],
            )));
        }
    });
    json("accumulator_rebuild", &timings.fields("rebuilds"));

    let timings = measure(REPEATS, 1_000_000, || {
        for _ in 0..1_000_000usize {
            let mut lanes = [0i32; ACCUMULATOR_SIZE];
            lanes.copy_from_slice(black_box(tensors.b1));
            black_box(lanes);
        }
    });
    json("accumulator_bias_copy", &timings.fields("copies"));

    // One input row accumulate: 256 `i16 as i32` adds, the rebuild's inner loop.
    let row = &tensors.w1[0..ACCUMULATOR_SIZE];
    let timings = measure(REPEATS, 1_000_000, || {
        let mut lanes = [0i32; ACCUMULATOR_SIZE];
        lanes.copy_from_slice(tensors.b1);
        for _ in 0..1_000_000usize {
            for (lane, weight) in lanes.iter_mut().zip(black_box(row)) {
                *lane += *weight as i32;
            }
            black_box(&lanes);
        }
    });
    json("accumulator_row_add", &timings.fields("row_adds"));

    // Prepared incremental delta pair: make and unmake with everything else
    // computed outside the timed region.
    let game = Game::from_fen(CORE_FENS[1]).unwrap();
    let mut prepared = *game.position();
    let movement = prepared.legal_moves()[0];
    let before = prepared;
    let undo = prepared.play(movement).expect("legal move");
    let after = prepared;
    let timings = measure(REPEATS, 200_000, || {
        let mut session = Session::new(&before);
        for _ in 0..200_000usize {
            session.on_make(undo.piece_changes(), &before, &after);
            session.on_unmake();
        }
        black_box(session.depth());
    });
    json("session_delta_pair_prepared", &timings.fields("pairs"));

    // Stage-by-stage head decomposition on fixed inputs.
    let acc = fixed.accumulator.values();
    let timings = measure(REPEATS, 200_000, || {
        for _ in 0..200_000usize {
            black_box(stages::stage_first_activation(black_box(acc)));
        }
    });
    json("stage_first_activation", &timings.fields("activations"));

    let a1 = &fixed.a1;
    let timings = measure(REPEATS, 100_000, || {
        for _ in 0..100_000usize {
            black_box(stages::stage_hidden_matmul(black_box(a1)));
        }
    });
    json("stage_hidden_matmul", &timings.fields("matmuls"));

    let h = &fixed.h;
    let timings = measure(REPEATS, 500_000, || {
        for _ in 0..500_000usize {
            black_box(stages::stage_hidden_activation(black_box(h)));
        }
    });
    json("stage_hidden_activation", &timings.fields("activations"));

    let a2 = &fixed.a2;
    let timings = measure(REPEATS, 500_000, || {
        for _ in 0..500_000usize {
            black_box(stages::stage_output_dot(black_box(a2)));
        }
    });
    json("stage_output_dot", &timings.fields("dots"));

    let output = fixed.output_numerator;
    let timings = measure(REPEATS, 1_000_000, || {
        for _ in 0..1_000_000usize {
            black_box(stages::stage_score_conversion(black_box(output)));
        }
    });
    json("stage_score_conversion", &timings.fields("conversions"));

    // Complete neural head, exactly `Session::evaluate`'s inference.
    let timings = measure(REPEATS, 100_000, || {
        for _ in 0..100_000usize {
            black_box(n2::forward(black_box(&fixed.accumulator)));
        }
    });
    json("complete_head", &timings.fields("heads"));

    // Session evaluate: head plus side-to-move sign and clamp.
    let session = Session::new(&middlegame);
    let timings = measure(REPEATS, 100_000, || {
        for _ in 0..100_000usize {
            black_box(session.evaluate(black_box(&middlegame)));
        }
    });
    json("session_evaluate", &timings.fields("evaluations"));

    // Copies and initialisation the head performs per evaluation. `forward_trace`
    // zips `row.iter()` with the 1 KiB `a1` *array by value*, so the compiler
    // copies all 256 lanes once per hidden neuron; both arrays are also zero
    // initialised before being fully overwritten.
    let a1_array = fixed.a1;
    let timings = measure(REPEATS, 20_000, || {
        for _ in 0..20_000usize {
            let mut sink = 0i32;
            for _ in 0..HIDDEN_SIZE {
                let copy = black_box(a1_array);
                sink ^= copy[0];
            }
            black_box(sink);
        }
    });
    json(
        "a1_array_copy_per_hidden_neuron",
        &timings.fields("forwards"),
    );

    let timings = measure(REPEATS, 200_000, || {
        for _ in 0..200_000usize {
            let mut a1 = [0i32; ACCUMULATOR_SIZE];
            let mut h = [0i32; HIDDEN_SIZE];
            a1[0] = 1;
            h[0] = 1;
            black_box((a1, h));
        }
    });
    json("activation_buffer_zeroing", &timings.fields("forwards"));

    // Stateless reference path: rebuild plus head plus sign plus clamp.
    let timings = measure(REPEATS, 20_000, || {
        for index in 0..20_000usize {
            black_box(N2Evaluator.evaluate(black_box(&all[index % all.len()])));
        }
    });
    json("stateless_evaluate", &timings.fields("evaluations"));

    // Per-position rebuild cost, so a report can separate position shape from
    // the fixed 256-lane overhead.
    for (position, fen) in all.iter().zip(&fens) {
        let features = n2::active_features(position).len();
        let timings = measure(REPEATS, 20_000, || {
            for _ in 0..20_000usize {
                black_box(Accumulator::from_position(black_box(position)));
            }
        });
        let mut fields = vec![
            ("fen", format!("{fen:?}")),
            ("active_features", features.to_string()),
        ];
        fields.extend(timings.fields("rebuilds"));
        json("accumulator_rebuild_per_position", &fields);
    }

    // Classical evaluator, unchanged, as the scale reference.
    let timings = measure(REPEATS, 200_000, || {
        for index in 0..200_000usize {
            black_box(HandcraftedEvaluator.evaluate(black_box(&all[index % all.len()])));
        }
    });
    json("classical_evaluate", &timings.fields("evaluations"));
}

/// Clock overhead the instrumented timings include twice per timed section.
#[test]
#[ignore = "performance measurement, run in release with --features profiling --ignored"]
fn timer_overhead() {
    let timings = measure(REPEATS, 1_000_000, || {
        let mut accumulated = 0u128;
        for _ in 0..1_000_000usize {
            let started = Instant::now();
            accumulated += started.elapsed().as_nanos();
        }
        black_box(accumulated);
    });
    json("timer_overhead", &timings.fields("pairs"));
}

/// Cold model load: parse, checksum and allocate the 409 527-byte asset.
///
/// Must be run alone so nothing has already initialised the `OnceLock`:
/// `--exact cold_model_load`. Warm calls are timed for contrast.
#[test]
#[ignore = "performance measurement, run in release with --features profiling --ignored"]
fn cold_model_load() {
    let started = Instant::now();
    let hash = n2::asset_sha256();
    let cold = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let warm_hash = n2::asset_sha256();
    let warm = started.elapsed().as_secs_f64();
    black_box(hash);
    black_box(warm_hash);
    json(
        "model_load",
        &[
            ("asset_bytes", n2::asset_bytes().len().to_string()),
            ("cold_seconds", f(cold)),
            ("warm_seconds", f(warm)),
            ("asset_sha256", format!("{:?}", n2::asset_sha256())),
        ],
    );
}
