//! Phase A2 per-stage same-run A/B, with the profile-only scaffolding.
//!
//! Each arm is the A1 stage function (original scalar arithmetic) or its
//! dispatched counterpart. Arms are interleaved and reversed on alternate
//! repeats, so the ratios are same-run and not cross-run load comparisons.
//!
//! ```text
//! taskset -c 6 cargo test --release --features profiling \
//!     --test n2_opt_a_a2_stage_bench -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "profiling")]

use std::hint::black_box;
use std::time::Instant;

use horseflight::eval::n2::stages;
use horseflight::eval::n2::{self, Accumulator, ACCUMULATOR_SIZE, HIDDEN_SIZE};
use horseflight::Position;

const FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "2kr3r/pp1q1ppp/2n1pn2/2bp4/8/1PN1PN2/PBPPQPPP/2KR3R w - - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3",
    "8/8/8/8/5k2/8/4P3/4K3 w - - 0 1",
];

struct Timings {
    label: &'static str,
    units_per_run: usize,
    samples: Vec<f64>,
}

impl Timings {
    fn median(&self) -> f64 {
        let mut ordered = self.samples.clone();
        ordered.sort_by(|left, right| left.partial_cmp(right).unwrap());
        ordered[ordered.len() / 2]
    }

    fn min(&self) -> f64 {
        self.samples.iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn max(&self) -> f64 {
        self.samples.iter().copied().fold(0.0, f64::max)
    }

    fn ns(&self) -> (f64, f64, f64) {
        let scale = 1e9 / self.units_per_run as f64;
        (
            self.min() * scale,
            self.median() * scale,
            self.max() * scale,
        )
    }
}

type Arm<'a> = (&'static str, Box<dyn FnMut() + 'a>);

fn measure_arms(repeats: usize, units_per_run: usize, mut arms: Vec<Arm<'_>>) -> Vec<Timings> {
    for (_, arm) in arms.iter_mut() {
        arm();
    }
    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(repeats); arms.len()];
    for repeat in 0..repeats {
        let order: Vec<usize> = if repeat % 2 == 0 {
            (0..arms.len()).collect()
        } else {
            (0..arms.len()).rev().collect()
        };
        for index in order {
            let started = Instant::now();
            (arms[index].1)();
            samples[index].push(started.elapsed().as_secs_f64());
        }
    }
    arms.into_iter()
        .zip(samples)
        .map(|((label, _), samples)| Timings {
            label,
            units_per_run,
            samples,
        })
        .collect()
}

fn json(stage: &str, timings: &[Timings]) {
    for entry in timings {
        let (min, median, max) = entry.ns();
        println!(
            "{{\"kind\":\"stage_same_run\",\"stage\":\"{stage}\",\"arm\":{:?},\
             \"units_per_run\":{},\"repeats\":{},\"min_ns\":{min:.3},\
             \"median_ns\":{median:.3},\"max_ns\":{max:.3},\
             \"raw_samples_seconds\":{:?}}}",
            entry.label,
            entry.units_per_run,
            entry.samples.len(),
            entry.samples
        );
    }
    let scalar = &timings[0];
    let optimized = timings.last().expect("at least one arm");
    println!(
        "{{\"kind\":\"stage_speedup\",\"stage\":\"{stage}\",\
         \"scalar_over_optimized_median\":{:.4}}}",
        scalar.median() / optimized.median()
    );
}

const REPEATS: usize = 15;

#[test]
#[ignore = "performance measurement, run with --ignored --nocapture"]
fn stage_attribution_same_run() {
    let positions: Vec<Position> = FENS
        .iter()
        .map(|fen| Position::from_fen(fen).expect("fen"))
        .collect();
    // Mix all eight positions inside each timed unit, as the A1 stage bench did
    // on one fixed kiwipete accumulator for the head records.
    let accumulators: Vec<Accumulator> = positions.iter().map(Accumulator::from_position).collect();
    let acc = accumulators;
    let traces: Vec<n2::ForwardTrace> = acc.iter().map(n2::forward_trace).collect();
    let a1_i32: Vec<[i32; ACCUMULATOR_SIZE]> = traces.iter().map(|trace| trace.a1).collect();
    let h_i32: Vec<[i32; HIDDEN_SIZE]> = traces.iter().map(|trace| trace.h).collect();
    let a2_i32: Vec<[i32; HIDDEN_SIZE]> = traces.iter().map(|trace| trace.a2).collect();
    let a1_i16: Vec<[i16; ACCUMULATOR_SIZE]> = acc
        .iter()
        .map(|accumulator| stages::stage_first_activation_dispatched(accumulator.values()))
        .collect();
    let a2_i16: Vec<[i16; HIDDEN_SIZE]> = h_i32
        .iter()
        .map(stages::stage_hidden_activation_dispatched)
        .collect();

    let timings = measure_arms(
        REPEATS,
        200_000,
        vec![
            (
                "scalar_i32",
                Box::new(|| {
                    for index in 0..200_000usize {
                        let acc = acc[index % acc.len()].values();
                        black_box(stages::stage_first_activation(black_box(acc)));
                    }
                }),
            ),
            (
                "dispatched_i16",
                Box::new(|| {
                    for index in 0..200_000usize {
                        let acc = acc[index % acc.len()].values();
                        black_box(stages::stage_first_activation_dispatched(black_box(acc)));
                    }
                }),
            ),
        ],
    );
    json("first_activation", &timings);

    let timings = measure_arms(
        REPEATS,
        100_000,
        vec![
            (
                "scalar_oracle_i32_a1",
                Box::new(|| {
                    for index in 0..100_000usize {
                        black_box(stages::stage_hidden_matmul(black_box(
                            &a1_i32[index % a1_i32.len()],
                        )));
                    }
                }),
            ),
            (
                "scalar_i16_weights",
                Box::new(|| {
                    for index in 0..100_000usize {
                        black_box(stages::stage_hidden_matmul_scalar_i16(black_box(
                            &a1_i16[index % a1_i16.len()],
                        )));
                    }
                }),
            ),
            (
                "dispatched_avx2",
                Box::new(|| {
                    for index in 0..100_000usize {
                        black_box(stages::stage_hidden_matmul_dispatched(black_box(
                            &a1_i16[index % a1_i16.len()],
                        )));
                    }
                }),
            ),
        ],
    );
    json("hidden_matmul", &timings);

    let timings = measure_arms(
        REPEATS,
        500_000,
        vec![
            (
                "scalar_i32",
                Box::new(|| {
                    for index in 0..500_000usize {
                        black_box(stages::stage_hidden_activation(black_box(
                            &h_i32[index % h_i32.len()],
                        )));
                    }
                }),
            ),
            (
                "dispatched_i16",
                Box::new(|| {
                    for index in 0..500_000usize {
                        black_box(stages::stage_hidden_activation_dispatched(black_box(
                            &h_i32[index % h_i32.len()],
                        )));
                    }
                }),
            ),
        ],
    );
    json("hidden_activation", &timings);

    let timings = measure_arms(
        REPEATS,
        500_000,
        vec![
            (
                "scalar_i32",
                Box::new(|| {
                    for index in 0..500_000usize {
                        black_box(stages::stage_output_dot(black_box(
                            &a2_i32[index % a2_i32.len()],
                        )));
                    }
                }),
            ),
            (
                "dispatched_i16",
                Box::new(|| {
                    for index in 0..500_000usize {
                        black_box(stages::stage_output_dot_dispatched(black_box(
                            &a2_i16[index % a2_i16.len()],
                        )));
                    }
                }),
            ),
        ],
    );
    json("output_dot", &timings);

    // Incremental accumulator row kernel: the per-node maintenance work.
    let base = *acc[1].values();
    let row = stages::row_for_feature(0);
    let timings = measure_arms(
        REPEATS,
        500_000,
        vec![
            (
                "scalar",
                Box::new(|| {
                    let mut lanes = base;
                    for _ in 0..500_000usize {
                        stages::stage_accumulator_row_add_scalar(&mut lanes, black_box(row));
                        black_box(lanes[0]);
                        lanes = base;
                    }
                }),
            ),
            (
                "dispatched_avx2",
                Box::new(|| {
                    let mut lanes = base;
                    for _ in 0..500_000usize {
                        stages::stage_accumulator_row_add_dispatched(&mut lanes, black_box(row));
                        black_box(lanes[0]);
                        lanes = base;
                    }
                }),
            ),
        ],
    );
    json("accumulator_row_add", &timings);
}
