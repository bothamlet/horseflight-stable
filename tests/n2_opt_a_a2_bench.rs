//! Phase A2 same-run A/B measurements for the dispatched N2-002 head.
//!
//! Both arms run in one process and one build, so the comparison is not exposed
//! to the host's cross-run load swings. Arms are alternated (and the order is
//! reversed on alternate repeats) so frequency ramp and background load hit both
//! arms. The scalar arm is the independent reference (`forward_trace`), the
//! optimized arm is production `forward` / [`Session`]. Search nodes, scores,
//! moves and PVs are asserted equal between arms; only elapsed time and NPS are
//! expected to differ.
//!
//! ```text
//! taskset -c 6 cargo test --release --test n2_opt_a_a2_bench \
//!     -- --ignored --nocapture --test-threads=1
//! ```

use std::hint::black_box;
use std::time::Instant;

use horseflight::eval::n2::session::Session;
use horseflight::eval::n2::{
    self, forward, forward_numerator, forward_trace, Accumulator, N2Evaluator,
};
use horseflight::eval::{clamp_evaluation, EvaluationSession, Evaluator};
use horseflight::search::{search_with_session, SearchConfig};
use horseflight::{Game, PieceChange, Position};

/// The A1 suite, unchanged.
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

fn positions() -> Vec<Position> {
    FENS.iter()
        .map(|fen| Position::from_fen(fen).unwrap_or_else(|error| panic!("{fen}: {error}")))
        .collect()
}

/// One arm's raw samples.
struct Timings {
    label: String,
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

    fn ns_per_unit(&self) -> (f64, f64, f64) {
        let scale = 1e9 / self.units_per_run as f64;
        (
            self.min() * scale,
            self.median() * scale,
            self.max() * scale,
        )
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        let (min, median, max) = self.ns_per_unit();
        vec![
            ("arm", format!("{:?}", self.label)),
            ("units_per_run", self.units_per_run.to_string()),
            ("repeats", self.samples.len().to_string()),
            ("min_seconds", format!("{:.9e}", self.min())),
            ("median_seconds", format!("{:.9e}", self.median())),
            ("max_seconds", format!("{:.9e}", self.max())),
            ("min_ns_per_unit", format!("{min:.3}")),
            ("median_ns_per_unit", format!("{median:.3}")),
            ("max_ns_per_unit", format!("{max:.3}")),
            ("raw_samples_seconds", format!("{:?}", self.samples)),
        ]
    }
}

/// Measure every arm in the same loop, reversing arm order on alternate
/// repeats. One untimed warm-up pass runs first for every arm.
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
            label: label.to_string(),
            units_per_run,
            samples,
        })
        .collect()
}

fn json(kind: &str, fields: &[(&'static str, String)]) {
    let body = fields
        .iter()
        .map(|(key, value)| format!("\"{key}\":{value}"))
        .collect::<Vec<_>>()
        .join(",");
    println!("{{\"kind\":\"{kind}\",{body}}}");
}

const REPEATS: usize = 15;

#[test]
#[ignore = "performance measurement, run with --ignored --nocapture"]
fn head_and_evaluation_same_run() {
    let all = positions();
    let accumulators: Vec<Accumulator> = all.iter().map(Accumulator::from_position).collect();

    let timings = measure_arms(
        REPEATS,
        50_000,
        vec![
            (
                "scalar_forward_trace",
                Box::new(|| {
                    for index in 0..50_000usize {
                        let accumulator = &accumulators[index % accumulators.len()];
                        black_box(forward_trace(black_box(accumulator)).output_numerator);
                    }
                }),
            ),
            (
                "optimized_forward_numerator",
                Box::new(|| {
                    for index in 0..50_000usize {
                        let accumulator = &accumulators[index % accumulators.len()];
                        black_box(forward_numerator(black_box(accumulator)));
                    }
                }),
            ),
            (
                "optimized_forward",
                Box::new(|| {
                    for index in 0..50_000usize {
                        let accumulator = &accumulators[index % accumulators.len()];
                        black_box(forward(black_box(accumulator)));
                    }
                }),
            ),
        ],
    );
    for entry in &timings {
        json("head_same_run", &entry.fields());
    }
    let head_scalar_median = timings[0].median();
    let head_optimized_median = timings[1].median();

    let timings = measure_arms(
        REPEATS,
        20_000,
        vec![
            (
                "scalar_evaluator",
                Box::new(|| {
                    for index in 0..20_000usize {
                        black_box(ScalarEvaluator.evaluate(black_box(&all[index % all.len()])));
                    }
                }),
            ),
            (
                "optimized_evaluator",
                Box::new(|| {
                    for index in 0..20_000usize {
                        black_box(N2Evaluator.evaluate(black_box(&all[index % all.len()])));
                    }
                }),
            ),
        ],
    );
    for entry in &timings {
        json("stateless_evaluation_same_run", &entry.fields());
    }
    json(
        "same_run_ratios",
        &[
            (
                "scalar_over_optimized_head_median",
                format!("{:.4}", head_scalar_median / head_optimized_median),
            ),
            (
                "scalar_over_optimized_eval_median",
                format!("{:.4}", timings[0].median() / timings[1].median()),
            ),
        ],
    );
}

/// Stateless scalar evaluator, for the same-run comparison only.
struct ScalarEvaluator;

impl Evaluator for ScalarEvaluator {
    fn evaluate(&self, position: &Position) -> i32 {
        let accumulator = Accumulator::from_position(position);
        let white_relative =
            n2::div_ties_even(forward_trace(&accumulator).output_numerator * 400, 32_512);
        let side_to_move = if position.side_to_move().is_white() {
            white_relative
        } else {
            -white_relative
        };
        clamp_evaluation(side_to_move)
    }
}

/// Incremental session whose `evaluate` uses the scalar head.
///
/// Make/unmake/reset delegate to the real [`Session`], so both arms share the
/// same accumulator maintenance and differ only in the forward pass.
struct ScalarSession {
    inner: Session,
}

impl ScalarSession {
    fn new(root: &Position) -> ScalarSession {
        ScalarSession {
            inner: Session::new(root),
        }
    }
}

impl EvaluationSession for ScalarSession {
    fn evaluate(&self, position: &Position) -> i32 {
        let white_relative = n2::div_ties_even(
            forward_trace(self.inner.accumulator()).output_numerator * 400,
            32_512,
        );
        let side_to_move = if position.side_to_move().is_white() {
            white_relative
        } else {
            -white_relative
        };
        clamp_evaluation(side_to_move)
    }

    fn on_make(&mut self, changes: &[PieceChange], before: &Position, after: &Position) {
        self.inner.on_make(changes, before, after);
    }

    fn on_unmake(&mut self) {
        self.inner.on_unmake();
    }

    fn reset(&mut self, root: &Position) {
        self.inner.reset(root);
    }
}

fn config(depth: u32) -> SearchConfig {
    SearchConfig {
        max_depth: depth,
        node_limit: 1_000_000_000,
        ordering: true,
        quiescence: true,
        q_depth_limit: 4,
    }
}

#[test]
#[ignore = "performance measurement, run with --ignored --nocapture"]
fn fixed_depth_search_same_run() {
    let depth = 5;
    for (label, fen) in FENS.iter().take(4).enumerate() {
        let game = Game::from_fen(fen).expect("search fen");
        let settings = config(depth);
        let mut scalar_samples = Vec::new();
        let mut optimized_samples = Vec::new();
        let mut nodes = 0u64;
        // First trial warms both arms; later trials alternate which arm runs
        // first so neither is systematically favoured by load drift.
        for trial in 0..4 {
            let scalar_first = trial % 2 == 0;
            let run_scalar = |samples: &mut Vec<f64>| {
                let mut session = ScalarSession::new(game.position());
                let started = Instant::now();
                let result = search_with_session(&game, &N2Evaluator, &mut session, &settings)
                    .expect("scalar search");
                let elapsed = started.elapsed().as_secs_f64();
                if trial > 0 {
                    samples.push(elapsed);
                }
                result
            };
            let run_optimized = |samples: &mut Vec<f64>| {
                let mut session = Session::new(game.position());
                let started = Instant::now();
                let result = search_with_session(&game, &N2Evaluator, &mut session, &settings)
                    .expect("optimized search");
                let elapsed = started.elapsed().as_secs_f64();
                if trial > 0 {
                    samples.push(elapsed);
                }
                result
            };
            let (scalar, optimized) = if scalar_first {
                let scalar = run_scalar(&mut scalar_samples);
                let optimized = run_optimized(&mut optimized_samples);
                (scalar, optimized)
            } else {
                let optimized = run_optimized(&mut optimized_samples);
                let scalar = run_scalar(&mut scalar_samples);
                (scalar, optimized)
            };
            assert_eq!(
                scalar.nodes, optimized.nodes,
                "node count must be identical"
            );
            assert_eq!(scalar.score, optimized.score, "score must be identical");
            assert_eq!(
                scalar.action, optimized.action,
                "best move must be identical"
            );
            assert_eq!(scalar.pv, optimized.pv, "PV must be identical");
            nodes = scalar.nodes;
        }
        let scalar = Timings {
            label: "scalar".to_string(),
            units_per_run: nodes.max(1) as usize,
            samples: scalar_samples,
        };
        let optimized = Timings {
            label: "optimized".to_string(),
            units_per_run: nodes.max(1) as usize,
            samples: optimized_samples,
        };
        for entry in [&scalar, &optimized] {
            let mut fields = vec![
                ("fen_index", label.to_string()),
                ("fen", format!("{fen:?}")),
                ("depth", depth.to_string()),
                ("nodes", entry.units_per_run.to_string()),
                ("median_ms", format!("{:.3}", entry.median() * 1e3)),
                (
                    "median_nps",
                    format!("{:.1}", entry.units_per_run as f64 / entry.median()),
                ),
            ];
            fields.extend(entry.fields());
            json("fixed_depth_search_same_run", &fields);
        }
        json(
            "fixed_depth_search_speedup",
            &[
                ("fen_index", label.to_string()),
                ("depth", depth.to_string()),
                (
                    "scalar_over_optimized_median",
                    format!("{:.4}", scalar.median() / optimized.median()),
                ),
            ],
        );
    }
}
