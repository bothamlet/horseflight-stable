//! Local NNUE inference and search benchmarks.
//!
//! Run with `cargo test --release --test n2_002_bench -- --ignored --nocapture`.
//! Measurements are emitted as JSON lines.

use std::hint::black_box;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use horseflight::eval::n2::session::Session;
use horseflight::eval::n2::{self, Accumulator, N2Evaluator};
use horseflight::eval::{EvaluationSession, Evaluator, HandcraftedEvaluator};
use horseflight::search::{search, search_controlled_with_session, SearchConfig, SearchControl};
use horseflight::{Game, Position};

const FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "2kr3r/pp1q1ppp/2n1pn2/2bp4/8/1PN1PN2/PBPPQPPP/2KR3R w - - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
];

fn positions() -> Vec<Position> {
    FENS.iter()
        .map(|fen| Position::from_fen(fen).unwrap_or_else(|error| panic!("{fen}: {error}")))
        .collect()
}

/// Repeated timings: worst, median and best wall-clock seconds plus every raw
/// sample, so the report never has to invent precision.
struct Timings {
    units_per_run: usize,
    repeats: usize,
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

    fn rate(&self, seconds: f64) -> f64 {
        self.units_per_run as f64 / seconds
    }

    /// Timing fields for one JSON record.
    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("units_per_run", self.units_per_run.to_string()),
            ("repeats", self.repeats.to_string()),
            ("min_seconds", f(self.min())),
            ("median_seconds", f(self.median())),
            ("max_seconds", f(self.max())),
            ("min_seconds_per_second", f(self.rate(self.min()))),
            ("median_per_second", f(self.rate(self.median()))),
            ("max_seconds_per_second", f(self.rate(self.max()))),
            ("raw_samples_seconds", format!("{:?}", self.samples)),
        ]
    }
}

fn measure(repeats: usize, units_per_run: usize, mut run: impl FnMut()) -> Timings {
    // Warm-up: one full pass, so the first measurement is not dominated by cold
    // caches or lazy initialisation of the embedded model.
    run();
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        run();
        samples.push(started.elapsed().as_secs_f64());
    }
    Timings {
        units_per_run,
        repeats,
        samples,
    }
}

fn json(kind: &str, fields: &[(&str, String)]) {
    let body = fields
        .iter()
        .map(|(key, value)| format!("\"{key}\":{value}"))
        .collect::<Vec<_>>()
        .join(",");
    println!("{{\"kind\":\"{kind}\",{body}}}");
}

fn f(value: f64) -> String {
    format!("{value:.1}")
}

#[test]
#[ignore = "performance measurement, run in release with --ignored"]
fn eval_throughput() {
    let positions = positions();
    let accumulator = Accumulator::from_position(&positions[1]);
    let game = Game::from_fen(FENS[1]).unwrap();

    // Classical evaluator, for scale.
    let timings = measure(5, 200_000, || {
        for index in 0..200_000usize {
            let position = &positions[index % positions.len()];
            black_box(HandcraftedEvaluator.evaluate(black_box(position)));
        }
    });
    let mut fields = vec![("units", format!("{:?}", "200000 evaluations"))];
    fields.extend(timings.fields());
    json("classical_evaluate", &fields);

    // Full accumulator rebuild from a position (the reference cost).
    let timings = measure(5, 50_000, || {
        for index in 0..50_000usize {
            black_box(Accumulator::from_position(black_box(
                &positions[index % positions.len()],
            )));
        }
    });
    let mut fields = vec![("units", format!("{:?}", "50000 rebuilds"))];
    fields.extend(timings.fields());
    json("n2_accumulator_rebuild", &fields);

    // Neural head only: activation + hidden layer + output on a ready accumulator.
    let timings = measure(5, 100_000, || {
        for _ in 0..100_000usize {
            black_box(n2::forward(black_box(&accumulator)));
        }
    });
    let mut fields = vec![
        ("units", format!("{:?}", "100000 forward passes")),
        (
            "median_nanoseconds_per_forward",
            f(1e9 * timings.median() / 100_000.0),
        ),
        (
            "min_nanoseconds_per_forward",
            f(1e9 * timings.min() / 100_000.0),
        ),
    ];
    fields.extend(timings.fields());
    json("n2_head_only", &fields);

    // Stateless evaluator: rebuild + head, the compatible `Evaluator` path.
    let timings = measure(5, 20_000, || {
        for index in 0..20_000usize {
            let position = &positions[index % positions.len()];
            black_box(N2Evaluator.evaluate(black_box(position)));
        }
    });
    let mut fields = vec![("units", format!("{:?}", "20000 evaluations"))];
    fields.extend(timings.fields());
    json("n2_stateless_evaluate", &fields);

    // Incremental maintenance cost only: the move, its undo record and both
    // positions are prepared outside the timed loop, so this measures
    // `on_make`/`on_unmake` (accumulator + delta bookkeeping) and nothing else.
    let mut prepared = *game.position();
    let movement = prepared.legal_moves()[0];
    let before = prepared;
    let undo = prepared.play(movement).expect("legal move");
    let after = prepared;
    let timings = measure(5, 200_000, || {
        let mut session = Session::new(&before);
        for _ in 0..200_000usize {
            session.on_make(undo.piece_changes(), &before, &after);
            session.on_unmake();
        }
        black_box(session.depth());
    });
    let mut fields = vec![
        (
            "units",
            format!("{:?}", "200000 prepared make+unmake delta pairs"),
        ),
        (
            "median_nanoseconds_per_pair",
            f(1e9 * timings.median() / 200_000.0),
        ),
    ];
    fields.extend(timings.fields());
    json("n2_session_delta_pair", &fields);

    // Full make/unmake cycle including move generation and board updates, for
    // comparison with the delta-only figure above.
    let timings = measure(5, 20_000, || {
        let mut position = *game.position();
        let mut session = Session::new(&position);
        for _ in 0..20_000usize {
            let movement = position.legal_moves()[0];
            let before = position;
            let undo = position.play(movement).expect("legal move");
            session.on_make(undo.piece_changes(), &before, &position);
            black_box(&session);
            position.unmake(&undo).expect("unmake");
            session.on_unmake();
        }
        black_box(session.depth());
    });
    let mut fields = vec![("units", format!("{:?}", "20000 movegen+make+unmake pairs"))];
    fields.extend(timings.fields());
    json("n2_session_make_unmake_pair", &fields);
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
#[ignore = "performance measurement, run in release with --ignored"]
fn fixed_depth_search_throughput() {
    for (label, game) in FENS
        .iter()
        .map(|fen| (fen, Game::from_fen(fen).unwrap()))
        .collect::<Vec<_>>()
    {
        let classical = search(&game, &HandcraftedEvaluator, &config(5)).expect("classical search");
        let mut session = Session::new(game.position());
        let neural =
            horseflight::search::search_with_session(&game, &N2Evaluator, &mut session, &config(5))
                .expect("nnue search");
        json(
            "fixed_depth_search",
            &[
                ("fen", format!("{label:?}")),
                ("depth", format!("{:?}", "5")),
                ("classical_nodes", classical.nodes.to_string()),
                ("classical_ms", f(classical.elapsed.as_secs_f64() * 1e3)),
                ("classical_nps", f(classical.nps())),
                ("nnue_nodes", neural.nodes.to_string()),
                ("nnue_ms", f(neural.elapsed.as_secs_f64() * 1e3)),
                ("nnue_nps", f(neural.nps())),
                (
                    "node_ratio_nnue_over_classical",
                    f(neural.nodes as f64 / classical.nodes as f64),
                ),
            ],
        );
    }
}

/// Nodes reached inside a fixed wall-clock budget, mirroring the UCI adapter's
/// deadline thread. This is a throughput measurement, not a strength result.
#[test]
#[ignore = "performance measurement, run in release with --ignored"]
fn fixed_time_search_nodes() {
    let budget = Duration::from_secs(2);
    for (label, game) in FENS
        .iter()
        .map(|fen| (fen, Game::from_fen(fen).unwrap()))
        .collect::<Vec<_>>()
    {
        let mut rows = Vec::new();
        for evaluator in ["classical", "nnue"] {
            let control = SearchControl::new();
            let canceller = control.clone();
            let (finished, wait) = mpsc::channel::<()>();
            let waiter = std::thread::spawn(move || {
                if matches!(
                    wait.recv_timeout(budget),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ) {
                    canceller.cancel();
                }
            });
            let started = Instant::now();
            let result = match evaluator {
                "classical" => horseflight::search::search_controlled(
                    &game,
                    &HandcraftedEvaluator,
                    &config(96),
                    &control,
                ),
                _ => {
                    let mut session = Session::new(game.position());
                    search_controlled_with_session(
                        &game,
                        &N2Evaluator,
                        &mut session,
                        &config(96),
                        &control,
                    )
                }
            }
            .expect("timed search");
            let elapsed = started.elapsed();
            let _ = finished.send(());
            let _ = waiter.join();
            rows.push((evaluator, result.nodes, result.completed_depth, elapsed));
        }
        for (evaluator, nodes, depth, elapsed) in rows {
            json(
                "fixed_time_search",
                &[
                    ("fen", format!("{label:?}")),
                    ("budget_ms", f(budget.as_secs_f64() * 1e3)),
                    ("evaluator", format!("{evaluator:?}")),
                    ("nodes", nodes.to_string()),
                    ("depth", depth.to_string()),
                    ("elapsed_ms", f(elapsed.as_secs_f64() * 1e3)),
                    ("nps", f(nodes as f64 / elapsed.as_secs_f64())),
                ],
            );
        }
    }
}

#[test]
#[ignore = "performance measurement, run in release with --ignored"]
fn model_identity() {
    let _ = Arc::new(());
    json(
        "model_identity",
        &[
            ("model_id", format!("{:?}", n2::MODEL_ID)),
            ("asset_sha256", format!("{:?}", n2::asset_sha256())),
            ("weights_sha256", format!("{:?}", n2::weights_sha256())),
            ("provenance_flags", n2::provenance_flags().to_string()),
            (
                "score_perspective_metadata",
                format!("{:?}", n2::score_perspective()),
            ),
            (
                "score_perspective_derived",
                format!("{:?}", n2::DERIVED_SCORE_PERSPECTIVE),
            ),
        ],
    );
    let _ = black_box(Position::startpos());
}
