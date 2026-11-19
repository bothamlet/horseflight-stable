//! Phase A1 search call distribution and activation sparsity.
//!
//! Builds only with the profile-only feature and is `#[ignore]`d as a
//! measurement:
//!
//! ```text
//! taskset -c 6 cargo test --release --features profiling \
//!     --test n2_opt_a_profiling -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Each line is one JSON object. The counters and their timers are the
//! instrumented numbers: they include atomic-counter, `Instant::now()` and
//! thread-local `HashSet` overhead that a normal build does not pay, so these
//! timings and rates are only for comparing *within* the instrumented run. The
//! reported score, move, PV and node count are deterministic and are compared
//! against an uninstrumented build separately.

#![cfg(feature = "profiling")]

use std::hint::black_box;
use std::time::Instant;

use horseflight::eval::n2::session::Session;
use horseflight::eval::n2::stages::{self, BLOCK_COUNT, BLOCK_LANES};
use horseflight::eval::n2::{self, Accumulator, N2Evaluator};
use horseflight::profiling;
use horseflight::search::{search_with_session, SearchAction, SearchConfig, SearchResult};
use horseflight::{Game, Move, Position};

/// The historical four-position suite plus deterministic extra coverage.
const SUITE: &[(&str, &str)] = &[
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ),
    (
        "middlegame",
        "2kr3r/pp1q1ppp/2n1pn2/2bp4/8/1PN1PN2/PBPPQPPP/2KR3R w - - 0 10",
    ),
    ("endgame_kp", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    (
        "promotion_tactic",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    ),
    (
        "mate_net",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    ),
    ("en_passant", "8/8/8/2k5/2pP4/8/B7/4K3 b - d3 0 3"),
    ("bare_kp", "8/8/8/8/5k2/8/4P3/4K3 w - - 0 1"),
];

fn config(depth: u32) -> SearchConfig {
    SearchConfig {
        max_depth: depth,
        node_limit: 1_000_000_000,
        ordering: true,
        quiescence: true,
        q_depth_limit: 4,
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

fn action_text(action: SearchAction) -> String {
    match action {
        SearchAction::Move(movement) | SearchAction::Fallback(movement) => movement.to_uci(),
        SearchAction::Claim(claim) => format!("claim:{claim:?}"),
        SearchAction::Terminal(outcome) => format!("terminal:{outcome:?}"),
    }
}

fn pv_text(pv: &[Move]) -> String {
    pv.iter()
        .map(|movement| movement.to_uci())
        .collect::<Vec<_>>()
        .join(" ")
}

fn result_fields(result: &SearchResult) -> Vec<(&'static str, String)> {
    vec![
        ("bestmove", format!("{:?}", action_text(result.action))),
        (
            "search_elapsed_seconds",
            format!("{:.9e}", result.elapsed.as_secs_f64()),
        ),
        (
            "score",
            result
                .score
                .map_or_else(|| "null".to_string(), |score| score.to_string()),
        ),
        ("completed_depth", result.completed_depth.to_string()),
        ("nodes", result.nodes.to_string()),
        ("q_nodes", result.q_nodes.to_string()),
        ("pv", format!("{:?}", pv_text(&result.pv))),
        ("pv_len", result.pv.len().to_string()),
    ]
}

/// Search one position at a fixed depth with counting enabled.
fn instrumented_search(game: &Game, settings: SearchConfig) -> (SearchResult, profiling::Snapshot) {
    let mut session = Session::new(game.position());
    profiling::reset();
    let result = search_with_session(game, &N2Evaluator, &mut session, &settings)
        .expect("instrumented search");
    (result, profiling::snapshot())
}

#[test]
#[ignore = "measurement, run in release with --features profiling --ignored"]
fn search_call_distribution() {
    for (label, fen) in SUITE {
        let game = Game::from_fen(fen).unwrap();
        let (result, counts) = instrumented_search(&game, config(6));
        assert_eq!(
            counts.nodes_main + counts.nodes_quiescence,
            result.nodes,
            "main + quiescence nodes must equal SearchResult::nodes"
        );
        let mut fields = vec![
            ("fen_label", format!("{label:?}")),
            ("fen", format!("{fen:?}")),
            ("depth", "6".to_string()),
            ("quiescence", "true".to_string()),
            (
                "nodes_split_sum",
                (counts.nodes_main + counts.nodes_quiescence).to_string(),
            ),
            ("nodes_main", counts.nodes_main.to_string()),
            ("nodes_quiescence", counts.nodes_quiescence.to_string()),
            ("nodes_frontier", counts.nodes_frontier.to_string()),
            ("static_evaluations", counts.static_evaluations.to_string()),
            ("forward_calls", counts.forward_calls.to_string()),
            ("make_calls", counts.make_calls.to_string()),
            ("unmake_calls", counts.unmake_calls.to_string()),
            ("rebuilds", counts.rebuilds.to_string()),
            (
                "prepared_unevaluated_own",
                counts.prepared_unevaluated_own.to_string(),
            ),
            (
                "prepared_unevaluated_own_ratio",
                format!("{:.9e}", counts.prepared_unevaluated_own_ratio()),
            ),
            (
                "prepared_unevaluated_subtree",
                counts.prepared_unevaluated_subtree.to_string(),
            ),
            (
                "prepared_unevaluated_subtree_ratio",
                format!("{:.9e}", counts.prepared_unevaluated_subtree_ratio()),
            ),
            (
                "static_evaluations_quiescence",
                counts.static_evaluations_quiescence.to_string(),
            ),
            (
                "static_evaluations_frontier",
                counts.static_evaluations_frontier.to_string(),
            ),
            (
                "repeated_evaluations",
                counts.repeated_evaluations.to_string(),
            ),
            ("unique_evaluations", counts.unique_evaluations.to_string()),
            (
                "repeated_evaluation_ratio",
                format!("{:.9e}", counts.repeated_evaluation_ratio()),
            ),
            (
                "evaluations_per_1000_nodes",
                format!("{:.9e}", counts.evaluations_per_1000_nodes()),
            ),
            ("evaluation_time_ns", counts.evaluation_time_ns.to_string()),
            ("make_time_ns", counts.make_time_ns.to_string()),
            ("unmake_time_ns", counts.unmake_time_ns.to_string()),
            (
                "evaluation_time_share_of_search",
                format!(
                    "{:.9e}",
                    counts.evaluation_time_ns as f64 / (result.elapsed.as_secs_f64() * 1e9)
                ),
            ),
            (
                "maintenance_time_share_of_search",
                format!(
                    "{:.9e}",
                    (counts.make_time_ns + counts.unmake_time_ns) as f64
                        / (result.elapsed.as_secs_f64() * 1e9)
                ),
            ),
        ];
        fields.extend(result_fields(&result));
        json("search_call_distribution", &fields);
    }

    // Quiescence disabled: the frontier path evaluates one leaf per node.
    let mut frontier_config = config(4);
    frontier_config.quiescence = false;
    for (label, fen) in SUITE {
        let game = Game::from_fen(fen).unwrap();
        let (result, counts) = instrumented_search(&game, frontier_config);
        assert_eq!(
            counts.nodes_main + counts.nodes_quiescence,
            result.nodes,
            "main + quiescence nodes must equal SearchResult::nodes"
        );
        assert_eq!(
            counts.nodes_frontier, counts.nodes_quiescence,
            "frontier leaves are a subset of quiescence nodes in the disabled path"
        );
        let mut fields = vec![
            ("fen_label", format!("{label:?}")),
            ("fen", format!("{fen:?}")),
            ("depth", "4".to_string()),
            ("quiescence", "false".to_string()),
            ("nodes_main", counts.nodes_main.to_string()),
            ("nodes_quiescence", counts.nodes_quiescence.to_string()),
            ("nodes_frontier", counts.nodes_frontier.to_string()),
            ("static_evaluations", counts.static_evaluations.to_string()),
            ("make_calls", counts.make_calls.to_string()),
            (
                "prepared_unevaluated_own",
                counts.prepared_unevaluated_own.to_string(),
            ),
            (
                "prepared_unevaluated_subtree",
                counts.prepared_unevaluated_subtree.to_string(),
            ),
            (
                "static_evaluations_quiescence",
                counts.static_evaluations_quiescence.to_string(),
            ),
            (
                "static_evaluations_frontier",
                counts.static_evaluations_frontier.to_string(),
            ),
            (
                "evaluations_per_1000_nodes",
                format!("{:.9e}", counts.evaluations_per_1000_nodes()),
            ),
        ];
        fields.extend(result_fields(&result));
        json("search_call_distribution_frontier", &fields);
    }
}

/// Deterministic legal-move walk, so sampled positions are reproducible.
struct Walk {
    game: Game,
    state: u64,
}

impl Walk {
    fn new(fen: &str, seed: u64) -> Walk {
        Walk {
            game: Game::from_fen(fen).unwrap(),
            state: seed,
        }
    }

    fn next_state(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Play one deterministic legal move; false when the game ended.
    fn step(&mut self) -> bool {
        let legal = self.game.position().legal_moves();
        if legal.is_empty() {
            return false;
        }
        let pick = (self.next_state() >> 33) as usize % legal.len();
        let movement = legal[pick];
        self.game.play(movement).is_ok()
    }
}

/// Activation sparsity samples: 8 suites plus 240 deterministic walk positions.
#[test]
#[ignore = "measurement, run in release with --features profiling --ignored"]
fn activation_sparsity() {
    let mut sampled: Vec<(String, Position)> = SUITE
        .iter()
        .map(|(label, fen)| (format!("suite:{label}"), Position::from_fen(fen).unwrap()))
        .collect();
    for seed in [1u64, 2, 3] {
        let mut walk = Walk::new(SUITE[0].1, seed);
        for ply in 0..80usize {
            if !walk.step() {
                break;
            }
            if ply % 3 == 0 {
                sampled.push((format!("walk{seed}:ply{ply}"), *walk.game.position()));
            }
        }
    }

    let mut zero_ratios = Vec::new();
    for (label, position) in &sampled {
        let accumulator = Accumulator::from_position(position);
        let a1 = stages::stage_first_activation(accumulator.values());
        let stats = stages::activation_stats(&a1);
        zero_ratios.push(stats.zero_ratio());
        json(
            "activation_sparsity_sample",
            &[
                ("label", format!("{label:?}")),
                ("fen", format!("{:?}", position.to_fen())),
                (
                    "active_features",
                    n2::active_features(position).len().to_string(),
                ),
                ("stats", stats.json()),
                ("zero_ratio", format!("{:.9e}", stats.zero_ratio())),
            ],
        );
    }

    zero_ratios.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let mean = zero_ratios.iter().sum::<f64>() / zero_ratios.len() as f64;
    json(
        "activation_sparsity_aggregate",
        &[
            ("samples", sampled.len().to_string()),
            ("block_lanes", BLOCK_LANES.to_string()),
            ("block_count", BLOCK_COUNT.to_string()),
            ("min_zero_ratio", format!("{:.9e}", zero_ratios[0])),
            (
                "median_zero_ratio",
                format!("{:.9e}", zero_ratios[zero_ratios.len() / 2]),
            ),
            (
                "max_zero_ratio",
                format!("{:.9e}", zero_ratios[zero_ratios.len() - 1]),
            ),
            ("mean_zero_ratio", format!("{:.9e}", mean)),
        ],
    );
}

/// Cost of finding the zero lanes, against the multiply it would have to save.
#[test]
#[ignore = "measurement, run in release with --features profiling --ignored"]
fn sparsity_detection_overhead() {
    let position = Position::from_fen(SUITE[1].1).unwrap();
    let accumulator = Accumulator::from_position(&position);
    let a1 = stages::stage_first_activation(accumulator.values());
    let stats = stages::activation_stats(&a1);

    let repeats = 9;
    let units = 200_000;
    let mut samples = Vec::new();
    for _ in 0..=repeats {
        let started = Instant::now();
        let mut zeros = 0usize;
        for _ in 0..units {
            zeros += black_box(&a1)
                .iter()
                .filter(|activation| **activation == 0)
                .count();
        }
        let elapsed = started.elapsed().as_secs_f64();
        black_box(zeros);
        samples.push(elapsed);
    }
    let scan = samples.split_off(1);
    let scan_median = median(&scan);
    json(
        "sparsity_detection_scan",
        &[
            ("zero_ratio", format!("{:.9e}", stats.zero_ratio())),
            ("units", units.to_string()),
            (
                "median_ns_per_scan",
                format!("{:.9e}", 1e9 * scan_median / units as f64),
            ),
            ("raw_samples_seconds", format!("{scan:?}")),
        ],
    );

    let mut samples = Vec::new();
    for _ in 0..=repeats {
        let started = Instant::now();
        for _ in 0..units {
            black_box(stages::stage_hidden_matmul(black_box(&a1)));
        }
        samples.push(started.elapsed().as_secs_f64());
    }
    black_box(&samples);
    let dense = samples.split_off(1);
    let dense_median = median(&dense);
    json(
        "sparsity_detection_reference",
        &[
            ("units", units.to_string()),
            (
                "median_ns_per_dense_matmul",
                format!("{:.9e}", 1e9 * dense_median / units as f64),
            ),
            ("raw_samples_seconds", format!("{dense:?}")),
        ],
    );
}

/// Per-event cost of the instrumentation itself, so the report can disclose it
/// instead of subtracting it silently from a measured search.
#[test]
#[ignore = "measurement, run in release with --features profiling --ignored"]
fn instrumentation_overhead() {
    let repeats = 9;
    let units = 1_000_000;

    let mut samples = Vec::new();
    for _ in 0..=repeats {
        let started = Instant::now();
        for _ in 0..units {
            profiling::record_forward();
        }
        samples.push(started.elapsed().as_secs_f64());
    }
    let atomic = samples.split_off(1);
    json(
        "instrumentation_atomic_add",
        &[
            ("units", units.to_string()),
            (
                "median_ns_per_event",
                format!("{:.9e}", 1e9 * median(&atomic) / units as f64),
            ),
            ("raw_samples_seconds", format!("{atomic:?}")),
        ],
    );

    profiling::reset();
    let mut samples = Vec::new();
    for _ in 0..=repeats {
        let started = Instant::now();
        for index in 0..units {
            profiling::record_static_evaluation(index as u64, false);
        }
        samples.push(started.elapsed().as_secs_f64());
    }
    let hashed = samples.split_off(1);
    json(
        "instrumentation_static_evaluation_count",
        &[
            ("units", units.to_string()),
            (
                "median_ns_per_event",
                format!("{:.9e}", 1e9 * median(&hashed) / units as f64),
            ),
            ("raw_samples_seconds", format!("{hashed:?}")),
        ],
    );
    profiling::reset();

    let mut samples = Vec::new();
    for _ in 0..=repeats {
        let started = Instant::now();
        let mut accumulated = 0u128;
        for _ in 0..units {
            let clock = Instant::now();
            accumulated += clock.elapsed().as_nanos();
        }
        black_box(accumulated);
        samples.push(started.elapsed().as_secs_f64());
    }
    let timers = samples.split_off(1);
    json(
        "instrumentation_timer_pair",
        &[
            ("units", units.to_string()),
            (
                "median_ns_per_pair",
                format!("{:.9e}", 1e9 * median(&timers) / units as f64),
            ),
            ("raw_samples_seconds", format!("{timers:?}")),
        ],
    );
}

fn median(samples: &[f64]) -> f64 {
    let mut ordered = samples.to_vec();
    ordered.sort_by(|left, right| left.partial_cmp(right).unwrap());
    ordered[ordered.len() / 2]
}
