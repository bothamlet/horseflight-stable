//! Compare NNUE inference and accumulator variants during search.
//!
//! All variants run in one process with the same instrumentation. The timings
//! are for comparison within that run, not for measuring release throughput.
//!
//! ```text
//! taskset -c 6 cargo test --release --features profiling \
//!     --test n2_opt_a_a2_attribution -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "profiling")]

use std::hint::black_box;
use std::time::Instant;

use horseflight::eval::n2::session::Session;
use horseflight::eval::n2::stages;
use horseflight::eval::n2::{self, N2Evaluator};
use horseflight::eval::{clamp_evaluation, EvaluationSession};
use horseflight::search::{search_with_session, SearchConfig};
use horseflight::{Game, PieceChange, Position};

const FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "2kr3r/pp1q1ppp/2n1pn2/2bp4/8/1PN1PN2/PBPPQPPP/2KR3R w - - 0 10",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
];

#[derive(Clone, Copy, Debug)]
enum Variant {
    ScalarFull,
    ScalarI16,
    Avx2Matmul,
    Avx2Head,
    ScalarRows,
    Production,
}

impl Variant {
    fn label(self) -> &'static str {
        match self {
            Variant::ScalarFull => "0_scalar_full",
            Variant::ScalarI16 => "1_scalar_i16_packing",
            Variant::Avx2Matmul => "2_avx2_matmul_only",
            Variant::Avx2Head => "3_avx2_head",
            Variant::ScalarRows => "4_production_head_scalar_rows",
            Variant::Production => "5_production",
        }
    }
}

struct VariantSession {
    inner: Session,
    variant: Variant,
}

fn to_cp(output_numerator: i32) -> i32 {
    n2::div_ties_even(output_numerator * 400, 32_512)
}

impl VariantSession {
    fn new(root: &Position, variant: Variant) -> VariantSession {
        let mut inner = Session::new(root);
        if matches!(variant, Variant::ScalarRows) {
            inner.set_scalar_rows(true);
        }
        VariantSession { inner, variant }
    }

    fn white_relative(&self) -> i32 {
        let accumulator = self.inner.accumulator();
        match self.variant {
            Variant::ScalarFull => to_cp(n2::forward_trace(accumulator).output_numerator),
            Variant::ScalarI16 => {
                let a1 = stages::stage_first_activation_scalar_i16(accumulator.values());
                let h = stages::stage_hidden_matmul_scalar_i16(&a1);
                let a2 = stages::stage_hidden_activation_scalar_i16(&h);
                to_cp(stages::stage_output_dot_scalar_i16(&a2))
            }
            Variant::Avx2Matmul => {
                let a1 = stages::stage_first_activation_scalar_i16(accumulator.values());
                let h = stages::stage_hidden_matmul_dispatched(&a1);
                let a2 = stages::stage_hidden_activation_scalar_i16(&h);
                to_cp(stages::stage_output_dot_scalar_i16(&a2))
            }
            Variant::Avx2Head => {
                let a1 = stages::stage_first_activation_dispatched(accumulator.values());
                let h = stages::stage_hidden_matmul_dispatched(&a1);
                let a2 = stages::stage_hidden_activation_dispatched(&h);
                to_cp(stages::stage_output_dot_dispatched(&a2))
            }
            Variant::ScalarRows | Variant::Production => n2::forward(accumulator),
        }
    }
}

impl EvaluationSession for VariantSession {
    fn evaluate(&self, position: &Position) -> i32 {
        let white_relative = self.white_relative();
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

fn median(values: &[f64]) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(|left, right| left.partial_cmp(right).unwrap());
    ordered[ordered.len() / 2]
}

#[test]
#[ignore = "performance measurement, run with --ignored --nocapture"]
fn cumulative_head_and_row_attribution() {
    let depth = 5;
    let variants = [
        Variant::ScalarFull,
        Variant::ScalarI16,
        Variant::Avx2Matmul,
        Variant::Avx2Head,
        Variant::ScalarRows,
        Variant::Production,
    ];
    for (index, fen) in FENS.iter().enumerate() {
        let game = Game::from_fen(fen).expect("fen");
        let settings = config(depth);
        let mut reference = None;
        for variant in variants {
            let mut samples = Vec::new();
            let mut nodes = 0u64;
            for trial in 0..4 {
                let mut session = VariantSession::new(game.position(), variant);
                let started = Instant::now();
                let result = search_with_session(&game, &N2Evaluator, &mut session, &settings)
                    .expect("search");
                let elapsed = started.elapsed().as_secs_f64();
                black_box(&result);
                if trial > 0 {
                    samples.push(elapsed);
                }
                nodes = result.nodes;
                let key = (
                    result.nodes,
                    result.score,
                    format!("{:?}", result.action),
                    result
                        .pv
                        .iter()
                        .map(|movement| movement.to_uci())
                        .collect::<Vec<_>>(),
                );
                match &reference {
                    None => reference = Some(key),
                    Some(expected) => assert_eq!(
                        expected,
                        &key,
                        "variant {} differs deterministically at {fen}",
                        variant.label()
                    ),
                }
            }
            println!(
                "{{\"kind\":\"attribution\",\"fen_index\":{index},\"variant\":\"{}\",\
                 \"nodes\":{nodes},\"trials\":{},\"median_seconds\":{:.9e},\
                 \"median_nps\":{:.3},\"raw_samples_seconds\":{:?}}}",
                variant.label(),
                samples.len(),
                median(&samples),
                nodes as f64 / median(&samples),
                samples
            );
        }
    }
}
