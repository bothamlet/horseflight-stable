//! Stage-level accessors for the Phase A1 profiling harness.
//!
//! Compiled only with `--features profiling`. It exists so the evidence
//! harness can time one stage of the forward pass at a time and inspect tensor
//! layout without changing any production kernel: the functions here are
//! *copies* of the arithmetic in [`super::forward_trace`] and the private
//! activation helpers it calls, split at the stage boundaries a profiler would
//! report.
//!
//! Each `stage_*` function runs exactly one stage, so a stage's measured cost is
//! that stage alone. The copies must stay in step with the production kernels
//! they mirror; [`tensors`] exposes the same private tensors [`super::model`]
//! validated, and nothing here is reachable from a normal build.

use super::{
    activate_accumulator, activate_hidden, div_ties_even_wide, model, ACCUMULATOR_SIZE, ACC_SCALE,
    ACT_MAX, CP_DENOMINATOR, CP_MULTIPLIER, HIDDEN_SCALE, HIDDEN_SIZE,
};

/// Immutable view of the six validated tensors in container order.
pub struct Tensors {
    /// Input weights, neuron-major: `w1[feature * ACCUMULATOR_SIZE + lane]`.
    pub w1: &'static [i16],
    /// First-layer biases, one per accumulator lane.
    pub b1: &'static [i32],
    /// Hidden weights, output-neuron-major: `w2[hidden * ACCUMULATOR_SIZE + lane]`.
    pub w2: &'static [i8],
    /// Hidden biases, one per hidden neuron.
    pub b2: &'static [i32],
    /// Output weights, one per hidden activation.
    pub w3: &'static [i16],
    /// Output bias.
    pub b3: &'static [i32],
}

/// The validated model's tensors, borrowed for the lifetime of the process.
pub fn tensors() -> Tensors {
    let model = model();
    Tensors {
        w1: &model.w1,
        b1: &model.b1,
        w2: &model.w2,
        b2: &model.b2,
        w3: &model.w3,
        b3: &model.b3,
    }
}

/// First activation only: clamp and ties-to-even rounding of every lane.
pub fn stage_first_activation(acc: &[i32; ACCUMULATOR_SIZE]) -> [i32; ACCUMULATOR_SIZE] {
    activate_accumulator(acc)
}

/// Hidden matrix multiply only, without the hidden activation.
pub fn stage_hidden_matmul(a1: &[i32; ACCUMULATOR_SIZE]) -> [i32; HIDDEN_SIZE] {
    let tensors = tensors();
    let mut h = [0i32; HIDDEN_SIZE];
    for (hidden, value) in h.iter_mut().enumerate() {
        let row = &tensors.w2[hidden * ACCUMULATOR_SIZE..][..ACCUMULATOR_SIZE];
        let mut sum = tensors.b2[hidden];
        for (weight, activation) in row.iter().zip(a1) {
            sum += *weight as i32 * activation;
        }
        *value = sum;
    }
    h
}

/// Hidden activation only: clamp and ties-to-even rounding of every neuron.
pub fn stage_hidden_activation(h: &[i32; HIDDEN_SIZE]) -> [i32; HIDDEN_SIZE] {
    activate_hidden(h)
}

/// Output dot product only.
pub fn stage_output_dot(a2: &[i32; HIDDEN_SIZE]) -> i32 {
    let tensors = tensors();
    let mut output_numerator = ACT_MAX * tensors.b3[0];
    for (weight, activation) in tensors.w3.iter().zip(a2) {
        output_numerator += *weight as i32 * activation;
    }
    output_numerator
}

/// White-relative centipawn conversion only (`400 * out / 32512`, ties to even).
pub fn stage_score_conversion(output_numerator: i32) -> i32 {
    div_ties_even_wide(
        output_numerator as i64 * CP_MULTIPLIER as i64,
        CP_DENOMINATOR as i64,
    ) as i32
}

/// Lanes per sparsity block, as used by [`ActivationStats`].
pub const BLOCK_LANES: usize = 16;
/// Number of sparsity blocks in one activation row.
pub const BLOCK_COUNT: usize = ACCUMULATOR_SIZE / BLOCK_LANES;

/// Zero/saturation and block-clustering statistics of one first activation.
///
/// Zeros are the only lanes a sparse kernel could skip: a zero activation
/// contributes nothing to the hidden exchange `w2[k][j] * a1[j]`. `saturated`
/// counts lanes pinned at [`ACT_MAX`], where clamping has discarded input
/// information. Block occupancy measures whether the nonzeros are clustered in
/// a way a block-granular kernel could exploit, or spread evenly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActivationStats {
    /// Activation lanes inspected (`ACCUMULATOR_SIZE`).
    pub lanes: usize,
    /// Lanes equal to zero.
    pub zeros: usize,
    /// Lanes equal to [`ACT_MAX`] (clamped saturation).
    pub saturated: usize,
    /// Nonzero lanes per [`BLOCK_LANES`]-lane block, in lane order.
    pub block_nonzeros: [usize; BLOCK_COUNT],
    /// Longest run of consecutive nonzero lanes.
    pub longest_nonzero_run: usize,
}

impl ActivationStats {
    /// Fraction of lanes that are zero.
    pub fn zero_ratio(&self) -> f64 {
        self.zeros as f64 / self.lanes as f64
    }

    /// Blocks holding at least one nonzero multiply.
    pub fn occupied_blocks(&self) -> usize {
        self.block_nonzeros
            .iter()
            .filter(|count| **count > 0)
            .count()
    }

    /// This row as `{lanes,zeros,saturated,occupied_blocks,longest_nonzero_run,block_nonzeros}`.
    pub fn json(&self) -> String {
        format!(
            "{{\"lanes\":{},\"zeros\":{},\"saturated\":{},\"occupied_blocks\":{},\
             \"longest_nonzero_run\":{},\"block_nonzeros\":{:?}}}",
            self.lanes,
            self.zeros,
            self.saturated,
            self.occupied_blocks(),
            self.longest_nonzero_run,
            &self.block_nonzeros[..],
        )
    }
}

/// Measure the sparsity of one first activation row.
pub fn activation_stats(a1: &[i32; ACCUMULATOR_SIZE]) -> ActivationStats {
    let mut stats = ActivationStats {
        lanes: ACCUMULATOR_SIZE,
        ..ActivationStats::default()
    };
    let mut run = 0usize;
    for (lane, value) in a1.iter().enumerate() {
        if *value == 0 {
            stats.zeros += 1;
            run = 0;
        } else {
            stats.block_nonzeros[lane / BLOCK_LANES] += 1;
            run += 1;
            stats.longest_nonzero_run = stats.longest_nonzero_run.max(run);
        }
        if *value == ACT_MAX {
            stats.saturated += 1;
        }
    }
    stats
}

/// Scales the stage harness relies on, so a report never hardcodes them.
pub fn scales() -> (i32, i32, i32) {
    (ACC_SCALE, ACT_MAX, HIDDEN_SCALE)
}

/// Dispatched first activation: the production kernel behind `forward`.
pub fn stage_first_activation_dispatched(acc: &[i32; ACCUMULATOR_SIZE]) -> [i16; ACCUMULATOR_SIZE] {
    super::simd::first_activation(acc)
}

/// Scalar first activation in the same `i16` form, for same-run comparison.
pub fn stage_first_activation_scalar_i16(acc: &[i32; ACCUMULATOR_SIZE]) -> [i16; ACCUMULATOR_SIZE] {
    super::simd::first_activation_scalar(acc)
}

/// Dispatched hidden matrix multiply over the packed `i16` weights.
pub fn stage_hidden_matmul_dispatched(a1: &[i16; ACCUMULATOR_SIZE]) -> [i32; HIDDEN_SIZE] {
    let model = model();
    super::simd::hidden_matmul(&model.w2_i16, &model.b2, a1)
}

/// Scalar hidden matrix multiply over the same packed `i16` weights.
pub fn stage_hidden_matmul_scalar_i16(a1: &[i16; ACCUMULATOR_SIZE]) -> [i32; HIDDEN_SIZE] {
    let model = model();
    super::simd::hidden_matmul_scalar(&model.w2_i16, &model.b2, a1)
}

/// Scalar accumulator row add, for same-run comparison with the kernel.
pub fn stage_accumulator_row_add_scalar(acc: &mut [i32; ACCUMULATOR_SIZE], row: &[i16]) {
    super::simd::accumulator_row_add_scalar(acc, row);
}

/// Dispatched accumulator row add.
pub fn stage_accumulator_row_add_dispatched(acc: &mut [i32; ACCUMULATOR_SIZE], row: &[i16]) {
    super::simd::accumulator_row_add(acc, row);
}

/// Scalar accumulator row subtract.
pub fn stage_accumulator_row_sub_scalar(acc: &mut [i32; ACCUMULATOR_SIZE], row: &[i16]) {
    super::simd::accumulator_row_sub_scalar(acc, row);
}

/// Dispatched accumulator row subtract.
pub fn stage_accumulator_row_sub_dispatched(acc: &mut [i32; ACCUMULATOR_SIZE], row: &[i16]) {
    super::simd::accumulator_row_sub(acc, row);
}

/// One validated input row, for the row-kernel comparison.
pub fn row_for_feature(feature: usize) -> &'static [i16] {
    let model = model();
    &model.w1[feature * ACCUMULATOR_SIZE..][..ACCUMULATOR_SIZE]
}

/// Dispatched hidden activation.
pub fn stage_hidden_activation_dispatched(h: &[i32; HIDDEN_SIZE]) -> [i16; HIDDEN_SIZE] {
    super::simd::hidden_activation(h)
}

/// Scalar hidden activation in the same `i16` form.
pub fn stage_hidden_activation_scalar_i16(h: &[i32; HIDDEN_SIZE]) -> [i16; HIDDEN_SIZE] {
    super::simd::hidden_activation_scalar(h)
}

/// Dispatched output dot product.
pub fn stage_output_dot_dispatched(a2: &[i16; HIDDEN_SIZE]) -> i32 {
    let model = model();
    super::simd::output_numerator(&model.w3, model.b3[0], a2)
}

/// Scalar output dot product over the same `i16` activations.
pub fn stage_output_dot_scalar_i16(a2: &[i16; HIDDEN_SIZE]) -> i32 {
    let model = model();
    super::simd::output_numerator_scalar(&model.w3, model.b3[0], a2)
}
