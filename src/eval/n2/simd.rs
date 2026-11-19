//! Dispatched SIMD kernels for the N2-002 integer head.
//!
//! The crate keeps `#![deny(unsafe_code)]`; this module is the only place that
//! opts out, because Rust's `std::arch` intrinsics are `unsafe` by
//! construction. Every `unsafe` block here is a fixed-width kernel over slices
//! whose lengths are compile-time constants of the N2-002 contract
//! ([`ACCUMULATOR_SIZE`] = 256, [`HIDDEN_SIZE`] = 32), so no kernel depends on
//! caller-provided lengths.
//!
//! # Exactness of the 16-bit hidden matmul
//!
//! The scalar reference computes, for each hidden neuron `k`,
//!
//! ```text
//! h[k] = b2[k] + sum_j w2[k][j] * a1[j]
//! ```
//!
//! with `w2` decoded as `i8` and `a1` the first activation in `0..=127`.
//! `vpmaddwd` replaces each adjacent pair of products with one
//! `_mm256_madd_epi16`, which multiplies two `i16` lanes in 32-bit and adds
//! adjacent products in 32-bit (no saturation anywhere). The substitutions are
//! exact because:
//!
//! - every `w2` element is an `i8`, so its `i16` widening is lossless;
//! - every `a1` lane is in `0..=127` (proof in [`super::activate_accumulator`])
//!   and fits `i16`;
//! - a product therefore lies in `[-128 * 127, 127 * 127]`, and a pair sum in
//!   `[-32512, 32258]`, far inside `i32`; the 256-term sum is at most
//!   `256 * 16256 = 4_161_536` in magnitude before the `i32` bias, so no
//!   intermediate wraps where the scalar code does not;
//! - the two 8-lane accumulators are added with wrapping `i32` addition, then
//!   summed with `i32` addition, matching the scalar order-insensitive sum for
//!   every value that does not overflow. For the frozen asset
//!   `max |w2| = 124` and `max |b2| = 857`, so the real bound is tighter still.
//!
//! # Exactness of the activation kernels
//!
//! Both activation helpers use the same round-to-nearest-ties-to-even identity
//! as [`super::div_ties_even`] for a power-of-two divisor:
//!
//! ```text
//! q = n >> shift; r = n & (2^shift - 1)
//! round = q + 1 when r > half, or r == half and q is odd
//! ```
//!
//! over non-negative numerators, so the vector form is the scalar form. The
//! first activation additionally clamps `acc` to `0..=256` and multiplies by
//! `127`; the product is at most `32512`, which still fits `i16` after the
//! rounding, which is how the matmul input is produced directly.
//!
//! # Dispatch and fallback
//!
//! [`first_activation`], [`hidden_matmul`], [`hidden_activation`] and
//! [`output_numerator`] check `is_x86_feature_detected!("avx2")` (cached by the
//! standard library) and otherwise run the scalar reference kernel. Both the
//! AVX2 and scalar kernels are exercised directly by the unit tests in this
//! module, so a SIMD-capable host still tests the fallback.

#![allow(unsafe_code)]

use super::{ACCUMULATOR_SIZE, ACC_SCALE, ACT_MAX, HIDDEN_SCALE, HIDDEN_SIZE};

/// First activation: clamp `acc` to `0..=ACC_SCALE`, scale by `ACT_MAX` and
/// round ties-to-even by `ACC_SCALE`, in `i16`.
pub(crate) fn first_activation(acc: &[i32; ACCUMULATOR_SIZE]) -> [i16; ACCUMULATOR_SIZE] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: `acc` is a 256-lane array and the kernel reads exactly
            // 16 lanes per iteration for 16 iterations. The output array is
            // likewise 256 lanes. AVX2 availability was checked at runtime.
            return unsafe { avx2::first_activation(acc) };
        }
    }
    first_activation_scalar(acc)
}

/// Hidden activation: clamp `h` to `0..=ACT_MAX*HIDDEN_SCALE`, divide by
/// `HIDDEN_SCALE` ties-to-even, in `i16`.
pub(crate) fn hidden_activation(h: &[i32; HIDDEN_SIZE]) -> [i16; HIDDEN_SIZE] {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: `h` and the returned array are 32 lanes; the kernel reads
            // one 8-lane vector per iteration for four iterations.
            return unsafe { avx2::hidden_activation(h) };
        }
    }
    hidden_activation_scalar(h)
}

/// Hidden matrix multiply: `h[k] = b2[k] + sum_j w2[k][j] * a1[j]`.
///
/// `w2` is the packed `i16` copy of the asset's `i8` tensor and must be exactly
/// [`HIDDEN_SIZE`] × [`ACCUMULATOR_SIZE`] elements; `b2` must hold
/// [`HIDDEN_SIZE`] biases.
pub(crate) fn hidden_matmul(
    w2: &[i16],
    b2: &[i32],
    a1: &[i16; ACCUMULATOR_SIZE],
) -> [i32; HIDDEN_SIZE] {
    assert_eq!(w2.len(), HIDDEN_SIZE * ACCUMULATOR_SIZE);
    assert_eq!(b2.len(), HIDDEN_SIZE);
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the caller contract above guarantees the 8192-element
            // weight region and 32 biases; each neuron's kernel reads exactly
            // its own 256-element row and the 256-lane activation array.
            return unsafe { avx2::hidden_matmul(w2, b2, a1) };
        }
    }
    hidden_matmul_scalar(w2, b2, a1)
}

/// Output numerator: `ACT_MAX * b3[0] + sum_k w3[k] * a2[k]`.
///
/// `w3` is the asset's `i16` output row and must hold [`HIDDEN_SIZE`] weights;
/// `a2` is the second activation in `0..=127`.
pub(crate) fn output_numerator(w3: &[i16], b3: i32, a2: &[i16; HIDDEN_SIZE]) -> i32 {
    assert_eq!(w3.len(), HIDDEN_SIZE);
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: `w3` holds 32 elements per the contract and `a2` is a
            // 32-lane array; the kernel reads two 16-lane vectors from each.
            return unsafe { avx2::output_numerator(w3, b3, a2) };
        }
    }
    output_numerator_scalar(w3, b3, a2)
}

/// Add one input row to the accumulator: `acc[j] += w1[f][j]`.
///
/// `row` must hold [`ACCUMULATOR_SIZE`] `i16` weights and `acc` is the fixed
/// 256-lane accumulator. Every weight is an `i16`, so widening to `i32` is
/// lossless and the addition is the scalar `+=` of the reference.
pub(crate) fn accumulator_row_add(acc: &mut [i32; ACCUMULATOR_SIZE], row: &[i16]) {
    assert_eq!(row.len(), ACCUMULATOR_SIZE);
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: `acc` is exactly 256 lanes and `row` is checked above;
            // the kernel reads and writes 8 lanes per iteration for 32
            // iterations. AVX2 availability was checked at runtime.
            return unsafe { avx2::accumulator_row_add(acc, row) };
        }
    }
    accumulator_row_add_scalar(acc, row);
}

/// Subtract one input row from the accumulator: `acc[j] -= w1[f][j]`.
pub(crate) fn accumulator_row_sub(acc: &mut [i32; ACCUMULATOR_SIZE], row: &[i16]) {
    assert_eq!(row.len(), ACCUMULATOR_SIZE);
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: same fixed-width contract as `accumulator_row_add`.
            return unsafe { avx2::accumulator_row_sub(acc, row) };
        }
    }
    accumulator_row_sub_scalar(acc, row);
}

/// Scalar reference for [`accumulator_row_add`].
///
/// Public within the crate so the dispatch fallback and the forced tests both
/// call this exact function; the AVX2 arm is never compared with a test-local
/// copy of the loop.
pub(crate) fn accumulator_row_add_scalar(acc: &mut [i32; ACCUMULATOR_SIZE], row: &[i16]) {
    for (lane, weight) in acc.iter_mut().zip(row.iter()) {
        *lane += *weight as i32;
    }
}

/// Scalar reference for [`accumulator_row_sub`].
pub(crate) fn accumulator_row_sub_scalar(acc: &mut [i32; ACCUMULATOR_SIZE], row: &[i16]) {
    for (lane, weight) in acc.iter_mut().zip(row.iter()) {
        *lane -= *weight as i32;
    }
}

/// Scalar reference for [`first_activation`].
///
/// Kept public within the crate so the unit tests can force it on an
/// AVX2-capable host.
pub(crate) fn first_activation_scalar(acc: &[i32; ACCUMULATOR_SIZE]) -> [i16; ACCUMULATOR_SIZE] {
    let mut out = [0i16; ACCUMULATOR_SIZE];
    for (lane, value) in out.iter_mut().zip(acc.iter()) {
        *lane = super::div_ties_even((*value).clamp(0, ACC_SCALE) * ACT_MAX, ACC_SCALE) as i16;
    }
    out
}

/// Scalar reference for [`hidden_activation`].
pub(crate) fn hidden_activation_scalar(h: &[i32; HIDDEN_SIZE]) -> [i16; HIDDEN_SIZE] {
    let mut out = [0i16; HIDDEN_SIZE];
    for (lane, value) in out.iter_mut().zip(h.iter()) {
        *lane =
            super::div_ties_even((*value).clamp(0, ACT_MAX * HIDDEN_SCALE), HIDDEN_SCALE) as i16;
    }
    out
}

/// Scalar reference for [`hidden_matmul`].
pub(crate) fn hidden_matmul_scalar(
    w2: &[i16],
    b2: &[i32],
    a1: &[i16; ACCUMULATOR_SIZE],
) -> [i32; HIDDEN_SIZE] {
    let mut out = [0i32; HIDDEN_SIZE];
    for (neuron, value) in out.iter_mut().enumerate() {
        let row = &w2[neuron * ACCUMULATOR_SIZE..][..ACCUMULATOR_SIZE];
        let mut sum = b2[neuron];
        for (weight, activation) in row.iter().zip(a1.iter()) {
            sum += *weight as i32 * *activation as i32;
        }
        *value = sum;
    }
    out
}

/// Scalar reference for [`output_numerator`].
pub(crate) fn output_numerator_scalar(w3: &[i16], b3: i32, a2: &[i16; HIDDEN_SIZE]) -> i32 {
    let mut sum = ACT_MAX * b3;
    for (weight, activation) in w3.iter().zip(a2.iter()) {
        sum += *weight as i32 * *activation as i32;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    //! AVX2 kernels. Each function is reachable only through the runtime
    //! dispatch above, which has already verified the feature.

    use super::{ACCUMULATOR_SIZE, ACC_SCALE, ACT_MAX, HIDDEN_SCALE, HIDDEN_SIZE};
    use core::arch::x86_64::*;

    /// Apply the ties-to-even power-of-two rounding identity to one vector.
    ///
    /// `numerator` must hold non-negative `n = v * multiplier`, where `v` was
    /// clamped to `0..=clamp_max`; `shift` is the log2 divisor.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn round_activation<const SHIFT: i32>(numerator: __m256i) -> __m256i {
        let zero = _mm256_setzero_si256();
        let half = _mm256_set1_epi32(1 << (SHIFT - 1));
        let mask = _mm256_set1_epi32((1 << SHIFT) - 1);
        let one = _mm256_set1_epi32(1);
        let quotient = _mm256_srai_epi32(numerator, SHIFT);
        let remainder = _mm256_and_si256(numerator, mask);
        // r > half, or r == half and quotient is odd.
        let above = _mm256_cmpgt_epi32(remainder, half);
        let tied = _mm256_and_si256(
            _mm256_cmpeq_epi32(remainder, half),
            _mm256_cmpgt_epi32(_mm256_and_si256(quotient, one), zero),
        );
        let bump = _mm256_or_si256(above, tied);
        _mm256_sub_epi32(quotient, bump)
    }

    /// Clamp one vector to `0..=max` and scale by `factor`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn clamp_scale_activation(value: __m256i, max: __m256i, factor: __m256i) -> __m256i {
        let clamped = _mm256_min_epi32(_mm256_max_epi32(value, _mm256_setzero_si256()), max);
        _mm256_mullo_epi32(clamped, factor)
    }

    /// Sum eight `i32` lanes using only SSE2 shuffles and adds.
    ///
    /// Deliberately avoids `_mm_hadd_epi32` (SSSE3), so the AVX2 dispatch
    /// requires no ISA beyond the feature it checked.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_sum(sum: __m256i) -> i32 {
        let low = _mm256_castsi256_si128(sum);
        let high = _mm256_extracti128_si256(sum, 1);
        let mut four = _mm_add_epi32(low, high);
        // [a,b,c,d] -> [a+c,b+d,a+c,b+d] -> [a+b+c+d, ...].
        four = _mm_add_epi32(four, _mm_shuffle_epi32(four, 0x4E));
        four = _mm_add_epi32(four, _mm_shuffle_epi32(four, 0xB1));
        _mm_cvtsi128_si32(four)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn first_activation(
        acc: &[i32; ACCUMULATOR_SIZE],
    ) -> [i16; ACCUMULATOR_SIZE] {
        let mut out = [0i16; ACCUMULATOR_SIZE];
        let max = _mm256_set1_epi32(ACC_SCALE);
        let factor = _mm256_set1_epi32(ACT_MAX);
        let mut lane = 0usize;
        while lane < ACCUMULATOR_SIZE {
            let low = _mm256_loadu_si256(acc.as_ptr().add(lane) as *const __m256i);
            let high = _mm256_loadu_si256(acc.as_ptr().add(lane + 8) as *const __m256i);
            let low = round_activation::<8>(clamp_scale_activation(low, max, factor));
            let high = round_activation::<8>(clamp_scale_activation(high, max, factor));
            // `packs` operates per 128-bit lane, so reorder 64-bit lanes to
            // recover the ascending order: [a0..a3, b0..b3, a4..a7, b4..b7]
            // becomes [a0..a7, b0..b7].
            let packed = _mm256_permute4x64_epi64(_mm256_packs_epi32(low, high), 0xD8);
            _mm256_storeu_si256(out.as_mut_ptr().add(lane) as *mut __m256i, packed);
            lane += 16;
        }
        out
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn hidden_activation(h: &[i32; HIDDEN_SIZE]) -> [i16; HIDDEN_SIZE] {
        let mut out = [0i16; HIDDEN_SIZE];
        let max = _mm256_set1_epi32(ACT_MAX * HIDDEN_SCALE);
        let mut lane = 0usize;
        while lane < HIDDEN_SIZE {
            let value = _mm256_loadu_si256(h.as_ptr().add(lane) as *const __m256i);
            let clamped = _mm256_min_epi32(_mm256_max_epi32(value, _mm256_setzero_si256()), max);
            let rounded = round_activation::<7>(clamped);
            // Pack the two 128-bit halves separately so the 8 results stay in
            // ascending lane order.
            let packed = _mm_packs_epi32(
                _mm256_castsi256_si128(rounded),
                _mm256_extracti128_si256(rounded, 1),
            );
            _mm_storeu_si128(out.as_mut_ptr().add(lane) as *mut __m128i, packed);
            lane += 8;
        }
        out
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn hidden_matmul(
        w2: &[i16],
        b2: &[i32],
        a1: &[i16; ACCUMULATOR_SIZE],
    ) -> [i32; HIDDEN_SIZE] {
        let mut out = [0i32; HIDDEN_SIZE];
        for neuron in 0..HIDDEN_SIZE {
            let row = w2.as_ptr().add(neuron * ACCUMULATOR_SIZE);
            let mut low_sum = _mm256_setzero_si256();
            let mut high_sum = _mm256_setzero_si256();
            let mut lane = 0usize;
            while lane < ACCUMULATOR_SIZE {
                let weight_low = _mm256_loadu_si256(row.add(lane) as *const __m256i);
                let weight_high = _mm256_loadu_si256(row.add(lane + 16) as *const __m256i);
                let activation_low = _mm256_loadu_si256(a1.as_ptr().add(lane) as *const __m256i);
                let activation_high =
                    _mm256_loadu_si256(a1.as_ptr().add(lane + 16) as *const __m256i);
                // Each madd multiplies 16 `i16` lanes in 32-bit and adds
                // adjacent pairs; no saturation is involved.
                low_sum = _mm256_add_epi32(low_sum, _mm256_madd_epi16(weight_low, activation_low));
                high_sum =
                    _mm256_add_epi32(high_sum, _mm256_madd_epi16(weight_high, activation_high));
                lane += 32;
            }
            out[neuron] = horizontal_sum(_mm256_add_epi32(low_sum, high_sum)) + b2[neuron];
        }
        out
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn output_numerator(w3: &[i16], b3: i32, a2: &[i16; HIDDEN_SIZE]) -> i32 {
        let weight_low = _mm256_loadu_si256(w3.as_ptr() as *const __m256i);
        let weight_high = _mm256_loadu_si256(w3.as_ptr().add(16) as *const __m256i);
        let activation_low = _mm256_loadu_si256(a2.as_ptr() as *const __m256i);
        let activation_high = _mm256_loadu_si256(a2.as_ptr().add(16) as *const __m256i);
        let sum = _mm256_add_epi32(
            _mm256_madd_epi16(weight_low, activation_low),
            _mm256_madd_epi16(weight_high, activation_high),
        );
        ACT_MAX * b3 + horizontal_sum(sum)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn accumulator_row_add(acc: &mut [i32; ACCUMULATOR_SIZE], row: &[i16]) {
        let mut lane = 0usize;
        while lane < ACCUMULATOR_SIZE {
            // Eight i16 weights widen losslessly to eight i32 lanes.
            let weights = _mm_loadu_si128(row.as_ptr().add(lane) as *const __m128i);
            let weights = _mm256_cvtepi16_epi32(weights);
            let values = _mm256_loadu_si256(acc.as_ptr().add(lane) as *const __m256i);
            _mm256_storeu_si256(
                acc.as_mut_ptr().add(lane) as *mut __m256i,
                _mm256_add_epi32(values, weights),
            );
            lane += 8;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn accumulator_row_sub(acc: &mut [i32; ACCUMULATOR_SIZE], row: &[i16]) {
        let mut lane = 0usize;
        while lane < ACCUMULATOR_SIZE {
            let weights = _mm_loadu_si128(row.as_ptr().add(lane) as *const __m128i);
            let weights = _mm256_cvtepi16_epi32(weights);
            let values = _mm256_loadu_si256(acc.as_ptr().add(lane) as *const __m256i);
            _mm256_storeu_si256(
                acc.as_mut_ptr().add(lane) as *mut __m256i,
                _mm256_sub_epi32(values, weights),
            );
            lane += 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic SplitMix64 stream, so adversarial vectors are reproducible.
    struct Rng(u64);

    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, bound: u64) -> u64 {
            self.next_u64() % bound
        }
    }

    #[test]
    fn first_activation_covers_every_distinct_clamp_value() {
        // `acc` only reaches the arithmetic through `clamp(0, ACC_SCALE)`, so
        // the 257 values `0..=256` are the complete input domain.
        let mut cases = Vec::new();
        for lane in 0..ACCUMULATOR_SIZE {
            cases.push(lane as i32);
        }
        cases.push(ACC_SCALE);
        cases.push(-1);
        for chunk in cases.chunks(ACCUMULATOR_SIZE) {
            let mut acc = [0i32; ACCUMULATOR_SIZE];
            for (lane, value) in acc.iter_mut().zip(chunk) {
                *lane = *value;
            }
            let scalar = first_activation_scalar(&acc);
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: fixed-width kernel on a 256-lane array; feature checked.
                let simd = unsafe { avx2::first_activation(&acc) };
                assert_eq!(scalar, simd, "first activation scalar vs AVX2");
            }
            let expected = super::super::activate_accumulator(&acc);
            for (lane, (actual, expected)) in scalar.iter().zip(expected.iter()).enumerate() {
                assert_eq!(*actual as i32, *expected, "lane {lane}");
            }
        }
    }

    #[test]
    fn hidden_activation_scalar_matches_simd() {
        // The rounding only depends on the division remainder and quotient
        // parity, so check every residue with an even and an odd quotient, plus
        // the clamp boundaries.
        let mut cases = Vec::new();
        for quotient_parity in 0..2i32 {
            for residue in 0..HIDDEN_SCALE {
                cases.push(quotient_parity * HIDDEN_SCALE + residue);
            }
        }
        cases.push(ACT_MAX * HIDDEN_SCALE);
        cases.push(ACT_MAX * HIDDEN_SCALE + 1);
        cases.push(-1);
        for chunk in cases.chunks(HIDDEN_SIZE) {
            let mut h = [0i32; HIDDEN_SIZE];
            for (lane, value) in h.iter_mut().zip(chunk) {
                *lane = *value;
            }
            let scalar = hidden_activation_scalar(&h);
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: fixed-width kernel on a 32-lane array; feature checked.
                let simd = unsafe { avx2::hidden_activation(&h) };
                assert_eq!(scalar, simd, "hidden activation scalar vs AVX2");
            }
            let expected = super::super::activate_hidden(&h);
            for (lane, (actual, expected)) in scalar.iter().zip(expected.iter()).enumerate() {
                assert_eq!(*actual as i32, *expected, "lane {lane}");
            }
        }
    }

    #[test]
    fn hidden_matmul_scalar_matches_simd_on_adversarial_inputs() {
        let mut rng = Rng(0x5EED_1234);
        let mut w2 = vec![0i16; HIDDEN_SIZE * ACCUMULATOR_SIZE];
        let mut a1 = [0i16; ACCUMULATOR_SIZE];
        let mut b2 = [0i32; HIDDEN_SIZE];
        for repeat in 0..40 {
            for weight in &mut w2 {
                // Alternate extremes and random i8 values (lossless i16 widen).
                *weight = match rng.below(8) {
                    0 => -128,
                    1 => 127,
                    2 => 0,
                    _ => rng.below(256) as i16 - 128,
                };
            }
            for lane in &mut a1 {
                *lane = match repeat % 4 {
                    0 => 127,
                    1 => 0,
                    _ => rng.below(128) as i16,
                };
            }
            for bias in &mut b2 {
                *bias = rng.next_u64() as i32;
            }
            let scalar = hidden_matmul_scalar(&w2, &b2, &a1);
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: buffers have exactly the contracted lengths.
                let simd = unsafe { avx2::hidden_matmul(&w2, &b2, &a1) };
                assert_eq!(scalar, simd, "hidden matmul scalar vs AVX2 repeat {repeat}");
            }
        }
    }

    #[test]
    fn accumulator_row_add_and_sub_match_the_scalar_loop() {
        let mut rng = Rng(0xADD5_0B5E);
        let mut row = [0i16; ACCUMULATOR_SIZE];
        for weight in &mut row {
            *weight = match rng.below(8) {
                0 => i16::MIN,
                1 => i16::MAX,
                _ => rng.below(65536) as i16,
            };
        }
        let mut base = [0i32; ACCUMULATOR_SIZE];
        for value in &mut base {
            *value = rng.next_u64() as i32;
        }

        let mut scalar_add = base;
        accumulator_row_add_scalar(&mut scalar_add, &row);
        let mut simd_add = base;
        accumulator_row_add(&mut simd_add, &row);
        assert_eq!(
            scalar_add, simd_add,
            "row add scalar fallback vs dispatched"
        );

        let mut scalar_sub = base;
        accumulator_row_sub_scalar(&mut scalar_sub, &row);
        let mut simd_sub = base;
        accumulator_row_sub(&mut simd_sub, &row);
        assert_eq!(
            scalar_sub, simd_sub,
            "row sub scalar fallback vs dispatched"
        );

        let mut round_trip = base;
        accumulator_row_add(&mut round_trip, &row);
        accumulator_row_sub(&mut round_trip, &row);
        assert_eq!(
            round_trip, base,
            "add then sub must restore the accumulator"
        );
    }

    #[test]
    fn output_numerator_scalar_matches_simd_on_adversarial_inputs() {
        let mut rng = Rng(0xC0FF_EE01);
        let mut w3 = [0i16; HIDDEN_SIZE];
        let mut a2 = [0i16; HIDDEN_SIZE];
        for _ in 0..40 {
            for weight in &mut w3 {
                *weight = match rng.below(6) {
                    0 => i16::MIN,
                    1 => i16::MAX,
                    _ => rng.below(65536) as i16,
                };
            }
            for lane in &mut a2 {
                *lane = rng.below(128) as i16;
            }
            // Keep adversarial biases in the non-overflow domain, not release wrapping.
            let bias = rng.below(2_000_001) as i32 - 1_000_000;
            let expected = i64::from(ACT_MAX) * i64::from(bias)
                + w3.iter()
                    .zip(a2)
                    .map(|(weight, activation)| i64::from(*weight) * i64::from(activation))
                    .sum::<i64>();
            let expected = i32::try_from(expected).expect("adversarial output must fit i32");
            let scalar = output_numerator_scalar(&w3, bias, &a2);
            assert_eq!(scalar, expected, "output numerator scalar vs wide oracle");
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: `w3` and `a2` hold exactly 32 elements.
                let simd = unsafe { avx2::output_numerator(&w3, bias, &a2) };
                assert_eq!(scalar, simd, "output numerator scalar vs AVX2");
            }
        }
    }
    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn hidden_matmul_rejects_short_weights() {
        let a1 = [0i16; ACCUMULATOR_SIZE];
        let b2 = [0i32; HIDDEN_SIZE];
        let _ = hidden_matmul(&[], &b2, &a1);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn hidden_matmul_rejects_short_biases() {
        let a1 = [0i16; ACCUMULATOR_SIZE];
        let w2 = [0i16; HIDDEN_SIZE * ACCUMULATOR_SIZE];
        let _ = hidden_matmul(&w2, &[], &a1);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn output_numerator_rejects_short_weights() {
        let a2 = [0i16; HIDDEN_SIZE];
        let _ = output_numerator(&[], 0, &a2);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn accumulator_row_add_rejects_short_row() {
        let mut acc = [0i32; ACCUMULATOR_SIZE];
        let row = [0i16; ACCUMULATOR_SIZE - 1];
        accumulator_row_add(&mut acc, &row);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn accumulator_row_sub_rejects_short_row() {
        let mut acc = [0i32; ACCUMULATOR_SIZE];
        let row = [0i16; ACCUMULATOR_SIZE - 1];
        accumulator_row_sub(&mut acc, &row);
    }
}
