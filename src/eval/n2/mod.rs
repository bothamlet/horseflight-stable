//! Integer NNUE evaluator with a 781 → 256 → 32 → 1 architecture.
//!
//! The model is embedded and validated once. [`N2Evaluator`] rebuilds the
//! accumulator for reference; UCI search uses the incremental [`session::Session`].
//!
//! # Input schema
//!
//! The schema is *not* [`crate::eval::features`] schema v1 (768 binary
//! piece-square inputs). This model uses 781 inputs with fixed color indices:
//!
//! ```text
//! index = ((color_offset + role) * 64) + square
//! color_offset = 0 white, 6 black; role = pawn 0 .. king 5; a1=0 .. h8=63
//! 768          = White to move (active iff the side to move is White)
//! 769..772     = castling rights K, Q, k, q
//! 773..780     = legal en-passant file index 0..7
//! ```
//!
//! The en-passant input is active only when a fully legal en-passant capture
//! exists, reusing [`Position::legal_ep_square`] (pin and x-ray safety
//! included); this module never re-implements king-safety logic.
//!
//! # Integer semantics
//!
//! ```text
//! acc[j] = b1[j] + sum over all active features f of w1[f][j]
//! a1[j]  = tie_even(clamp(acc[j], 0, 256) * 127 / 256)
//! h[k]   = b2[k] + sum_j w2[k][j] * a1[j]
//! a2[k]  = tie_even(clamp(h[k], 0, 16256) / 128)
//! out    = w3[0..31] · a2 + 127 * b3[0]
//! cp     = tie_even(400 * out / 32512)
//! ```
//!
//! `tie_even` is round-to-nearest with ties to even, implemented with floor
//! division ([`div_ties_even`]): `q = floor(n/d)`, `r = n - q*d` in `[0, d)`;
//! bump `q` when `2r > d`, or when `2r == d` and `q` is odd. For negative
//! numerators this matches Python's `divmod` floor semantics.
//!
//! The asset's `out` is treated as White-relative. [`N2Evaluator`] negates it
//! for Black to move so `Evaluator` contract (side-to-move centipawns) holds.
//! The supplied metadata string
//! `white_relative_pending_provenance_verification` is preserved verbatim in
//! [`score_perspective`]; [`DERIVED_SCORE_PERSPECTIVE`] records the derived
//! interpretation separately and never overwrites it.
//!
//! # Provenance
//!
//! The embedded asset carries the three supplied-member SHA256 values as source
//! pointers and `provenance_flags = 0` (`original_export_verified = false`).
//! Those values are recomputed observed digests, not an original export
//! manifest: internal golden self-consistency is not training provenance.

use std::fmt;
use std::sync::OnceLock;

use crate::eval::{clamp_evaluation, Evaluator};
use crate::position::{PieceChange, Position};
use crate::types::{CastlingSide, Color, Piece};

pub mod session;
pub mod sha256;
mod simd;
#[cfg(feature = "profiling")]
pub mod stages;

/// Embedded portable asset. Tiny, immutable and validated once in [`model`].
const ASSET_BYTES: &[u8] = include_bytes!("../../../assets/n2-002.bin");

/// Container version accepted by this runtime.
pub const CONTAINER_VERSION: u32 = 1;
/// Stable model identifier stored in the asset.
pub const MODEL_ID: &str = "horseflight-n2-002";
/// N2 absolute-white schema version accepted by this runtime.
pub const SCHEMA_VERSION: u32 = 1;
/// Stable N2 schema identifier stored in the asset.
pub const SCHEMA_ID: &str = "horseflight-n2-absolute-white-781";

/// Number of input features.
pub const INPUT_FEATURES: usize = 781;
/// Accumulator width.
pub const ACCUMULATOR_SIZE: usize = 256;
/// Hidden layer width.
pub const HIDDEN_SIZE: usize = 32;
/// Output width.
pub const OUTPUT_SIZE: usize = 1;

/// Accumulator activation scale (clamp upper bound).
pub const ACC_SCALE: i32 = 256;
/// Activation maximum for both hidden layers.
pub const ACT_MAX: i32 = 127;
/// Hidden activation divisor.
pub const HIDDEN_SCALE: i32 = 128;
/// Output weight scale.
pub const OUT_SCALE: i32 = 256;
/// Centipawn multiplier applied to the output numerator.
pub const CP_MULTIPLIER: i32 = 400;
/// Output denominator behind [`CP_MULTIPLIER`]: `127 * 256`.
pub const CP_DENOMINATOR: i32 = ACT_MAX * OUT_SCALE;

/// Index of the White-to-move input.
pub const WHITE_TO_MOVE_FEATURE: usize = 768;
/// First castling input (K, Q, k, q at 769..772).
pub const CASTLING_FEATURE_BASE: usize = 769;
/// First legal-en-passant input (files 0..7 at 773..780).
pub const EP_FEATURE_BASE: usize = 773;

/// The supplied metadata's score-perspective string, preserved verbatim.
pub const METADATA_SCORE_PERSPECTIVE: &str = "white_relative_pending_provenance_verification";
/// Derived interpretation recorded separately from the supplied metadata.
pub const DERIVED_SCORE_PERSPECTIVE: &str = "derived: white-relative";

/// Fixed tensor order and byte lengths used by the container.
const W1_LEN: usize = INPUT_FEATURES * ACCUMULATOR_SIZE;
const B1_LEN: usize = ACCUMULATOR_SIZE;
const W2_LEN: usize = HIDDEN_SIZE * ACCUMULATOR_SIZE;
const B2_LEN: usize = HIDDEN_SIZE;
const W3_LEN: usize = OUTPUT_SIZE * HIDDEN_SIZE;
const B3_LEN: usize = OUTPUT_SIZE;
const TENSOR_BYTES: usize = W1_LEN * 2 + B1_LEN * 4 + W2_LEN + B2_LEN * 4 + W3_LEN * 2 + B3_LEN * 4;
const CHECKSUM_BYTES: usize = 32;

/// Typed asset validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssetError {
    /// The eight magic bytes are not `HFLNNUE\x01`.
    BadMagic,
    /// The container version is not supported.
    UnsupportedContainerVersion(u32),
    /// The stored model identifier is not [`MODEL_ID`].
    BadModelId,
    /// The stored N2 schema version is not supported.
    UnsupportedSchemaVersion(u32),
    /// The stored N2 schema identifier is not [`SCHEMA_ID`].
    BadSchemaId,
    /// Dimensions are not 781 → 256 → 32 → 1.
    WrongDimensions {
        in_features: u32,
        accumulator: u32,
        hidden: u32,
        out: u32,
    },
    /// Scales differ from the pinned integer contract.
    WrongScales {
        acc_scale: i32,
        act_max: i32,
        hidden_scale: i32,
        out_scale: i32,
        cp_multiplier: i32,
    },
    /// The buffer ends before a complete header, tensor region or checksum.
    TruncatedBytes,
    /// The tensor region is shorter than the fixed tensor sizes.
    TensorLength { expected: usize, actual: usize },
    /// Bytes follow the fixed tensor region (before the trailing checksum).
    TrailingBytes,
    /// The trailing SHA-256 does not cover the preceding bytes.
    ChecksumMismatch,
}

impl fmt::Display for AssetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssetError::BadMagic => write!(formatter, "bad asset magic"),
            AssetError::UnsupportedContainerVersion(version) => {
                write!(formatter, "unsupported container version {version}")
            }
            AssetError::BadModelId => write!(formatter, "unexpected model id"),
            AssetError::UnsupportedSchemaVersion(version) => {
                write!(formatter, "unsupported N2 schema version {version}")
            }
            AssetError::BadSchemaId => write!(formatter, "unexpected N2 schema id"),
            AssetError::WrongDimensions {
                in_features,
                accumulator,
                hidden,
                out,
            } => write!(
                formatter,
                "wrong dimensions {in_features}->{accumulator}->{hidden}->{out}"
            ),
            AssetError::WrongScales {
                acc_scale,
                act_max,
                hidden_scale,
                out_scale,
                cp_multiplier,
            } => write!(
                formatter,
                "wrong scales acc={acc_scale} act={act_max} hidden={hidden_scale} \
                 out={out_scale} cp={cp_multiplier}"
            ),
            AssetError::TruncatedBytes => write!(formatter, "truncated asset bytes"),
            AssetError::TensorLength { expected, actual } => write!(
                formatter,
                "wrong tensor length: expected {expected} bytes, found {actual}"
            ),
            AssetError::TrailingBytes => write!(formatter, "trailing bytes after tensors"),
            AssetError::ChecksumMismatch => write!(formatter, "asset checksum mismatch"),
        }
    }
}

impl std::error::Error for AssetError {}

/// Validated header metadata plus source pointers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelMeta {
    pub model_id: String,
    pub schema_id: String,
    pub in_features: u32,
    pub accumulator: u32,
    pub hidden: u32,
    pub out: u32,
    pub acc_scale: i32,
    pub act_max: i32,
    pub hidden_scale: i32,
    pub out_scale: i32,
    pub cp_multiplier: i32,
    pub provenance_flags: u32,
    pub source_hashes: [[u8; 32]; 3],
}

/// Fully validated immutable model.
pub struct Model {
    meta: ModelMeta,
    w1: Vec<i16>,
    b1: Vec<i32>,
    w2: Vec<i8>,
    /// Deterministic `i16` widening of `w2`, so the hidden matmul can use
    /// 16-bit multiply-add instructions. The `i8` tensor stays for the scalar
    /// reference and for asset parity.
    w2_i16: Vec<i16>,
    b2: Vec<i32>,
    w3: Vec<i16>,
    b3: Vec<i32>,
    asset_sha256: String,
    weights_sha256: String,
}

impl Model {
    /// Validate `bytes` and return the parsed model.
    fn parse(bytes: &[u8]) -> Result<Model, AssetError> {
        let mut reader = Reader::new(bytes);

        if reader.take(8)? != b"HFLNNUE\x01" {
            return Err(AssetError::BadMagic);
        }
        let container_version = reader.u32()?;
        if container_version != CONTAINER_VERSION {
            return Err(AssetError::UnsupportedContainerVersion(container_version));
        }
        let model_id = String::from_utf8(reader.string_bytes()?.to_vec())
            .map_err(|_| AssetError::BadModelId)?;
        if model_id != MODEL_ID {
            return Err(AssetError::BadModelId);
        }
        let schema_version = reader.u32()?;
        if schema_version != SCHEMA_VERSION {
            return Err(AssetError::UnsupportedSchemaVersion(schema_version));
        }
        let schema_id = String::from_utf8(reader.string_bytes()?.to_vec())
            .map_err(|_| AssetError::BadSchemaId)?;
        if schema_id != SCHEMA_ID {
            return Err(AssetError::BadSchemaId);
        }

        let in_features = reader.u32()?;
        let accumulator = reader.u32()?;
        let hidden = reader.u32()?;
        let out = reader.u32()?;
        if (in_features, accumulator, hidden, out) != (781, 256, 32, 1) {
            return Err(AssetError::WrongDimensions {
                in_features,
                accumulator,
                hidden,
                out,
            });
        }

        let acc_scale = reader.i32()?;
        let act_max = reader.i32()?;
        let hidden_scale = reader.i32()?;
        let out_scale = reader.i32()?;
        let cp_multiplier = reader.i32()?;
        if (acc_scale, act_max, hidden_scale, out_scale, cp_multiplier)
            != (ACC_SCALE, ACT_MAX, HIDDEN_SCALE, OUT_SCALE, CP_MULTIPLIER)
        {
            return Err(AssetError::WrongScales {
                acc_scale,
                act_max,
                hidden_scale,
                out_scale,
                cp_multiplier,
            });
        }

        let provenance_flags = reader.u32()?;
        let mut source_hashes = [[0u8; 32]; 3];
        for hash in &mut source_hashes {
            hash.copy_from_slice(reader.take(32)?);
        }

        let remaining = bytes.len().saturating_sub(reader.position());
        if remaining < CHECKSUM_BYTES {
            return Err(AssetError::TruncatedBytes);
        }
        let actual_tensors = remaining - CHECKSUM_BYTES;
        if actual_tensors < TENSOR_BYTES {
            return Err(AssetError::TensorLength {
                expected: TENSOR_BYTES,
                actual: actual_tensors,
            });
        }
        if actual_tensors > TENSOR_BYTES {
            return Err(AssetError::TrailingBytes);
        }

        let w1 = reader.i16_vec(W1_LEN)?;
        let b1 = reader.i32_vec(B1_LEN)?;
        let w2 = reader.i8_vec(W2_LEN)?;
        let w2_i16: Vec<i16> = w2.iter().map(|weight| i16::from(*weight)).collect();
        let b2 = reader.i32_vec(B2_LEN)?;
        let w3 = reader.i16_vec(W3_LEN)?;
        let b3 = reader.i32_vec(B3_LEN)?;

        let stored_checksum = reader.take(CHECKSUM_BYTES)?;
        let computed = sha256::digest(&bytes[..bytes.len() - CHECKSUM_BYTES]);
        if stored_checksum != computed {
            return Err(AssetError::ChecksumMismatch);
        }

        Ok(Model {
            meta: ModelMeta {
                model_id,
                schema_id,
                in_features,
                accumulator,
                hidden,
                out,
                acc_scale,
                act_max,
                hidden_scale,
                out_scale,
                cp_multiplier,
                provenance_flags,
                source_hashes,
            },
            w1,
            b1,
            w2,
            w2_i16,
            b2,
            w3,
            b3,
            asset_sha256: sha256::hex(&sha256::digest(bytes)),
            weights_sha256: sha256::hex(&source_hashes[0]),
        })
    }

    /// Validated header metadata.
    pub fn meta(&self) -> &ModelMeta {
        &self.meta
    }
}

/// Validate a model buffer against the embedded NNUE format.
///
/// This performs exactly the same checks as the one-time embedded-asset
/// validation and returns only header metadata, so tests and tooling can feed
/// malformed buffers without constructing a model.
pub fn validate_bytes(bytes: &[u8]) -> Result<ModelMeta, AssetError> {
    Model::parse(bytes).map(|model| model.meta)
}

/// The embedded asset bytes, for validation tests.
pub fn asset_bytes() -> &'static [u8] {
    ASSET_BYTES
}

static MODEL: OnceLock<Model> = OnceLock::new();

/// The validated embedded model. Parsing and checksum verification happen once.
pub fn model() -> &'static Model {
    MODEL.get_or_init(|| {
        Model::parse(ASSET_BYTES)
            .unwrap_or_else(|error| panic!("embedded NNUE weights failed validation: {error}"))
    })
}

/// SHA-256 of the complete embedded asset file, lowercase hex.
pub fn asset_sha256() -> &'static str {
    &model().asset_sha256
}

/// SHA-256 of the supplied `weights.npz`, recorded as a source pointer.
pub fn weights_sha256() -> &'static str {
    &model().weights_sha256
}

/// Provenance flags from the asset header (bit0 = original_export_verified).
pub fn provenance_flags() -> u32 {
    model().meta.provenance_flags
}

/// Supplied metadata score-perspective string, reported verbatim.
pub fn score_perspective() -> &'static str {
    METADATA_SCORE_PERSPECTIVE
}

/// Index of one piece-plane input.
pub fn piece_feature_index(piece: Piece, square: crate::types::Square) -> usize {
    (piece.color.index() * 6 + piece.role.index()) * 64 + square.index() as usize
}

/// Visit every active input index for `position` in ascending order.
pub fn for_each_active_feature(position: &Position, mut visit: impl FnMut(usize)) {
    let board = position.board();
    for color in Color::ALL {
        for role in crate::types::Role::ALL {
            for square in board.pieces(color, role).iter() {
                visit(piece_feature_index(Piece::new(color, role), square));
            }
        }
    }

    if position.side_to_move().is_white() {
        visit(WHITE_TO_MOVE_FEATURE);
    }

    let rights = position.castling_rights();
    for (offset, (color, side)) in CASTLING_RIGHTS.iter().enumerate() {
        if rights.has(*color, *side) {
            visit(CASTLING_FEATURE_BASE + offset);
        }
    }

    if let Some(square) = position.legal_ep_square() {
        visit(EP_FEATURE_BASE + square.file() as usize);
    }
}

/// Collect the active input indices for `position` in ascending order.
pub fn active_features(position: &Position) -> Vec<usize> {
    let mut features = Vec::with_capacity(32);
    for_each_active_feature(position, |index| features.push(index));
    features
}

/// Dense binary input vector. Test helper: production accumulation visits only
/// active inputs.
#[doc(hidden)]
pub fn dense_feature_vector(position: &Position) -> [u8; INPUT_FEATURES] {
    let mut dense = [0u8; INPUT_FEATURES];
    for_each_active_feature(position, |index| dense[index] = 1);
    dense
}

const CASTLING_RIGHTS: [(Color, CastlingSide); 4] = [
    (Color::White, CastlingSide::KingSide),
    (Color::White, CastlingSide::QueenSide),
    (Color::Black, CastlingSide::KingSide),
    (Color::Black, CastlingSide::QueenSide),
];

/// Integer round-to-nearest, ties-to-even, using floor division.
///
/// `denominator` must be positive. The remainder is `numerator - q*denominator`
/// in `[0, denominator)`, so negative numerators round symmetrically around
/// the floor of the exact quotient.
pub fn div_ties_even(numerator: i32, denominator: i32) -> i32 {
    debug_assert!(denominator > 0, "rounding denominator must be positive");
    let quotient = numerator.div_euclid(denominator);
    let remainder = numerator - quotient * denominator;
    let twice = remainder * 2;
    if twice > denominator || (twice == denominator && quotient % 2 != 0) {
        quotient + 1
    } else {
        quotient
    }
}

/// Ties-to-even rounding of an exact wide quotient.
///
/// [`div_ties_even`] works on `i32`; the centipawn conversion multiplies the
/// output numerator by [`CP_MULTIPLIER`] first, so it uses this wider form to
/// keep even a much larger output layer exact instead of wrapping in release
/// builds. For the shipped asset `|out| <= 32*127*234 + 127*25 = 954_055`, so
/// `|400*out| <= 381_622_000`: already inside `i32`, and provably inside `i64`.
fn div_ties_even_wide(numerator: i64, denominator: i64) -> i64 {
    debug_assert!(denominator > 0, "rounding denominator must be positive");
    let quotient = numerator.div_euclid(denominator);
    let remainder = numerator - quotient * denominator;
    let twice = remainder * 2;
    if twice > denominator || (twice == denominator && quotient % 2 != 0) {
        quotient + 1
    } else {
        quotient
    }
}

fn activate_accumulator(acc: &[i32; ACCUMULATOR_SIZE]) -> [i32; ACCUMULATOR_SIZE] {
    let mut a1 = [0i32; ACCUMULATOR_SIZE];
    for (index, value) in acc.iter().enumerate() {
        a1[index] = div_ties_even((*value).clamp(0, ACC_SCALE) * ACT_MAX, ACC_SCALE);
    }
    a1
}

fn activate_hidden(h: &[i32; HIDDEN_SIZE]) -> [i32; HIDDEN_SIZE] {
    let mut a2 = [0i32; HIDDEN_SIZE];
    for (index, value) in h.iter().enumerate() {
        a2[index] = div_ties_even((*value).clamp(0, ACT_MAX * HIDDEN_SCALE), HIDDEN_SCALE);
    }
    a2
}

/// Every intermediate tensor of one forward pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForwardTrace {
    pub a1: [i32; ACCUMULATOR_SIZE],
    pub h: [i32; HIDDEN_SIZE],
    pub a2: [i32; HIDDEN_SIZE],
    pub output_numerator: i32,
}

/// Incremental accumulator for the 781-input schema.
///
/// [`Accumulator::rebuild`] is the reference computation.
/// [`Accumulator::apply_move`] and [`Accumulator::revert_move`] take both
/// positions around the move so the three metadata inputs (side to move,
/// castling rights, legal en-passant file) are never forgotten; they update the
/// accumulator from piece deltas and metadata differences without allocating or
/// rebuilding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Accumulator {
    acc: [i32; ACCUMULATOR_SIZE],
    /// Profiling-only knob for the A2 attribution harness: force the scalar
    /// row kernels while keeping every other path unchanged.
    #[cfg(feature = "profiling")]
    scalar_rows: bool,
}

impl Accumulator {
    /// Build a fresh accumulator from a full recomputation.
    pub fn from_position(position: &Position) -> Accumulator {
        let mut accumulator = Accumulator {
            acc: [0; ACCUMULATOR_SIZE],
            #[cfg(feature = "profiling")]
            scalar_rows: false,
        };
        accumulator.rebuild(position);
        accumulator
    }

    /// Recompute every accumulator lane from scratch.
    pub fn rebuild(&mut self, position: &Position) {
        #[cfg(feature = "profiling")]
        crate::profiling::record_rebuild();
        self.acc.copy_from_slice(&model().b1);
        for_each_active_feature(position, |feature| {
            self.add_row(feature);
        });
    }

    /// Accumulator lanes.
    pub fn values(&self) -> &[i32; ACCUMULATOR_SIZE] {
        &self.acc
    }

    /// Profiling-only: force scalar row add/sub (A2 attribution harness).
    #[cfg(feature = "profiling")]
    pub fn set_scalar_rows(&mut self, enabled: bool) {
        self.scalar_rows = enabled;
    }

    /// Apply one legal move's piece deltas plus its metadata deltas.
    ///
    /// `before` is the position the move was played from and `after` the
    /// resulting position. Both are required: the side to move, castling
    /// rights and legal en-passant file are model inputs.
    pub fn apply_move(&mut self, changes: &[PieceChange], before: &Position, after: &Position) {
        for change in changes {
            if let Some(piece) = change.before {
                self.sub_row(piece_feature_index(piece, change.square));
            }
            if let Some(piece) = change.after {
                self.add_row(piece_feature_index(piece, change.square));
            }
        }
        self.update_metadata(after, before);
    }

    /// Revert one legal move's piece deltas plus its metadata deltas.
    ///
    /// `before` and `after` are the same positions passed to
    /// [`Accumulator::apply_move`] for this move.
    pub fn revert_move(&mut self, changes: &[PieceChange], before: &Position, after: &Position) {
        for change in changes.iter().rev() {
            if let Some(piece) = change.after {
                self.sub_row(piece_feature_index(piece, change.square));
            }
            if let Some(piece) = change.before {
                self.add_row(piece_feature_index(piece, change.square));
            }
        }
        self.update_metadata(before, after);
    }

    fn add_row(&mut self, feature: usize) {
        let row = &model().w1[feature * ACCUMULATOR_SIZE..][..ACCUMULATOR_SIZE];
        #[cfg(feature = "profiling")]
        if self.scalar_rows {
            simd::accumulator_row_add_scalar(&mut self.acc, row);
            return;
        }
        simd::accumulator_row_add(&mut self.acc, row);
    }

    fn sub_row(&mut self, feature: usize) {
        let row = &model().w1[feature * ACCUMULATOR_SIZE..][..ACCUMULATOR_SIZE];
        #[cfg(feature = "profiling")]
        if self.scalar_rows {
            simd::accumulator_row_sub_scalar(&mut self.acc, row);
            return;
        }
        simd::accumulator_row_sub(&mut self.acc, row);
    }

    fn update_metadata(&mut self, add: &Position, remove: &Position) {
        let add_white = add.side_to_move().is_white();
        let remove_white = remove.side_to_move().is_white();
        if add_white && !remove_white {
            self.add_row(WHITE_TO_MOVE_FEATURE);
        } else if !add_white && remove_white {
            self.sub_row(WHITE_TO_MOVE_FEATURE);
        }

        let add_rights = add.castling_rights();
        let remove_rights = remove.castling_rights();
        for (offset, (color, side)) in CASTLING_RIGHTS.iter().enumerate() {
            let added = add_rights.has(*color, *side);
            let removed = remove_rights.has(*color, *side);
            if added && !removed {
                self.add_row(CASTLING_FEATURE_BASE + offset);
            } else if removed && !added {
                self.sub_row(CASTLING_FEATURE_BASE + offset);
            }
        }

        let add_ep = add
            .legal_ep_square()
            .map(|square| EP_FEATURE_BASE + square.file() as usize);
        let remove_ep = remove
            .legal_ep_square()
            .map(|square| EP_FEATURE_BASE + square.file() as usize);
        if add_ep != remove_ep {
            if let Some(feature) = add_ep {
                self.add_row(feature);
            }
            if let Some(feature) = remove_ep {
                self.sub_row(feature);
            }
        }
    }
}

/// Run the network on a ready accumulator with the scalar reference kernels.
///
/// This is the independent arithmetic oracle for the dispatched kernels used
/// by [`forward`]: it keeps the original `i32` loops, the exact
/// [`div_ties_even`] rounding and the asset's `i8`/`i16` tensor types. Tests
/// compare [`forward_trace_optimized`] against it on every golden fixture and
/// on the incremental session's positions.
pub fn forward_trace(accumulator: &Accumulator) -> ForwardTrace {
    #[cfg(feature = "profiling")]
    crate::profiling::record_forward();
    let model = model();
    let a1 = activate_accumulator(&accumulator.acc);

    let mut h = [0i32; HIDDEN_SIZE];
    for (hidden, value) in h.iter_mut().enumerate() {
        let row = &model.w2[hidden * ACCUMULATOR_SIZE..][..ACCUMULATOR_SIZE];
        let mut sum = model.b2[hidden];
        for (weight, activation) in row.iter().zip(a1) {
            sum += *weight as i32 * activation;
        }
        *value = sum;
    }

    let a2 = activate_hidden(&h);

    let mut output_numerator = ACT_MAX * model.b3[0];
    for (weight, activation) in model.w3.iter().zip(a2) {
        output_numerator += *weight as i32 * activation;
    }

    ForwardTrace {
        a1,
        h,
        a2,
        output_numerator,
    }
}

/// The dispatched head shared by [`forward_numerator`] and
/// [`forward_trace_optimized`].
///
/// Returns `(a1, h, a2, output_numerator)` with the activations in their `i16`
/// kernel form. The arithmetic is exact (module-level proof in [`simd`]);
/// widening to `i32` is only needed by the trace view, so a production
/// evaluation does not pay for it.
fn forward_head(
    accumulator: &Accumulator,
) -> (
    [i16; ACCUMULATOR_SIZE],
    [i32; HIDDEN_SIZE],
    [i16; HIDDEN_SIZE],
    i32,
) {
    let model = model();
    let a1 = simd::first_activation(&accumulator.acc);
    let h = simd::hidden_matmul(&model.w2_i16, &model.b2, &a1);
    let a2 = simd::hidden_activation(&h);
    let output_numerator = simd::output_numerator(&model.w3, model.b3[0], &a2);
    (a1, h, a2, output_numerator)
}

/// Exact optimized counterpart of [`forward_trace`], exposed for parity tests.
///
/// Production reaches the same kernels through [`forward_numerator`]; this view
/// exists so tests can compare every intermediate tensor of the dispatched path
/// against the scalar oracle without a second implementation.
#[doc(hidden)]
pub fn forward_trace_optimized(accumulator: &Accumulator) -> ForwardTrace {
    let (a1, h, a2, output_numerator) = forward_head(accumulator);
    let mut a1_wide = [0i32; ACCUMULATOR_SIZE];
    for (wide, narrow) in a1_wide.iter_mut().zip(a1) {
        *wide = narrow as i32;
    }
    let mut a2_wide = [0i32; HIDDEN_SIZE];
    for (wide, narrow) in a2_wide.iter_mut().zip(a2) {
        *wide = narrow as i32;
    }
    ForwardTrace {
        a1: a1_wide,
        h,
        a2: a2_wide,
        output_numerator,
    }
}

/// Raw white-relative output numerator before centipawn conversion.
///
/// This runs the dispatched kernels: AVX2 when the CPU supports it at runtime,
/// otherwise the scalar reference. Results are bit-identical to
/// `forward_trace(accumulator).output_numerator`.
pub fn forward_numerator(accumulator: &Accumulator) -> i32 {
    #[cfg(feature = "profiling")]
    crate::profiling::record_forward();
    forward_head(accumulator).3
}

/// White-relative centipawn score for a ready accumulator.
///
/// `cp = round_ties_even(400 * out / 32512)`. The multiplication happens in
/// `i64` (see `div_ties_even_wide`) and the result is far below the reserved
/// mate band; search still clamps every evaluator result with
/// [`clamp_evaluation`].
pub fn forward(accumulator: &Accumulator) -> i32 {
    let scaled = div_ties_even_wide(
        forward_numerator(accumulator) as i64 * CP_MULTIPLIER as i64,
        CP_DENOMINATOR as i64,
    );
    scaled as i32
}

/// Stateless reference evaluator: full rebuild plus forward pass.
///
/// This path allocates nothing but recomputes the whole accumulator on every
/// call. It converts the White-relative output to side-to-move centipawns
/// (negated for Black) and clamps with [`clamp_evaluation`]. Search uses the
/// incremental session instead.
pub struct N2Evaluator;

impl Evaluator for N2Evaluator {
    fn evaluate(&self, position: &Position) -> i32 {
        let accumulator = Accumulator::from_position(position);
        let white_relative = forward(&accumulator);
        let side_to_move = if position.side_to_move().is_white() {
            white_relative
        } else {
            -white_relative
        };
        clamp_evaluation(side_to_move)
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Reader<'a> {
        Reader { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], AssetError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or(AssetError::TruncatedBytes)?;
        if end > self.bytes.len() {
            return Err(AssetError::TruncatedBytes);
        }
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, AssetError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn i32(&mut self) -> Result<i32, AssetError> {
        let bytes = self.take(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn string_bytes(&mut self) -> Result<&'a [u8], AssetError> {
        let length = self.u32()? as usize;
        self.take(length)
    }

    fn i16_vec(&mut self, length: usize) -> Result<Vec<i16>, AssetError> {
        let bytes = self.take(length * 2)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
    }

    fn i32_vec(&mut self, length: usize) -> Result<Vec<i32>, AssetError> {
        let bytes = self.take(length * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    fn i8_vec(&mut self, length: usize) -> Result<Vec<i8>, AssetError> {
        let bytes = self.take(length)?;
        Ok(bytes.iter().map(|byte| *byte as i8).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_even_rounds_half_to_even() {
        assert_eq!(div_ties_even(0, 2), 0);
        assert_eq!(div_ties_even(1, 2), 0);
        assert_eq!(div_ties_even(3, 2), 2);
        assert_eq!(div_ties_even(5, 2), 2);
        assert_eq!(div_ties_even(7, 2), 4);
        assert_eq!(div_ties_even(2, 2), 1);
    }

    #[test]
    fn ties_even_handles_negative_numerators_via_floor_division() {
        assert_eq!(div_ties_even(-1, 2), 0);
        assert_eq!(div_ties_even(-3, 2), -2);
        assert_eq!(div_ties_even(-5, 2), -2);
        assert_eq!(div_ties_even(-7, 2), -4);
        assert_eq!(div_ties_even(-2, 2), -1);
    }

    #[test]
    fn ties_even_matches_power_of_two_shift_definition() {
        for numerator in -4000i32..4000 {
            for shift in 1..=8u32 {
                let expected = {
                    let q = numerator >> shift;
                    let r = numerator & ((1 << shift) - 1);
                    let half = 1 << (shift - 1);
                    if r > half || (r == half && q % 2 != 0) {
                        q + 1
                    } else {
                        q
                    }
                };
                assert_eq!(
                    div_ties_even(numerator, 1 << shift),
                    expected,
                    "numerator {numerator} shift {shift}"
                );
            }
        }
    }

    #[test]
    fn accumulator_activation_clamp_and_rounding_boundaries() {
        let mut acc = [0i32; ACCUMULATOR_SIZE];
        acc[0] = -1;
        acc[1] = 0;
        acc[2] = 1;
        acc[3] = ACC_SCALE;
        acc[4] = ACC_SCALE + 1;
        acc[5] = 255;
        acc[6] = 256;
        let a1 = activate_accumulator(&acc);
        assert_eq!(a1[0], 0);
        assert_eq!(a1[1], 0);
        assert_eq!(a1[2], 0);
        assert_eq!(a1[3], ACT_MAX);
        assert_eq!(a1[4], ACT_MAX);
        // 255*127/256 = 126.5039..., remainder 129 > half 128, so 127.
        assert_eq!(a1[5], 127);
        assert_eq!(a1[6], ACT_MAX);
    }

    #[test]
    fn hidden_activation_clamp_and_rounding_boundaries() {
        let mut h = [0i32; HIDDEN_SIZE];
        h[0] = -1;
        h[1] = 0;
        h[2] = 1;
        h[3] = ACT_MAX * HIDDEN_SCALE;
        h[4] = ACT_MAX * HIDDEN_SCALE + 1;
        h[5] = 64;
        h[6] = 192;
        let a2 = activate_hidden(&h);
        assert_eq!(a2[0], 0);
        assert_eq!(a2[1], 0);
        assert_eq!(a2[2], 0);
        assert_eq!(a2[3], ACT_MAX);
        assert_eq!(a2[4], ACT_MAX);
        assert_eq!(a2[5], 0); // exact 0.5 ties to even 0
        assert_eq!(a2[6], 2); // exact 1.5 ties to even 2
    }

    #[test]
    fn packed_hidden_weights_are_the_lossless_i16_copy() {
        let model = model();
        assert_eq!(model.w2_i16.len(), model.w2.len());
        for (index, (packed, original)) in model.w2_i16.iter().zip(model.w2.iter()).enumerate() {
            assert_eq!(*packed, i16::from(*original), "w2 index {index}");
        }
        assert_eq!(model.w2_i16.len(), HIDDEN_SIZE * ACCUMULATOR_SIZE);
    }

    #[test]
    fn embedded_asset_metadata_and_provenance() {
        let meta = model().meta();
        assert_eq!(meta.model_id, MODEL_ID);
        assert_eq!(meta.schema_id, SCHEMA_ID);
        assert_eq!(
            (meta.in_features, meta.accumulator, meta.hidden, meta.out),
            (781, 256, 32, 1)
        );
        assert_eq!(meta.provenance_flags, 0);
        assert_eq!(provenance_flags(), 0);
        assert_eq!(score_perspective(), METADATA_SCORE_PERSPECTIVE);
        assert_eq!(DERIVED_SCORE_PERSPECTIVE, "derived: white-relative");
        assert_eq!(
            weights_sha256(),
            "8dcb4f2908c37cb7ce800eb3e6d170aedeb7a5ab89f2b41d2c57e32340454250"
        );
    }
}
