//! Incremental NNUE evaluation with reversible move updates.
//!
//! [`Session`] owns one [`Accumulator`] plus a fixed-capacity undo stack. Search
//! resets it once at the root, then calls [`EvaluationSession::on_make`] after
//! every successful move and [`EvaluationSession::on_unmake`] on every exit
//! path, so evaluation is a forward pass over a maintained accumulator instead
//! of a full rebuild per leaf. The stack is a fixed array sized for
//! [`crate::search::MAX_PLY`] plus four spare levels; make and unmake allocate
//! nothing.
//!
//! Each undo record is the compact delta of one move: the board-square
//! [`PieceChange`]s from [`crate::position::Undo::piece_changes`] plus the
//! metadata input indices (side to move, castling rights, legal en-passant
//! file) that the move activated or deactivated. Unmake removes exactly the
//! inputs the make added and restores exactly the ones it removed, so the
//! accumulator is bit-identical to a full rebuild rather than merely close.
//! [`Session::matches_rebuild`] is the reference check used by tests.

use crate::eval::n2::{
    piece_feature_index, Accumulator, CASTLING_FEATURE_BASE, EP_FEATURE_BASE, WHITE_TO_MOVE_FEATURE,
};
use crate::eval::{clamp_evaluation, EvaluationSession};
use crate::position::{PieceChange, Position, MAX_PIECE_CHANGES};
use crate::search::MAX_PLY;
use crate::types::Square;

/// Undo levels reserved at construction: the absolute search ply bound plus
/// spare levels for a claim probe made at the deepest search ply.
pub const MAX_SESSION_DEPTH: usize = MAX_PLY as usize + 4;

/// At most six metadata inputs are active: side to move, four castling rights
/// and one legal en-passant file.
const MAX_METADATA_FEATURES: usize = 6;

/// One move's compact, reversible delta.
#[derive(Clone, Copy, Debug)]
struct Delta {
    changes: [PieceChange; MAX_PIECE_CHANGES],
    change_len: usize,
    /// Metadata inputs the move activated.
    added: [u16; MAX_METADATA_FEATURES],
    added_len: usize,
    /// Metadata inputs the move deactivated.
    removed: [u16; MAX_METADATA_FEATURES],
    removed_len: usize,
}

const EMPTY_DELTA: Delta = Delta {
    changes: [PieceChange {
        square: Square::A1,
        before: None,
        after: None,
    }; MAX_PIECE_CHANGES],
    change_len: 0,
    added: [0; MAX_METADATA_FEATURES],
    added_len: 0,
    removed: [0; MAX_METADATA_FEATURES],
    removed_len: 0,
};

/// Non-piece inputs: side to move, castling rights, and a legal en-passant file.
#[derive(Clone, Copy, PartialEq, Eq)]
struct MetaInputs {
    white_to_move: bool,
    castling: [bool; 4],
    ep_file: Option<u8>,
}

impl MetaInputs {
    fn read(position: &Position) -> MetaInputs {
        let rights = position.castling_rights();
        MetaInputs {
            white_to_move: position.side_to_move().is_white(),
            castling: [
                rights.has(
                    crate::types::Color::White,
                    crate::types::CastlingSide::KingSide,
                ),
                rights.has(
                    crate::types::Color::White,
                    crate::types::CastlingSide::QueenSide,
                ),
                rights.has(
                    crate::types::Color::Black,
                    crate::types::CastlingSide::KingSide,
                ),
                rights.has(
                    crate::types::Color::Black,
                    crate::types::CastlingSide::QueenSide,
                ),
            ],
            ep_file: position.legal_ep_square().map(|square| square.file()),
        }
    }

    /// Active metadata input indices, ascending as the schema orders them.
    fn features(&self) -> ([u16; MAX_METADATA_FEATURES], usize) {
        let mut features = [0u16; MAX_METADATA_FEATURES];
        let mut len = 0;
        if self.white_to_move {
            features[len] = WHITE_TO_MOVE_FEATURE as u16;
            len += 1;
        }
        for (offset, active) in self.castling.iter().enumerate() {
            if *active {
                features[len] = (CASTLING_FEATURE_BASE + offset) as u16;
                len += 1;
            }
        }
        if let Some(file) = self.ep_file {
            features[len] = (EP_FEATURE_BASE + file as usize) as u16;
            len += 1;
        }
        (features, len)
    }
}

/// Incremental evaluation state for one search.
pub struct Session {
    accumulator: Accumulator,
    undo: [Delta; MAX_SESSION_DEPTH],
    depth: usize,
}

impl Session {
    /// Build a session for `root` with an empty undo stack.
    pub fn new(root: &Position) -> Session {
        Session {
            accumulator: Accumulator::from_position(root),
            undo: [EMPTY_DELTA; MAX_SESSION_DEPTH],
            depth: 0,
        }
    }

    /// The maintained accumulator.
    pub fn accumulator(&self) -> &Accumulator {
        &self.accumulator
    }

    /// Outstanding make levels: zero exactly when every make was unmatched.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Profiling-only: force scalar accumulator row kernels on this session.
    #[cfg(feature = "profiling")]
    pub fn set_scalar_rows(&mut self, enabled: bool) {
        self.accumulator.set_scalar_rows(enabled);
    }

    /// Whether the maintained accumulator equals a full rebuild of `position`.
    ///
    /// Test helper and debug oracle; search never calls it.
    pub fn matches_rebuild(&self, position: &Position) -> bool {
        self.accumulator.values() == Accumulator::from_position(position).values()
    }
}

impl EvaluationSession for Session {
    fn evaluate(&self, position: &Position) -> i32 {
        let white_relative = crate::eval::n2::forward(&self.accumulator);
        let side_to_move = if position.side_to_move().is_white() {
            white_relative
        } else {
            -white_relative
        };
        clamp_evaluation(side_to_move)
    }

    fn on_make(&mut self, changes: &[PieceChange], before: &Position, after: &Position) {
        #[cfg(feature = "profiling")]
        let started = std::time::Instant::now();
        assert!(
            changes.len() <= MAX_PIECE_CHANGES,
            "a chess move changes at most {MAX_PIECE_CHANGES} squares"
        );
        self.accumulator.apply_move(changes, before, after);

        let (before_features, before_len) = MetaInputs::read(before).features();
        let (after_features, after_len) = MetaInputs::read(after).features();
        let mut delta = EMPTY_DELTA;
        delta.change_len = changes.len();
        delta.changes[..changes.len()].copy_from_slice(changes);
        for &feature in &after_features[..after_len] {
            if !before_features[..before_len].contains(&feature) {
                delta.added[delta.added_len] = feature;
                delta.added_len += 1;
            }
        }
        for &feature in &before_features[..before_len] {
            if !after_features[..after_len].contains(&feature) {
                delta.removed[delta.removed_len] = feature;
                delta.removed_len += 1;
            }
        }

        assert!(
            self.depth < MAX_SESSION_DEPTH,
            "evaluation session undo overflow at depth {}",
            self.depth
        );
        self.undo[self.depth] = delta;
        self.depth += 1;
        #[cfg(feature = "profiling")]
        {
            crate::profiling::record_make();
            crate::profiling::record_make_time(started.elapsed());
        }
    }

    fn on_unmake(&mut self) {
        #[cfg(feature = "profiling")]
        let started = std::time::Instant::now();
        assert!(
            self.depth > 0,
            "evaluation session unmake without a matching make"
        );
        self.depth -= 1;
        let delta = self.undo[self.depth];
        for &feature in &delta.added[..delta.added_len] {
            self.accumulator.sub_row(feature as usize);
        }
        for &feature in &delta.removed[..delta.removed_len] {
            self.accumulator.add_row(feature as usize);
        }
        for change in delta.changes[..delta.change_len].iter().rev() {
            if let Some(piece) = change.after {
                self.accumulator
                    .sub_row(piece_feature_index(piece, change.square));
            }
            if let Some(piece) = change.before {
                self.accumulator
                    .add_row(piece_feature_index(piece, change.square));
            }
        }
        #[cfg(feature = "profiling")]
        {
            crate::profiling::record_unmake();
            crate::profiling::record_unmake_time(started.elapsed());
        }
    }

    fn reset(&mut self, root: &Position) {
        self.accumulator.rebuild(root);
        self.depth = 0;
    }
}
