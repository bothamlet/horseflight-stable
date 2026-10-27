#![forbid(unsafe_code)]

//! Horseflight: an original, dependency-free chess foundation and first search.
//!
//! Phase 1A provides typed squares/pieces/colours, a hybrid square-array plus
//! occupancy-bitboard board, original ray-walking legal move generation,
//! reversible in-place make/unmake with fixed-size undo and piece deltas,
//! deterministic Zobrist hashing, FEN parsing/serialization, history-aware draw
//! rules and perft. Phase 1B adds an original replaceable evaluation and a
//! full-window negamax alpha-beta search with iterative deepening, previous-PV
//! and MVV/LVA ordering and bounded quiescence.
//!
//! There is no transposition table, principal-variation search, aspiration
//! window, null move, late-move reduction, selective pruning, neural-network,
//! tablebase, UCI, GPU or threading infrastructure in this crate.
//!
//! # Search and evaluation boundary
//!
//! [`search::search`] takes a [`Game`], an [`eval::Evaluator`] and a
//! [`search::SearchConfig`], clones the game once and reports a truthful
//! [`search::SearchResult`]. Search own only rule adjudication and terminal
//! scoring; every leaf score comes from the evaluator. A caller that needs to
//! interrupt a running search can pass a [`search::SearchControl`] to
//! [`search::search_controlled`]; the token is standard-library-only, polled at
//! node and claim-probe boundaries, and a never-cancelled token leaves
//! [`search::search`] deterministic. Both search entry points are generic over
//! `E: Evaluator + ?Sized`, so a concrete evaluator avoids mandatory dynamic
//! dispatch while `&dyn Evaluator` callers keep working unchanged.
//! [`eval::features`] fixes the versioned 768-input binary piece-square schema
//! a future neural evaluator will consume, with allocation-free extraction and
//! delta updates. It is a boundary, not an implemented model.
//! [`Undo::piece_changes`] remains the interface a future incremental evaluator
//! can consume; a future transposition table can key on [`Position::hash`], but
//! draw-sensitive scores would also need the halfmove clock and path-dependent
//! repetition context.
//!
//! # Hashes versus repetition identity
//!
//! [`Position::hash`] includes piece placement, side to move, castling rights
//! and an en-passant file only when a fully legal en-passant capture exists. It
//! excludes both clocks and all history. [`Position::same_repetition_identity`]
//! compares exact state after a hash match, so a hash collision cannot by
//! itself be adjudicated as a draw.
//!
//! # FEN validation boundary
//!
//! FEN parsing validates structure and basic invariants: six fields, eight
//! ranks of width eight, legal piece characters, exactly one king per side, no
//! pawns on back ranks, castling rights backed by the king and rook on their
//! home squares, a structurally valid en-passant target and `u32` counters.
//! It does not prove reachability and does not reject every illegal chess
//! position (for example a side-not-to-move in check). Counters are `u32`; a
//! legal move that would overflow one is rejected with
//! [`MoveError::CounterOverflow`] without mutation; search reports that as a
//! [`search::SearchStop::Transition`] interruption rather than hiding it.
//!
//! Draw adjudication distinguishes automatic results (checkmate, stalemate,
//! conservative insufficient material, fivefold repetition, seventy-five-move)
//! from claimable rules (threefold repetition, fifty-move) in [`Game`]. The
//! material test is deliberately conservative and is not a complete FIDE dead
//! position proof. Search treats a claim as an optional zero-score action.

mod attacks;
pub mod bitboard;
pub mod board;
pub mod eval;
pub mod fen;
mod hash;
pub mod history;
pub mod movegen;
pub mod perft;
pub mod position;
pub mod search;
pub mod types;

pub use bitboard::Bitboard;
pub use board::{Board, BoardInvariantError};
pub use eval::{Evaluator, HandcraftedEvaluator, MaterialEvaluator};
pub use fen::FenError;
pub use hash::ZOBRIST_SEED;
pub use history::{DrawClaims, Game, Outcome};
pub use movegen::{legal_moves, resolve_legal_move, MoveList};
pub use perft::{perft, perft_divide, PerftError};
pub use position::{
    MoveError, PieceChange, Position, PositionInvariantError, Undo, UnmakeError, STARTPOS_FEN,
};
pub use search::{
    search, search_controlled, ConfigError, DrawClaim, SearchAction, SearchConfig, SearchControl,
    SearchResult, SearchStop,
};
pub use types::{CastlingRights, CastlingSide, Color, Move, MoveKind, Piece, Role, Square};
