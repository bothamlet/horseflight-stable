#![forbid(unsafe_code)]

//! Horseflight Phase 1A: an original, dependency-free chess-state foundation.
//!
//! The crate provides typed squares/pieces/colours, a hybrid square-array plus
//! occupancy-bitboard board, original ray-walking legal move generation,
//! reversible in-place make/unmake with fixed-size undo and piece deltas,
//! deterministic Zobrist hashing, FEN parsing/serialization, history-aware draw
//! rules and perft. It contains no search, evaluation, neural-network, UCI or
//! tablebase infrastructure.
//!
//! # Future search and evaluation boundary
//!
//! A future searcher can own a [`Position`] and apply legal moves through
//! [`Position::play`], keeping the returned [`Undo`] records on its own LIFO
//! stack. [`Undo::piece_changes`] exposes the exact before/after change at every
//! touched square, so an incremental evaluator can replay forward changes on
//! make and reverse them on unmake without knowing move semantics. [`Game`] is
//! the history/draw-rule layer, not a search API. Nothing here presumes a
//! particular evaluator, search algorithm, transposition table or threading
//! model.
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
//! [`MoveError::CounterOverflow`] without mutation.
//!
//! Draw adjudication distinguishes automatic results (checkmate, stalemate,
//! conservative insufficient material, fivefold repetition, seventy-five-move)
//! from claimable rules (threefold repetition, fifty-move) in [`Game`]. The
//! material test is deliberately conservative and is not a complete FIDE dead
//! position proof.

mod attacks;
pub mod bitboard;
pub mod board;
pub mod fen;
mod hash;
pub mod history;
pub mod movegen;
pub mod perft;
pub mod position;
pub mod types;

pub use bitboard::Bitboard;
pub use board::{Board, BoardInvariantError};
pub use fen::FenError;
pub use hash::ZOBRIST_SEED;
pub use history::{DrawClaims, Game, Outcome};
pub use movegen::{legal_moves, resolve_legal_move, MoveList};
pub use perft::{perft, perft_divide, PerftError};
pub use position::{
    MoveError, PieceChange, Position, PositionInvariantError, Undo, UnmakeError, STARTPOS_FEN,
};
pub use types::{CastlingRights, CastlingSide, Color, Move, MoveKind, Piece, Role, Square};
