use core::fmt;

use crate::position::{MoveError, Position, UnmakeError};
use crate::types::Move;

/// Errors from perft traversal. Counts are accumulated with checked arithmetic,
/// so an exhausted `u64` is reported rather than wrapped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PerftError {
    Move(MoveError),
    Unmake(UnmakeError),
    CountOverflow { depth: u32 },
}

impl fmt::Display for PerftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PerftError::Move(error) => write!(f, "perft move failed: {error}"),
            PerftError::Unmake(error) => write!(f, "perft unmake failed: {error}"),
            PerftError::CountOverflow { depth } => {
                write!(f, "perft count overflowed u64 at depth {depth}")
            }
        }
    }
}

impl std::error::Error for PerftError {}

/// Count legal move paths of exactly `depth` plies. Depth zero is one.
/// Draw adjudication is deliberately not applied: perft counts positions, not
/// game results.
pub fn perft(position: &Position, depth: u32) -> Result<u64, PerftError> {
    let mut position = *position;
    perft_inner(&mut position, depth)
}

/// Per-root-move counts. At depth zero there are no root moves, so the result is
/// empty; `perft(position, 1)` is the sum of a depth-one divide.
pub fn perft_divide(position: &Position, depth: u32) -> Result<Vec<(Move, u64)>, PerftError> {
    let mut position = *position;
    if depth == 0 {
        return Ok(Vec::new());
    }

    let moves = position.legal_moves();
    let mut result = Vec::with_capacity(moves.len());
    for movement in moves {
        let undo = position
            .make_unchecked(movement)
            .map_err(PerftError::Move)?;
        let count = perft_inner(&mut position, depth - 1)?;
        position.unmake(&undo).map_err(PerftError::Unmake)?;
        result.push((movement, count));
    }
    Ok(result)
}

fn perft_inner(position: &mut Position, depth: u32) -> Result<u64, PerftError> {
    if depth == 0 {
        return Ok(1);
    }

    let moves = position.legal_moves();
    let mut total = 0u64;
    for movement in moves {
        let undo = position
            .make_unchecked(movement)
            .map_err(PerftError::Move)?;
        let count = perft_inner(position, depth - 1)?;
        position.unmake(&undo).map_err(PerftError::Unmake)?;
        total = total
            .checked_add(count)
            .ok_or(PerftError::CountOverflow { depth })?;
    }
    Ok(total)
}
