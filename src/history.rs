use core::fmt;

use crate::fen::FenError;
use crate::position::{MoveError, Position, Undo, UnmakeError};
use crate::types::{Color, Move};

/// Automatic game outcomes. Claim-based threefold and fifty-move draws are
/// reported separately by [`DrawClaims`]; they are not automatic results.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
    Ongoing,
    Checkmate { winner: Color },
    Stalemate,
    InsufficientMaterial,
    FivefoldRepetition,
    SeventyFiveMoveDraw,
}

impl Outcome {
    pub const fn is_over(self) -> bool {
        !matches!(self, Outcome::Ongoing)
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Outcome::Ongoing => f.write_str("ongoing"),
            Outcome::Checkmate { winner } => write!(f, "checkmate ({winner} wins)"),
            Outcome::Stalemate => f.write_str("stalemate"),
            Outcome::InsufficientMaterial => f.write_str("insufficient material"),
            Outcome::FivefoldRepetition => f.write_str("fivefold repetition"),
            Outcome::SeventyFiveMoveDraw => f.write_str("seventy-five-move draw"),
        }
    }
}

/// Rule-based draw claims available to the player to move.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct DrawClaims {
    /// The current position has occurred at least three times, or the intended
    /// move would create the third occurrence.
    pub threefold: bool,
    /// The current position has reached 100 halfmoves, or the intended move
    /// would reach 100 halfmoves without a capture or pawn move.
    pub fifty_move: bool,
}

impl DrawClaims {
    pub const fn none() -> DrawClaims {
        DrawClaims {
            threefold: false,
            fifty_move: false,
        }
    }

    pub const fn any(self) -> bool {
        self.threefold || self.fifty_move
    }
}

/// A position plus reversible game history.
///
/// The current [`Position`] is separate from history. The history stack keeps
/// exact positions for repetition identity and undo records for reversible
/// play; this is intentionally not just an irreversible game wrapper.
///
/// This is not a tournament controller: [`Game::play`] applies a legal move
/// reversibly and records it, but does not adjudicate claims or stop because an
/// automatic outcome has appeared. A caller driving a real game must inspect
/// [`Game::outcome`] and honor claims/outcomes itself.
#[derive(Clone, Debug)]
pub struct Game {
    current: Position,
    positions: Vec<Position>,
    undos: Vec<Undo>,
}

impl Game {
    pub fn new(start: Position) -> Game {
        Game {
            current: start,
            positions: vec![start],
            undos: Vec::new(),
        }
    }

    pub fn startpos() -> Game {
        Game::new(Position::startpos())
    }

    pub fn from_fen(fen: &str) -> Result<Game, FenError> {
        Ok(Game::new(Position::from_fen(fen)?))
    }

    pub const fn position(&self) -> &Position {
        &self.current
    }

    pub fn start_position(&self) -> Position {
        self.positions[0]
    }

    /// Exact position stack, starting with the initial position and ending with
    /// the current one.
    pub fn positions(&self) -> &[Position] {
        &self.positions
    }

    pub fn history_len(&self) -> usize {
        self.undos.len()
    }

    pub const fn side_to_move(&self) -> Color {
        self.current.side_to_move()
    }

    pub fn moves(&self) -> impl Iterator<Item = Move> + '_ {
        self.undos.iter().map(Undo::movement)
    }

    /// Play a legal move and record it reversibly. This is a board/history
    /// operation, not a tournament controller: it does not execute draw claims
    /// and does not refuse a move because the previous position is already over.
    /// Actual-game callers must honor [`Game::outcome`] and claim results.
    pub fn play(&mut self, movement: Move) -> Result<(), MoveError> {
        let undo = self.current.play(movement)?;
        self.undos.push(undo);
        self.positions.push(self.current);
        Ok(())
    }

    /// Undo the most recent move and restore the exact previous position.
    /// Returns the undone move, or `None` at the initial position.
    pub fn undo(&mut self) -> Result<Option<Move>, UnmakeError> {
        let Some(undo) = self.undos.pop() else {
            return Ok(None);
        };
        if let Err(error) = self.current.unmake(&undo) {
            self.undos.push(undo);
            return Err(error);
        }
        self.positions.pop();
        Ok(Some(undo.movement()))
    }

    /// Number of times the current position appears in the exact-position
    /// history, including the current one.
    pub fn current_repetition_count(&self) -> usize {
        self.positions
            .iter()
            .filter(|position| position.same_repetition_identity(&self.current))
            .count()
    }

    fn raw_threefold_claim(&self) -> bool {
        self.current_repetition_count() >= 3
    }

    fn raw_fifty_move_claim(&self) -> bool {
        self.current.halfmove_clock() >= 100
    }

    /// Threefold claim availability for the current position. False once an
    /// automatic outcome has already ended the game.
    pub fn threefold_claim_available(&self) -> bool {
        !self.outcome().is_over() && self.raw_threefold_claim()
    }

    pub fn fivefold_repetition(&self) -> bool {
        self.current_repetition_count() >= 5
    }

    /// Fifty-move claim availability for the current position. False once an
    /// automatic outcome has already ended the game.
    pub fn fifty_move_claim_available(&self) -> bool {
        !self.outcome().is_over() && self.raw_fifty_move_claim()
    }

    pub fn seventy_five_move_draw(&self) -> bool {
        self.current.halfmove_clock() >= 150
    }

    /// Claims available without moving. If the current game is already over by
    /// an automatic rule, no claim is available and the direct per-rule helpers
    /// agree.
    pub fn claimable_now(&self) -> DrawClaims {
        if self.outcome().is_over() {
            return DrawClaims::none();
        }
        DrawClaims {
            threefold: self.raw_threefold_claim(),
            fifty_move: self.raw_fifty_move_claim(),
        }
    }

    /// Claims available by writing an intended legal move and claiming before
    /// playing it. Order matters: if the current game is already over, no claim
    /// is available and the intended move is not validated. Otherwise the
    /// intended move is simulated without mutation
    /// and only the resulting threefold and 100-halfmove conditions are
    /// reported. A future mate, stalemate, fifth repetition or 150-halfmove
    /// threshold does not suppress an otherwise valid intended-move claim: if
    /// the claim is accepted, the game is immediately drawn before that move is
    /// played. If the claim is incorrect, only then is the move actually made.
    pub fn claimable_after(&self, movement: Move) -> Result<DrawClaims, MoveError> {
        if self.outcome().is_over() {
            return Ok(DrawClaims::none());
        }
        let mut next = self.current;
        next.play(movement)?;
        let repetitions = self
            .positions
            .iter()
            .filter(|position| position.same_repetition_identity(&next))
            .count()
            + 1;
        Ok(DrawClaims {
            threefold: repetitions >= 3,
            fifty_move: next.halfmove_clock() >= 100,
        })
    }

    /// Automatic outcome with mate precedence. Claim-based draws are not
    /// returned here; call [`Game::claimable_now`] or
    /// [`Game::claimable_after`] for those.
    pub fn outcome(&self) -> Outcome {
        if self.current.is_checkmate() {
            return Outcome::Checkmate {
                winner: self.current.side_to_move().opposite(),
            };
        }
        if self.current.is_stalemate() {
            return Outcome::Stalemate;
        }
        if self.current.has_insufficient_material() {
            return Outcome::InsufficientMaterial;
        }
        if self.seventy_five_move_draw() {
            return Outcome::SeventyFiveMoveDraw;
        }
        if self.fivefold_repetition() {
            return Outcome::FivefoldRepetition;
        }
        Outcome::Ongoing
    }
}
