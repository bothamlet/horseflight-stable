//! Original full-window negamax alpha-beta search with iterative deepening.
//!
//! The searcher owns no chess knowledge beyond rule adjudication: every leaf
//! score comes from an [`Evaluator`], and the handcrafted evaluator is only a
//! default. It reports a legal selected action, the side-to-move score, the
//! completed depth, a legal principal variation, node counts, a stop reason and
//! elapsed time.
//!
//! # Algorithm
//!
//! Iterative deepening calls a plain full-window negamax alpha-beta search with
//! fail-soft bounds. There is no transposition table, principal-variation
//! search, aspiration window, null move or selective pruning. Only the last
//! fully completed iteration's score, move and PV are retained if a later
//! iteration is interrupted. Move ordering (toggleable) tries the previous
//! iteration's PV move first when the search path matches that PV, then
//! MVV/LVA-style captures and promotions, then quiet moves; ordering off uses
//! move-generation order.
//!
//! # Quiescence
//!
//! At depth zero a toggleable quiescence search resolves captures (including
//! en passant), promotions (including quiet underpromotions) and, when in
//! check, **every** legal evasion. A checked node never stands pat and never
//! cuts off on a static score. Non-check frontiers stand pat and stop after at
//! most [`SearchConfig::q_depth_limit`] capture plies; that limit is a horizon
//! approximation, not an exact tactical resolution. Check evasions are still
//! searched at a q-depth boundary. Recursion is absolutely bounded by
//! [`MAX_PLY`]: a *nonterminal* frontier reached at the cap — checked or quiet —
//! aborts the iteration with [`SearchStop::PlyCap`] instead of a fabricated
//! static, exact or mate score, because the remaining continuations are unknown.
//! With quiescence
//! disabled, depth-zero pages return a static evaluation after rule checks: an
//! explicit diagnostic baseline, not a claim of tactical accuracy.
//!
//! # Terminal, draw and claim adjudication
//!
//! Rule state is checked before any evaluation, including at depth zero and at
//! q-depth limits, in the Phase 1A precedence order: checkmate, stalemate,
//! insufficient material, seventy-five-move, fivefold repetition. Scores are
//! root-relative: a loss at `ply` is `-MATE_SCORE + ply`, so shorter wins and
//! longer losses are preferred.
//!
//! Claimable threefold and fifty-move draws follow the Phase 1A intended-move
//! semantics and are modelled as **optional zero-score actions**: the side to
//! move may claim a draw, but search keeps any positive continuation. A current
//! claim needs only the position; an intended claim is probed on every quiet
//! legal move, including at a quiet q frontier, because a quiet move that
//! creates the third occurrence or reaches 100 halfmoves must not vanish just
//! because quiescence expands captures. A recommended claim is reported as
//! [`SearchAction::Claim`] and is never executed or placed in the PV.
//!
//! The repetition scan is skipped below four plies of quiet history: a
//! recurring board cannot be separated by a pawn move or a capture (pawns never
//! move backwards and captures do not undo), so any repetition sits inside the
//! current halfmove window at least four halfmoves after the previous
//! occurrence. Three occurrences therefore need at least eight quiet plies and
//! five need at least sixteen.
//!
//! # Budgets and purity
//!
//! [`SearchConfig::node_limit`] is a hard aggregate budget: [`SearchResult::nodes`]
//! counts every main and quiescence entry (the root of each iteration included)
//! and never exceeds it, while [`SearchResult::q_nodes`] is the subset that were
//! quiescence or q-disabled frontier entries. Rule-claim probes and PV bookkeeping
//! are not counted; they are not search nodes.
//!
//! Search clones the caller's [`Game`] once and works on the clone, so the
//! caller's position, history and hash are unchanged on completion, node/ply
//! stop and errors.
//!
//! # Approximations
//!
//! A bounded quiescence search is a horizon approximation. A draw claim is
//! optional, so search may decline a certain draw for a positive searched line
//! that later fails to hold. Claim handling here is a library policy, not a
//! tournament controller: no clocks, arbiter or interface state is modelled.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::eval::{self, Evaluator, INFINITY, MATE_SCORE};
use crate::history::{Game, Outcome};
use crate::position::Position;
use crate::types::{Move, Role};

/// Absolute recursion bound in plies, shared with the mate-score band.
pub const MAX_PLY: u32 = eval::MATE_MAX_PLY;

/// Largest accepted [`SearchConfig::max_depth`].
pub const MAX_MAIN_DEPTH: u32 = 96;

/// Largest accepted [`SearchConfig::q_depth_limit`].
pub const MAX_Q_DEPTH: u32 = 48;

/// Three occurrences cannot appear below this halfmove clock.
const THREEFOLD_MIN_CLOCK: u32 = 8;

/// An intended quiet move creates the third occurrence only above this clock.
const INTENDED_THREEFOLD_MIN_CLOCK: u32 = 7;

/// Five occurrences cannot appear below this halfmove clock.
const FIVEFOLD_MIN_CLOCK: u32 = 16;

/// Deterministic search configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchConfig {
    /// Requested iterative-deepening depth, at least one.
    pub max_depth: u32,
    /// Hard aggregate main-plus-quiescence node budget; zero means "no search".
    pub node_limit: u64,
    /// Previous-PV and MVV/LVA ordering; off uses move-generation order.
    pub ordering: bool,
    /// Quiescence search at depth zero; off is the static frontier baseline.
    pub quiescence: bool,
    /// Capture plies allowed at a non-check quiescence frontier.
    pub q_depth_limit: u32,
}

impl Default for SearchConfig {
    fn default() -> SearchConfig {
        SearchConfig {
            max_depth: 4,
            node_limit: 1_000_000,
            ordering: true,
            quiescence: true,
            q_depth_limit: 4,
        }
    }
}

impl SearchConfig {
    /// Validate the depth, quiescence and absolute-ply settings.
    pub fn validate(self) -> Result<SearchConfig, ConfigError> {
        if self.max_depth == 0 {
            return Err(ConfigError::ZeroDepth);
        }
        if self.max_depth > MAX_MAIN_DEPTH {
            return Err(ConfigError::DepthTooLarge {
                max_depth: self.max_depth,
                limit: MAX_MAIN_DEPTH,
            });
        }
        if self.q_depth_limit > MAX_Q_DEPTH {
            return Err(ConfigError::QDepthTooLarge {
                q_depth_limit: self.q_depth_limit,
                limit: MAX_Q_DEPTH,
            });
        }
        let needed = self.max_depth + self.q_depth_limit;
        if needed > MAX_PLY {
            return Err(ConfigError::PlyCapExceeded {
                needed,
                limit: MAX_PLY,
            });
        }
        Ok(self)
    }
}

/// Rejected search configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigError {
    ZeroDepth,
    DepthTooLarge { max_depth: u32, limit: u32 },
    QDepthTooLarge { q_depth_limit: u32, limit: u32 },
    PlyCapExceeded { needed: u32, limit: u32 },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            ConfigError::ZeroDepth => f.write_str("search depth must be at least one"),
            ConfigError::DepthTooLarge { max_depth, limit } => {
                write!(f, "search depth {max_depth} exceeds the limit of {limit}")
            }
            ConfigError::QDepthTooLarge {
                q_depth_limit,
                limit,
            } => write!(
                f,
                "quiescence depth {q_depth_limit} exceeds the limit of {limit}"
            ),
            ConfigError::PlyCapExceeded { needed, limit } => write!(
                f,
                "depth plus quiescence depth {needed} exceeds the {limit}-ply search bound"
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Why an iteration or the whole search stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchStop {
    /// The requested depth completed.
    DepthReached,
    /// The aggregate node budget ran out, possibly mid-iteration.
    NodeLimit,
    /// The root position is already an automatic game result; nothing was searched.
    TerminalRoot,
    /// The absolute ply cap was reached at a nonterminal frontier, checked or quiet.
    PlyCap,
    /// A position transition failed (counter overflow or an unmake mismatch).
    Transition,
    /// The caller's [`SearchControl`] requested cancellation.
    Cancelled,
}

impl fmt::Display for SearchStop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            SearchStop::DepthReached => "depth reached",
            SearchStop::NodeLimit => "node limit",
            SearchStop::TerminalRoot => "terminal root",
            SearchStop::PlyCap => "ply cap",
            SearchStop::Transition => "transition error",
            SearchStop::Cancelled => "cancelled",
        })
    }
}

/// A draw claim available to the side to move.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawClaim {
    /// The current position can be claimed as a draw without moving.
    Current,
    /// Writing this legal quiet move and claiming before playing it draws.
    /// The move is *not* played and must not be treated as the selected move.
    Intended(Move),
}

/// The recommended root action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchAction {
    /// Play this legal move: a searched choice.
    Move(Move),
    /// Claim a draw instead of moving.
    Claim(DrawClaim),
    /// The root is already an automatic result; no move is played.
    Terminal(Outcome),
    /// No iteration completed; this deterministic legal move is not a searched choice.
    Fallback(Move),
}

/// A complete, truthful search report.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Recommended action, distinguishing searched moves, claims, terminals and fallbacks.
    pub action: SearchAction,
    /// A draw claim available at the root, whether or not it was recommended.
    pub claim: Option<DrawClaim>,
    /// Side-to-move score in centipawns, or a mate score.
    ///
    /// `None` only for an unsearched fallback (`completed_depth == 0` with no root
    /// claim). A reported draw claim is exactly zero and needs no search, so it
    /// carries `Some(0)` even at `completed_depth == 0`.
    pub score: Option<i32>,
    /// Number of fully completed iterative-deepening depths.
    pub completed_depth: u32,
    /// Legal principal variation of the last completed iteration; empty for claims,
    /// terminals and fallbacks, and no quiescence moves are reported.
    pub pv: Vec<Move>,
    /// Aggregate main-plus-quiescence node count, never above the configured budget.
    pub nodes: u64,
    /// The part of [`SearchResult::nodes`] spent on quiescence or frontier entries.
    pub q_nodes: u64,
    /// Why the search stopped.
    pub stop: SearchStop,
    /// Wall-clock duration of this search call.
    pub elapsed: Duration,
}

impl SearchResult {
    /// Aggregate nodes per second, zero for a zero-length measurement.
    pub fn nps(&self) -> f64 {
        let seconds = self.elapsed.as_secs_f64();
        if seconds > 0.0 {
            self.nodes as f64 / seconds
        } else {
            0.0
        }
    }
}

/// A small, engine-independent cooperative cancellation token.
///
/// Cloning shares the flag, so a caller can hand one clone to a search thread
/// and cancel it from another thread. The standard library is the only
/// dependency. A token that is never cancelled leaves search deterministic and
/// identical to [`search`].
#[derive(Clone, Debug, Default)]
pub struct SearchControl {
    cancelled: Arc<AtomicBool>,
}

impl SearchControl {
    pub fn new() -> SearchControl {
        SearchControl::default()
    }

    /// Request cancellation. A running search stops at its next bounded check
    /// and restores any position it had temporarily played; a later search with
    /// this token does not begin claim probes.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// Search `game` without mutating it, returning a truthful report.
///
/// This entry point has no cancellation and is deterministic: adding
/// [`search_controlled`] does not change its results.
pub fn search(
    game: &Game,
    evaluator: &dyn Evaluator,
    config: &SearchConfig,
) -> Result<SearchResult, ConfigError> {
    search_inner(game, evaluator, config, None)
}

/// Search `game` cooperatively, stopping at a bounded check when `control` is
/// cancelled.
///
/// A cancelled search reports [`SearchStop::Cancelled`]. The last fully
/// completed iteration is retained exactly as for a node-limit stop; if no
/// iteration completed, the result is an unsearched legal [`SearchAction::Fallback`]
/// with no score. Cancellation is polled before and inside root/intended-claim
/// probes and at node boundaries, so a pre-cancelled search does not begin a
/// costly claim scan. The caller's game is never mutated.
pub fn search_controlled(
    game: &Game,
    evaluator: &dyn Evaluator,
    config: &SearchConfig,
    control: &SearchControl,
) -> Result<SearchResult, ConfigError> {
    search_inner(game, evaluator, config, Some(control))
}

fn search_inner(
    game: &Game,
    evaluator: &dyn Evaluator,
    config: &SearchConfig,
    control: Option<&SearchControl>,
) -> Result<SearchResult, ConfigError> {
    let config = config.validate()?;
    let mut worker = Worker {
        game: game.clone(),
        evaluator,
        config,
        control,
        nodes: 0,
        q_nodes: 0,
        previous_pv: Vec::new(),
        pv: vec![Vec::new(); MAX_PLY as usize + 1],
    };
    Ok(worker.run())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stop {
    NodeLimit,
    PlyCap,
    Transition,
    Cancelled,
}

impl From<Stop> for SearchStop {
    fn from(stop: Stop) -> SearchStop {
        match stop {
            Stop::NodeLimit => SearchStop::NodeLimit,
            Stop::PlyCap => SearchStop::PlyCap,
            Stop::Transition => SearchStop::Transition,
            Stop::Cancelled => SearchStop::Cancelled,
        }
    }
}

struct Iteration {
    value: i32,
    best_move: Option<Move>,
    pv: Vec<Move>,
}

struct Worker<'a> {
    game: Game,
    evaluator: &'a dyn Evaluator,
    config: SearchConfig,
    control: Option<&'a SearchControl>,
    nodes: u64,
    q_nodes: u64,
    previous_pv: Vec<Move>,
    pv: Vec<Vec<Move>>,
}

impl Worker<'_> {
    fn run(&mut self) -> SearchResult {
        let started = Instant::now();
        let outcome = self.game.outcome();
        if outcome.is_over() {
            let score = match outcome {
                Outcome::Checkmate { .. } => -MATE_SCORE,
                _ => 0,
            };
            return self.finish(
                SearchAction::Terminal(outcome),
                None,
                Some(score),
                0,
                Vec::new(),
                SearchStop::TerminalRoot,
                started,
            );
        }

        if self.cancelled() {
            return self.cancelled_without_iteration(started);
        }

        let root_position = *self.game.position();
        // The game is not over, so the root has at least one legal move.
        let root_moves: Vec<Move> = root_position.legal_moves().to_vec();
        let claim = match self.root_claim() {
            Ok(claim) => claim,
            Err(Stop::Cancelled) => return self.cancelled_without_iteration(started),
            Err(stop) => {
                let fallback = root_moves[0];
                return self.finish(
                    SearchAction::Fallback(fallback),
                    None,
                    None,
                    0,
                    Vec::new(),
                    stop.into(),
                    started,
                );
            }
        };
        let claim_zero = claim.is_some();

        let mut completed: Option<Iteration> = None;
        let mut completed_depth = 0;
        let mut stop = SearchStop::DepthReached;
        for depth in 1..=self.config.max_depth {
            match self.iterate(depth, claim_zero, &root_moves) {
                Ok(iteration) => {
                    self.previous_pv = iteration.pv.clone();
                    completed = Some(iteration);
                    completed_depth = depth;
                }
                Err(reason) => {
                    stop = reason.into();
                    break;
                }
            }
        }

        let Some(iteration) = completed else {
            // Nothing was searched. A rule claim is still a fact, so report it honestly
            // instead of pretending a fallback move is a claim or a claim is a search.
            if let Some(claim) = claim {
                return self.finish(
                    SearchAction::Claim(claim),
                    Some(claim),
                    Some(0),
                    0,
                    Vec::new(),
                    stop,
                    started,
                );
            }
            let fallback = root_moves[0];
            return self.finish(
                SearchAction::Fallback(fallback),
                None,
                None,
                0,
                Vec::new(),
                stop,
                started,
            );
        };

        if claim_zero && iteration.value <= 0 {
            let claim = claim.expect("a zero claim option implies a root claim");
            return self.finish(
                SearchAction::Claim(claim),
                Some(claim),
                Some(iteration.value),
                completed_depth,
                Vec::new(),
                stop,
                started,
            );
        }
        let movement = iteration
            .best_move
            .expect("a completed iteration without a claim option selects a move");
        self.finish(
            SearchAction::Move(movement),
            claim,
            Some(iteration.value),
            completed_depth,
            iteration.pv,
            stop,
            started,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        action: SearchAction,
        claim: Option<DrawClaim>,
        score: Option<i32>,
        completed_depth: u32,
        pv: Vec<Move>,
        stop: SearchStop,
        started: Instant,
    ) -> SearchResult {
        SearchResult {
            action,
            claim,
            score,
            completed_depth,
            pv,
            nodes: self.nodes,
            q_nodes: self.q_nodes,
            stop,
            elapsed: started.elapsed(),
        }
    }

    /// Current and intended-move claims at the root, using the Phase 1A rule API.
    ///
    /// Cancellation is polled before each costly intended-move simulation, so a
    /// token cancelled during this scan stops before another probe begins.
    fn root_claim(&mut self) -> Result<Option<DrawClaim>, Stop> {
        if self.cancelled() {
            return Err(Stop::Cancelled);
        }
        if self.game.claimable_now().any() {
            return Ok(Some(DrawClaim::Current));
        }
        let moves = self.game.position().legal_moves();
        for movement in moves.iter().copied() {
            if self.cancelled() {
                return Err(Stop::Cancelled);
            }
            if self
                .game
                .claimable_after(movement)
                .is_ok_and(|claims| claims.any())
            {
                return Ok(Some(DrawClaim::Intended(movement)));
            }
        }
        Ok(None)
    }

    fn cancelled(&self) -> bool {
        self.control.is_some_and(SearchControl::is_cancelled)
    }

    fn cancelled_without_iteration(&self, started: Instant) -> SearchResult {
        // The game is not over on this path, so the root has a legal move to
        // report as an explicitly unsearched fallback.
        let fallback = self.game.position().legal_moves()[0];
        self.finish(
            SearchAction::Fallback(fallback),
            None,
            None,
            0,
            Vec::new(),
            SearchStop::Cancelled,
            started,
        )
    }

    fn iterate(
        &mut self,
        depth: u32,
        claim_zero: bool,
        root_moves: &[Move],
    ) -> Result<Iteration, Stop> {
        self.reserve_node(false)?;
        for line in &mut self.pv {
            line.clear();
        }
        let hint = self.previous_pv.first().copied();
        let ordered = self.order_moves(self.game.position(), root_moves, hint);

        let mut best_value = -INFINITY;
        let mut best_move = None;
        let mut alpha = -INFINITY;
        if claim_zero {
            best_value = 0;
            alpha = 0;
        }
        for movement in ordered {
            let on_pv_path = hint == Some(movement);
            let score = -self.search_child(movement, |worker| {
                worker.negamax(1, depth - 1, -INFINITY, -alpha, on_pv_path)
            })?;
            if score > best_value {
                best_value = score;
                best_move = Some(movement);
            }
            if score > alpha {
                alpha = score;
                let (head, tail) = self.pv.split_at_mut(1);
                head[0].clear();
                head[0].push(movement);
                head[0].extend_from_slice(&tail[0]);
            }
        }
        Ok(Iteration {
            value: best_value,
            best_move,
            pv: self.pv[0].clone(),
        })
    }

    fn negamax(
        &mut self,
        ply: u32,
        depth: u32,
        alpha: i32,
        beta: i32,
        on_previous_pv: bool,
    ) -> Result<i32, Stop> {
        // Clear this ply's line first: terminal and leaf nodes return before
        // writing a PV, and a parent must never splice a stale sibling line.
        self.pv[ply as usize].clear();
        if depth == 0 {
            return if self.config.quiescence {
                self.quiescence(ply, self.config.q_depth_limit, alpha, beta)
            } else {
                self.evaluate_frontier(ply)
            };
        }
        self.reserve_node(false)?;
        if ply >= MAX_PLY {
            return Err(Stop::PlyCap);
        }

        let position = *self.game.position();
        let legal = position.legal_moves();
        let in_check = position.in_check();
        if legal.is_empty() {
            return Ok(if in_check {
                -MATE_SCORE + ply as i32
            } else {
                0
            });
        }
        if self.automatic_draw(&position) {
            return Ok(0);
        }

        let mut best = -INFINITY;
        let mut alpha = alpha;
        if self.claim_available(&position, &legal)? {
            if 0 >= beta {
                return Ok(0);
            }
            if 0 > alpha {
                alpha = 0;
            }
            best = 0;
        }

        let hint = if self.config.ordering && on_previous_pv {
            self.previous_pv.get(ply as usize).copied()
        } else {
            None
        };
        for movement in self.order_moves(&position, &legal, hint) {
            let on_pv_path = on_previous_pv && hint == Some(movement);
            let score = -self.search_child(movement, |worker| {
                worker.negamax(ply + 1, depth - 1, -beta, -alpha, on_pv_path)
            })?;
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    let (head, tail) = self.pv.split_at_mut(ply as usize + 1);
                    let line = &mut head[ply as usize];
                    line.clear();
                    line.push(movement);
                    line.extend_from_slice(&tail[0]);
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        Ok(best)
    }

    fn quiescence(
        &mut self,
        ply: u32,
        q_depth_left: u32,
        alpha: i32,
        beta: i32,
    ) -> Result<i32, Stop> {
        self.reserve_node(true)?;
        let position = *self.game.position();
        let legal = position.legal_moves();
        let in_check = position.in_check();
        if legal.is_empty() {
            return Ok(if in_check {
                -MATE_SCORE + ply as i32
            } else {
                0
            });
        }
        if self.automatic_draw(&position) {
            return Ok(0);
        }
        if ply >= MAX_PLY {
            // A nonterminal frontier cannot be resolved inside the ply cap, so the
            // iteration aborts instead of inventing a static, exact or mate score.
            return Err(Stop::PlyCap);
        }

        let mut best = -INFINITY;
        let mut alpha = alpha;
        if self.claim_available(&position, &legal)? {
            if 0 >= beta {
                return Ok(0);
            }
            if 0 > alpha {
                alpha = 0;
            }
            best = 0;
        }

        if !in_check {
            let stand_pat = self.static_evaluation(&position);
            if stand_pat >= beta {
                return Ok(stand_pat);
            }
            if stand_pat > alpha {
                alpha = stand_pat;
            }
            if stand_pat > best {
                best = stand_pat;
            }
            if q_depth_left == 0 {
                return Ok(best);
            }
        }

        let mut tactical = Vec::with_capacity(legal.len());
        for movement in legal.iter().copied() {
            if in_check || is_tactical(&position, movement) {
                tactical.push(movement);
            }
        }
        for movement in self.order_moves(&position, &tactical, None) {
            // Check evasions never consume capture-ply budget.
            let next_budget = if in_check {
                q_depth_left
            } else {
                q_depth_left - 1
            };
            let score = -self.search_child(movement, |worker| {
                worker.quiescence(ply + 1, next_budget, -beta, -alpha)
            })?;
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        Ok(best)
    }

    /// Depth-zero leaf when quiescence is disabled: rule checks, claims and a
    /// static evaluation only. This is a diagnostic baseline, not a search.
    fn evaluate_frontier(&mut self, ply: u32) -> Result<i32, Stop> {
        self.reserve_node(true)?;
        let position = *self.game.position();
        let legal = position.legal_moves();
        let in_check = position.in_check();
        if legal.is_empty() {
            return Ok(if in_check {
                -MATE_SCORE + ply as i32
            } else {
                0
            });
        }
        if self.automatic_draw(&position) {
            return Ok(0);
        }
        if ply >= MAX_PLY {
            return Err(Stop::PlyCap);
        }
        let claim = self.claim_available(&position, &legal)?;
        let static_score = self.static_evaluation(&position);
        Ok(if claim {
            static_score.max(0)
        } else {
            static_score
        })
    }

    fn reserve_node(&mut self, quiescence: bool) -> Result<(), Stop> {
        if self.cancelled() {
            return Err(Stop::Cancelled);
        }
        if self.nodes >= self.config.node_limit {
            return Err(Stop::NodeLimit);
        }
        self.nodes += 1;
        if quiescence {
            self.q_nodes += 1;
        }
        Ok(())
    }

    fn search_child(
        &mut self,
        movement: Move,
        search: impl FnOnce(&mut Self) -> Result<i32, Stop>,
    ) -> Result<i32, Stop> {
        self.game
            .play_generated(movement)
            .map_err(|_| Stop::Transition)?;
        let searched = search(self);
        let restored = self.game.undo();
        match (searched, restored) {
            (Ok(score), Ok(_)) => Ok(score),
            (Err(stop), _) => Err(stop),
            (Ok(_), Err(_)) => Err(Stop::Transition),
        }
    }

    fn static_evaluation(&self, position: &Position) -> i32 {
        eval::clamp_evaluation(self.evaluator.evaluate(position))
    }

    /// Automatic draws in Phase 1A precedence order, after checkmate/stalemate.
    fn automatic_draw(&self, position: &Position) -> bool {
        if position.has_insufficient_material() {
            return true;
        }
        if position.halfmove_clock() >= 150 {
            return true;
        }
        if position.halfmove_clock() < FIVEFOLD_MIN_CLOCK {
            return false;
        }
        self.repetition_count(position) >= 5
    }

    /// Occurrences of `position` in the exact position stack, including itself.
    ///
    /// A board that recurs cannot have been separated by a pawn move or a
    /// capture, so every occurrence sits inside the current halfmove window and
    /// at least four halfmoves after the previous one; below four quiet plies no
    /// repetition exists and the scan is skipped.
    fn repetition_count(&self, position: &Position) -> usize {
        if position.halfmove_clock() < 4 {
            return 1;
        }
        self.game
            .positions()
            .iter()
            .filter(|previous| previous.same_repetition_identity(position))
            .count()
    }

    /// Whether a current or intended-move draw claim is an available action.
    fn claim_available(&mut self, position: &Position, legal: &[Move]) -> Result<bool, Stop> {
        if self.current_claim(position) {
            return Ok(true);
        }
        self.intended_claim(position, legal)
    }

    fn current_claim(&self, position: &Position) -> bool {
        if position.halfmove_clock() >= 100 {
            return true;
        }
        if position.halfmove_clock() < THREEFOLD_MIN_CLOCK {
            return false;
        }
        self.repetition_count(position) >= 3
    }

    fn intended_claim(&mut self, position: &Position, legal: &[Move]) -> Result<bool, Stop> {
        let clock = position.halfmove_clock();
        let fifty = (99..u32::MAX).contains(&clock);
        let threefold = clock >= INTENDED_THREEFOLD_MIN_CLOCK;
        if !fifty && !threefold {
            return Ok(false);
        }
        for movement in legal.iter().copied() {
            if self.cancelled() {
                return Err(Stop::Cancelled);
            }
            if is_zeroing(position, movement) {
                continue;
            }
            if fifty {
                return Ok(true);
            }
            self.game
                .play_generated(movement)
                .map_err(|_| Stop::Transition)?;
            let repetitions = self.repetition_count(self.game.position());
            self.game.undo().map_err(|_| Stop::Transition)?;
            if repetitions >= 3 {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn order_moves(&self, position: &Position, legal: &[Move], hint: Option<Move>) -> Vec<Move> {
        if !self.config.ordering {
            return legal.to_vec();
        }
        let mut keyed: Vec<(i32, Move)> = legal
            .iter()
            .map(|&movement| (self.order_key(position, movement, hint), movement))
            .collect();
        keyed.sort_by_key(|entry| std::cmp::Reverse(entry.0));
        keyed.into_iter().map(|(_, movement)| movement).collect()
    }

    fn order_key(&self, position: &Position, movement: Move, hint: Option<Move>) -> i32 {
        if Some(movement) == hint {
            return i32::MAX;
        }
        let mut key = 0;
        if let Some(victim) = capture_victim(position, movement) {
            key += 16 * order_units(victim);
            if let Some(attacker) = position.board().piece_at(movement.from()) {
                key -= order_units(attacker.role);
            }
        }
        if let Some(promoted) = movement.promotion_role() {
            key += 8 * order_units(promoted);
        }
        key
    }
}

/// Capture target of `movement`, including an en-passant pawn.
fn capture_victim(position: &Position, movement: Move) -> Option<Role> {
    if movement.is_en_passant() {
        return Some(Role::Pawn);
    }
    position
        .board()
        .piece_at(movement.to())
        .map(|piece| piece.role)
}

fn is_tactical(position: &Position, movement: Move) -> bool {
    capture_victim(position, movement).is_some() || movement.is_promotion()
}

/// Move resets the halfmove clock: a pawn move or a capture.
fn is_zeroing(position: &Position, movement: Move) -> bool {
    if movement.is_en_passant() {
        return true;
    }
    let board = position.board();
    board
        .piece_at(movement.from())
        .is_some_and(|piece| piece.role == Role::Pawn)
        || board.piece_at(movement.to()).is_some()
}

/// MVV/LVA ordering units: rough relative piece values, not evaluation values.
const fn order_units(role: Role) -> i32 {
    match role {
        Role::Pawn => 1,
        Role::Knight | Role::Bishop => 3,
        Role::Rook => 5,
        Role::Queen => 9,
        Role::King => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{HandcraftedEvaluator, MaterialEvaluator};
    use crate::types::Square;
    use std::cell::Cell;

    /// Evaluator that requests cancellation after a fixed number of leaves.
    struct CancelAfter {
        control: SearchControl,
        remaining: Cell<u32>,
    }

    impl Evaluator for CancelAfter {
        fn evaluate(&self, position: &Position) -> i32 {
            let left = self.remaining.get();
            if left == 0 {
                self.control.cancel();
            } else {
                self.remaining.set(left - 1);
            }
            HandcraftedEvaluator.evaluate(position)
        }
    }

    fn make_worker(game: Game, evaluator: &'static dyn Evaluator) -> Worker<'static> {
        Worker {
            game,
            evaluator,
            config: SearchConfig {
                max_depth: 1,
                node_limit: 100_000,
                ordering: true,
                quiescence: true,
                q_depth_limit: 2,
            },
            control: None,
            nodes: 0,
            q_nodes: 0,
            previous_pv: Vec::new(),
            pv: vec![Vec::new(); MAX_PLY as usize + 1],
        }
    }

    fn play_uci(game: &mut Game, text: &str) {
        let movement = game
            .position()
            .legal_moves()
            .iter()
            .copied()
            .find(|movement| movement.to_uci() == text)
            .unwrap_or_else(|| panic!("{text} is not legal in {}", game.position().to_fen()));
        game.play(movement).expect("coordinate move is legal");
    }

    #[test]
    fn a_pre_cancelled_search_does_not_probe_or_search() {
        let control = SearchControl::new();
        control.cancel();
        // A position with seven quiet plies would otherwise trigger the costly
        // intended-claim scan at the root.
        let mut history = Game::from_fen("7k/8/8/8/8/8/8/K1Q5 w - - 0 1").unwrap();
        for text in ["a1a2", "h8g8", "a2a1", "g8h8", "a1a2", "h8g8", "a2a1"] {
            play_uci(&mut history, text);
        }
        assert_eq!(history.position().halfmove_clock(), 7);
        let before = history.clone();
        let result = search_controlled(
            &history,
            &HandcraftedEvaluator,
            &SearchConfig::default(),
            &control,
        )
        .expect("a valid configuration is still required");
        assert_eq!(result.stop, SearchStop::Cancelled);
        assert_eq!(result.nodes, 0);
        assert_eq!(result.completed_depth, 0);
        assert_eq!(result.score, None);
        let expected = history.position().legal_moves()[0];
        assert_eq!(result.action, SearchAction::Fallback(expected));
        assert_eq!(history.position().to_fen(), before.position().to_fen());
        assert_eq!(history.history_len(), before.history_len());
    }

    #[test]
    fn cancelling_an_intended_claim_scan_restores_the_worker() {
        let mut history = Game::from_fen("7k/8/8/8/8/8/8/K1Q5 w - - 0 1").unwrap();
        for text in ["a1a2", "h8g8", "a2a1", "g8h8", "a1a2", "h8g8", "a2a1"] {
            play_uci(&mut history, text);
        }
        let control = SearchControl::new();
        control.cancel();
        let mut worker = Worker {
            game: history,
            evaluator: &HandcraftedEvaluator,
            config: SearchConfig::default(),
            control: Some(&control),
            nodes: 0,
            q_nodes: 0,
            previous_pv: Vec::new(),
            pv: vec![Vec::new(); MAX_PLY as usize + 1],
        };
        let before_fen = worker.game.position().to_fen();
        let before_history = worker.game.history_len();
        let before_positions = worker.game.positions().len();
        let position = *worker.game.position();
        let legal = position.legal_moves();
        assert_eq!(
            worker.intended_claim(&position, &legal),
            Err(Stop::Cancelled)
        );
        assert_eq!(worker.game.position().to_fen(), before_fen);
        assert_eq!(worker.game.history_len(), before_history);
        assert_eq!(worker.game.positions().len(), before_positions);
    }

    #[test]
    fn a_mid_search_cancellation_stops_at_a_bounded_check_and_restores_history() {
        let control = SearchControl::new();
        let evaluator = CancelAfter {
            control: control.clone(),
            remaining: Cell::new(200),
        };
        let game =
            Game::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                .unwrap();
        let before = game.clone();
        let config = SearchConfig {
            max_depth: 8,
            node_limit: 2_000_000,
            ordering: true,
            quiescence: true,
            q_depth_limit: 4,
        };
        let result = search_controlled(&game, &evaluator, &config, &control)
            .expect("the configuration is valid");
        assert_eq!(result.stop, SearchStop::Cancelled);
        assert!(result.nodes > 0);
        assert!(result.completed_depth < config.max_depth);
        if result.completed_depth > 0 {
            assert!(result.score.is_some());
        }
        // Purity: the caller's game, history and repetition context are untouched.
        assert_eq!(game.position().to_fen(), before.position().to_fen());
        assert_eq!(game.history_len(), before.history_len());
        assert_eq!(game.positions().len(), before.positions().len());
        assert_eq!(
            game.current_repetition_count(),
            before.current_repetition_count()
        );
    }

    #[test]
    fn a_cancelled_child_unwinds_without_mutating_the_worker() {
        let control = SearchControl::new();
        let evaluator: &'static dyn Evaluator = &HandcraftedEvaluator;
        let mut worker = Worker {
            game: Game::from_fen("7k/8/8/8/8/8/8/K1Q5 w - - 0 1").unwrap(),
            evaluator,
            config: SearchConfig::default(),
            control: Some(&control),
            nodes: 0,
            q_nodes: 0,
            previous_pv: Vec::new(),
            pv: vec![Vec::new(); MAX_PLY as usize + 1],
        };
        let movement = worker.game.position().legal_moves()[0];
        let before_fen = worker.game.position().to_fen();
        let before_history = worker.game.history_len();
        let stop = worker
            .search_child(movement, |inner| {
                inner.control.expect("control is installed").cancel();
                Err(Stop::Cancelled)
            })
            .expect_err("cancellation propagates");
        assert!(matches!(stop, Stop::Cancelled));
        assert_eq!(worker.game.position().to_fen(), before_fen);
        assert_eq!(worker.game.history_len(), before_history);
    }

    #[test]
    fn the_absolute_ply_cap_aborts_a_nonterminal_frontier() {
        let evaluator: &'static dyn Evaluator = &HandcraftedEvaluator;
        // Quiet frontier one ply inside the cap: a stand-pat score is legal.
        let mut inside = make_worker(
            Game::from_fen("7k/8/8/8/8/8/8/K1Q5 w - - 0 1").unwrap(),
            evaluator,
        );
        let inside_score = inside
            .quiescence(MAX_PLY - 1, 1, -INFINITY, INFINITY)
            .expect("one ply inside the cap must score");
        assert!(inside_score.abs() <= eval::EVAL_MAX);

        // The same nonterminal pattern at the cap must abort, not invent a score.
        let mut quiet = make_worker(
            Game::from_fen("7k/8/8/8/8/8/8/K1Q5 w - - 0 1").unwrap(),
            evaluator,
        );
        assert!(matches!(
            quiet.quiescence(MAX_PLY, 1, -INFINITY, INFINITY),
            Err(Stop::PlyCap)
        ));

        // A checked nonterminal frontier at the cap must not fabricate a
        // static or mate score either.
        let mut checked = make_worker(
            Game::from_fen("k7/8/Q1K5/8/8/8/8/8 b - - 1 1").unwrap(),
            evaluator,
        );
        assert!(checked.game.position().in_check());
        assert!(!checked.game.position().legal_moves().is_empty());
        assert!(matches!(
            checked.quiescence(MAX_PLY, 2, -INFINITY, INFINITY),
            Err(Stop::PlyCap)
        ));

        // A terminal frontier at the cap is resolved exactly, not aborted.
        let mut mated = make_worker(
            Game::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").unwrap(),
            evaluator,
        );
        assert_eq!(
            mated
                .quiescence(MAX_PLY, 1, -INFINITY, INFINITY)
                .expect("checkmate is exact at any ply"),
            -MATE_SCORE + MAX_PLY as i32
        );

        // The internal stop maps to the public stop reason reported by search.
        assert_eq!(SearchStop::from(Stop::PlyCap), SearchStop::PlyCap);
        assert_eq!(SearchStop::from(Stop::NodeLimit), SearchStop::NodeLimit);
        assert_eq!(SearchStop::from(Stop::Transition), SearchStop::Transition);
        assert_eq!(SearchStop::from(Stop::Cancelled), SearchStop::Cancelled);
    }

    #[test]
    fn a_ply_cap_stop_unwinds_the_worker_without_mutating_it() {
        let mut worker = make_worker(
            Game::from_fen("7k/8/8/8/8/8/8/K1Q5 w - - 0 1").unwrap(),
            &HandcraftedEvaluator,
        );
        let movement = worker.game.position().legal_moves()[0];
        let before_fen = worker.game.position().to_fen();
        let before_hash = worker.game.position().hash();
        let before_history = worker.game.history_len();
        let before_positions = worker.game.positions().len();
        let stop = worker
            .search_child(movement, |inner| {
                inner.quiescence(MAX_PLY, 1, -INFINITY, INFINITY)
            })
            .expect_err("the capped frontier must abort");
        assert!(matches!(stop, Stop::PlyCap));
        assert_eq!(worker.game.position().to_fen(), before_fen);
        assert_eq!(worker.game.position().hash(), before_hash);
        assert_eq!(worker.game.history_len(), before_history);
        assert_eq!(worker.game.positions().len(), before_positions);
        assert_eq!(worker.game.current_repetition_count(), 1);
    }

    #[test]
    fn a_quiet_frontier_honours_an_intended_threefold_claim() {
        // Real history: white shuffles, and black is to move at seven halfmoves
        // where the quiet g8h8 writes the third occurrence.
        let mut history = Game::from_fen("7k/8/8/8/8/8/8/K1Q5 w - - 0 1").unwrap();
        for text in ["a1a2", "h8g8", "a2a1", "g8h8", "a1a2", "h8g8", "a2a1"] {
            play_uci(&mut history, text);
        }
        assert_eq!(history.position().halfmove_clock(), 7);
        assert!(!history.claimable_now().any(), "no current claim yet");
        let intended_move = history
            .position()
            .legal_moves()
            .iter()
            .copied()
            .find(|movement| movement.to_uci() == "g8h8")
            .expect("g8h8 is legal");
        assert!(history.claimable_after(intended_move).unwrap().threefold);

        let mut worker = make_worker(history.clone(), &HandcraftedEvaluator);
        let position = *worker.game.position();
        assert!(!worker.current_claim(&position));
        assert!(worker
            .intended_claim(&position, &position.legal_moves())
            .unwrap());
        assert_eq!(
            worker.quiescence(2, 2, -INFINITY, INFINITY).unwrap(),
            0,
            "the quiet intended claim is a zero-score option at the frontier"
        );

        // The identical board without history is simply lost for black, so the
        // zero above can only come from the intended-claim probe.
        let fresh_game = Game::from_fen(&history.position().to_fen()).unwrap();
        let expected_static =
            eval::clamp_evaluation(eval::HandcraftedEvaluator.evaluate(fresh_game.position()));
        let mut fresh = make_worker(fresh_game, &HandcraftedEvaluator);
        let fresh_score = fresh.quiescence(2, 2, -INFINITY, INFINITY).unwrap();
        assert_eq!(fresh_score, expected_static);
        assert!(fresh_score < -500, "{fresh_score}");
    }

    #[test]
    fn a_frontier_reports_automatic_results_before_evaluation() {
        let evaluator: &'static dyn Evaluator = &HandcraftedEvaluator;
        let cases = [
            // Checkmate for the side to move: mate distance carries the ply.
            ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", -MATE_SCORE + 3),
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 0),
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0),
            ("8/8/4k3/8/8/4K3/8/6R1 w - - 150 100", 0),
        ];
        for (fen, expected) in cases {
            let mut worker = make_worker(Game::from_fen(fen).unwrap(), evaluator);
            let score = worker.quiescence(3, 2, -INFINITY, INFINITY).unwrap();
            assert_eq!(score, expected, "{fen}");
        }
    }

    #[test]
    fn a_frontier_treats_a_draw_claim_as_a_zero_option() {
        // Black is a queen down but can claim the fifty-move draw.
        let mut worker = make_worker(
            Game::from_fen("7k/8/8/8/q7/8/8/7K w - - 100 60").unwrap(),
            &MaterialEvaluator,
        );
        assert_eq!(worker.quiescence(1, 2, -INFINITY, INFINITY).unwrap(), 0);
        // Without the claim the same board is hopeless for white.
        let mut quiet = make_worker(
            Game::from_fen("7k/8/8/8/q7/8/8/7K w - - 0 60").unwrap(),
            &MaterialEvaluator,
        );
        let score = quiet.quiescence(1, 2, -INFINITY, INFINITY).unwrap();
        assert_eq!(score, -900);
    }

    #[test]
    fn the_frontier_tactical_filter_selects_captures_and_promotions() {
        let position = Position::from_fen("4k3/8/8/3pP3/8/8/8/4K1N1 w - d6 0 1").unwrap();
        let mut found_ep = false;
        let mut found_quiet = false;
        for movement in position.legal_moves().iter().copied() {
            if movement.is_en_passant() {
                found_ep = true;
                assert_eq!(capture_victim(&position, movement), Some(Role::Pawn));
                assert!(is_tactical(&position, movement));
                assert!(is_zeroing(&position, movement));
            } else if movement.to_uci() == "g1f3" {
                found_quiet = true;
                assert_eq!(capture_victim(&position, movement), None);
                assert!(!is_tactical(&position, movement));
                assert!(!is_zeroing(&position, movement));
            }
        }
        assert!(found_ep && found_quiet);

        let promoting = Position::from_fen("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let promotion = promoting
            .legal_moves()
            .iter()
            .copied()
            .find(|movement| movement.promotion_role() == Some(Role::Knight))
            .expect("a quiet underpromotion exists");
        assert_eq!(capture_victim(&promoting, promotion), None);
        assert!(is_tactical(&promoting, promotion));
        assert!(is_zeroing(&promoting, promotion));

        let castling = Position::from_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1").unwrap();
        let castle = castling
            .legal_moves()
            .iter()
            .copied()
            .find(|movement| movement.is_castle())
            .expect("castling is legal");
        assert!(!is_tactical(&castling, castle));
        assert!(!is_zeroing(&castling, castle));
        assert_eq!(castle.from(), Square::from_ascii("e1").unwrap());
    }
}
