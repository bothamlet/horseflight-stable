//! Profiling-only instrumentation (Cargo feature `profiling`, off by default).
//!
//! This module is compiled only when `--features profiling` is passed, so a
//! normal build contains neither these counters nor any hook that updates them.
//! It exists because a sampling profiler is unavailable on this host
//! (`perf_event_paranoid = 4`), so Phase A1 measures search call distribution
//! and per-evaluation time with explicit counters and timers instead.
//!
//! # What is counted
//!
//! [`Snapshot`] reports the counted events of one measured search: search nodes
//! split into main and quiescence/frontier entries, static evaluations, neural
//! head (`forward`) calls, incremental make/unmake/delta updates, full
//! accumulator rebuilds, made positions whose subtree never performed a static
//! evaluation, and how many static evaluations landed on a position hash this
//! search had already evaluated. The last number is the only table-reuse signal
//! available: the engine has no transposition table, so it is a count of *would
//! have been* repeat evaluations, not a reuse the search performs.
//!
//! # Cost this instrumentation adds (must be disclosed, never subtracted)
//!
//! - one relaxed atomic addition per counted event;
//! - two `Instant::now()` calls per timed evaluation and per make/unmake;
//! - one thread-local `HashSet<u64>` insert plus lookup per static evaluation to
//!   detect repeated positions;
//! - one bool push/pop per made frame and one bool write per static evaluation to
//!   track whether that made position's own accumulator was evaluated.
//!
//! Because of these costs an instrumented run's timings and NPS are *not*
//! comparable with an uninstrumented run's. Instrumented runs are used for
//! distribution, the uninstrumented tree for throughput, and the two are
//! compared only on deterministic search results (score, move, PV, nodes).
//!
//! Counters are process-global and updated with `Relaxed` ordering, which is
//! exact for the single-threaded measured searches and never a claim about
//! multi-threaded sums by itself.

use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

macro_rules! counters {
    ($($counter:ident => $field:ident),* $(,)?) => {
        $(
            static $counter: AtomicU64 = AtomicU64::new(0);
        )*

        /// Reset every counter and the repeated-position set.
        pub fn reset() {
            $(
                $counter.store(0, Ordering::Relaxed);
            )*
            EVALUATED_HASHES.with(|hashes| hashes.borrow_mut().clear());
            MADE_FRAMES.with(|frames| frames.borrow_mut().evaluated.clear());
        }

        /// One plain load per counter; no field is derived or timed here.
        pub fn snapshot() -> Snapshot {
            Snapshot {
                $(
                    $field: $counter.load(Ordering::Relaxed),
                )*
            }
        }
    };
}

counters!(
    NODES_MAIN => nodes_main,
    NODES_QUIESCENCE => nodes_quiescence,
    NODES_FRONTIER => nodes_frontier,
    STATIC_EVALUATIONS => static_evaluations,
    FORWARD_CALLS => forward_calls,
    MAKE_CALLS => make_calls,
    UNMAKE_CALLS => unmake_calls,
    REBUILDS => rebuilds,
    PREPARED_UNEVALUATED_SUBTREE => prepared_unevaluated_subtree,
    PREPARED_UNEVALUATED_OWN => prepared_unevaluated_own,
    STATIC_EVALUATIONS_QUIESCENCE => static_evaluations_quiescence,
    STATIC_EVALUATIONS_FRONTIER => static_evaluations_frontier,
    REPEATED_EVALUATIONS => repeated_evaluations,
    UNIQUE_EVALUATIONS => unique_evaluations,
    EVALUATION_TIME_NS => evaluation_time_ns,
    MAKE_TIME_NS => make_time_ns,
    UNMAKE_TIME_NS => unmake_time_ns,
);

/// Per-thread stack of outstanding made frames and whether each frame's own
/// accumulator has been evaluated.
///
/// Kept as a plain struct so the frame semantics (a descendant evaluating does
/// not mark its ancestors) are unit-testable without global counters.
#[derive(Debug, Default)]
struct MadeFrames {
    evaluated: Vec<bool>,
}

impl MadeFrames {
    fn push(&mut self) {
        self.evaluated.push(false);
    }

    fn mark_current(&mut self) {
        if let Some(evaluated) = self.evaluated.last_mut() {
            *evaluated = true;
        }
    }

    /// Pop one frame and report whether its own accumulator was never
    /// evaluated. `None` means an unmake without a matching make.
    fn pop(&mut self) -> Option<bool> {
        self.evaluated.pop().map(|evaluated| !evaluated)
    }
}

thread_local! {
    /// Position hashes already statically evaluated in this measurement.
    ///
    /// Thread-local rather than global so the count is exact for the search
    /// thread and the map is never shared; it is cleared by [`reset`].
    static EVALUATED_HASHES: RefCell<HashSet<u64>> = RefCell::new(HashSet::new());

    /// One bool per outstanding session make: whether that made position's own
    /// accumulator was statically evaluated. A descendant evaluating does not
    /// mark its ancestors.
    static MADE_FRAMES: RefCell<MadeFrames> = RefCell::new(MadeFrames::default());
}

/// One measurement's counted events.
///
/// Every field is a plain counter: nothing here is derived from a timer and no
/// field is a rate, so the report can compute whatever ratio it needs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    /// Search nodes entered with `quiescence == false` (main nodes).
    pub nodes_main: u64,
    /// Search nodes entered with `quiescence == true` (quiescence plies).
    pub nodes_quiescence: u64,
    /// Depth-zero leaves scored by the quiescence-disabled frontier path.
    pub nodes_frontier: u64,
    /// Calls to the worker's `static_evaluation` (one neural head each).
    pub static_evaluations: u64,
    /// Calls to [`crate::eval::n2::forward`], the neural head entry point.
    pub forward_calls: u64,
    /// Incremental session `on_make` calls, including claim probes.
    pub make_calls: u64,
    /// Incremental session `on_unmake` calls, including claim probes.
    pub unmake_calls: u64,
    /// Full accumulator rebuilds (`Accumulator::rebuild`).
    pub rebuilds: u64,
    /// Made nodes whose *whole subtree* performed no static evaluation.
    ///
    /// Historical A1 metric, preserved under an explicit name: a parent that
    /// is itself unevaluated but has an evaluated descendant is not counted
    /// here. See [`Snapshot::prepared_unevaluated_subtree`] and
    /// [`Snapshot::prepared_unevaluated_own`].
    pub prepared_unevaluated_subtree: u64,
    /// Made frames whose own accumulator was never statically evaluated,
    /// regardless of descendants (the metric the A1 report intended).
    pub prepared_unevaluated_own: u64,
    /// Static evaluations performed at a quiescence stand-pat call site.
    pub static_evaluations_quiescence: u64,
    /// Static evaluations performed at the quiescence-disabled frontier site.
    pub static_evaluations_frontier: u64,
    /// Static evaluations on a position hash this measurement already evaluated.
    pub repeated_evaluations: u64,
    /// Static evaluations on a position hash not seen before in this measurement.
    pub unique_evaluations: u64,
    /// Nanoseconds attributed to `static_evaluation` (timer-inclusive).
    pub evaluation_time_ns: u64,
    /// Nanoseconds attributed to session `on_make` (timer-inclusive).
    pub make_time_ns: u64,
    /// Nanoseconds attributed to session `on_unmake` (timer-inclusive).
    pub unmake_time_ns: u64,
}

impl Snapshot {
    /// Static evaluations per 1000 search nodes, zero for an empty measurement.
    pub fn evaluations_per_1000_nodes(&self) -> f64 {
        // `nodes_frontier` is a subset of `nodes_quiescence` (both are entered
        // with `reserve_node(true)`), so adding it would double-count frontier
        // nodes when quiescence is disabled.
        let nodes = self.nodes_main + self.nodes_quiescence;
        if nodes == 0 {
            0.0
        } else {
            self.static_evaluations as f64 * 1000.0 / nodes as f64
        }
    }

    /// Share of made frames whose own accumulator was never evaluated.
    pub fn prepared_unevaluated_own_ratio(&self) -> f64 {
        if self.make_calls == 0 {
            0.0
        } else {
            self.prepared_unevaluated_own as f64 / self.make_calls as f64
        }
    }

    /// Historical subtree-level share; kept for A1 comparability only.
    pub fn prepared_unevaluated_subtree_ratio(&self) -> f64 {
        if self.make_calls == 0 {
            0.0
        } else {
            self.prepared_unevaluated_subtree as f64 / self.make_calls as f64
        }
    }

    /// Share of static evaluations that repeated an already-seen position hash.
    pub fn repeated_evaluation_ratio(&self) -> f64 {
        let total = self.repeated_evaluations + self.unique_evaluations;
        if total == 0 {
            0.0
        } else {
            self.repeated_evaluations as f64 / total as f64
        }
    }
}

/// Count one entered search node.
pub fn record_node(quiescence: bool) {
    let counter = if quiescence {
        &NODES_QUIESCENCE
    } else {
        &NODES_MAIN
    };
    counter.fetch_add(1, Ordering::Relaxed);
}

/// Count one quiescence-disabled depth-zero frontier leaf.
pub fn record_frontier_node() {
    NODES_FRONTIER.fetch_add(1, Ordering::Relaxed);
}

/// Count one static evaluation and record its position hash.
pub fn record_static_evaluation(position_hash: u64, frontier: bool) {
    STATIC_EVALUATIONS.fetch_add(1, Ordering::Relaxed);
    let site_counter = if frontier {
        &STATIC_EVALUATIONS_FRONTIER
    } else {
        &STATIC_EVALUATIONS_QUIESCENCE
    };
    site_counter.fetch_add(1, Ordering::Relaxed);
    MADE_FRAMES.with(|frames| frames.borrow_mut().mark_current());
    let repeated = EVALUATED_HASHES.with(|hashes| !hashes.borrow_mut().insert(position_hash));
    if repeated {
        REPEATED_EVALUATIONS.fetch_add(1, Ordering::Relaxed);
    } else {
        UNIQUE_EVALUATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Count one neural head call.
pub fn record_forward() {
    FORWARD_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// Count one incremental session make and push its own-evaluation frame.
pub fn record_make() {
    MAKE_CALLS.fetch_add(1, Ordering::Relaxed);
    MADE_FRAMES.with(|frames| frames.borrow_mut().push());
}

/// Count one incremental session unmake and close its own-evaluation frame.
pub fn record_unmake() {
    UNMAKE_CALLS.fetch_add(1, Ordering::Relaxed);
    let unevaluated = MADE_FRAMES.with(|frames| frames.borrow_mut().pop());
    if unevaluated == Some(true) {
        PREPARED_UNEVALUATED_OWN.fetch_add(1, Ordering::Relaxed);
    }
}

/// Count one full accumulator rebuild.
pub fn record_rebuild() {
    REBUILDS.fetch_add(1, Ordering::Relaxed);
}

/// Count one made node whose whole subtree never evaluated itself.
pub fn record_prepared_unevaluated_subtree() {
    PREPARED_UNEVALUATED_SUBTREE.fetch_add(1, Ordering::Relaxed);
}

/// Add one timed static evaluation.
pub fn record_evaluation_time(elapsed: Duration) {
    EVALUATION_TIME_NS.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
}

/// Add one timed session make.
pub fn record_make_time(elapsed: Duration) {
    MAKE_TIME_NS.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
}

/// Add one timed session unmake.
pub fn record_unmake_time(elapsed: Duration) {
    UNMAKE_TIME_NS.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
}

/// Live static-evaluation count, used to bracket subtree work.
pub fn static_evaluations() -> u64 {
    STATIC_EVALUATIONS.load(Ordering::Relaxed)
}

/// Live make count, used to bracket one `search_child` call.
pub fn make_calls() -> u64 {
    MAKE_CALLS.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluations_per_1000_nodes_does_not_double_count_frontier_subset() {
        let snapshot = Snapshot {
            nodes_main: 100,
            nodes_quiescence: 900,
            // Frontier leaves are entered as quiescence nodes too.
            nodes_frontier: 900,
            static_evaluations: 500,
            ..Snapshot::default()
        };
        assert!((snapshot.evaluations_per_1000_nodes() - 500.0).abs() < 1e-12);
    }

    #[test]
    fn parent_frame_stays_unevaluated_when_only_its_child_evaluates() {
        let mut frames = MadeFrames::default();
        frames.push(); // parent A
        frames.push(); // child B
        frames.mark_current(); // B's own accumulator is evaluated
        assert_eq!(frames.pop(), Some(false), "child B was evaluated");
        assert_eq!(frames.pop(), Some(true), "parent A was never evaluated");
        assert_eq!(frames.pop(), None);
    }
}
