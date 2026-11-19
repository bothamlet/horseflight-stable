//! Minimal UCI adapter for benchmarking Horseflight.
//!
//! The `Eval` option selects the classical evaluator (default) or the embedded
//! N2-002 NNUE with incremental accumulation. This adapter translates UCI
//! commands and clock limits into engine search requests.
//!
//! Time management is deliberately simple and conservative: a fraction of the
//! remaining clock plus the increment, cancelled by the engine's own
//! cooperative [`SearchControl`]. Cancellation keeps the last fully completed
//! iteration, so a timeout costs depth, never legality.

use std::io::{self, BufRead, Write};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use horseflight::eval::mate_distance_plies;
use horseflight::eval::n2::session::Session;
use horseflight::eval::n2::{self, N2Evaluator};
use horseflight::search::search_controlled_with_session;
use horseflight::{
    search_controlled, Game, HandcraftedEvaluator, Move, Role, SearchAction, SearchConfig,
    SearchControl, SearchResult, Square,
};

const ENGINE_NAME: &str = "Horseflight 0.1.0";
const ENGINE_AUTHOR: &str = "Horseflight";
/// The engine's own hard limits; UCI "infinite" searches simply use them.
const MAX_DEPTH: u32 = 96;
const NODE_CEILING: u64 = 1_000_000_000;

fn main() {
    let stdin = io::stdin();
    let mut game = Game::startpos();
    let mut active: Option<ActiveSearch> = None;
    let mut evaluator = EvaluatorChoice::Classical;
    // The per-search evaluator diagnostic is only emitted for a client that
    // opened a real UCI session. A driver that never sends `uci` (for example
    // the pinned N1 search-output regression) therefore keeps seeing exactly the
    // original `info`/`bestmove` stream, while every GUI and match runner still
    // gets the active-evaluator evidence it needs. `setoption` replies are
    // treated the same way: a client that never handshook never asked for them.
    let mut handshake = false;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => break,
        };
        let mut tokens = line.split_whitespace();
        let Some(command) = tokens.next() else {
            continue;
        };
        match command {
            "uci" => {
                handshake = true;
                out(&format!("id name {ENGINE_NAME}"));
                out(&format!("id author {ENGINE_AUTHOR}"));
                out("option name Eval type combo default classical var classical var nnue");
                out(&format!(
                    "info string n2 model={} asset_sha256={} perspective={} \
                     training_data=lichess-open-database",
                    n2::MODEL_ID,
                    n2::asset_sha256(),
                    n2::DERIVED_SCORE_PERSPECTIVE
                ));
                out("uciok");
            }
            "isready" => out("readyok"),
            "ucinewgame" => {
                stop_search(&mut active);
                game = Game::startpos();
                // `ucinewgame` resets the game, never the selected evaluator.
            }
            "setoption" => match parse_setoption(&tokens.collect::<Vec<_>>()) {
                // Only a real selection change interrupts a running search; an
                // unknown option or value is reported and changes nothing, as
                // it always has.
                Ok((name, value)) if name == "Eval" => match EvaluatorChoice::parse(&value) {
                    Some(choice) => {
                        stop_search(&mut active);
                        evaluator = choice;
                        if handshake {
                            match choice {
                                EvaluatorChoice::Classical => {
                                    out("info string Eval=classical");
                                }
                                EvaluatorChoice::Nnue => out(&format!(
                                    "info string Eval=nnue model={} asset_sha256={} \
                                     perspective={} training_data=lichess-open-database",
                                    n2::MODEL_ID,
                                    n2::asset_sha256(),
                                    n2::DERIVED_SCORE_PERSPECTIVE
                                )),
                            }
                        }
                    }
                    None => {
                        if handshake {
                            out(&format!(
                                "info string unknown Eval value {value:?}; keeping {}",
                                evaluator.as_str()
                            ));
                        }
                    }
                },
                Ok((name, _)) => {
                    if handshake {
                        out(&format!("info string unknown option {name:?}"));
                    }
                }
                Err(message) => {
                    if handshake {
                        out(&format!("info string {message}"));
                    }
                }
            },
            "position" => match parse_position(tokens.collect::<Vec<_>>()) {
                Ok(new_game) => {
                    stop_search(&mut active);
                    game = new_game;
                }
                Err(message) => out(&format!("info string {message}")),
            },
            "go" => {
                stop_search(&mut active);
                let limits = parse_go(tokens.collect::<Vec<_>>());
                active = Some(spawn_search(game.clone(), limits, evaluator, handshake));
            }
            "stop" => stop_search(&mut active),
            "ponderhit" => {}
            "quit" => {
                stop_search(&mut active);
                break;
            }
            _ => {}
        }
    }
    // stdin closed: never leave a search thread writing after the process ends.
    stop_search(&mut active);
}

fn out(text: &str) {
    let mut stdout = io::stdout().lock();
    let _ = writeln!(stdout, "{text}");
    let _ = stdout.flush();
}

struct ActiveSearch {
    control: SearchControl,
    handle: thread::JoinHandle<()>,
}

fn stop_search(active: &mut Option<ActiveSearch>) {
    if let Some(search) = active.take() {
        search.control.cancel();
        let _ = search.handle.join();
    }
}

/// A parsed `position` command: start position or FEN, then UCI moves.
fn parse_position(tokens: Vec<&str>) -> Result<Game, String> {
    let mut index;
    let mut game = match tokens.first().copied() {
        Some("startpos") => {
            index = 1;
            Game::startpos()
        }
        Some("fen") => {
            index = 1;
            let mut fields = Vec::new();
            while index < tokens.len() && tokens[index] != "moves" {
                fields.push(tokens[index]);
                index += 1;
            }
            if fields.len() != 6 {
                return Err(format!("fen needs six fields, got {}", fields.len()));
            }
            Game::from_fen(&fields.join(" ")).map_err(|error| format!("invalid fen: {error}"))?
        }
        _ => return Err("position expects startpos or fen".to_string()),
    };

    match tokens.get(index) {
        None => return Ok(game),
        Some(&"moves") => index += 1,
        Some(other) => return Err(format!("position expects moves, got {other:?}")),
    }
    for token in &tokens[index..] {
        let movement = find_legal_move(&game, token)
            .ok_or_else(|| format!("illegal or unknown move {token:?}"))?;
        game.play(movement)
            .map_err(|error| format!("move {token:?} rejected: {error}"))?;
    }
    Ok(game)
}

fn find_legal_move(game: &Game, text: &str) -> Option<Move> {
    // Moves carry their generation kind (double push, castle, en passant), so
    // compare the canonical UCI text rather than a bare from/to pair.
    let candidate = move_from_uci(text)?;
    game.position()
        .legal_moves()
        .iter()
        .copied()
        .find(|legal| legal.to_uci() == candidate.to_uci())
}

fn move_from_uci(token: &str) -> Option<Move> {
    if !token.is_ascii() || (token.len() != 4 && token.len() != 5) {
        return None;
    }
    let from = Square::from_ascii(&token[0..2])?;
    let to = Square::from_ascii(&token[2..4])?;
    match token[4..].chars().next() {
        Some(letter) => Some(Move::promotion(from, to, Role::from_uci_char(letter)?)),
        None => Some(Move::normal(from, to)),
    }
}

#[derive(Default, Clone, Copy)]
struct GoLimits {
    depth: Option<u32>,
    nodes: Option<u64>,
    movetime: Option<u64>,
    wtime: Option<u64>,
    btime: Option<u64>,
    winc: Option<u64>,
    binc: Option<u64>,
    movestogo: Option<u64>,
}

fn parse_go(tokens: Vec<&str>) -> GoLimits {
    let mut limits = GoLimits::default();
    let mut index = 0;
    while index < tokens.len() {
        let key = tokens[index];
        let value = tokens
            .get(index + 1)
            .and_then(|text| text.parse::<u64>().ok());
        match (key, value) {
            ("depth", Some(value)) => limits.depth = Some(value.clamp(1, MAX_DEPTH as u64) as u32),
            ("nodes", Some(value)) => limits.nodes = Some(value.max(1)),
            ("movetime", Some(value)) => limits.movetime = Some(value),
            ("wtime", Some(value)) => limits.wtime = Some(value),
            ("btime", Some(value)) => limits.btime = Some(value),
            ("winc", Some(value)) => limits.winc = Some(value),
            ("binc", Some(value)) => limits.binc = Some(value),
            ("movestogo", Some(value)) => limits.movestogo = Some(value.max(1)),
            ("infinite", _) | ("ponder", _) => {}
            _ => {}
        }
        index += if value.is_some() { 2 } else { 1 };
    }
    limits
}

/// Fraction of the clock spent this move. Kept simple, fixed and conservative:
/// never a whole clock, always a safety reserve, no opponent-dependent tuning.
fn time_budget_ms(game: &Game, limits: &GoLimits) -> Option<u64> {
    if let Some(movetime) = limits.movetime {
        return Some(movetime.saturating_sub(10));
    }
    let (remaining, increment) = if game.position().side_to_move() == horseflight::Color::White {
        (limits.wtime?, limits.winc.unwrap_or(0))
    } else {
        (limits.btime?, limits.binc.unwrap_or(0))
    };
    let reserve = 40;
    let spendable = remaining.saturating_sub(reserve);
    let planned = match limits.movestogo {
        Some(moves) => spendable / moves.saturating_add(1),
        // 25 moves of headroom plus most of the increment.
        None => (spendable / 25).saturating_add(increment.saturating_mul(3) / 4),
    };
    // Never plan more than the clock actually leaves after the reserve. A
    // near-zero clock therefore yields an immediate, legal fallback move
    // instead of a budget that overruns it.
    Some(planned.min(spendable))
}

fn search_config(limits: &GoLimits) -> SearchConfig {
    SearchConfig {
        max_depth: limits.depth.unwrap_or(MAX_DEPTH),
        node_limit: limits.nodes.unwrap_or(NODE_CEILING),
        ordering: true,
        quiescence: true,
        q_depth_limit: 4,
    }
}

/// The selected evaluation path. Classical is the default and reproduces the
/// pre-NNUE engine exactly; `nnue` is opt-in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EvaluatorChoice {
    Classical,
    Nnue,
}

impl EvaluatorChoice {
    fn parse(value: &str) -> Option<EvaluatorChoice> {
        match value {
            "classical" => Some(EvaluatorChoice::Classical),
            "nnue" => Some(EvaluatorChoice::Nnue),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            EvaluatorChoice::Classical => "classical",
            EvaluatorChoice::Nnue => "nnue",
        }
    }
}

/// Parse the `name <name> value <value>` body of a `setoption` command.
fn parse_setoption(tokens: &[&str]) -> Result<(String, String), String> {
    let Some(rest) = tokens.strip_prefix(&["name"][..]) else {
        return Err("setoption expects name".to_string());
    };
    let Some(position) = rest.iter().position(|token| *token == "value") else {
        return Err("setoption expects value".to_string());
    };
    if position == 0 {
        return Err("setoption expects an option name".to_string());
    }
    let name = rest[..position].join(" ");
    let value = rest[position + 1..].join(" ");
    Ok((name, value))
}

fn spawn_search(
    game: Game,
    limits: GoLimits,
    evaluator: EvaluatorChoice,
    report_evaluator: bool,
) -> ActiveSearch {
    let budget = time_budget_ms(&game, &limits);
    let config = search_config(&limits);
    let control = SearchControl::new();
    let worker_control = control.clone();

    // External logs can prove which evaluator actually ran. Printed before the
    // search thread starts so the line always precedes its info/bestmove pair.
    if report_evaluator {
        match evaluator {
            EvaluatorChoice::Classical => out("info string eval=classical"),
            EvaluatorChoice::Nnue => out(&format!(
                "info string eval=nnue model_sha256={}",
                n2::asset_sha256()
            )),
        }
    }

    let handle = thread::spawn(move || {
        // The deadline waiter parks on the channel, so it leaves as soon as the
        // search finishes instead of sleeping out the whole budget. One waiter
        // exists per running search and none survives it.
        let deadline = budget.map(|budget| {
            let deadline_control = worker_control.clone();
            let (finished, wait) = mpsc::channel::<()>();
            let waiter = thread::spawn(move || {
                if matches!(
                    wait.recv_timeout(Duration::from_millis(budget)),
                    Err(RecvTimeoutError::Timeout)
                ) {
                    deadline_control.cancel();
                }
            });
            (finished, waiter)
        });
        let started = Instant::now();
        let result = match evaluator {
            EvaluatorChoice::Classical => {
                search_controlled(&game, &HandcraftedEvaluator, &config, &worker_control)
            }
            EvaluatorChoice::Nnue => {
                // The session is owned by this search: reset happens inside the
                // search, and the session is dropped (balanced) when it ends.
                let mut session = Session::new(game.position());
                search_controlled_with_session(
                    &game,
                    &N2Evaluator,
                    &mut session,
                    &config,
                    &worker_control,
                )
            }
        };
        if let Some((finished, waiter)) = deadline {
            let _ = finished.send(());
            let _ = waiter.join();
        }
        report(&game, result, started, budget);
        let _ = io::stdout().flush();
    });

    ActiveSearch { control, handle }
}

/// Print genuine diagnostics plus a legal `bestmove`.
fn report(
    game: &Game,
    result: Result<SearchResult, horseflight::search::ConfigError>,
    started: Instant,
    budget: Option<u64>,
) {
    let result = match result {
        Ok(result) => result,
        Err(error) => {
            out(&format!("info string search rejected: {error}"));
            out(&format!("bestmove {}", first_legal_or_null(game)));
            return;
        }
    };

    let elapsed_ms = started.elapsed().as_millis();
    if let Some(score) = result.score {
        let score_text = match mate_distance_plies(score) {
            Some(0) => "mate 0".to_string(),
            Some(plies) => {
                let moves = plies.div_ceil(2) as i32;
                format!("mate {}", if score > 0 { moves } else { -moves })
            }
            None => format!("cp {score}"),
        };
        let pv = result
            .pv
            .iter()
            .map(|movement| movement.to_uci())
            .collect::<Vec<_>>()
            .join(" ");
        out(&format!(
            "info depth {} score {score_text} nodes {} nps {:.0} time {elapsed_ms} pv {pv}",
            result.completed_depth,
            result.nodes,
            result.nps()
        ));
    } else {
        out(&format!(
            "info depth 0 nodes {} time {elapsed_ms} string no completed iteration",
            result.nodes
        ));
    }
    if let Some(budget) = budget {
        if elapsed_ms as u64 > budget {
            out(&format!(
                "info string search stopped after {elapsed_ms}ms of a {budget}ms budget"
            ));
        }
    }

    let chosen = match result.action {
        SearchAction::Move(movement) | SearchAction::Fallback(movement) => Some(movement),
        // UCI has no draw claim; the claim's own legal move is played instead.
        SearchAction::Claim(horseflight::DrawClaim::Intended(movement)) => Some(movement),
        SearchAction::Claim(horseflight::DrawClaim::Current) => None,
        // The game already ended automatically. No move may be played even if
        // legal moves still exist (insufficient material, 75-move, fivefold).
        SearchAction::Terminal(_) => {
            out("bestmove 0000");
            return;
        }
    };
    let legal = chosen.filter(|movement| {
        game.position()
            .legal_moves()
            .iter()
            .any(|candidate| candidate.to_uci() == movement.to_uci())
    });
    match legal {
        Some(movement) => out(&format!("bestmove {}", movement.to_uci())),
        None => out(&format!("bestmove {}", first_legal_or_null(game))),
    }
}

fn first_legal_or_null(game: &Game) -> String {
    game.position()
        .legal_moves()
        .iter()
        .next()
        .map(|movement| movement.to_uci())
        .unwrap_or_else(|| "0000".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use horseflight::Color;

    fn limits(args: &str) -> GoLimits {
        parse_go(args.split_whitespace().collect())
    }

    #[test]
    fn depth_is_clamped_to_a_valid_search_configuration() {
        assert_eq!(limits("depth 0").depth, Some(1));
        assert_eq!(limits("depth 12").depth, Some(12));
        assert_eq!(limits("depth 1000").depth, Some(MAX_DEPTH));
    }

    #[test]
    fn time_budget_never_claims_more_than_the_remaining_clock() {
        let start = Game::startpos();
        assert_eq!(time_budget_ms(&start, &limits("wtime 0")), Some(0));
        assert_eq!(time_budget_ms(&start, &limits("wtime 39")), Some(0));
        assert_eq!(time_budget_ms(&start, &limits("wtime 40")), Some(0));
        assert_eq!(time_budget_ms(&start, &limits("wtime 45")), Some(0));
        assert_eq!(time_budget_ms(&start, &limits("wtime 65")), Some(1));
        assert_eq!(time_budget_ms(&start, &limits("movetime 0")), Some(0));
        assert_eq!(time_budget_ms(&start, &limits("movetime 5")), Some(0));
        assert_eq!(time_budget_ms(&start, &limits("movetime 100")), Some(90));
    }

    #[test]
    fn time_budget_arithmetic_saturates_on_extreme_input() {
        let start = Game::startpos();
        let huge = u64::MAX.to_string();
        let max_clock = format!("wtime {huge} btime {huge}");
        assert_eq!(
            time_budget_ms(&start, &limits(&max_clock)),
            Some((u64::MAX - 40) / 25)
        );
        let max_increment = format!("wtime {huge} winc {huge}");
        let clock_budget = (u64::MAX - 40) / 25;
        let increment_budget = u64::MAX / 4;
        assert_eq!(
            time_budget_ms(&start, &limits(&max_increment)),
            Some(clock_budget + increment_budget)
        );
        assert!(clock_budget + increment_budget < u64::MAX - 40);
        let max_moves = format!("wtime {huge} movestogo {huge}");
        assert_eq!(time_budget_ms(&start, &limits(&max_moves)), Some(0));
    }

    #[test]
    fn time_budget_uses_the_side_to_move_clock() {
        let black = Game::from_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1").expect("fen");
        assert_eq!(black.position().side_to_move(), Color::Black);
        assert_eq!(
            time_budget_ms(&black, &limits("wtime 10000 btime 5000")),
            Some(4960 / 25)
        );

        let start = Game::startpos();
        assert_eq!(
            time_budget_ms(&start, &limits("wtime 60000")),
            Some(59960 / 25)
        );
        assert_eq!(
            time_budget_ms(&start, &limits("wtime 60000 movestogo 40")),
            Some(59960 / 41)
        );
    }

    #[test]
    fn setoption_parses_name_and_value() {
        assert_eq!(
            parse_setoption(&["name", "Eval", "value", "nnue"]),
            Ok(("Eval".to_string(), "nnue".to_string()))
        );
        assert_eq!(
            parse_setoption(&["name", "My", "Option", "value", "two", "words"]),
            Ok(("My Option".to_string(), "two words".to_string()))
        );
        assert!(parse_setoption(&["value", "nnue"]).is_err());
        assert!(parse_setoption(&["name", "Eval"]).is_err());
        assert!(parse_setoption(&["name", "value", "nnue"]).is_err());
    }

    #[test]
    fn evaluator_choice_is_exact_and_classical_by_default() {
        assert_eq!(
            EvaluatorChoice::parse("classical"),
            Some(EvaluatorChoice::Classical)
        );
        assert_eq!(EvaluatorChoice::parse("nnue"), Some(EvaluatorChoice::Nnue));
        assert_eq!(EvaluatorChoice::parse("NNUE"), None);
        assert_eq!(EvaluatorChoice::parse(""), None);
        assert_eq!(EvaluatorChoice::Classical.as_str(), "classical");
        assert_eq!(EvaluatorChoice::Nnue.as_str(), "nnue");
    }

    #[test]
    fn parse_go_ignores_malformed_values_without_panicking() {
        let parsed = limits("depth nope nodes -1 movetime 50");
        assert_eq!(parsed.depth, None);
        assert_eq!(parsed.nodes, None);
        assert_eq!(parsed.movetime, Some(50));
    }
}
