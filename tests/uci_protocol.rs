//! Protocol-level tests for the benchmark UCI adapter.
//!
//! These exercise the real binary over pipes: no chess logic is re-implemented
//! here, and every `bestmove` is checked against the engine's own legal moves.
//! Every read is bounded and every harness kills its child on timeout, so a
//! regression fails instead of hanging the suite.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use horseflight::{Color, Game, Outcome};

/// Upper bound for a single protocol message. Searches in these tests are
/// deliberately tiny, so exceeding this means the adapter stopped responding.
const OUTPUT_TIMEOUT: Duration = Duration::from_secs(10);
const EXIT_TIMEOUT: Duration = Duration::from_secs(5);

struct Uci {
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    seen: Vec<String>,
}

impl Uci {
    fn spawn() -> Uci {
        let mut child = Command::new(env!("CARGO_BIN_EXE_uci"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn uci");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let (sender, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => {
                        if sender.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });
        Uci {
            child,
            stdin: Some(stdin),
            lines,
            seen: Vec::new(),
        }
    }

    fn send(&mut self, command: &str) {
        let stdin = self.stdin.as_mut().expect("stdin still open");
        writeln!(stdin, "{command}").expect("write command");
        stdin.flush().expect("flush command");
    }

    fn send_all(&mut self, commands: &[&str]) {
        for command in commands {
            self.send(command);
        }
    }

    fn next_line(&mut self, timeout: Duration) -> String {
        match self.lines.recv_timeout(timeout) {
            Ok(line) => {
                self.seen.push(line.clone());
                line
            }
            Err(error) => {
                self.kill();
                panic!(
                    "no UCI output within {timeout:?} ({error}); saw {:?}",
                    self.seen
                );
            }
        }
    }

    fn wait_for(&mut self, prefix: &str, timeout: Duration) -> String {
        let deadline = Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                self.kill();
                panic!("timed out waiting for {prefix:?}; saw {:?}", self.seen);
            }
            let line = self.next_line(remaining);
            if line.starts_with(prefix) {
                return line;
            }
        }
    }

    /// Send commands, then wait for the resulting `bestmove` and time the wait.
    fn search(&mut self, commands: &[&str]) -> (String, Duration) {
        self.send_all(commands);
        let started = Instant::now();
        let line = self.wait_for("bestmove ", OUTPUT_TIMEOUT);
        let chosen = line
            .split_whitespace()
            .nth(1)
            .expect("bestmove token")
            .to_string();
        (chosen, started.elapsed())
    }

    fn no_bestmove_window(&mut self, duration: Duration) {
        let deadline = Instant::now() + duration;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return;
            }
            match self.lines.recv_timeout(remaining) {
                Ok(line) => {
                    self.seen.push(line.clone());
                    assert!(!line.starts_with("bestmove "), "unexpected {line}");
                }
                Err(mpsc::RecvTimeoutError::Timeout) => return,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    panic!("UCI disconnected: {:?}", self.seen)
                }
            }
        }
    }

    fn close_stdin(&mut self) {
        self.stdin.take();
    }

    fn wait_exit(&mut self, timeout: Duration) -> ExitStatus {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.child.try_wait().expect("try_wait") {
                return status;
            }
            if Instant::now() >= deadline {
                self.kill();
                panic!(
                    "process did not exit within {timeout:?}; saw {:?}",
                    self.seen
                );
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn quit(&mut self) -> ExitStatus {
        self.send("quit");
        self.close_stdin();
        self.wait_exit(EXIT_TIMEOUT)
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn mark(&self) -> usize {
        self.seen.len()
    }

    /// Read a numeric `info` field only from output produced after `start`, so a
    /// value from an earlier search cannot satisfy the assertion.
    fn info_field_since(&self, start: usize, name: &str) -> Option<u64> {
        self.seen[start..].iter().rev().find_map(|line| {
            let tail = line.split_once(&format!(" {name} "))?.1;
            tail.split_whitespace().next()?.parse().ok()
        })
    }
}

impl Drop for Uci {
    fn drop(&mut self) {
        self.stdin.take();
        if self.child.try_wait().ok().flatten().is_none() {
            self.kill();
        }
    }
}

fn legal_uci(game: &Game) -> Vec<String> {
    game.position()
        .legal_moves()
        .iter()
        .map(|movement| movement.to_uci())
        .collect()
}

fn assert_legal(game: &Game, chosen: &str) {
    assert!(
        legal_uci(game).contains(&chosen.to_string()),
        "{chosen} must be legal in {}",
        game.position().to_fen()
    );
}

fn apply(game: &mut Game, text: &str) {
    let movement = game
        .position()
        .legal_moves()
        .iter()
        .copied()
        .find(|candidate| candidate.to_uci() == text)
        .unwrap_or_else(|| panic!("{text} is not legal"));
    game.play(movement).expect("legal play");
}

fn info_score_mate(line: &str) -> Option<i32> {
    let mut tokens = line.split_whitespace();
    while let Some(token) = tokens.next() {
        if token == "mate" {
            return tokens.next()?.parse().ok();
        }
    }
    None
}

fn info_pv(line: &str) -> Vec<String> {
    match line.split_once(" pv ") {
        Some((_, pv)) => pv.split_whitespace().map(str::to_string).collect(),
        None => Vec::new(),
    }
}

fn replay(moves: &[&str]) -> Game {
    let mut game = Game::startpos();
    for text in moves {
        apply(&mut game, text);
    }
    game
}

#[cfg(target_os = "linux")]
fn thread_count(pid: u32) -> usize {
    std::fs::read_dir(format!("/proc/{pid}/task"))
        .map(|entries| entries.count())
        .unwrap_or(0)
}

#[test]
fn handshake_readiness_and_bounded_depth_search() {
    let mut uci = Uci::spawn();
    uci.send("uci");
    uci.wait_for("uciok", OUTPUT_TIMEOUT);
    assert!(uci
        .seen
        .iter()
        .any(|line| line.starts_with("id name Horseflight")));
    assert!(uci.seen.iter().any(|line| line.starts_with("id author")));

    uci.send("isready");
    uci.wait_for("readyok", OUTPUT_TIMEOUT);

    uci.send("position startpos");
    let (chosen, elapsed) = uci.search(&["go depth 2"]);
    assert!(elapsed < OUTPUT_TIMEOUT);
    assert!(uci
        .seen
        .iter()
        .any(|line| line.starts_with("info depth 2 score")));
    assert_legal(&Game::startpos(), &chosen);

    let status = uci.quit();
    assert!(status.success(), "clean quit exits successfully");
}

#[test]
fn readiness_is_answered_while_an_infinite_search_runs() {
    let mut uci = Uci::spawn();
    uci.send_all(&["uci", "isready"]);
    uci.wait_for("uciok", OUTPUT_TIMEOUT);
    uci.wait_for("readyok", OUTPUT_TIMEOUT);

    uci.send("position startpos");
    uci.send("go infinite");
    // The infinite search has produced no bestmove yet; this readiness probe is
    // therefore answered while the search thread is still running.
    assert!(!uci.seen.iter().any(|line| line.starts_with("bestmove ")));
    uci.send("isready");
    let ready = Instant::now();
    uci.wait_for("readyok", OUTPUT_TIMEOUT);
    assert!(!uci.seen.iter().any(|line| line.starts_with("bestmove ")));
    assert!(ready.elapsed() < OUTPUT_TIMEOUT);

    uci.send("stop");
    let line = uci.wait_for("bestmove ", OUTPUT_TIMEOUT);
    let chosen = line.split_whitespace().nth(1).expect("bestmove token");
    assert_legal(&Game::startpos(), chosen);

    assert!(uci.quit().success());
}

#[test]
fn node_limit_and_zero_depth_are_bounded_searches() {
    let mut uci = Uci::spawn();
    let start = uci.mark();
    uci.send("position startpos");
    let (chosen, elapsed) = uci.search(&["go nodes 1"]);
    assert!(elapsed < OUTPUT_TIMEOUT);
    assert_legal(&Game::startpos(), &chosen);
    let nodes = uci
        .info_field_since(start, "nodes")
        .expect("info nodes field");
    assert!(nodes <= 1, "node budget 1 reported {nodes} nodes");

    let start = uci.mark();
    uci.send("position startpos");
    let (chosen, elapsed) = uci.search(&["go depth 0"]);
    assert!(elapsed < OUTPUT_TIMEOUT);
    assert_legal(&Game::startpos(), &chosen);
    assert!(
        uci.seen[start..]
            .iter()
            .any(|line| line.starts_with("info depth 1 score")),
        "depth 0 is clamped to a single completed depth: {:?}",
        &uci.seen[start..]
    );

    assert!(uci.quit().success());
}

#[test]
fn movetime_and_both_clocks_bound_the_search() {
    let mut uci = Uci::spawn();
    let game = replay(&["e2e4", "e7e5", "g1f3"]);

    let start = uci.mark();
    let (chosen, elapsed) =
        uci.search(&["position startpos moves e2e4 e7e5 g1f3", "go movetime 150"]);
    assert!(
        elapsed < Duration::from_secs(1),
        "movetime overran: {elapsed:?}"
    );
    assert_legal(&game, &chosen);
    let reported = uci
        .info_field_since(start, "time")
        .expect("info time field");
    assert!(reported <= 500, "movetime 150 search reported {reported}ms");

    // White to move with only the white clock running.
    let (chosen, elapsed) = uci.search(&["position startpos", "go wtime 1000 btime 1000"]);
    assert!(
        elapsed < Duration::from_secs(1),
        "white clock overran: {elapsed:?}"
    );
    assert_legal(&Game::startpos(), &chosen);

    // Black to move with increments; the budget must come from btime/binc.
    let black = replay(&["e2e4"]);
    let (chosen, elapsed) = uci.search(&[
        "position startpos moves e2e4",
        "go wtime 1000 btime 1000 winc 10 binc 10",
    ]);
    assert!(
        elapsed < Duration::from_secs(1),
        "black clock overran: {elapsed:?}"
    );
    assert_legal(&black, &chosen);

    // Extreme clock values exercise the saturating arithmetic without searching
    // beyond the depth bound.
    let huge = u64::MAX.to_string();
    let extreme =
        format!("go wtime {huge} btime {huge} winc {huge} binc {huge} movestogo {huge} depth 1");
    let (chosen, elapsed) = uci.search(&["position startpos", &extreme]);
    assert!(
        elapsed < Duration::from_secs(1),
        "extreme clocks overran: {elapsed:?}"
    );
    assert_legal(&Game::startpos(), &chosen);

    // A malformed depth value falls back to the other limits instead of hanging.
    let (chosen, elapsed) = uci.search(&["position startpos", "go depth nope movetime 50"]);
    assert!(
        elapsed < Duration::from_secs(1),
        "malformed go overran: {elapsed:?}"
    );
    assert_legal(&Game::startpos(), &chosen);

    assert!(uci.quit().success());
}

#[test]
fn replay_supports_castling_promotion_and_en_passant() {
    let mut uci = Uci::spawn();

    let castle_fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
    let castle_move = "e1g1";
    let mut castled = Game::from_fen(castle_fen).expect("castle fen");
    let (chosen, _) = uci.search(&[
        "position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1 moves e1g1",
        "go depth 2",
    ]);
    apply(&mut castled, castle_move);
    assert_legal(&castled, &chosen);

    let promotion_fen = "8/P7/8/8/8/8/8/K6k w - - 0 1";
    let mut promoted = Game::from_fen(promotion_fen).expect("promotion fen");
    let (chosen, _) = uci.search(&[
        "position fen 8/P7/8/8/8/8/8/K6k w - - 0 1 moves a7a8q",
        "go depth 2",
    ]);
    apply(&mut promoted, "a7a8q");
    assert_legal(&promoted, &chosen);

    let ep_fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1";
    let mut en_passant = Game::from_fen(ep_fen).expect("en passant fen");
    let (chosen, _) = uci.search(&[
        "position fen 4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1 moves e5d6",
        "go depth 2",
    ]);
    apply(&mut en_passant, "e5d6");
    assert_legal(&en_passant, &chosen);

    assert!(uci.quit().success());
}

#[test]
fn stop_infinite_repeated_searches_reset_quit_and_eof() {
    let mut uci = Uci::spawn();
    uci.send("position startpos");
    uci.send("go infinite");
    // No output can arrive before `stop`, so this bestmove proves the stopped
    // infinite search still reports a legal move.
    uci.send("stop");
    let line = uci.wait_for("bestmove ", OUTPUT_TIMEOUT);
    let stopped = line.split_whitespace().nth(1).expect("bestmove token");
    assert_legal(&Game::startpos(), stopped);

    // Repeated searches on distinct positions reuse the engine cleanly: a
    // stale or duplicated `bestmove` cannot look legal for the next position.
    let repeats = [
        ("position startpos", Game::startpos()),
        ("position startpos moves e2e4", replay(&["e2e4"])),
        (
            "position startpos moves e2e4 e7e5",
            replay(&["e2e4", "e7e5"]),
        ),
    ];
    for (position, expected) in repeats {
        let start = uci.mark();
        let (chosen, _) = uci.search(&[position, "go depth 1"]);
        assert_legal(&expected, &chosen);
        assert_eq!(
            uci.seen[start..]
                .iter()
                .filter(|line| line.starts_with("bestmove "))
                .count(),
            1,
            "one bestmove per search: {:?}",
            &uci.seen[start..]
        );
    }

    // ucinewgame resets the game to the start position.
    uci.send("ucinewgame");
    let (chosen, _) = uci.search(&["go depth 1"]);
    assert_legal(&Game::startpos(), &chosen);

    assert!(uci.quit().success());

    // Closing stdin during a search still prints a legal move and exits.
    let mut eof = Uci::spawn();
    eof.send("position startpos");
    eof.send("go depth 3");
    eof.close_stdin();
    let line = eof.wait_for("bestmove ", OUTPUT_TIMEOUT);
    let chosen = line.split_whitespace().nth(1).expect("bestmove token");
    assert_legal(&Game::startpos(), chosen);
    assert!(eof.wait_exit(EXIT_TIMEOUT).success());
}

#[cfg(target_os = "linux")]
#[test]
fn completed_searches_do_not_leave_deadline_threads() {
    let mut uci = Uci::spawn();
    uci.send("uci");
    uci.wait_for("uciok", OUTPUT_TIMEOUT);
    let baseline = thread_count(uci.pid());

    // A large movetime with a tiny depth finishes immediately. The old
    // implementation left one sleeping deadline thread per finished search.
    for _ in 0..8 {
        uci.send("position startpos");
        let (chosen, _) = uci.search(&["go depth 2 movetime 5000"]);
        assert_legal(&Game::startpos(), &chosen);
    }

    let after = thread_count(uci.pid());
    assert!(
        after <= baseline + 3,
        "finished searches leaked deadline threads: baseline {baseline}, after {after}"
    );
    assert!(uci.quit().success());
}

#[test]
fn terminal_roots_report_null_move() {
    let mut uci = Uci::spawn();

    // Checkmate: Kh8 is in check from Qg7, g7 is defended by Kg6, and both
    // escape squares (g8, h7) are covered. This is an automatic loss.
    let checkmate = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1";
    assert_eq!(
        Game::from_fen(checkmate).expect("checkmate fen").outcome(),
        Outcome::Checkmate {
            winner: Color::White
        },
        "the fixture must really be checkmate"
    );
    let start = uci.mark();
    let (chosen, _) = uci.search(&["position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "go depth 2"]);
    assert_eq!(chosen, "0000", "checkmate root must not play a move");
    let mated_score = uci.seen[start..]
        .iter()
        .rev()
        .find(|line| line.starts_with("info depth "))
        .and_then(|line| info_score_mate(line));
    assert_eq!(
        mated_score,
        Some(0),
        "a mated root must report mate 0: {:?}",
        &uci.seen[start..]
    );

    // Stalemate: independently checked as a draw with no legal moves.
    let stalemate = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1";
    assert_eq!(
        Game::from_fen(stalemate).expect("stalemate fen").outcome(),
        Outcome::Stalemate,
        "the fixture must really be stalemate"
    );
    let (chosen, _) = uci.search(&["position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "go depth 2"]);
    assert_eq!(chosen, "0000", "stalemate root must not play a move");

    // Insufficient material: the game is over even though legal moves exist.
    let insufficient = "8/8/8/8/8/8/8/K1B4k w - - 0 1";
    let insufficient_game = Game::from_fen(insufficient).expect("insufficient fen");
    assert_eq!(insufficient_game.outcome(), Outcome::InsufficientMaterial);
    assert!(!insufficient_game.position().legal_moves().is_empty());
    let (chosen, _) = uci.search(&["position fen 8/8/8/8/8/8/8/K1B4k w - - 0 1", "go depth 2"]);
    assert_eq!(chosen, "0000", "insufficient material must not play a move");

    // Seventy-five-move draw: legal moves remain, but the game is already over.
    let seventy_five = "7k/8/8/8/8/8/8/KQ6 w - - 150 200";
    let seventy_five_game = Game::from_fen(seventy_five).expect("seventy-five-move fen");
    assert_eq!(seventy_five_game.outcome(), Outcome::SeventyFiveMoveDraw);
    assert!(!seventy_five_game.position().legal_moves().is_empty());
    let (chosen, _) = uci.search(&[
        "position fen 7k/8/8/8/8/8/8/KQ6 w - - 150 200",
        "go depth 2",
    ]);
    assert_eq!(
        chosen, "0000",
        "seventy-five-move draw must not play a move"
    );

    assert!(uci.quit().success());
}

#[test]
fn searched_forced_mate_reports_sign_distance_and_legal_pv() {
    let mut uci = Uci::spawn();

    // White mates in one: Qf8#. The score is positive and the PV delivers mate.
    let mate_in_one = "7k/8/6K1/8/8/8/8/5Q2 w - - 0 1";
    let start = uci.mark();
    let (chosen, _) = uci.search(&["position fen 7k/8/6K1/8/8/8/8/5Q2 w - - 0 1", "go depth 3"]);
    let line = uci.seen[start..]
        .iter()
        .rev()
        .find(|line| line.starts_with("info depth "))
        .expect("mate search reports an info line");
    assert_eq!(info_score_mate(line), Some(1), "info line: {line}");
    let pv = info_pv(line);
    assert_eq!(pv.first(), Some(&chosen), "bestmove starts the PV: {line}");
    let mut game = Game::from_fen(mate_in_one).expect("mate-in-one fen");
    for movement in &pv {
        apply(&mut game, movement);
    }
    assert_eq!(
        game.outcome(),
        Outcome::Checkmate {
            winner: Color::White
        },
        "PV must end in checkmate: {line}"
    );

    // White is mated in one whatever it plays; the same conversion keeps the
    // negative sign and the distance (one move for Black).
    let mated_in_one = "8/8/8/8/1q6/8/P1k5/K7 w - - 0 1";
    let start = uci.mark();
    let (chosen, _) = uci.search(&["position fen 8/8/8/8/1q6/8/P1k5/K7 w - - 0 1", "go depth 4"]);
    let line = uci.seen[start..]
        .iter()
        .rev()
        .find(|line| line.starts_with("info depth "))
        .expect("lost-mate search reports an info line");
    assert_eq!(info_score_mate(line), Some(-1), "info line: {line}");
    let pv = info_pv(line);
    assert_eq!(
        pv.len(),
        2,
        "mate -1 is one own move plus Black's mate: {line}"
    );
    assert_eq!(pv.first(), Some(&chosen), "bestmove starts the PV: {line}");
    let mut game = Game::from_fen(mated_in_one).expect("mated-in-one fen");
    for movement in &pv {
        apply(&mut game, movement);
    }
    assert_eq!(
        game.outcome(),
        Outcome::Checkmate {
            winner: Color::Black
        },
        "PV must end in checkmate: {line}"
    );

    assert!(uci.quit().success());
}

#[test]
fn claimable_draw_still_returns_a_legal_move() {
    // UCI has no draw-claim command, so a claimable threefold repetition is
    // answered with a legal move instead of a claim or a null move.
    let moves = [
        "g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8",
    ];
    let game = replay(&moves);
    assert!(game.claimable_now().any(), "threefold must be claimable");

    let mut uci = Uci::spawn();
    let (chosen, _) = uci.search(&[
        "position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8",
        "go depth 2",
    ]);
    assert_ne!(chosen, "0000");
    assert_legal(&game, &chosen);

    assert!(uci.quit().success());
}

#[test]
fn malformed_positions_are_rejected_and_keep_previous_state() {
    let mut uci = Uci::spawn();
    uci.send("position startpos moves e2e4");

    let before = uci.seen.len();
    uci.send("position startpos moves e2e5");
    uci.wait_for("info string", OUTPUT_TIMEOUT);
    assert!(uci.seen[before..]
        .iter()
        .any(|line| line.contains("illegal or unknown move")));

    let before = uci.seen.len();
    uci.send("position fen not-a-fen");
    uci.wait_for("info string", OUTPUT_TIMEOUT);
    assert!(uci.seen[before..]
        .iter()
        .any(|line| line.contains("invalid fen") || line.contains("needs six fields")));

    let before = uci.seen.len();
    uci.send("position startpos e2e4");
    uci.wait_for("info string", OUTPUT_TIMEOUT);
    assert!(uci.seen[before..]
        .iter()
        .any(|line| line.contains("position expects moves")));

    let expected = replay(&["e2e4"]);
    let (chosen, _) = uci.search(&["go depth 1"]);
    // The game is still the one after e2e4 (Black to move), not the start
    // position suggested by the rejected commands.
    assert_legal(&expected, &chosen);

    assert!(uci.quit().success());
}

#[test]
fn draw_claim_recommendations_are_not_reported_as_fallback_scores() {
    let start = "7k/8/8/8/8/8/8/K1Q5 w - - 0 1";
    let moves = ["a1a2", "h8g8", "a2a1", "g8h8", "a1a2", "h8g8", "a2a1"];
    let mut intended = Game::from_fen(start).unwrap();
    for movement in moves {
        apply(&mut intended, movement);
    }
    let mut uci = Uci::spawn();
    let before = uci.seen.len();
    let (chosen, _) = uci.search(&[
        "position fen 7k/8/8/8/8/8/8/K1Q5 w - - 0 1 moves a1a2 h8g8 a2a1 g8h8 a1a2 h8g8 a2a1",
        "go depth 3",
    ]);
    let lines = &uci.seen[before..];
    assert_legal(&intended, &chosen);
    assert!(
        lines
            .iter()
            .any(|line| line.contains("draw claim recommended")
                && line.contains("g8h8")
                && line.contains("controller")),
        "{lines:?}"
    );
    assert!(
        lines.iter().any(|line| line.contains("unsearched")),
        "{lines:?}"
    );
    assert!(
        lines
            .iter()
            .all(|line| !line.contains(" score ") && !line.contains(" pv ")),
        "{lines:?}"
    );

    let mut current = Game::from_fen("7k/8/8/8/8/8/8/K1Q5 b - - 0 1").unwrap();
    for movement in [
        "h8g8", "a1a2", "g8h8", "a2a1", "h8g8", "a1a2", "g8h8", "a2a1",
    ] {
        apply(&mut current, movement);
    }
    assert!(current.claimable_now().any());
    let before = uci.seen.len();
    let (chosen, _) = uci.search(&[
        "position fen 7k/8/8/8/8/8/8/K1Q5 b - - 0 1 moves h8g8 a1a2 g8h8 a2a1 h8g8 a1a2 g8h8 a2a1",
        "go depth 3 searchmoves h8h7",
    ]);
    let lines = &uci.seen[before..];
    assert_eq!(chosen, "h8h7");
    assert_legal(&current, &chosen);
    assert!(
        lines
            .iter()
            .any(|line| line.contains("draw claim recommended")
                && line.contains("current")
                && line.contains("controller")),
        "{lines:?}"
    );
    assert!(
        lines.iter().any(|line| line.contains("unsearched")),
        "{lines:?}"
    );
    assert!(
        lines
            .iter()
            .all(|line| !line.contains(" score ") && !line.contains(" pv ")),
        "{lines:?}"
    );
    apply(&mut current, &chosen);
    assert!(
        !current.claimable_now().any(),
        "playing the fallback loses the current claim"
    );
    assert!(uci.quit().success());
}

#[test]
fn retained_terminal_and_completed_ponder_wait_for_release() {
    let mut uci = Uci::spawn();
    uci.send_all(&[
        "position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
        "go infinite",
        "isready",
    ]);
    uci.wait_for("readyok", OUTPUT_TIMEOUT);
    assert!(!uci.seen.iter().any(|line| line.starts_with("bestmove ")));
    uci.no_bestmove_window(Duration::from_millis(300));
    uci.send("stop");
    assert_eq!(uci.wait_for("bestmove ", OUTPUT_TIMEOUT), "bestmove 0000");
    let mark = uci.mark();
    uci.send_all(&["position startpos", "go ponder depth 1", "isready"]);
    uci.wait_for("readyok", OUTPUT_TIMEOUT);
    assert!(!uci.seen[mark..]
        .iter()
        .any(|line| line.starts_with("bestmove ")));
    uci.no_bestmove_window(Duration::from_millis(300));
    uci.send("ponderhit");
    uci.wait_for("bestmove ", OUTPUT_TIMEOUT);
    uci.no_bestmove_window(Duration::from_millis(200));
    uci.send("isready");
    uci.wait_for("readyok", OUTPUT_TIMEOUT);
    assert_eq!(
        uci.seen[mark..]
            .iter()
            .filter(|line| line.starts_with("bestmove "))
            .count(),
        1
    );
    assert!(uci.quit().success());
}

#[test]
fn mate_diagnostic_and_nnue_root_restriction() {
    let mut uci = Uci::spawn();
    uci.send_all(&[
        "setoption name Eval value nnue",
        "position startpos",
        "go mate 1 depth 1 searchmoves a2a3",
    ]);
    let diagnostic = uci.wait_for("info string go mate", OUTPUT_TIMEOUT);
    assert!(diagnostic.contains("unsupported"));
    assert_eq!(uci.wait_for("bestmove ", OUTPUT_TIMEOUT), "bestmove a2a3");
    assert!(uci.quit().success());
}

#[test]
fn restricted_special_moves_and_f06_intended_claim() {
    let mut uci = Uci::spawn();
    for (fen, movement) in [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("8/P7/8/8/8/8/8/K6k w - - 0 1", "a7a8q"),
        ("8/P7/8/8/8/8/8/K6k w - - 0 1", "a7a8n"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
    ] {
        for eval in ["classical", "nnue"] {
            let option = format!("setoption name Eval value {eval}");
            let position = format!("position fen {fen}");
            let go = format!("go depth 1 searchmoves {movement}");
            assert_eq!(uci.search(&[&option, &position, &go]).0, movement);
        }
    }
    let position =
        "position fen 7k/8/8/8/8/8/8/K1Q5 w - - 0 1 moves a1a2 h8g8 a2a1 g8h8 a1a2 h8g8 a2a1";
    for (movement, claim) in [("g8h8", true), ("g8h7", false)] {
        let mark = uci.mark();
        let go = format!("go depth 2 searchmoves {movement}");
        assert_eq!(uci.search(&[position, &go]).0, movement);
        assert_eq!(
            uci.seen[mark..]
                .iter()
                .any(|line| line.contains("draw claim recommended")),
            claim
        );
    }
    assert!(uci.quit().success());
}

#[test]
fn live_ponder_hit_starts_fresh_movetime() {
    let mut uci = Uci::spawn();
    uci.send_all(&["position startpos", "go ponder movetime 200", "isready"]);
    uci.wait_for("readyok", OUTPUT_TIMEOUT);
    uci.no_bestmove_window(Duration::from_millis(400));
    let started = Instant::now();
    uci.send("ponderhit");
    uci.wait_for("bestmove ", OUTPUT_TIMEOUT);
    let elapsed = started.elapsed();
    assert!(
        elapsed >= Duration::from_millis(100) && elapsed < Duration::from_secs(3),
        "{elapsed:?}"
    );
    uci.no_bestmove_window(Duration::from_millis(200));
    assert!(uci.quit().success());
}

#[test]
fn held_result_replacement_and_exit_cleanup() {
    let mut uci = Uci::spawn();
    uci.send_all(&["position startpos", "go ponder depth 1", "isready"]);
    uci.wait_for("readyok", OUTPUT_TIMEOUT);
    uci.no_bestmove_window(Duration::from_millis(300));
    uci.send_all(&["position startpos moves e2e4", "isready"]);
    uci.wait_for("readyok", OUTPUT_TIMEOUT);
    uci.no_bestmove_window(Duration::from_millis(200));
    let mark = uci.mark();
    let (chosen, _) = uci.search(&["go depth 1 searchmoves a7a6"]);
    assert_eq!(chosen, "a7a6");
    uci.no_bestmove_window(Duration::from_millis(200));
    assert_eq!(
        uci.seen[mark..]
            .iter()
            .filter(|l| l.starts_with("bestmove "))
            .count(),
        1
    );
    assert!(uci.quit().success());

    for quit in [true, false] {
        let mut uci = Uci::spawn();
        uci.send_all(&["position startpos", "go ponder depth 1", "isready"]);
        uci.wait_for("readyok", OUTPUT_TIMEOUT);
        uci.no_bestmove_window(Duration::from_millis(300));
        if quit {
            uci.send("quit");
        }
        uci.close_stdin();
        assert!(uci.wait_exit(EXIT_TIMEOUT).success());
        while let Ok(line) = uci.lines.try_recv() {
            uci.seen.push(line);
        }
        assert!(
            uci.seen
                .iter()
                .filter(|l| l.starts_with("bestmove "))
                .count()
                <= 1
        );
    }
}
