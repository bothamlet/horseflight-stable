//! UCI integration tests for evaluator selection.
//!
//! These tests drive the built `uci` binary exactly as a GUI or match runner
//! would, proving the handshake, the opt-in `Eval` option, the truthful
//! active-model diagnostics and the unchanged default (classical) behaviour.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(20);

struct Uci {
    child: Child,
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
        let stdout = child.stdout.take().expect("piped stdout");
        let (sender, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        Uci {
            child,
            lines,
            seen: Vec::new(),
        }
    }

    fn send(&mut self, command: &str) {
        let stdin = self.child.stdin.as_mut().expect("piped stdin");
        writeln!(stdin, "{command}").expect("write command");
        stdin.flush().expect("flush command");
    }

    fn next_line(&mut self) -> String {
        let line = self
            .lines
            .recv_timeout(TIMEOUT)
            .unwrap_or_else(|_| panic!("timed out; saw {:?}", self.seen));
        self.seen.push(line.clone());
        line
    }

    fn wait_for(&mut self, prefix: &str) -> String {
        loop {
            let line = self.next_line();
            if line.starts_with(prefix) || line == prefix {
                return line;
            }
        }
    }

    fn handshake(&mut self) -> Vec<String> {
        self.send("uci");
        let mut lines = Vec::new();
        loop {
            let line = self.next_line();
            lines.push(line.clone());
            if line == "uciok" {
                return lines;
            }
        }
    }

    fn ready(&mut self) {
        self.send("isready");
        self.wait_for("readyok");
    }

    /// Run `go depth <depth>` and return (info lines, bestmove).
    fn search(&mut self, position: &str, depth: u32) -> (Vec<String>, String) {
        self.send(position);
        self.send(&format!("go depth {depth}"));
        let mut info = Vec::new();
        loop {
            let line = self.next_line();
            if line.starts_with("bestmove") {
                return (info, line);
            }
            if line.starts_with("info ") {
                info.push(line);
            }
        }
    }
}

impl Drop for Uci {
    fn drop(&mut self) {
        if let Some(stdin) = self.child.stdin.as_mut() {
            let _ = writeln!(stdin, "quit");
            let _ = stdin.flush();
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            if self.child.try_wait().ok().flatten().is_some() {
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Parse a UCI `bestmove` line and prove it is legal in `game`.
fn assert_legal_bestmove(game: &horseflight::Game, bestmove: &str) {
    let mut tokens = bestmove.split_whitespace();
    assert_eq!(tokens.next(), Some("bestmove"));
    let text = tokens.next().expect("bestmove value");
    let legal = game.position().legal_moves();
    if text == "0000" {
        assert!(
            legal.is_empty(),
            "0000 reported although {} legal moves exist",
            legal.len()
        );
        return;
    }
    let from = horseflight::Square::from_ascii(&text[..2]).expect("origin square");
    let to = horseflight::Square::from_ascii(&text[2..4]).expect("target square");
    let promotion = text
        .as_bytes()
        .get(4)
        .map(|letter| horseflight::Role::from_uci_char(*letter as char).expect("promotion role"));
    let chosen = legal
        .iter()
        .copied()
        .find(|movement| {
            movement.from() == from && movement.to() == to && movement.promotion_role() == promotion
        })
        .unwrap_or_else(|| {
            panic!(
                "reported move {text} is not legal in {}",
                game.position().to_fen()
            )
        });
    assert!(game.position().is_legal(chosen));
}

#[test]
fn handshake_advertises_the_eval_option_and_truthful_model_diagnostics() {
    let mut uci = Uci::spawn();
    let lines = uci.handshake();

    assert!(
        lines
            .iter()
            .any(|line| line.starts_with("id name Horseflight")),
        "missing engine id: {lines:?}"
    );
    let option = lines
        .iter()
        .find(|line| line.starts_with("option name Eval "))
        .unwrap_or_else(|| panic!("missing Eval option: {lines:?}"));
    assert!(option.contains("type combo"), "wrong option type: {option}");
    assert!(
        option.contains("default classical"),
        "classical must stay the default: {option}"
    );
    assert!(
        option.contains("var classical"),
        "missing classical: {option}"
    );
    assert!(option.contains("var nnue"), "missing nnue: {option}");

    let diagnostics = lines
        .iter()
        .find(|line| line.contains("asset_sha256="))
        .unwrap_or_else(|| panic!("missing model diagnostics: {lines:?}"));
    assert!(
        diagnostics.contains("training_data=lichess-open-database"),
        "training data source not reported: {diagnostics}"
    );
    assert!(
        diagnostics.contains("derived: white-relative"),
        "the derived (not verified) perspective must be explicit: {diagnostics}"
    );
    uci.ready();
}

#[test]
fn default_search_reports_the_classical_evaluator() {
    let mut uci = Uci::spawn();
    uci.handshake();
    uci.ready();
    let (info, bestmove) = uci.search("position startpos", 2);
    assert!(
        info.iter().any(|line| line == "info string eval=classical"),
        "the active evaluator must be reported: {info:?}"
    );
    assert!(
        !info.iter().any(|line| line.contains("eval=nnue")),
        "classical must not claim the neural evaluator: {info:?}"
    );
    assert_legal_bestmove(&horseflight::Game::startpos(), &bestmove);
    // Pinned classical output from the unchanged engine (N1 regression values).
    assert!(
        info.iter()
            .any(|line| line.starts_with("info depth 2 score cp 0 nodes 96")),
        "classical depth-2 search changed: {info:?}"
    );
}

#[test]
fn switching_to_nnue_reports_the_embedded_model_and_plays_legal_moves() {
    let mut uci = Uci::spawn();
    uci.handshake();
    uci.ready();
    uci.send("setoption name Eval value nnue");
    let confirmation = uci.wait_for("info string Eval=nnue");
    assert!(
        confirmation.contains("model=horseflight-nnue"),
        "model id missing: {confirmation}"
    );
    assert!(
        confirmation.contains("asset_sha256="),
        "model checksum missing: {confirmation}"
    );
    assert!(
        confirmation.contains("training_data=lichess-open-database"),
        "training data source not reported: {confirmation}"
    );

    let (info, bestmove) = uci.search("position startpos", 3);
    assert!(
        info.iter()
            .any(|line| line.starts_with("info string eval=nnue model_sha256=")),
        "the neural evaluator must prove it ran: {info:?}"
    );
    assert_legal_bestmove(&horseflight::Game::startpos(), &bestmove);
    // A searched line must carry a score and a node count from the session path.
    let searched = info
        .iter()
        .find(|line| line.starts_with("info depth"))
        .unwrap_or_else(|| panic!("no search info: {info:?}"));
    assert!(searched.contains("score"), "missing score: {searched}");
    assert!(searched.contains("nodes"), "missing nodes: {searched}");

    // The engine still honours a later switch back to classical.
    uci.send("setoption name Eval value classical");
    uci.wait_for("info string Eval=classical");
    let (info, bestmove) = uci.search("position startpos", 2);
    assert!(info.iter().any(|line| line == "info string eval=classical"));
    assert_legal_bestmove(&horseflight::Game::startpos(), &bestmove);
}

#[test]
fn an_unknown_eval_value_is_rejected_without_changing_the_selection() {
    let mut uci = Uci::spawn();
    uci.handshake();
    uci.ready();
    uci.send("setoption name Eval value Stockfish");
    let rejection = uci.wait_for("info string unknown Eval value");
    assert!(
        rejection.contains("keeping classical"),
        "the default must be kept: {rejection}"
    );
    let (info, bestmove) = uci.search("position startpos", 2);
    assert!(info.iter().any(|line| line == "info string eval=classical"));
    assert_legal_bestmove(&horseflight::Game::startpos(), &bestmove);

    // An unknown option name is reported and changes nothing either.
    uci.send("setoption name UCI_Elo value 3000");
    let rejection = uci.wait_for("info string unknown option");
    assert!(rejection.contains("UCI_Elo"), "wrong name: {rejection}");
}

#[test]
fn ucinewgame_keeps_the_selected_evaluator() {
    let mut uci = Uci::spawn();
    uci.handshake();
    uci.ready();
    uci.send("setoption name Eval value nnue");
    uci.wait_for("info string Eval=nnue");
    uci.send("ucinewgame");
    uci.ready();
    let (info, bestmove) = uci.search("position startpos", 2);
    assert!(
        info.iter()
            .any(|line| line.starts_with("info string eval=nnue")),
        "ucinewgame must not silently reset the evaluator: {info:?}"
    );
    assert_legal_bestmove(&horseflight::Game::startpos(), &bestmove);
}

#[test]
fn nnue_search_reports_mates_and_terminals_truthfully() {
    let mut uci = Uci::spawn();
    uci.handshake();
    uci.ready();
    uci.send("setoption name Eval value nnue");
    uci.wait_for("info string Eval=nnue");

    // Mate in one for the side to move, with the neural evaluator active.
    let (info, bestmove) = uci.search("position fen 6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 3);
    assert!(
        info.iter().any(|line| line.contains("score mate 1")),
        "mate score missing: {info:?}"
    );
    assert_eq!(bestmove, "bestmove d1d8");
    let mate_game = horseflight::Game::from_fen("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1").unwrap();
    assert_legal_bestmove(&mate_game, &bestmove);

    // An already-finished game still returns the protocol's terminal answer.
    let (info, bestmove) = uci.search("position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", 2);
    assert!(
        info.iter().any(|line| line.contains("score mate 0")),
        "terminal score missing: {info:?}"
    );
    assert_eq!(bestmove, "bestmove 0000");
}
