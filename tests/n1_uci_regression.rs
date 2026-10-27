//! Mission N1 real-process UCI regression.
//!
//! The reference output below was captured from the unchanged original binary
//! at `master` HEAD 1446a74. Only `nps` and `time` (wall-clock measurements)
//! are removed before comparison; every deterministic field, including depth,
//! score, node count, PV and `bestmove`, is fixed.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

const OUTPUT_TIMEOUT: Duration = Duration::from_secs(10);

const SEARCHES: &[(&str, &[&str], &[&str])] = &[
    (
        "position startpos",
        &["go depth 2"],
        &[
            "info depth 2 score cp 0 nodes 96 pv a2a4 a7a5",
            "bestmove a2a4",
        ],
    ),
    (
        "position startpos",
        &["go nodes 50"],
        &["info depth 1 score cp 12 nodes 50 pv a2a4", "bestmove a2a4"],
    ),
    (
        "position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
        &["go depth 2"],
        &["info depth 0 score mate 0 nodes 0 pv", "bestmove 0000"],
    ),
];

/// Drop only the two wall-clock measurements from a UCI `info` line.
fn normalize_line(line: &str) -> String {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let mut kept = Vec::with_capacity(tokens.len());
    let mut index = 0;
    while index < tokens.len() {
        if tokens[index] == "nps" || tokens[index] == "time" {
            // A malformed line with a trailing key would otherwise hide it.
            index += 2;
            continue;
        }
        kept.push(tokens[index]);
        index += 1;
    }
    kept.join(" ")
}

struct Uci {
    child: Child,
    lines: Receiver<String>,
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
        Uci { child, lines }
    }

    fn send(&mut self, command: &str) {
        let stdin = self.child.stdin.as_mut().expect("piped stdin");
        writeln!(stdin, "{command}").expect("write command");
        stdin.flush().expect("flush command");
    }

    /// Run one search and return its normalized `info`/`bestmove` lines.
    fn search(&mut self, position: &str, go: &[&str]) -> Vec<String> {
        self.send(position);
        for command in go {
            self.send(command);
        }
        let deadline = Instant::now() + OUTPUT_TIMEOUT;
        let mut lines = Vec::new();
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "timed out; saw {lines:?}");
            let line = self.lines.recv_timeout(remaining).expect("UCI output");
            let normalized = normalize_line(&line);
            let is_bestmove = line.starts_with("bestmove ");
            if line.starts_with("info ") || is_bestmove {
                lines.push(normalized);
            }
            if is_bestmove {
                return lines;
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

#[test]
fn normalized_depth_and_node_limited_uci_output_matches_the_original_process() {
    let mut uci = Uci::spawn();
    for (position, go, expected) in SEARCHES {
        assert_eq!(uci.search(position, go), *expected);
    }
}
