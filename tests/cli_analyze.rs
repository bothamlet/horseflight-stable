use std::process::Command;

fn analyze() -> Command {
    Command::new(env!("CARGO_BIN_EXE_analyze"))
}

fn stdout_of(command: &mut Command) -> String {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn stderr_of(command: &mut Command) -> String {
    let output = command.output().unwrap();
    assert!(!output.status.success(), "command unexpectedly succeeded");
    String::from_utf8(output.stderr).unwrap()
}

const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

#[test]
fn cli_reports_a_completed_search() {
    let stdout = stdout_of(analyze().args(["--depth", "2"]));
    for needle in [
        "fen: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "moves: -",
        "depth: requested 2 completed 2",
        "stop: depth reached",
        "action: move ",
        "claim: none",
        "score: ",
        "pv: ",
        "nodes: ",
        "(q ",
        "time: ",
        "nps: ",
    ] {
        assert!(stdout.contains(needle), "missing {needle:?} in:\n{stdout}");
    }
}

#[test]
fn cli_accepts_a_fen_and_both_ablations() {
    let stdout = stdout_of(analyze().args([
        "--fen",
        KIWIPETE,
        "--depth",
        "2",
        "--no-ordering",
        "--no-quiescence",
    ]));
    assert!(stdout.contains("stop: depth reached"), "{stdout}");
    assert!(
        stdout.contains("depth: requested 2 completed 2"),
        "{stdout}"
    );
    assert!(stdout.contains("(q "), "{stdout}");
}

#[test]
fn cli_builds_history_from_coordinate_moves() {
    // Two shuffling cycles make the start-of-cycle position a current claim.
    let stdout = stdout_of(analyze().args([
        "--fen",
        "7k/Q7/8/8/8/8/8/K7 b - - 0 1",
        "--moves",
        "h8g8 a1b1 g8h8 b1a1 h8g8 a1b1 g8h8 b1a1",
        "--depth",
        "2",
    ]));
    assert!(stdout.contains("moves: h8g8 a1b1 g8h8 b1a1"), "{stdout}");
    assert!(stdout.contains("claim: current"), "{stdout}");
    assert!(stdout.contains("action: claim current draw"), "{stdout}");
    assert!(stdout.contains("pv: -"), "{stdout}");
}

#[test]
fn cli_reports_a_node_limited_fallback_truthfully() {
    let stdout = stdout_of(analyze().args(["--depth", "3", "--nodes", "0"]));
    assert!(stdout.contains("stop: node limit"), "{stdout}");
    assert!(stdout.contains("completed 0"), "{stdout}");
    assert!(stdout.contains("action: fallback move "), "{stdout}");
    assert!(stdout.contains("score: none"), "{stdout}");
    assert!(stdout.contains("pv: -"), "{stdout}");
}

#[test]
fn cli_failures_are_nonzero_and_understandable() {
    let stderr = stderr_of(analyze().args(["--fen", "not a fen", "--depth", "2"]));
    assert!(stderr.contains("invalid FEN"), "{stderr}");

    let stderr = stderr_of(&mut analyze());
    assert!(stderr.contains("missing --depth"), "{stderr}");

    let stderr = stderr_of(analyze().args(["--depth", "0"]));
    assert!(stderr.contains("depth"), "{stderr}");

    let stderr = stderr_of(analyze().args(["--depth", "2", "--nodes", "many"]));
    assert!(stderr.contains("node limit"), "{stderr}");

    let stderr = stderr_of(analyze().args(["--depth", "2", "--moves", "e2e5"]));
    assert!(stderr.contains("illegal move"), "{stderr}");

    let stderr = stderr_of(analyze().args(["--depth", "2", "--moves", "e2e4 notamove"]));
    assert!(stderr.contains("not a coordinate move"), "{stderr}");

    let stderr = stderr_of(analyze().args(["--depth", "2", "--q-depth", "999"]));
    assert!(stderr.contains("quiescence depth"), "{stderr}");

    let stderr = stderr_of(analyze().args(["--depth", "2", "--unknown"]));
    assert!(stderr.contains("unknown option"), "{stderr}");
}

#[test]
fn cli_reports_mate_distances_and_terminal_checkmate() {
    let stdout = stdout_of(analyze().args([
        "--fen",
        "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        "--depth",
        "1",
    ]));
    assert!(
        stdout.contains("score: mate in 1 for white (31999)"),
        "{stdout}"
    );
    assert!(stdout.contains("action: move a1a8"), "{stdout}");

    let stdout =
        stdout_of(analyze().args(["--fen", "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "--depth", "1"]));
    assert!(
        stdout.contains("score: checkmated for black (-32000)"),
        "{stdout}"
    );
    assert!(
        stdout.contains("action: terminal: checkmate (white wins)"),
        "{stdout}"
    );
    assert!(stdout.contains("stop: terminal root"), "{stdout}");
}

#[test]
fn cli_rejects_non_ascii_move_tokens_without_panicking() {
    // "aéb" and "aébc" are four and five bytes yet contain a multi-byte
    // character, so a naive 0..2/2..4 slice would panic on a char boundary.
    for token in ["aéb", "aébc", "é2e4", "e2é4", "e2e4é"] {
        let output = analyze()
            .args(["--depth", "1", "--moves", token])
            .output()
            .unwrap();
        assert_eq!(
            output.status.code(),
            Some(2),
            "{token:?} must fail like any other invalid move, not panic: {output:?}"
        );
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            stderr.contains("not a coordinate move"),
            "{token:?}: {stderr}"
        );
    }
}

#[test]
fn cli_help_succeeds_without_searching() {
    let stdout = stdout_of(analyze().arg("--help"));
    assert!(stdout.contains("Usage: analyze"), "{stdout}");
    assert!(stdout.contains("--moves"), "{stdout}");
    assert!(stdout.contains("--no-quiescence"), "{stdout}");
}
