use std::process::Command;

fn perft_command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_perft"))
}

#[test]
fn cli_reports_standard_perft_counts() {
    let output = perft_command().arg("2").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("nodes: 400"), "stdout: {stdout}");

    let output = perft_command().arg("0").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("nodes: 1"), "stdout: {stdout}");
}

#[test]
fn cli_divide_prints_root_moves_and_total() {
    let output = perft_command().args(["--divide", "2"]).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("a2a3: 20"), "stdout: {stdout}");
    assert!(stdout.contains("total: 400"), "stdout: {stdout}");
}

#[test]
fn cli_accepts_a_fen_argument() {
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    let output = perft_command().args(["--fen", fen, "1"]).output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("nodes: 48"), "stdout: {stdout}");
}

#[test]
fn cli_failures_are_nonzero_and_understandable() {
    let output = perft_command()
        .args(["--fen", "not a fen", "1"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("invalid FEN"), "stderr: {stderr}");

    let output = perft_command().output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("missing depth"), "stderr: {stderr}");

    let output = perft_command().arg("99").output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("exceeds"), "stderr: {stderr}");
}

#[test]
fn cli_help_succeeds() {
    let output = perft_command().arg("--help").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Usage: perft"), "stdout: {stdout}");
}
