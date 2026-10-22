use std::process;
use std::time::Instant;

use horseflight::{perft, perft_divide, Position};

const MAX_DEPTH: u32 = 12;

fn main() {
    if let Err(message) = run() {
        eprintln!("error: {message}");
        process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut fen: Option<String> = None;
    let mut divide = false;
    let mut depth: Option<u32> = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            "--divide" => divide = true,
            "--fen" => {
                fen = Some(
                    args.next()
                        .ok_or_else(|| "--fen requires a quoted FEN argument".to_string())?,
                );
            }
            other if other.starts_with("--") => {
                return Err(format!("unknown option {other:?}; try --help"));
            }
            other => {
                if depth.is_some() {
                    return Err(format!("unexpected extra argument {other:?}"));
                }
                let parsed = other
                    .parse::<u32>()
                    .map_err(|_| format!("depth must be a non-negative integer, got {other:?}"))?;
                if parsed > MAX_DEPTH {
                    return Err(format!(
                        "depth {parsed} exceeds the development CLI limit of {MAX_DEPTH}"
                    ));
                }
                depth = Some(parsed);
            }
        }
    }

    let depth = depth.ok_or_else(|| "missing depth; try --help".to_string())?;
    let position = match fen {
        Some(text) => Position::from_fen(&text).map_err(|error| format!("invalid FEN: {error}"))?,
        None => Position::startpos(),
    };

    let started = Instant::now();
    let nodes = if divide {
        let rows = perft_divide(&position, depth)
            .map_err(|error| format!("perft divide failed: {error}"))?;
        let mut rows = rows;
        rows.sort_by_key(|(movement, _)| movement.to_uci());
        let mut total = 0u64;
        for (movement, count) in &rows {
            println!("{}: {}", movement.to_uci(), count);
            total = total
                .checked_add(*count)
                .ok_or_else(|| "divide total overflowed u64".to_string())?;
        }
        if depth == 0 {
            total = 1;
        }
        println!("total: {total}");
        total
    } else {
        perft(&position, depth).map_err(|error| format!("perft failed: {error}"))?
    };
    let elapsed = started.elapsed();
    let seconds = elapsed.as_secs_f64();
    let nps = if seconds > 0.0 {
        nodes as f64 / seconds
    } else {
        0.0
    };

    println!("fen: {}", position.to_fen());
    println!("depth: {depth}");
    println!("nodes: {nodes}");
    println!("time: {seconds:.6}s");
    println!("nps: {nps:.0}");
    Ok(())
}

fn print_usage() {
    println!(
        "Usage: perft [--fen <FEN>] [--divide] <depth>\n\
         \n\
         Count legal move paths from a position (default: standard start position).\n\
         \n\
         Options:\n\
           --fen <FEN>   Quoted FEN; default is the standard start position.\n\
           --divide      Print a count for each root move plus the total.\n\
           -h, --help    Print this help.\n\
         \n\
         Depth 0 is one node. The development CLI caps depth at {MAX_DEPTH}."
    );
}
