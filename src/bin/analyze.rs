use std::process;
use std::time::Instant;

use horseflight::eval::mate_distance_plies;
use horseflight::{
    search, Color, DrawClaim, Game, HandcraftedEvaluator, Move, Role, SearchAction, SearchConfig,
    SearchResult, Square,
};

const DEFAULT_NODE_LIMIT: u64 = 5_000_000;

fn main() {
    if let Err(message) = run() {
        eprintln!("error: {message}");
        process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let mut fen: Option<String> = None;
    let mut moves: Option<String> = None;
    let mut depth: Option<u32> = None;
    let mut node_limit = DEFAULT_NODE_LIMIT;
    let mut q_depth_limit = 4;
    let mut ordering = true;
    let mut quiescence = true;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            "--fen" => {
                fen = Some(
                    args.next()
                        .ok_or_else(|| "--fen requires a quoted FEN argument".to_string())?,
                );
            }
            "--moves" => {
                moves = Some(
                    args.next()
                        .ok_or_else(|| "--moves requires a quoted move sequence".to_string())?,
                );
            }
            "--depth" => depth = Some(parse_count(&mut args, "--depth")?),
            "--nodes" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--nodes requires a non-negative integer".to_string())?;
                node_limit = value.parse::<u64>().map_err(|_| {
                    format!("node limit must be a non-negative integer, got {value:?}")
                })?;
            }
            "--q-depth" => q_depth_limit = parse_count(&mut args, "--q-depth")?,
            "--no-ordering" => ordering = false,
            "--no-quiescence" => quiescence = false,
            other if other.starts_with("--") => {
                return Err(format!("unknown option {other:?}; try --help"));
            }
            other => {
                return Err(format!(
                    "unexpected argument {other:?}; depth is passed as --depth <N>"
                ));
            }
        }
    }

    let depth = depth.ok_or_else(|| "missing --depth; try --help".to_string())?;
    let config = SearchConfig {
        max_depth: depth,
        node_limit,
        ordering,
        quiescence,
        q_depth_limit,
    }
    .validate()
    .map_err(|error| format!("invalid search limits: {error}"))?;

    let mut game = match &fen {
        Some(text) => Game::from_fen(text).map_err(|error| format!("invalid FEN: {error}"))?,
        None => Game::startpos(),
    };
    let move_text = moves.unwrap_or_default();
    let mut played: Vec<Move> = Vec::new();
    for token in move_text.split_whitespace() {
        let movement = parse_move(token)?;
        game.play(movement)
            .map_err(|error| format!("illegal move {token:?}: {error}"))?;
        played.push(movement);
    }

    let started = Instant::now();
    let result = search(&game, &HandcraftedEvaluator, &config)
        .map_err(|error| format!("search failed: {error}"))?;
    let elapsed = started.elapsed();

    println!("fen: {}", game.position().to_fen());
    let played: Vec<String> = played.iter().map(|movement| movement.to_uci()).collect();
    println!(
        "moves: {}",
        if played.is_empty() {
            "-".to_string()
        } else {
            played.join(" ")
        }
    );
    println!(
        "depth: requested {} completed {}",
        config.max_depth, result.completed_depth
    );
    println!("stop: {}", result.stop);
    println!("action: {}", describe_action(&result));
    println!("claim: {}", describe_claim(&result));
    println!(
        "score: {}",
        describe_score(&result, game.position().side_to_move())
    );
    println!("pv: {}", describe_pv(&result));
    println!(
        "nodes: {} (q {}, main {})",
        result.nodes,
        result.q_nodes,
        result.nodes - result.q_nodes
    );
    println!("time: {:.6}s", elapsed.as_secs_f64());
    println!("nps: {:.0}", result.nps());
    Ok(())
}

fn parse_count(args: &mut impl Iterator<Item = String>, option: &str) -> Result<u32, String> {
    let value = args
        .next()
        .ok_or_else(|| format!("{option} requires a non-negative integer"))?;
    value
        .parse::<u32>()
        .map_err(|_| format!("{option} must be a non-negative integer, got {value:?}"))
}

fn parse_move(token: &str) -> Result<Move, String> {
    // Coordinate moves are ASCII, so reject non-ASCII before slicing: a
    // multi-byte character straddling a slice boundary would panic instead of
    // producing the documented error.
    let not_a_move = || format!("move {token:?} is not a coordinate move such as e2e4 or e7e8q");
    if !token.is_ascii() || (token.len() != 4 && token.len() != 5) {
        return Err(not_a_move());
    }
    let from = Square::from_ascii(&token[0..2]).ok_or_else(not_a_move)?;
    let to = Square::from_ascii(&token[2..4]).ok_or_else(not_a_move)?;
    if let Some(promotion) = token[4..].chars().next() {
        let role = Role::from_uci_char(promotion).ok_or_else(not_a_move)?;
        return Ok(Move::promotion(from, to, role));
    }
    Ok(Move::normal(from, to))
}

fn describe_action(result: &SearchResult) -> String {
    match result.action {
        SearchAction::Move(movement) => format!("move {}", movement.to_uci()),
        SearchAction::Claim(DrawClaim::Current) => "claim current draw".to_string(),
        SearchAction::Claim(DrawClaim::Intended(movement)) => {
            format!("claim draw with intended move {}", movement.to_uci())
        }
        SearchAction::Terminal(outcome) => format!("terminal: {outcome}"),
        SearchAction::Fallback(movement) => {
            format!("fallback move {} (not searched)", movement.to_uci())
        }
    }
}

fn describe_claim(result: &SearchResult) -> String {
    match result.claim {
        None => "none".to_string(),
        Some(DrawClaim::Current) => "current".to_string(),
        Some(DrawClaim::Intended(movement)) => format!("intended {}", movement.to_uci()),
    }
}

fn describe_score(result: &SearchResult, side: Color) -> String {
    let Some(score) = result.score else {
        return "none (no completed search)".to_string();
    };
    match mate_distance_plies(score) {
        // Only a terminal root reaches distance zero: the side to move is mated.
        Some(0) => format!("checkmated for {side} ({score})"),
        Some(plies) => {
            let moves = plies.div_ceil(2);
            if score > 0 {
                format!("mate in {moves} for {side} ({score})")
            } else {
                format!("mated in {moves} for {side} ({score})")
            }
        }
        None => format!("{score} cp for {side}"),
    }
}

fn describe_pv(result: &SearchResult) -> String {
    if result.pv.is_empty() {
        return "-".to_string();
    }
    result
        .pv
        .iter()
        .map(|movement| movement.to_uci())
        .collect::<Vec<String>>()
        .join(" ")
}

fn print_usage() {
    println!(
        "Usage: analyze [--fen <FEN>] [--moves \"<uci> ...\"] --depth <N> [options]\n\
         \n\
         Search a chess position and report the selected action, score, principal\n\
         variation, node counts, stop reason and timing. No interface protocol.\n\
         \n\
         Options:\n\
           --fen <FEN>        Quoted FEN; default is the standard start position.\n\
           --moves \"<uci>\"    Space-separated coordinate moves played from the FEN to\n\
                              build real history (for repetitions); FEN alone starts fresh.\n\
           --depth <N>        Requested iterative-deepening depth (required, 1..={}).\n\
           --nodes <N>        Hard aggregate main-plus-quiescence node budget (default {}).\n\
           --q-depth <N>      Capture plies allowed at a non-check q frontier (default 4).\n\
           --no-ordering      Use move-generation order instead of PV/MVV-LVA ordering.\n\
           --no-quiescence    Evaluate depth-zero leaves statically (diagnostic baseline).\n\
           -h, --help         Print this help.\n\
         \n\
         Scores are centipawns from the side to move's perspective; mate scores are\n\
         reported as a distance in moves. Exits nonzero on invalid options, moves or FEN.",
        horseflight::search::MAX_MAIN_DEPTH,
        DEFAULT_NODE_LIMIT
    );
}
