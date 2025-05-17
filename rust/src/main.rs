//! Hierarchical information aggregation model implementation
//!
//! This simulation is based on the model described in Michael A. Moore's comps paper 
//! *"Bayesian Decision Making in Intelligence Bureaucracies."* It implements a
//! continuous-space simulation of information flow in organizational hierarchies.
//!
//! The model explores two aggregation strategies:
//! * **bayesian** - field officers average their previous estimate with a new noisy sample (recursive Bayes)
//! * **nonlearning** - each level takes the plain arithmetic mean of the most recent messages

mod agents;
mod model;
mod utils;

use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering}; 
use std::sync::Arc;
use std::time::Instant;

use crate::agents::PartyStatus;
use crate::model::{Algorithm, OrgModel};
use crate::utils::print_progress;

/// Command-line argument representation of party status
///
/// This enum is used for parsing command-line arguments and is later
/// converted to the internal PartyStatus enum.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum PartyStatusArg {
    /// Neither party is biased
    Neutral,
    /// Blue party agents provide inverted samples
    BlueWrong,
    /// Red party agents provide inverted samples
    RedWrong,
}

impl From<PartyStatusArg> for PartyStatus {
    fn from(status: PartyStatusArg) -> Self {
        match status {
            PartyStatusArg::Neutral => PartyStatus::Neutral,
            PartyStatusArg::BlueWrong => PartyStatus::BlueWrong,
            PartyStatusArg::RedWrong => PartyStatus::RedWrong,
        }
    }
}

/// Command-line arguments for the simulation
///
/// This struct defines all the parameters that can be configured
/// when running the simulation from the command line.
#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Algorithm to use: bayesian or nonlearning
    #[arg(long, default_value = "bayesian")]
    algo: String,
    
    /// Run batch mode with both algorithms for several noise levels
    #[arg(long)]
    batch: bool,
    
    /// Number of runs to average in batch mode
    #[arg(long, default_value_t = 5)]
    runs: usize,
    
    /// Maximum number of ticks to run before stopping
    #[arg(long, default_value_t = 2000)]
    max_ticks: usize,
    
    /// Reliability (noise level) of observations
    #[arg(long, default_value_t = 1.5)]
    reliability: f64,
    
    /// Ratio of blue party agents (0.0 to 1.0)
    #[arg(long, default_value_t = 0.5)]
    party_ratio: f64,
    
    /// Party status (neutral, blue-wrong, red-wrong)
    #[arg(long, value_enum, default_value = "neutral")]
    party_status: PartyStatusArg,
    
    /// Filter subordinates by same party
    #[arg(long)]
    same_party_filter: bool,
    
    /// Threshold for equilibrium detection
    #[arg(long, default_value_t = 0.01)]
    eq_threshold: f64,
    
    /// Number of managers
    #[arg(long, default_value_t = 3)]
    n_managers: usize,
    
    /// Number of analysts per manager
    #[arg(long, default_value_t = 4)]
    n_analysts: usize,
    
    /// Number of agents per analyst
    #[arg(long, default_value_t = 15)]
    n_agents: usize,
    
    /// Random seed
    #[arg(long)]
    seed: Option<u64>,
}

/// Run a single simulation and return the results
///
/// This function creates a new model with the specified parameters and
/// runs it until completion or until the maximum number of ticks is reached.
///
/// # Returns
/// A tuple containing:
/// - The number of ticks (time steps) until completion
/// - The final error between the director's belief and the ground truth
///
/// # Arguments
/// * `args` - Configuration parameters
/// * `algo` - The aggregation algorithm to use
/// * `seed` - Optional random seed for reproducibility
fn run_once(args: &Args, algo: Algorithm, seed: Option<u64>) -> (usize, f64) {
    let mut model = OrgModel::new(
        args.reliability,
        args.party_ratio,
        algo,
        args.party_status.into(),
        args.same_party_filter,
        args.eq_threshold,
        args.n_managers,
        args.n_analysts,
        args.n_agents,
        seed,
    );
    
    model.run(args.max_ticks)
}

/// Run a batch of simulations with different parameters
///
/// This function runs multiple simulations with different parameter combinations:
/// - Both Bayesian and NonLearning algorithms
/// - Three different reliability levels (0.5, 1.5, 3.0)
/// - Multiple random seeds for each combination
///
/// Results are aggregated and printed in a table format showing the mean
/// number of ticks until completion and mean final error for each configuration.
/// Simulations are run in parallel using the rayon crate for better performance.
fn run_batch(args: &Args) {
    println!("Running batch simulations with {} runs each...", args.runs);
    
    // Reliability levels to test
    let reliabilities = vec![0.5, 1.5, 3.0];
    let algorithms = vec![Algorithm::Bayesian, Algorithm::NonLearning];
    
    let total_sims = reliabilities.len() * algorithms.len() * args.runs;
    let completed = Arc::new(AtomicUsize::new(0));
    
    println!("Will run {} total simulations", total_sims);
    
    // Results storage
    let mut results = Vec::new();
    
    let start_time = Instant::now();
    
    for reliability in &reliabilities {
        for &algo in &algorithms {
            let algo_str = algo.as_str();
            
            // Need a thread-safe counter for progress updates
            let completed_clone = Arc::clone(&completed);
            
            // Run multiple simulations in parallel
            let run_results: Vec<(usize, f64)> = (0..args.runs)
                .into_par_iter()
                .map(|seed| {
                    // Create a new Args with updated reliability
                    let mut new_args = args.clone();
                    new_args.reliability = *reliability;
                    
                    let result = run_once(
                        &new_args,
                        algo,
                        Some(seed as u64),
                    );
                    
                    // Update progress (using atomic counter)
                    let current = completed_clone.fetch_add(1, Ordering::SeqCst) + 1;
                    print_progress(current, total_sims, "Progress");
                    
                    result
                })
                .collect();
            
            // Extract ticks from results
            let ticks: Vec<usize> = run_results.iter().map(|(t, _)| *t).collect();
            let errors: Vec<f64> = run_results.iter().map(|(_, e)| *e).collect();
            
            // Compute mean values
            let mean_ticks: f64 = ticks.iter().sum::<usize>() as f64 / args.runs as f64;
            let mean_error: f64 = errors.iter().sum::<f64>() / args.runs as f64;
            
            // Compute standard deviation for ticks
            let tick_variance: f64 = ticks.iter()
                .map(|&t| (t as f64 - mean_ticks).powi(2))
                .sum::<f64>() / args.runs as f64;
            let tick_stddev: f64 = tick_variance.sqrt();
            
            // Calculate how many runs hit the max ticks limit
            let max_tick_count = ticks.iter().filter(|&&t| t >= args.max_ticks).count();
            let max_tick_percent = (max_tick_count as f64 / args.runs as f64) * 100.0;
            
            results.push((algo_str, *reliability, mean_ticks, tick_stddev, mean_error, max_tick_percent));
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("\nCompleted in {:.2?}", elapsed);
    
    // Print results in a table format
    println!("\n{:<12} {:<12} {:<12} {:<12} {:<12} {:<12}", 
             "Algorithm", "Reliability", "Mean Ticks", "StdDev", "Mean Error", "Hit Max %");
    println!("{:-<72}", "");
    
    for (algo, reliability, ticks, tick_stddev, error, max_pct) in &results {
        println!("{:<12} {:<12.1} {:<12.1} {:<12.1} {:<12.3} {:<12.1}%", 
                 algo, reliability, ticks, tick_stddev, error, max_pct);
    }
    
    // Warn the user if many simulations are hitting the max ticks limit
    let any_hitting_max = results.iter().any(|(_, _, _, _, _, max_pct)| *max_pct > 50.0);
    if any_hitting_max {
        println!("\nWARNING: Many simulations are hitting the maximum tick limit ({}). ", args.max_ticks);
        println!("Consider increasing --max-ticks if you want to see natural convergence.");
    }
}

/// Main entry point for the simulation
///
/// Parses command line arguments and either runs a single simulation
/// or a batch of simulations based on the provided arguments.
fn main() {
    // Parse command line arguments
    let args = Args::parse();
    
    // If batch mode is enabled, run multiple simulations
    if args.batch {
        run_batch(&args);
        return;
    }
    
    // For single simulation, parse the algorithm string
    let algo = match Algorithm::from_str(&args.algo) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    
    println!("Running single simulation with algorithm: {}", args.algo);
    let (ticks, error) = run_once(&args, algo, args.seed);
    
    println!("Finished in {} ticks; director final error {:.3}", ticks, error);
}
