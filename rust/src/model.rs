//! Model implementation for the hierarchical information aggregation simulation
//!
//! This module contains the core simulation logic and organizational model structure.
//! It defines the world parameters, organizational hierarchy, and manages the
//! step-by-step evolution of the simulation.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::agents::{Agent, Party, PartyStatus, Rank, TruthMarker};
use crate::utils::euclidean_distance;

/// World and grid constants for the simulation
/// Number of cells per side in the grid (≈ 0.5 NetLogo patch)
#[allow(dead_code)]
pub const GRID_RES: usize = 41;
/// Minimum world coordinate value
pub const WORLD_MIN: f64 = -10.0;
/// Maximum world coordinate value
pub const WORLD_MAX: f64 = 10.0;
/// Scale factor for converting between world coordinates and grid cells
#[allow(dead_code)]
pub const CELL_SCALE: f64 = (GRID_RES as f64 - 1.0) / (WORLD_MAX - WORLD_MIN);

/// Aggregation algorithm for belief updating
///
/// The simulation supports two different algorithms for updating beliefs:
/// - Bayesian: agents combine previous beliefs with new observations
/// - NonLearning: agents use only the most recent information
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Algorithm {
    /// Recursive Bayesian update (average of previous belief and new observation)
    Bayesian,
    /// Simple arithmetic mean of subordinate reports (no learning over time)
    NonLearning,
}

impl Algorithm {
    /// Convert algorithm enum to string representation
    ///
    /// Used for command-line arguments and serialization
    pub fn as_str(&self) -> &'static str {
        match self {
            Algorithm::Bayesian => "bayesian",
            Algorithm::NonLearning => "nonlearning",
        }
    }
    
    /// Parse algorithm from string representation
    ///
    /// # Arguments
    /// * `s` - String representation of the algorithm ("bayesian" or "nonlearning")
    ///
    /// # Returns
    /// * `Ok(Algorithm)` - If the string matches a known algorithm
    /// * `Err(String)` - If the string does not match a known algorithm
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "bayesian" => Ok(Algorithm::Bayesian),
            "nonlearning" => Ok(Algorithm::NonLearning),
            _ => Err(format!("Unknown algorithm: {}", s)),
        }
    }
}

/// Organizational model for the hierarchical information aggregation simulation
///
/// This is the central structure of the simulation, containing all agents,
/// their relationships, and the parameters controlling their behavior.
/// The model manages the step-by-step execution of the simulation and
/// tracks the overall state and progress.
#[derive(Debug, Serialize, Deserialize)]
pub struct OrgModel {
    /// Reliability (noise level) of observations - lower values mean more noise
    pub reliability: f64,
    /// Ratio of blue party agents (0.0 to 1.0) - controls political distribution
    pub party_ratio: f64,
    /// Aggregation algorithm used for belief updates (Bayesian or NonLearning)
    pub algo: Algorithm,
    /// Party status - determines whether agents of certain parties provide biased information
    pub party_status: PartyStatus,
    /// Whether agents should only consider subordinates of their same party
    pub same_party_filter: bool,
    /// Threshold for detecting equilibrium (convergence) - smaller means more precision
    pub eq_threshold: f64,
    
    /// Ground truth coordinates (H₀, H₁) that agents are trying to estimate
    pub truth: (f64, f64),
    /// Truth marker object representing the ground truth location
    pub truth_marker: TruthMarker,
    
    /// All agents in the model, with index 0 always being the director
    pub agents: Vec<Agent>,
    /// Organizational structure: maps from agent ID to IDs of their direct subordinates
    pub subordinates: HashMap<usize, Vec<usize>>,
    
    /// Number of cycles without significant movement (for equilibrium detection)
    pub cycles_since_move: usize,
    /// Current simulation time step
    pub tick: usize,
    /// Whether the simulation is still running or has reached equilibrium
    pub running: bool,
    
    /// Random number generator for stochastic elements (excluded from serialization)
    #[serde(skip, default = "StdRng::from_entropy")]
    pub rng: StdRng,
}

impl OrgModel {
    /// Create a new organizational model with the specified parameters
    ///
    /// This initializes a complete simulation with randomized ground truth,
    /// and builds a full organizational hierarchy with the specified number
    /// of managers, analysts, and field agents.
    ///
    /// # Arguments
    /// * `reliability` - Noise level parameter for observations (lower = more noise)
    /// * `party_ratio` - Fraction of agents that belong to the blue party (0.0 to 1.0)
    /// * `algo` - Aggregation algorithm to use (Bayesian or NonLearning)
    /// * `parties_status` - Whether agents of certain parties provide biased information
    /// * `same_party_filter` - Whether to only consider subordinates of the same party
    /// * `eq_threshold` - Threshold for detecting equilibrium (convergence)
    /// * `n_managers` - Number of managers under the director
    /// * `n_analysts` - Number of analysts per manager
    /// * `n_agents` - Number of field agents per analyst
    /// * `seed` - Optional random seed for reproducibility
    pub fn new(
        reliability: f64,
        party_ratio: f64,
        algo: Algorithm,
        parties_status: PartyStatus,
        same_party_filter: bool,
        eq_threshold: f64,
        n_managers: usize,
        n_analysts: usize, 
        n_agents: usize,
        seed: Option<u64>,
    ) -> Self {
        let seed = seed.unwrap_or_else(|| rand::random());
        let mut rng = StdRng::seed_from_u64(seed);
        
        // Generate ground truth coordinates
        let h0: f64 = rng.gen_range(WORLD_MIN..WORLD_MAX);
        let h1: f64 = rng.gen_range(WORLD_MIN..WORLD_MAX);
        let truth = (h0, h1);
        let truth_marker = TruthMarker::new(h0, h1);
        
        let mut model = Self {
            reliability,
            party_ratio,
            algo,
            party_status: parties_status,
            same_party_filter,
            eq_threshold,
            truth,
            truth_marker,
            agents: Vec::new(),
            subordinates: HashMap::new(),
            cycles_since_move: 0,
            tick: 0,
            running: true,
            rng,
        };
        
        // Build the organizational hierarchy
        model.build_hierarchy(n_managers, n_analysts, n_agents);
        
        model
    }
    
    /// Randomly assign a party affiliation based on party_ratio
    ///
    /// Returns "blue" with probability equal to party_ratio, 
    /// otherwise returns "red".
    ///
    /// # Returns
    /// Party::Blue or Party::Red according to the configured ratio
    fn rand_party(&mut self) -> Party {
        if self.rng.gen::<f64>() < self.party_ratio {
            Party::Blue
        } else {
            Party::Red
        }
    }
    
    /// Build the full organizational hierarchy
    ///
    /// Constructs the complete four-tier hierarchical organization:
    /// 1. One Director at the top
    /// 2. Multiple Managers reporting to the Director
    /// 3. Multiple Analysts reporting to each Manager
    /// 4. Multiple field Agents reporting to each Analyst
    ///
    /// # Arguments
    /// * `n_mgr` - Number of managers under the director
    /// * `n_anal` - Number of analysts per manager
    /// * `n_ag` - Number of field agents per analyst
    fn build_hierarchy(&mut self, n_mgr: usize, n_anal: usize, n_ag: usize) {
        let mut next_id = 0;
        
        // Create director
        let director = Agent::new(
            next_id,
            Rank::Director,
            Party::Red, // Director is always red
            None,
            self.truth,
            self.reliability,
            &mut self.rng,
        );
        self.agents.push(director);
        next_id += 1;
        
        // Create managers
        let director_id = 0;
        let mut manager_ids = Vec::new();
        
        for _ in 0..n_mgr {
            let manager = Agent::new(
                next_id, 
                Rank::Manager, 
                Party::Red, // Managers are always red
                Some(director_id),
                self.truth,
                self.reliability,
                &mut self.rng,
            );
            manager_ids.push(next_id);
            self.agents.push(manager);
            next_id += 1;
        }
        
        self.subordinates.insert(director_id, manager_ids.clone());
        
        // Create analysts
        let mut analyst_ids = Vec::new();
        
        for &manager_id in &manager_ids {
            let mut manager_analysts = Vec::new();
            
            for _ in 0..n_anal {
                let analyst = Agent::new(
                    next_id,
                    Rank::Analyst,
                    Party::Red, // Analysts are always red
                    Some(manager_id),
                    self.truth,
                    self.reliability,
                    &mut self.rng,
                );
                manager_analysts.push(next_id);
                analyst_ids.push(next_id);
                self.agents.push(analyst);
                next_id += 1;
            }
            
            self.subordinates.insert(manager_id, manager_analysts);
        }
        
        // Create agents
        for &analyst_id in &analyst_ids {
            let mut analyst_agents = Vec::new();
            
            for _ in 0..n_ag {
                let agent = Agent::new(
                    next_id,
                    Rank::Agent,
                    self.rand_party(), // Field agents can be either party
                    Some(analyst_id),
                    self.truth,
                    self.reliability,
                    &mut self.rng,
                );
                analyst_agents.push(next_id);
                self.agents.push(agent);
                next_id += 1;
            }
            
            self.subordinates.insert(analyst_id, analyst_agents);
        }
    }
    
    /// Calculate the director's error in estimating the ground truth
    ///
    /// This measures how close the director's final belief is to the actual
    /// ground truth, which is a key performance metric for the organization.
    ///
    /// # Returns
    /// The Euclidean distance between the director's belief and the ground truth
    pub fn director_error(&self) -> f64 {
        // Director is always the first agent
        let director = &self.agents[0];
        euclidean_distance((director.ph_zero, director.ph_one), self.truth)
    }
    
    /// Get all direct subordinates for a given agent
    ///
    /// Retrieves the agents that directly report to the specified agent.
    /// This is used during belief updating to aggregate information.
    ///
    /// # Arguments
    /// * `agent_id` - The ID of the agent whose subordinates to retrieve
    ///
    /// # Returns
    /// A vector containing clones of all direct subordinate agents
    fn get_subordinates(&self, agent_id: usize) -> Vec<Agent> {
        if let Some(sub_ids) = self.subordinates.get(&agent_id) {
            sub_ids.iter()
                .map(|&id| self.agents[id].clone())
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Advance the simulation by one time step
    ///
    /// This is the core simulation function that performs one complete update cycle.
    /// It uses a two-phase update process:
    /// 1. All agents compute their next beliefs based on subordinate information
    /// 2. All agents commit their new beliefs simultaneously
    ///
    /// The simulation automatically detects equilibrium (when beliefs stop changing
    /// significantly) and will set `running` to false when convergence is achieved.
    pub fn step(&mut self) {
        if !self.running {
            return;
        }
        
        // Director is always the first agent
        let director = &self.agents[0];
        let prev_dir_belief = (director.ph_zero, director.ph_one);
        
        // First pass: all agents compute their next belief (step phase)
        for i in 0..self.agents.len() {
            let agent_id = self.agents[i].id;
            let subordinates = self.get_subordinates(agent_id);
            
            // Clone the agent to avoid borrowing issues
            let mut agent = self.agents[i].clone();
            
            agent.step(
                self.truth,
                self.reliability,
                self.party_status,
                self.algo.as_str(),
                &subordinates,
                self.same_party_filter,
                &mut self.rng,
            );
            
            // Update the agent in the agents vector
            self.agents[i] = agent;
        }
        
        // Second pass: all agents commit their beliefs (advance phase)
        for agent in &mut self.agents {
            agent.advance();
        }
        
        // Check for equilibrium
        let director = &self.agents[0];
        let new_dir_belief = (director.ph_zero, director.ph_one);
        let move_dist = euclidean_distance(prev_dir_belief, new_dir_belief);
        
        if move_dist < self.eq_threshold {
            self.cycles_since_move += 1;
        } else {
            self.cycles_since_move = 0;
        }
        
        // Stop if equilibrium reached
        if self.cycles_since_move >= 10 {
            self.running = false;
        }
        
        self.tick += 1;
    }
    
    /// Run the simulation until equilibrium or maximum time steps
    ///
    /// Repeatedly calls step() until either:
    /// - The simulation reaches equilibrium (director's belief stops changing)
    /// - The maximum number of time steps is reached
    ///
    /// # Arguments
    /// * `max_ticks` - Maximum number of time steps to run
    ///
    /// # Returns
    /// A tuple containing:
    /// - The number of time steps executed
    /// - The final error between the director's belief and ground truth
    pub fn run(&mut self, max_ticks: usize) -> (usize, f64) {
        while self.running && self.tick < max_ticks {
            self.step();
        }
        
        (self.tick, self.director_error())
    }
    
    /// Run multiple simulations with different random seeds
    ///
    /// Creates and runs multiple simulation instances with the same parameters
    /// but different random seeds, to gather statistical results.
    ///
    /// # Arguments
    /// * `algo` - The aggregation algorithm to use
    /// * `reliability` - The noise level parameter
    /// * `n_runs` - Number of simulation runs to perform
    /// * `max_ticks` - Maximum ticks per simulation
    ///
    /// # Returns
    /// A vector of (ticks, error) pairs, one for each simulation run
    #[allow(dead_code)]
    pub fn batch_run(
        algo: Algorithm,
        reliability: f64,
        n_runs: usize,
        max_ticks: usize,
    ) -> Vec<(usize, f64)> {
        let mut results = Vec::with_capacity(n_runs);
        
        for seed in 0..n_runs as u64 {
            let mut model = OrgModel::new(
                reliability,
                0.5, // Default party ratio
                algo,
                PartyStatus::Neutral,
                false, // No same-party filter
                0.01, // Default eq threshold
                3, // Default n_managers
                4, // Default n_analysts
                15, // Default n_agents
                Some(seed),
            );
            
            let (ticks, error) = model.run(max_ticks);
            results.push((ticks, error));
        }
        
        results
    }
}
