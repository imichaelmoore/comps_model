//! Agent definitions for the information aggregation model
//!
//! This module contains the definitions for the different types of agents
//! in the organizational hierarchy, including their behavior and properties.

use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::model::{WORLD_MAX, WORLD_MIN};
use crate::utils::euclidean_distance;

/// The party affiliation of an agent.
///
/// Agents belong to either the Red or Blue party, representing different
/// political affiliations or organizational groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Party {
    /// Red party affiliation
    Red,
    /// Blue party affiliation
    Blue,
}

/// The party status in the simulation.
///
/// Determines whether agents from a specific party provide biased/inverted
/// observations of the ground truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartyStatus {
    /// Neither party is biased (truthful observations)
    Neutral,
    /// Blue party agents provide inverted observations
    BlueWrong,
    /// Red party agents provide inverted observations
    RedWrong,
}

/// The rank of an agent in the organization.
///
/// The organizational hierarchy consists of four tiers, from lowest to highest:
/// Agents (field officers), Analysts, Managers, and a single Director.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Rank {
    /// Field officer (lowest rank)
    Agent,
    /// Middle management - analyzes field reports
    Analyst,
    /// Upper management - oversees analysts
    Manager,
    /// Executive leadership (highest rank)
    Director,
}

/// An organizational agent in the hierarchy.
///
/// Agents maintain beliefs about a two-dimensional ground truth (H₀, H₁).
/// Each agent has a rank, party affiliation, and a supervisor (except the Director).
/// Agents update their beliefs based on noisy observations and information from subordinates.  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    /// Unique identifier for the agent
    pub id: usize,
    /// Agent's rank in the hierarchy (Agent, Analyst, Manager, or Director)
    pub rank: Rank,
    /// Agent's party affiliation (Red or Blue)
    pub party: Party,
    /// Supervisor ID, None if this is the director
    pub supervisor_id: Option<usize>,
    /// Current belief in dimension 0 (H₀)
    pub ph_zero: f64,
    /// Current belief in dimension 1 (H₁)
    pub ph_one: f64,
    /// Next belief to commit in the advance step (two-phase update)
    pub next_belief: Option<(f64, f64)>,
}

impl Agent {
    /// Create a new agent with an initial belief
    ///
    /// The agent starts with a noisy observation of the ground truth,
    /// where the noise level is determined by the reliability parameter.
    /// Lower reliability values result in higher noise/uncertainty.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the agent
    /// * `rank` - The agent's position in the organizational hierarchy
    /// * `party` - The agent's party affiliation
    /// * `supervisor_id` - Optional ID of the agent's direct supervisor
    /// * `truth` - The ground truth coordinates (H₀, H₁) 
    /// * `reliability` - Noise level parameter (lower = more noise)
    /// * `rng` - Random number generator for sampling initial beliefs
    pub fn new(
        id: usize,
        rank: Rank,
        party: Party,
        supervisor_id: Option<usize>,
        truth: (f64, f64),
        reliability: f64,
        rng: &mut StdRng,
    ) -> Self {
        let (h0, h1) = truth;
        let normal = Normal::new(0.0, reliability).unwrap();
        
        // Initial noisy observation of ground truth
        let ph_zero = h0 + normal.sample(rng);
        let ph_one = h1 + normal.sample(rng);
        
        Self {
            id,
            rank,
            party,
            supervisor_id,
            ph_zero,
            ph_one,
            next_belief: None,
        }
    }

    /// Take a biased sample based on party status
    ///
    /// Creates a noisy observation of the ground truth. If the agent's party
    /// is marked as "wrong" (based on party_status), the observation is inverted.
    /// This models systemic bias in information gathering based on party affiliation.
    ///
    /// # Arguments
    /// * `truth` - The ground truth coordinates (H₀, H₁)
    /// * `reliability` - Noise level parameter (lower = more noise)
    /// * `party_status` - Whether agents of a certain party provide biased observations
    /// * `rng` - Random number generator for sampling
    ///
    /// # Returns
    /// A tuple containing the (possibly biased) observation in both dimensions
    pub fn biased_sample(
        &self,
        truth: (f64, f64),
        reliability: f64,
        party_status: PartyStatus,
        rng: &mut StdRng,
    ) -> (f64, f64) {
        let (h0, h1) = truth;
        let normal = Normal::new(0.0, reliability).unwrap();
        
        let mut x = h0 + normal.sample(rng);
        let mut y = h1 + normal.sample(rng);
        
        // Invert the sample based on party status
        match (party_status, self.party) {
            (PartyStatus::BlueWrong, Party::Blue) => {
                x = -x;
                y = -y;
            }
            (PartyStatus::RedWrong, Party::Red) => {
                x = -x;
                y = -y;
            }
            _ => {}
        }
        
        (x, y)
    }
    
    /// Compute the next belief (for step phase)
    ///
    /// This is the first phase of the two-phase update process. Agents compute
    /// their next belief but don't commit it yet. The computation depends on
    /// the agent's rank and the chosen algorithm:
    /// - Field agents take a new (possibly biased) observation and combine it with 
    ///   their previous belief (Bayesian) or use it directly (Non-learning)
    /// - Higher ranks aggregate information from their subordinates
    ///
    /// # Arguments
    /// * `truth` - The ground truth coordinates (H₀, H₁)
    /// * `reliability` - Noise level parameter
    /// * `party_status` - Whether agents of a certain party provide biased observations
    /// * `algo` - The algorithm to use ("bayesian" or "nonlearning")
    /// * `subordinates` - The agents directly reporting to this agent
    /// * `same_party_filter` - Whether to only consider subordinates of the same party
    /// * `rng` - Random number generator for sampling
    pub fn step(
        &mut self,
        truth: (f64, f64),
        reliability: f64,
        party_status: PartyStatus,
        algo: &str,
        subordinates: &[Agent],
        same_party_filter: bool,
        rng: &mut StdRng,
    ) {
        match self.rank {
            Rank::Agent => {
                let (x_new, y_new) = self.biased_sample(truth, reliability, party_status, rng);
                
                if algo == "bayesian" {
                    // Bayesian update: average with previous belief
                    let x_new = (self.ph_zero + x_new) / 2.0;
                    let y_new = (self.ph_one + y_new) / 2.0;
                    self.next_belief = Some((x_new, y_new));
                } else {
                    // Non-learning: use the new sample directly
                    self.next_belief = Some((x_new, y_new));
                }
            }
            _ => {
                // Filter subordinates if same_party_filter is enabled
                let filtered_subs = if same_party_filter {
                    subordinates.iter()
                        .filter(|a| a.party == self.party)
                        .collect::<Vec<_>>()
                } else {
                    subordinates.iter().collect::<Vec<_>>()
                };
                
                // If no subordinates after filtering, just maintain current belief
                if filtered_subs.is_empty() {
                    self.next_belief = Some((self.ph_zero, self.ph_one));
                    return;
                }
                
                // Calculate mean of subordinates' beliefs
                let sum_x: f64 = filtered_subs.iter().map(|a| a.ph_zero).sum::<f64>() + self.ph_zero;
                let sum_y: f64 = filtered_subs.iter().map(|a| a.ph_one).sum::<f64>() + self.ph_one;
                
                let count = filtered_subs.len() as f64 + 1.0; // Include self
                let x_mean = sum_x / count;
                let y_mean = sum_y / count;
                
                self.next_belief = Some((x_mean, y_mean));
            }
        }
    }
    
    /// Commit the next belief (for advance phase)
    ///
    /// This is the second phase of the two-phase update process.
    /// The agent commits the belief computed in the step phase, ensuring that
    /// all agents update their beliefs simultaneously rather than sequentially.
    /// The new belief is clamped to remain within the world boundaries.
    pub fn advance(&mut self) {
        if let Some((x, y)) = self.next_belief {
            // Clamp values to world limits
            let x = x.max(WORLD_MIN).min(WORLD_MAX);
            let y = y.max(WORLD_MIN).min(WORLD_MAX);
            
            self.ph_zero = x;
            self.ph_one = y;
            self.next_belief = None;
        }
    }
    
    /// Calculate error (distance from truth)
    ///
    /// Computes the Euclidean distance between the agent's current belief
    /// and the ground truth, providing a measure of accuracy.
    ///
    /// # Arguments
    /// * `truth` - The ground truth coordinates (H₀, H₁)
    ///
    /// # Returns
    /// The Euclidean distance between the agent's belief and truth
    #[allow(dead_code)]
    pub fn error(&self, truth: (f64, f64)) -> f64 {
        euclidean_distance((self.ph_zero, self.ph_one), truth)
    }
}

/// The passive marker for the ground truth location
///
/// This structure represents the actual ground truth (H₀, H₁) that agents
/// are trying to estimate through their noisy observations and aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthMarker {
    /// Ground truth coordinate for dimension 0 (H₀)
    pub x: f64,
    /// Ground truth coordinate for dimension 1 (H₁)
    pub y: f64,
}

impl TruthMarker {
    /// Create a new truth marker at the specified coordinates
    ///
    /// # Arguments
    /// * `x` - The ground truth value for dimension 0 (H₀)
    /// * `y` - The ground truth value for dimension 1 (H₁)
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}
