# Hierarchical Information Aggregation Model

This is a high-performance simulation of hierarchical information aggregation described in Michael A. Moore's comps paper *"Bayesian Decision Making in Intelligence Bureaucracies."* It implements a continuous-space simulation using Rust for excellent performance and parallel processing capabilities.

## Model Overview

The model simulates a four-tier hierarchical organization where agents try to estimate a two-dimensional ground truth through noisy observations and information aggregation:

- **Field Agents** (lowest level): Take noisy observations of the ground truth
- **Analysts**: Aggregate information from their assigned field agents
- **Managers**: Aggregate information from their assigned analysts 
- **Director** (top level): Makes the final estimate based on manager reports

The simulation explores two information aggregation strategies:
- **Bayesian**: Field officers average their previous estimate with a new noisy sample (recursive Bayes)
- **Non-Learning**: Each level takes the plain arithmetic mean of the most recent messages

Party affiliations ("Red" and "Blue") can influence how information flows through the hierarchy, especially when:
- One party provides systematically biased (inverted) observations
- Agents filter information based on party affiliation

## Usage

### Basic Usage

Run a single simulation with default parameters:

```bash
cargo run --release
```

### Algorithm Selection

Choose between Bayesian and Non-Learning algorithms:

```bash
cargo run --release -- --algo bayesian
cargo run --release -- --algo nonlearning
```

### Batch Mode

Run multiple simulations with both algorithms at several noise levels and print statistics:

```bash
cargo run --release -- --batch
```

### Advanced Configuration

The model supports many configuration parameters:

```bash
cargo run --release -- \
  --reliability 1.5 \
  --party-ratio 0.5 \
  --party-status neutral \
  --same-party-filter \
  --eq-threshold 0.01 \
  --n-managers 3 \
  --n-analysts 4 \
  --n-agents 15 \
  --seed 42
```

Run with `--help` to see all available options:

```bash
cargo run --release -- --help
```

## Performance

This implementation is optimized for performance, especially in batch mode where it takes advantage of parallel processing using the Rayon crate. This makes it well-suited for large-scale parameter sweeps and extensive simulation runs.

## Building

Build the optimized release version:

```bash
cargo build --release
```

## Implementation Details

The simulation is implemented in Rust with a focus on performance and scalability. It uses:

- **Rayon** for parallel processing of batch simulations
- **Clap** for a robust command-line interface
- **Serde** for serialization capabilities
- **Statrs** for statistical functions

The core simulation is designed around a clean separation of concerns between agents, model logic, and utility functions.
