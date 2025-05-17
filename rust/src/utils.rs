//! Utility functions for the hierarchical information aggregation model
//!
//! This module contains various helper functions used throughout the simulation,
//! including statistical utilities, coordinate transformations, and display helpers.

/// Calculate the Euclidean distance between two points in 2D space
///
/// This is used to measure the error between an agent's belief and the ground truth,
/// as well as to detect movement/convergence in the simulation.
///
/// # Arguments
/// * `point1` - First point coordinates (x, y)
/// * `point2` - Second point coordinates (x, y)
///
/// # Returns
/// The Euclidean distance between the two points: √((x₂-x₁)² + (y₂-y₁)²)
pub fn euclidean_distance(point1: (f64, f64), point2: (f64, f64)) -> f64 {
    let dx = point1.0 - point2.0;
    let dy = point1.1 - point2.1;
    (dx * dx + dy * dy).sqrt()
}

/// Format a vector of floating-point values as a simple statistical summary
///
/// Calculates and formats key statistical measures (mean, standard deviation,
/// min, max) for a set of values. Used for reporting simulation results.
///
/// # Arguments
/// * `values` - Slice of floating-point values to summarize
///
/// # Returns
/// A formatted string containing the statistical summary
#[allow(dead_code)]
pub fn format_stats(values: &[f64]) -> String {
    if values.is_empty() {
        return "N/A (no data)".to_string();
    }
    
    let n = values.len();
    let sum: f64 = values.iter().sum();
    let mean = sum / n as f64;
    
    let var_sum: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum();
    let std_dev = (var_sum / n as f64).sqrt();
    
    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    format!(
        "mean: {:.3}, std: {:.3}, min: {:.3}, max: {:.3}, n: {}",
        mean, std_dev, min, max, n
    )
}

/// Format a vector of integer values as a simple statistical summary
///
/// Converts integer values to floating-point and then uses format_stats to generate
/// a statistical summary. Primarily used for reporting time steps to convergence.
///
/// # Arguments
/// * `values` - Slice of integer values to summarize
///
/// # Returns
/// A formatted string containing the statistical summary
#[allow(dead_code)]
pub fn format_int_stats(values: &[usize]) -> String {
    let values_f64: Vec<f64> = values.iter().map(|&x| x as f64).collect();
    format_stats(&values_f64)
}

/// Convert continuous world coordinates to discrete grid cell indices
///
/// The simulation operates in a continuous space, but for visualization and
/// certain operations, these coordinates need to be mapped to discrete grid cells.
/// This function performs that conversion and ensures the result is within grid bounds.
///
/// # Arguments
/// * `x` - X-coordinate in world space
/// * `y` - Y-coordinate in world space
/// * `grid_res` - Resolution of the grid (number of cells per side)
///
/// # Returns
/// A tuple of grid cell indices (column, row)
#[allow(dead_code)]
pub fn world_to_cell(x: f64, y: f64, grid_res: usize) -> (usize, usize) {
    use crate::model::{CELL_SCALE, WORLD_MIN};
    
    let cx = ((x - WORLD_MIN) * CELL_SCALE).round() as usize;
    let cy = ((y - WORLD_MIN) * CELL_SCALE).round() as usize;
    
    // Clamp to grid bounds
    let cx = cx.min(grid_res - 1);
    let cy = cy.min(grid_res - 1);
    
    (cx, cy)
}

/// Create a simple ASCII progress bar on the console
///
/// Used to display the progress of batch simulations. The progress bar updates
/// in-place on the same line until completion, providing visual feedback during
/// long-running batch processes.
///
/// # Arguments
/// * `current` - Current progress value
/// * `total` - Total expected value for completion
/// * `prefix` - Text to display before the progress bar
///
/// # Example output
/// ```text
/// Progress [====================          ] 67% (67/100)
/// ```
pub fn print_progress(current: usize, total: usize, prefix: &str) {
    const BAR_WIDTH: usize = 30;
    
    let progress = (current as f64 / total as f64).min(1.0);
    let filled_width = (BAR_WIDTH as f64 * progress).round() as usize;
    let empty_width = BAR_WIDTH - filled_width;
    
    let percent = (progress * 100.0).round() as usize;
    let filled = "=".repeat(filled_width);
    let empty = " ".repeat(empty_width);
    
    // Print and return to start of line
    print!("\r{} [{}{}] {}% ({}/{})", prefix, filled, empty, percent, current, total);
    if current >= total {
        println!();
    }
    
    // Flush to ensure immediate output
    use std::io::Write;
    std::io::stdout().flush().unwrap();
}
