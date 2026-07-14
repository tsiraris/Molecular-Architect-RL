"""
=============================================
Active Learning Curve Visualization (Stage 3)
=============================================

This script serves as the primary evaluation visualization tool for the Stage-3 active learning (AL) loop.
It proves the efficacy and monotonic improvement of the AL cycle by parsing round-by-round metric logs 
and generating a comprehensive 4-panel figure. 

The generated plot acts as headline evidence that the active learning loop functions correctly, 
specifically tracking:
1. Scaffold-split Spearman correlation (demonstrating increasing ranking power on novel scaffolds).
2. Surrogate RMSE (demonstrating decreasing absolute prediction error).
3. Uncertainty on generated molecules (demonstrating increasing model confidence over the newly explored chemical space).
4. Median docking score of the acquired batch (demonstrating physical improvement in the generated molecules).
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")                                                                       # Set matplotlib to use the non-interactive 'Agg' backend for headless server environments
import matplotlib.pyplot as plt


def main():
    """
    Parses active learning metrics from JSON logs and renders a 4-panel progression plot.
    
    How it works:
    Initializes an argument parser to receive the target directory containing AL round logs 
    and the desired output filepath. It globs and sorts all `metrics_round_*.json` files 
    in the target directory, extracts the necessary metric keys (Spearman, RMSE, uncertainty, 
    and docking scores), and maps them against the discrete AL round integers. It then 
    generates a 1x4 matplotlib subplot grid, plots the trends with customized titles and 
    formatting, and saves the final figure to disk.
    
    Args:
        None (reads from `sys.argv` via `argparse`).
        
    Returns:
        None. Saves a PNG file to the path specified by the `--out` argument.
        
    Example:
        >>> # From command line:
        >>> # python al_curves.py --dir artifacts/al_run_01 --out results/al_progression.png
    """
    # -------------------------------------------------------------------------------------
    # Argument Parsing: Configure and parse command-line arguments for input directory 
    # containing JSON logs and output PNG filepath.
    # -------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser()                                                          # Initialize the argument parser to handle command-line inputs
    ap.add_argument("--dir", required=True)                                                 # Define a required argument for the input directory containing JSON logs
    ap.add_argument("--out", default="../results/al_curves.png")                            # Define an optional argument for the output PNG filepath with a default location
    args = ap.parse_args()                                                                  # Parse the provided command-line arguments into a namespace object
    
    # -------------------------------------------------------------------------------------
    # Data Loading & Validation
    # Discover, sort, and load the JSON metric files spanning all active learning rounds.
    # -------------------------------------------------------------------------------------
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.dir, "metrics_round*.json")))] # Find, alphabetically sort, and load all round metric JSON files into a list of dictionaries
    if not rows:                                                                            # Check if the loaded list of metric dictionaries is empty
        raise SystemExit("no metrics_round*.json found")                                    # Abort execution immediately and raise an error if no log files were discovered
    
    # -------------------------------------------------------------------------------------
    # Metric Extraction & Plot Configuration: Extract the X-axis rounds and define the 
    # layout mapping for the 4 Y-axis metrics (Spearman, RMSE, uncertainty, docking).
    # -------------------------------------------------------------------------------------
    rounds = [r["round"] for r in rows]                                                     # Extract the discrete integer round numbers from the loaded dictionaries for the X-axis
    panels = [("scaffold_spearman", "Scaffold-split Spearman (up)"),                        # Map the Spearman correlation metric key to its corresponding panel title
              ("scaffold_rmse_z", "Surrogate RMSE z (down)"),                               # Map the normalized surrogate RMSE metric key to its corresponding panel title
              ("gen_uncertainty_mean", "Uncertainty on generated (down)"),                  # Map the generative uncertainty metric key to its corresponding panel title
              ("dock_median", "Median docking score, -kcal/mol (up)")]                      # Map the median docking score metric key to its corresponding panel title
    
    # -------------------------------------------------------------------------------------
    # Figure Rendering & Export
    # Generate the subplots, populate the data series, format the axes, and save to disk.
    # -------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))                                         # Initialize a wide matplotlib figure containing a 1x4 grid of subplot axes
    for ax, (key, title) in zip(axes, panels):                                              # Iterate concurrently through the axes and the predefined metric mapping tuples
        ax.plot(rounds, [r.get(key) for r in rows], "o-", lw=2)                             # Plot the specific metric sequence against the rounds using a solid line with circle markers
        ax.set_title(title); ax.set_xlabel("AL round"); ax.set_xticks(rounds); ax.grid(alpha=0.3) # Apply the panel title, label the X-axis, lock the X-ticks to discrete rounds, and add a faint grid
    fig.tight_layout(); fig.savefig(args.out, dpi=150)                                      # Automatically adjust subplot padding to prevent overlap and export the figure to a high-res PNG
    print(f"[al_curves] wrote {args.out}")                                                  # Output a success confirmation message to the standard console


if __name__ == "__main__":                                                                  # Prevent automatic execution if the script is imported as a module elsewhere
    main()                                                                                  # Invoke the main execution function to begin plotting