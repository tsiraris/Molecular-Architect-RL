"""
=====================================================================
PPO vs GFlowNet Diversity and Reward Comparison (Stage-3 Evaluation)
=====================================================================

This script evaluates and compares two distinct generative policies—Proximal Policy 
Optimization (PPO) and Generative Flow Networks (GFlowNet)—to assess their respective 
exploration/exploitation trade-offs. 

GFlowNets are not reward maximisers. Trajectory balance makes the policy sample x with 
probability proportional to R(x), so a converged GFlowNet trades PEAK reward for MODE COVERAGE. 
Comparing the two methods on mean reward alone is therefore meaningless, as it measures 
the property GFlowNets deliberately give up. The literature-standard axes are:
  - Top-K reward           : what PPO optimises (exploitation capability).
  - Number of modes        : molecules above a reward threshold that are pairwise Tanimoto-separated 
                             (< 0.7 by convention, Bengio et al. 2021 / Jain et al. 2022). 
                             This is what GFlowNets optimise (mode coverage and exploration).
  - Internal diversity     : mean pairwise (1 - Tanimoto) over the top-K.
  - Validity / uniqueness  : read from the *_meta.json sidecars written by sample_policy.py.

Methodological Guards:
  1. EQUAL-N: The two policies had very different validity rates in earlier runs. Because mode 
     counts and diversity scale with sample size, every method is subsampled to the same N 
     before any metric is computed to prevent manufactured differences.
  2. MULTI-SEED: Passes several CSVs per method (one per seed). Every metric is reported as 
     mean +/- sd across seeds to ensure honest, empirical reporting.
  3. IDENTICAL CONFIG: The reward column must have been produced by sample_policy.py under 
     an identical reward config for both methods.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")                                                              # Mute RDKit's benign per-molecule chatter to keep stdout clean


def _fp(smi):
    """
    Computes a 2048-bit Morgan fingerprint (radius 2) for a given SMILES string.

    Translates the SMILES string into an RDKit molecule object. If the parsing is successful, 
    it delegates to RDKit's AllChem module to generate a bit vector representation of the 
    molecule's circular substructures. This fingerprint is the basis for all downstream 
    Tanimoto similarity and diversity calculations.

    Args:
        smi (str): The input SMILES string to process.

    Returns:
        ExplicitBitVect or None: The calculated fingerprint, or None if the SMILES cannot be parsed.
        
    Example:
        >>> fp = _fp("CC1=CC=CC=C1")
        >>> type(fp)
        <class 'rdkit.DataStructs.cDataStructs.ExplicitBitVect'>
    """
    # -------------------------------------------------------------------------------------
    # Fingerprint Generation
    # Convert string to molecule and extract the bit-vector representation.
    # -------------------------------------------------------------------------------------
    m = Chem.MolFromSmiles(str(smi))                                                            # Parse the raw SMILES string into an active RDKit molecule object
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m is not None else None   # Fingerprint valid molecules only using a radius of 2 and 2048 bits


def _int_div(smiles):
    """
    Calculates the internal topological diversity (1 - mean pairwise Tanimoto similarity) 
    of a batch of molecules.

    Converts a list of SMILES strings into Morgan fingerprints, dropping invalid ones. 
    It computes the upper-triangle Tanimoto similarities between all unique pairs in the set. 
    The final internal diversity is defined as 1.0 minus the mathematical mean of these 
    pairwise similarities.

    Args:
        smiles (List[str]): A list of SMILES strings representing the molecular batch.

    Returns:
        float: Internal diversity score in [0.0, 1.0]; higher means structurally more varied.
               Returns 0.0 if fewer than 2 valid molecules exist.
               
    Example:
        >>> pop = ["CCO", "CCN", "CCC"]
        >>> div = _int_div(pop)
    """
    # -------------------------------------------------------------------------------------
    # Pairwise Similarity Calculation: Compute the Tanimoto similarity for all unique 
    # molecule pairs in the batch to determine population variance.
    # -------------------------------------------------------------------------------------
    fps = [f for f in (_fp(s) for s in smiles) if f is not None]                            # Fingerprint everything parseable, filtering out invalid SMILES strings
    if len(fps) < 2:                                                                        # Diversity is undefined for fewer than two molecules
        return 0.0                                                                          # Degenerate case: return zero diversity if comparison is impossible
    sims = []                                                                               # Accumulate the upper-triangle similarities in an empty list
    for i in range(1, len(fps)):                                                            # Walk the triangle once by iterating through fingerprints starting from the second element
        sims += list(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i]))                   # Bulk call is far faster than pairwise Python loops for similarity computation
    return float(1.0 - np.mean(sims))                                                       # Convert mean similarity into a diversity score by subtracting from 1.0


def _n_modes(smiles, cutoff=0.7):
    """
    Counts Tanimoto-separated modes via greedy leader clustering.

    This is the standard GFlowNet mode-coverage metric. A molecule opens a new "mode" 
    only if its Tanimoto similarity to every previously discovered mode is below the `cutoff`. 
    Molecules passed to this function are expected to be pre-filtered to those above 
    the targeted reward threshold.

    Args:
        smiles (List[str]): Above-threshold molecules.
        cutoff (float): Tanimoto similarity above which two molecules count as the same mode.

    Returns:
        int: The number of distinct structural modes discovered.
        
    Example:
        >>> top_smiles = ["CCO", "c1ccccc1", "c1ccccc1O"]
        >>> modes = _n_modes(top_smiles, cutoff=0.7)
    """
    # -------------------------------------------------------------------------------------
    # Greedy Leader Clustering
    # Sequentially assign molecules to new or existing modes based on Tanimoto distance.
    # -------------------------------------------------------------------------------------
    leaders = []                                                                            # Fingerprints of the modes discovered so far stored as cluster leaders
    for s in smiles:                                                                        # Consider each candidate in turn from the provided list
        f = _fp(s)                                                                          # Fingerprint it into a 2048-bit vector
        if f is None:                                                                       # Skip unparseable molecules
            continue                                                                        # Nothing to cluster if the graph fails to compile
        if not leaders:                                                                     # The first valid molecule always opens a mode
            leaders.append(f)                                                               # Register it as the very first leader
            continue                                                                        # Move on to the next candidate
        if max(DataStructs.BulkTanimotoSimilarity(f, leaders)) < cutoff:                    # Sufficiently unlike every known mode? (i.e. if Tanimoto is above the cutoff --> a new mode)
            leaders.append(f)                                                               # Then it is a new mode; append to the leaders list
    # Returns an integer count of the number of distinct modes
    return len(leaders)                                                                     # The mode count equals the final length of the leaders list


def summarise(csv_path, thresh, n_equal, topk, seed=0):
    """
    Computes all comparison metrics for one sample file, subsampled to a common N (n_equal) molecules.

    Loads a single seed's CSV, drops missing rewards, and randomly subsamples it to 
    `n_equal` to prevent sample-size artifacts. It isolates the high-reward subset 
    (>= `thresh`) to count modes and the top `topk` subset to calculate peak exploitation 
    and internal diversity. Finally, it attempts to load upstream validity/uniqueness 
    metrics from a JSON sidecar file.

    Args:
        csv_path (str): A smiles,reward CSV written by sample_policy.py.
        thresh (float): Reward threshold defining a "high-reward" molecule.
        n_equal (int): Common sample size every method is cut to before metrics are computed.
        topk (int): How many top-reward molecules define the Top-K statistics.
        seed (int): Deterministic subsampling seed for reproducible splits.

    Returns:
        dict: A dictionary mapping evaluation metric names to their computed scalar values:
            - "n_used": Sample size actually used (post equal-N cut)
            - "validity":  Fraction of sampled episodes that yielded a valid molecule
            - "uniqueness_of_valid": Unique fraction among the valid molecules
            - "reward_mean":  Mean reward overall
            - "topk_reward_mean": Mean reward for the Top-K slice
            - "single_best_reward": Best single molecule found overall
            - "n_above_thresh": Number of molecules above the reward threshold
            - "n_modes": How many modes in the high-reward subset
            - "topk_internal_diversity": Internal diversity among the best molecules (rounded to 4 decimals)
              
    """
    # -------------------------------------------------------------------------------------
    # Data Filtering, Subsampling, and Sorting
    # Enforce Equal-N data size and extract critical analytical subsets.
    # -------------------------------------------------------------------------------------
    df = pd.read_csv(csv_path).dropna(subset=["reward"])                                    # Load and drop unscored rows directly upon ingest
    if len(df) > n_equal:                                                                   # EQUAL-N guard: never compare a big sample against a small one
        df = df.sample(n_equal, random_state=seed)                                          # Deterministic subsample to the common size using the provided random seed
    df = df.sort_values("reward", ascending=False)                                          # Rank by reward for the Top-K statistics in descending order
    # Keep only candidates above the reward threshold
    hi = df[df.reward >= thresh]                                                            # The above-threshold subset that defines modes
    # and the top-k subset
    top = df.head(topk)                                                                     # The Top-K slice containing the highest scoring candidates

    # -------------------------------------------------------------------------------------
    # Sidecar Metadata Ingestion & Metric Aggregation
    # Load upstream generation stats and compile final mathematical evaluations.
    # -------------------------------------------------------------------------------------
    meta_path = csv_path.replace(".csv", "_meta.json")                                      # sample_policy.py writes validity/uniqueness here
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}                  # Read it when present; absence is non-fatal to the evaluation

    return {"n_used": int(len(df)),                                                         # Sample size actually used (post equal-N cut)
            "validity": meta.get("validity"),                                               # Fraction of sampled episodes that yielded a valid molecule
            "uniqueness_of_valid": meta.get("uniqueness_of_valid"),                         # Unique fraction among the valid molecules
            "reward_mean": float(df.reward.mean()),                                         # Mean reward: expected to favour PPO by construction
            "topk_reward_mean": float(top.reward.mean()) if len(top) else None,             # Top-K reward: the exploitation axis
            "single_best_reward": float(df.reward.max()) if len(df) else None,              # Best single molecule found overall
            "n_above_thresh": int(len(hi)),                                                 # How many cleared the bar at all
            "n_modes": _n_modes(list(hi.smiles)),                                           # Mode coverage: the axis GFlowNets optimise
            "topk_internal_diversity": round(_int_div(list(top.smiles)), 4)}                # Structural variety among the best molecules rounded to 4 decimals


def aggregate(rows):
    """
    Reduces metric rows from multiple seeds into statistical aggregates (mean +/- sd).

    Iterates over a list of dictionary outputs from `summarise`. It identifies keys 
    that possess valid numerical data across all seeds and computes their sample mean 
    and standard deviation, returning a nested dictionary structure.

    Args:
        rows (List[dict]): A list containing one `summarise` output dictionary per seed.

    Returns:
        dict: Format {metric: {"mean": float, "sd": float, "n_seeds": int}} for valid metrics.
    """
    # -------------------------------------------------------------------------------------
    # Multi-Seed Statistical Reduction
    # Condense individual run data into reliable means and standard deviations.
    # -------------------------------------------------------------------------------------
    out = {}                                                                                # Aggregated result dictionary initialized empty
    keys = [k for k in rows[0] if all(isinstance(r.get(k), (int, float)) and r.get(k) is not None
                                      for r in rows)]                                       # Only aggregate metrics present and numeric in every seed
    for k in keys:                                                                          # Reduce each metric independently by looping through valid keys
        vals = np.array([float(r[k]) for r in rows], float)                                 # Gather the per-seed values into a NumPy float array
        out[k] = {"mean": round(float(vals.mean()), 4),                                     # Central tendency rounded for readability
                  "sd": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4),        # Spread across seeds (sample sd), safely zeroing out single-item arrays
                  "n_seeds": int(len(vals))}                                                # How many replicates back the number
    return out                                                                              # The aggregated block returned as a nested dictionary


def main():
    """
    CLI wrapper: Aggregates multi-seed samples for each method and emits the reward/diversity Pareto.
    
    How it works:
    Parses arguments allowing multiple CSVs per policy (via `nargs="+"`). It runs the 
    `summarise` protocol on every file individually, aggregates the metrics via `aggregate`, 
    dumps the full nested dictionary to a JSON artifact, and prints a human-readable 
    trade-off table to the console.

    Args:
        None.
        
    Returns:
        None.
    """
    # -------------------------------------------------------------------------------------
    # Argument Parsing: 
    # Define and extract the required command-line inputs accommodating multiple seeds.
    # -------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="PPO vs GFlowNet reward/diversity Pareto.")
    ap.add_argument("--ppo_csv", nargs="+", required=True, help="one smiles,reward CSV per seed")     # Multi-seed by design using nargs="+"
    ap.add_argument("--gfn_csv", nargs="+", required=True, help="one CSV per seed (may be per-beta)") # Multi-seed by design using nargs="+"
    ap.add_argument("--gfn_label", default="GFlowNet", help="e.g. 'GFlowNet b=0.5'")                  # Lets several betas be compared as separate Pareto points
    ap.add_argument("--thresh", type=float, default=6.0, help="reward threshold defining a mode")     # Scale-dependent: inspect the reward distribution first
    ap.add_argument("--n_equal", type=int, default=800, help="common sample size for every method")   # EQUAL-N guard to prevent statistical artifacts
    ap.add_argument("--topk", type=int, default=100)                                                  # Top-K slice size for local search diversity
    ap.add_argument("--out", default="../results/ppo_vs_gfn.json")                                    # Where the Pareto lands on disk
    a = ap.parse_args()

    # -------------------------------------------------------------------------------------------
    # Process Execution and JSON Export: For each seed independently, compute all the comparison 
    # metrics, aggregate them for PPO and GFlowNet, and save to disk along with the protocol 
    # (i.e. threshold, n_equal, topk, PPO and GFlowNet metrics and every per-seed rows).
    # -------------------------------------------------------------------------------------------
    ppo_rows = [summarise(p, a.thresh, a.n_equal, a.topk, seed=i) for i, p in enumerate(a.ppo_csv)]   # Per-seed PPO metrics evaluated via list comprehension
    gfn_rows = [summarise(p, a.thresh, a.n_equal, a.topk, seed=i) for i, p in enumerate(a.gfn_csv)]   # Per-seed GFlowNet metrics evaluated via list comprehension
    res = {"threshold": a.thresh, "n_equal": a.n_equal, "topk": a.topk,                               # The protocol, recorded with the result
           "PPO": aggregate(ppo_rows), a.gfn_label: aggregate(gfn_rows),
           "_per_seed": {"PPO": ppo_rows, a.gfn_label: gfn_rows}}                                     # Raw per-seed rows for full auditability
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)                                         # Ensure the destination exists before attempting write
    json.dump(res, open(a.out, "w"), indent=2)                                                        # Persist the master dictionary as cleanly indented JSON

    # -------------------------------------------------------------------------------------
    # Console Output (Human-Readable Pareto)
    # Print the fundamental trade-off: exploitation vs mode coverage.
    # -------------------------------------------------------------------------------------
    # Human-readable Pareto: exploitation (Top-K reward) against coverage (modes / diversity).
    print(f"\n{'method':22s}{'topK_reward':>14s}{'n_modes':>10s}{'intdiv':>9s}{'validity':>10s}")
    for name in ("PPO", a.gfn_label):                                                                 # One line per Pareto point
        m = res[name]                                                                                 # Its aggregated metrics mapped to a local variable
        def g(k):                                                                                     # Formats mean+/-sd compactly, tolerating absent metrics
            return f"{m[k]['mean']:.2f}+/-{m[k]['sd']:.2f}" if k in m else "n/a"
        print(f"{name:22s}{g('topk_reward_mean'):>14s}{g('n_modes'):>10s}"
              f"{g('topk_internal_diversity'):>9s}{g('validity'):>10s}")
    print(f"\n[compare] wrote {a.out}")                                                               # Confirm the artefact physical location
    print("[compare] Trade-off, not a winner: PPO is expected to lead on Top-K reward\n"
          "          and the GFlowNet on modes/diversity.")                                           # Human-readable reminder


if __name__ == "__main__":
    main()