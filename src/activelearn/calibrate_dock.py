"""
==========================================
Docking to Label Calibration (Stage 3 Fix)
==========================================

In the active-learning (AL) loop, newly generated molecules are scored via docking and added 
to a training set whose ground-truth labels are z-scored ChEMBL pChEMBL values. Previously, 
docking scores were z-normalized *within each batch* before merging. This was scientifically flawed:
    1. Loss of Absolute Affinity: Normalizing within a batch forces every round to span ~N(0,1). 
       A round of terrible molecules would have its "best" scored as +2, exactly like the "best" 
       of an excellent batch, blinding the surrogate model to absolute improvements.
    2. Drift (Lack of Round-Invariance): The normalizing constants drifted based on acquisition bias.
    3. Inconsistency: The same molecule evaluated in two different rounds received two different labels.

Fix: 
This script establishes a SINGLE, FROZEN, round-invariant affine map (label_z ~= a * dock_score + b). 
It fits this linear regression once against a stratified reference subset of molecules with known 
measured pChEMBL affinities. 

What is fitted:
Because covalent docking (Cys12 warheads) and non-covalent docking rely on fundamentally different 
estimators, this script splits the reference set and fits independent (a, b) parameters for each mode. 
These maps are frozen into `dock_calibration.json` and applied universally in all subsequent AL rounds.

Which docking score is used:
The script strongly prefers `gnina`'s `CNNaffinity` because it directly outputs a predicted pKd 
(higher = better), matching the physical quantity of pChEMBL and resisting lipophilic macrocycle hacks. 
If forced to fall back to `smina` (Vina score), it thermodynamically converts the binding free energy 
(kcal/mol) to log-affinity via pKd = -dG / 1.364 (at 298 K).

Note:
Crucially, the script explicitly calculates and prints the R^2 and Spearman rho between the docking 
scores and the measured affinities. Since docking is a noisy proxy, these metrics are expected to be 
modest. Reporting them honestly provides a strict upper bound on how much the AL loop can actually 
learn from docking labels.
"""

import argparse
import json
import os
import sys

# -----------------------------------------------------------------------------------------
# Environment & Imports
# Configure the path to resolve internal modules and import necessary scientific libraries.
# -----------------------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                     # Inject the parent `src/` directory into the python path to allow direct module execution

import numpy as np                                                                                  # Import Numpy for least-squares linear regression and summary statistics
import pandas as pd                                                                                 # Import Pandas for efficient CSV I/O and stratified dataframe sampling
from rdkit import Chem                                                                              # Import RDKit Chem module for SMILES parsing, required for warhead identification
from rdkit import RDLogger                                                                          # Import RDKit Logger to suppress excessive and benign console warnings

RDLogger.DisableLog("rdApp.*")                                                                      # Mute RDKit's C++ valence warnings to save massive amounts of I/O overhead during batch loops

from oracle.dock import dock_many_fast, which_engine                                                # Import non-covalent parallel docking routine and environment engine probe
from oracle.covalent_dock import covalent_dock_many                                                 # Import covalent-specific docking routine targeting Cys12
from oracle.prepare_receptor import ligand_com                                                      # Import center-of-mass calculator for bounding box resolution
from reward.warhead import has_warhead                                                              # Import structural filter to route molecules to covalent vs non-covalent pipelines

KCAL_PER_LOG_UNIT = 1.364                                                                           # Constant: 2.303 * R * T at 298 K, used to mathematically map free energy (-dG) to log-affinity (pKd)


def dock_score_of(result, engine):
    """
    Extracts and standardizes the docking score into a pKd-scaled format.

    If the engine is gnina, it pulls the `CNNaffinity` (natively a predicted pKd). If falling 
    back to a Vina-family engine (smina), it extracts the raw affinity (binding free energy, dG 
    in kcal/mol) and converts it to a log-affinity scale using the formula `pKd = -dG / 1.364`. 
    This ensures both estimators land on the same interpretable physical scale before calibration.

    Args:
        result (DockResult): The structured result object returned from the docking oracle.
        engine (str): The name of the docking engine utilized ("gnina" or "smina").

    Returns:
        float or None: The harmonized pKd-scaled score, or None if the docking simulation failed.

    Example:
        >>> score = dock_score_of(result, "smina") # e.g., result.affinity = -8.18 kcal/mol
        >>> print(score) # 8.18 / 1.364
        5.997
    """
    # -------------------------------------------------------------------------------------
    # Score Extraction & Thermodynamic Scaling: Normalize varied engine outputs into a 
    # universal pKd metric (CNNaffinity for gnina, else pKd = -dG / 1.364 for Vina).
    # -------------------------------------------------------------------------------------
    if not result.ok:                                                                               # Check the boolean flag indicating if the docking simulation completed successfully
        return None                                                                                 # Return None to signal the caller that no valid score exists for this molecule
    if engine == "gnina" and result.cnn_affinity is not None:                                       # Check if gnina was used and the CNN routing successfully produced an affinity prediction
        return float(result.cnn_affinity)                                                           # Return the CNNaffinity directly as a float, requiring no thermodynamic unit conversion
    if result.affinity is not None:                                                                 # Fallback condition evaluating if a standard Vina free-energy score exists
        return float(-result.affinity / KCAL_PER_LOG_UNIT)                                          # Convert the kcal/mol free energy into pKd scale to match CNNaffinity
    return None                                                                                     # Fallback return if neither metric could be successfully parsed from the result object


def _fit(x, y):
    """
    Performs the Ordinary Least Squares (OLS) linear regression mapping docking scores to labels.
    Returns the slope `a` and intercept `b` for the affine map `y ~= a*x + b`, along with R^2 and Spearman rho.

    Computes the slope `a` and intercept `b` for the affine map `y ~= a*x + b`. It calculates 
    the Coefficient of Determination (R^2) and Spearman rank correlation (rho) to honestly report 
    how well docking orders the measured affinities. Refuses to fit on degenerate/tiny datasets.

    Args:
        x (np.ndarray): 1D array of pKd-scaled docking scores.
        y (np.ndarray): 1D array of measured ground-truth labels (pChEMBL affinities).

    Returns:
        dict: A dictionary containing the fitted parameters ("a", "b"), statistical 
        diagnostics ("r2", "spearman"), and the sample size ("n").
    """
    # ------------------------------------------------------------------------------------------
    # Least-Squares Regression & Diagnostics
    # Compute the affine transformation parameters (a, b) and quantify proxy quality (R^2, rho).
    # ------------------------------------------------------------------------------------------
    x = np.asarray(x, float)                                                                        # Cast the input docking scores to a strict floating-point numpy array for linear algebra
    y = np.asarray(y, float)                                                                        # Cast the target z-scored labels to a strict floating-point numpy array
    if len(x) < 5 or float(np.std(x)) < 1e-8:                                                       # Abort fitting if the dataset is statistically insignificant or entirely uniform/degenerate
        return {"a": None, "b": None, "r2": None, "spearman": None, "n": int(len(x))}               # Return null parameters forcing the pipeline to fall back to a hardcoded heuristic map
    a, b = np.polyfit(x, y, 1)                                                                      # Execute a 1st-degree polynomial fit (OLS) to extract the slope (a) and y-intercept (b)
    pred = a * x + b                                                                                # Project the predicted z-labels using the newly fitted affine transformation
    ss_res = float(np.sum((y - pred) ** 2))                                                         # Compute the Residual Sum of Squares (variance unexplained by the model)
    ss_tot = float(np.sum((y - y.mean()) ** 2))                                                     # Compute the Total Sum of Squares (total variance of the ground-truth labels)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None                                          # Compute the R^2 coefficient, guarding against division by zero if target labels are constant
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))                                 # Compute the Spearman rank correlation to assess monotonic ordering capabilities of docking
    return {"a": float(a), "b": float(b), "r2": (float(r2) if r2 is not None else None),            # Pack the slope, intercept, and R^2 metric into the output dictionary
            "spearman": rho, "n": int(len(x))}                                                      # Pack the Spearman correlation and sample size, completing the frozen map definition


def calibrate(base_csv, cfg, out_json, n_ref=200, seed=0, cache=None):
    """
    Orchestrates the initial OLS calibration process: samples reference data, splits them by 
    covalent/non-covalent, docks them, fits the regression line, and freezes the map parameters 
    (a, b, R^2, rho and n for each mode).

    How it works:
    1. Loads the labeled seed CSV and filters out massive peptide macrocycles (MW > 700) because 
       they dominate runtime (300s each) and fall outside the domain reachable by the generator.
    2. Performs rank-based stratified sampling across the label range to ensure the regression 
       line is constrained across the entire affinity span, not just clustered near the mean.
    3. Splits the sampled molecules into covalent (warhead-bearing) and non-covalent subsets.
    4. Executes the respective docking pipelines against the target defined in the config.
       Crucially, covalent docking requires a raw PDB (not PDBQT) because `obabel` destroys 
       the chain/residue naming required by gnina's `--covalent_rec_atom` flag.
    5. Feeds the results to `_fit()` and writes the independent (a, b) parameters to disk.

    Args:
        base_csv (str): Path to the labeled seed CSV (SMILES, label_z).
        cfg (dict): Target configuration dictionary (receptor paths, box dims).
        out_json (str): Destination path for the frozen calibration JSON.
        n_ref (int, optional): The target budget of reference molecules to dock. Defaults to 200.
        seed (int, optional): Random seed for deterministic stratified sampling. Defaults to 0.
        cache (DockCache, optional): Database to avoid re-docking known molecules. Defaults to None.

    Returns:
        dict: The complete nested calibration dictionary written to disk.
    """
    # ----------------------------------------------------------------------------------------------
    # Dataset Curation & Domain Matching
    # Load labels, clean missing data, and enforce molecular weight applicability limits (MW < 700).
    # ----------------------------------------------------------------------------------------------
    df = pd.read_csv(base_csv)                                                                      # Ingest the base labeled seed dataset from disk into a Pandas dataframe
    smi_col, lab_col = df.columns[0], df.columns[1]                                                 # Identify the columns dynamically assuming the strict convention: SMILES first, label second
    df = df.dropna(subset=[smi_col, lab_col])                                                       # Purge any dataframe rows missing either structural or label data, as they break regression
    
    from rdkit.Chem import Descriptors as _D                                                        # Import RDKit physical descriptors locally to keep the master file header clean
    _mw = df[smi_col].astype(str).map(lambda x: (lambda m: _D.MolWt(m) if m else 1e9)(Chem.MolFromSmiles(x))) # Safely calculate the molecular weight of every valid SMILES string, defaulting to infinity on failure
    df = df[_mw <= 700]                                                                             # Filter out macrocycles (MW > 700) to keep the fit bounded to the generator's actual drug-like domain

    # -------------------------------------------------------------------------------------
    # Stratified Sampling
    # Bin the data using quantiles and sample evenly to capture the full label variance.
    # -------------------------------------------------------------------------------------
    df = df.copy()                                                                                  # Create an explicit dataframe copy to avoid mutating the caller's memory references
    df["_bin"] = pd.qcut(df[lab_col].rank(method="first"), q=min(10, max(2, len(df) // 20)),        # Assign rows to rank-based quantile bins, robustly handling heavily skewed label distributions
                         labels=False, duplicates="drop")                                           # Drop duplicate edge bins safely and return raw integer indices
    per_bin = max(1, n_ref // max(1, df["_bin"].nunique()))                                         # Calculate the exact allocation of samples required per bin to reach the global budget evenly
    ref = (df.groupby("_bin", group_keys=False)                                                     # Group the dataframe by the calculated quantile bins
             .apply(lambda g: g.sample(min(len(g), per_bin), random_state=seed)))                   # Draw samples deterministically from each bin, taking all available if the bin is underpopulated
    ref = ref.head(n_ref)                                                                           # Enforce a hard truncation to ensure the requested docking budget is respected exactly

    # -------------------------------------------------------------------------------------
    # Configuration Resolution
    # Extract engine specs, receptors, bounding boxes, and covalent constraints.
    # -------------------------------------------------------------------------------------
    engine = which_engine(cfg["docking"].get("engine", "gnina")) or "smina"                         # Detect the configured docking engine dynamically, falling back to smina if gnina is missing
    receptor = cfg["receptor_pdbqt"]                                                                # Extract the path to the PDBQT-formatted receptor for standard non-covalent computations
    receptor_cov = cfg.get("receptor_pdb") or receptor                                              # COVALENT MUST USE THE PDB: gnina's --covalent_rec_atom resolves "A:12:SG" from chain/residue naming, which an obabel -xr PDBQT does not preserve
    box = cfg["docking_box"]                                                                        # Extract the search-box spatial bounding definitions from the configuration dictionary
    center = box["center"] or ligand_com(cfg["ref_ligand_sdf"])                                     # Resolve the 3D center coordinate either explicitly or by measuring the reference ligand's center of mass
    size = tuple(box["size_angstrom"])                                                              # Extract the spatial dimension limits of the search box as a static tuple
    cys = cfg.get("covalent", {}).get("cys_spec", "A:12:SG")                                        # Extract the precise chain/residue nomenclature targeting Cys12 for covalent docking hooks
    cnn_mode = cfg["docking"].get("cnn_scoring", "rescore")                                         # Identify the operational mode for gnina's convolutional neural network scoring head

    # -------------------------------------------------------------------------------------
    # Target Splitting (Covalent vs Non-Covalent)
    # -------------------------------------------------------------------------------------
    smis = ref[smi_col].astype(str).tolist()                                                        # Extract the column of sampled SMILES strings into a flat Python list
    labs = dict(zip(smis, ref[lab_col].astype(float).tolist()))                                     # Construct a fast lookup dictionary mapping exact SMILES strings to their measured ground-truth labels
    cov = [s for s in smis if has_warhead(Chem.MolFromSmiles(s))]                                   # Filter the list to isolate only sequences containing active covalent warhead motifs
    noncov = [s for s in smis if s not in set(cov)]                                                 # Define the non-covalent set as the complement of the covalent list
    print(f"[calibrate] engine={engine} reference n={len(smis)} (covalent={len(cov)}, "             # Emit a status header logging the engine and the exact distribution of the split sets
          f"noncovalent={len(noncov)})")                                                            # Close the print statement providing clear transparency into the upcoming workload

    out = {"engine": engine, "score": ("cnn_affinity" if engine == "gnina" else "pKd_from_vina")}   # Initialize the output payload marking precisely WHICH score type the map rests upon (crucial provenance)

    def _collect(results, inputs, tag):
        """
        This helper function collates a list of DockResult objects into a list of (x, y) tuples.

        Args:
            results (List[DockResult]): A list of DockResult objects to collate.
            inputs (List[str]): A list of SMILES strings corresponding to the ordering of the results.
            tag (str): A string identifier for the input set (covalent or noncovalent).

        Returns:
            List[Tuple[float, float]]: A list of (x, y) tuples, where x is the predicted score and y is the measured label.
        """
        # ---------------------------------------------------------------------------------
        # Result Collation
        # Sift successes from failures, pairing predicted scores with ground-truth labels.
        # ---------------------------------------------------------------------------------
        xs, ys = [], []                                                                             # Initialize separate accumulators for the x (predicted) and y (measured) variables of the fit
        n_fail = n_noscore = n_nolabel = 0                                                          # Initialize diagnostic counters tracking the exact failure modes leading to dropped data points
        examples = []                                                                               # Initialize a list to hold a few string representations of failed molecules for operator debugging
        for r, s_in in zip(results, inputs):                                                        # Iterate over the paired alignment of the raw docking results and original input strings
            if not r.ok:                                                                            # Evaluate if the underlying docking engine failed entirely for this specific molecule
                n_fail += 1                                                                         # Increment the terminal failure counter
                if len(examples) < 3:                                                               # Cap the number of captured diagnostic examples to avoid console flooding
                    examples.append(s_in[:70])                                                      # Append a truncated version of the failing SMILES string to the diagnostic list
                continue                                                                            # Skip further processing and advance to the next molecule in the batch
            v = dock_score_of(r, engine)                                                            # Extract the harmonized pKd-scaled score from the successful result object
            if v is None:                                                                           # Evaluate if the extraction logic yielded an unusable null score
                n_noscore += 1; continue                                                            # Increment the missing score counter and skip to the next molecule
            key = r.smiles if r.smiles in labs else s_in                                            # Select the lookup key, preferring the result's internal SMILES over the input string
            if key not in labs:                                                                     # Verify the resolved key actually maps to a known ground-truth label in the dictionary
                n_nolabel += 1; continue                                                            # Increment the missing label counter and skip to the next molecule
            xs.append(v); ys.append(labs[key])                                                      # Successfully push the valid (docked, measured) pairing into the regression accumulators
        print(f"[calibrate] {tag}: paired={len(xs)} dock_failed={n_fail} no_score={n_noscore} "     # Emit an extensive accounting line summarizing exactly how much data survived the filtration
              f"no_label={n_nolabel}")                                                              # Close the formatted string tracking data loss
        if n_fail and examples:                                                                     # Check if there were any hard docking failures warranting diagnostic output
            print(f"[calibrate]   example failed inputs: {examples}")                               # Surface the concrete SMILES strings of the failures so the operator can diagnose structural issues
        return xs, ys                                                                               # Return the finalized, parallel arrays ready for linear regression

    # -------------------------------------------------------------------------------------
    # Execution & Map Freezing
    # Execute batch docking, run OLS map regressions via `_collect()`, and save to JSON.
    # -------------------------------------------------------------------------------------
    cov_res = list(covalent_dock_many(cov, receptor_cov, cys, autobox_ligand=cfg["ref_ligand_sdf"], engine="gnina", cnn_scoring=cnn_mode, # Dispatch the covalent batch to the gnina target processing queue
                                      progress=True, cache=cache)) if (cov and engine == "gnina") else []       # Execute only if targets exist and the engine supports covalent hooks
    out["covalent"] = _fit(*_collect(cov_res, cov, "covalent"))                                                 # Compress the results, execute linear regression, and freeze the covalent mapping dictionary

    non_res = list(dock_many_fast(noncov, receptor, center, size, engine=engine, cnn_scoring=cnn_mode,  # Dispatch the non-covalent batch to the standard parallel docking grid
                                  gpu=cfg["docking"].get("gpu", True), progress=True, cache=cache))     # Accelerate utilizing hardware GPUs and invoke the cache
    out["noncovalent"] = _fit(*_collect(non_res, noncov, "noncovalent"))                                # Compress the results, execute linear regression, and freeze the non-covalent mapping dictionary

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)                                    # Ensure the parent directory for the target JSON configuration file physically exists
    json.dump(out, open(out_json, "w"), indent=2)                                                   # Serialize the nested output dictionary and persist it to disk with clean human-readable indentation
    for mode in ("covalent", "noncovalent"):                                                        # Iterate over the two primary docking modalities to report regression diagnostics
        f = out[mode]                                                                               # Retrieve the isolated mapping dictionary corresponding to the current mode
        if f["a"] is None:                                                                          # Evaluate if the linear regression collapsed due to insufficient or degenerate paired data
            print(f"[calibrate] {mode:12s}: NOT FITTED (n={f['n']}) -> loop will use the fallback map") # Emit a warning that this mode failed to regress and will default to the static heuristic affine
        else:                                                                                       # Execute if the regression resolved into a mathematically sound fit
            print(f"[calibrate] {mode:12s}: label ~= {f['a']:+.4f}*dock {f['b']:+.4f}   "           # Emit the exact mathematical transformation equation mapping the score to the z-scale
                  f"R2={f['r2']:.3f}  spearman={f['spearman']:.3f}  n={f['n']}")                    # Append the critical honesty metrics (R^2 and Spearman correlation) defining the proxy's reliability
    print(f"[calibrate] wrote {out_json}")                                                          # Confirm successful serialization specifying the output file path
    return out                                                                                      # Return the complete programmatic dictionary back to the invoking caller


def apply_calibration(cal, score, mode):
    """
    Transforms a raw docking score into the training z-scale using the frozen calibration.

    Retrieves the slope (a) and intercept (b) from the frozen map for the specified mode (covalent or noncovalent).
    If the map failed to fit during calibration, it defaults to a static heuristic formula:
    `z = (score - 6.0) / 1.5`, which reasonably centers a typical pKd of 6.0 at zero.

    Args:
        cal (dict or None): The nested calibration dictionary generated by `calibrate()`.
        score (float): The standardized pKd-scaled docking score.
        mode (str): "covalent" or "noncovalent".

    Returns:
        float: The final, round-invariant z-scored label.
    """
    # -------------------------------------------------------------------------------------
    # Calibration Application
    # -------------------------------------------------------------------------------------
    fit = (cal or {}).get(mode) or {}                                                               # Safely extract the target mode's mapping dictionary traversing any null checks
    a, b = fit.get("a"), fit.get("b")                                                               # Retrieve the regressed slope and y-intercept variables
    if a is None or b is None:                                                                      # Evaluate if the extracted mapping variables are invalid indicating a failed initial fit
        return float((score - 6.0) / 1.5)                                                           # Fallback affine: fixed, round-invariant scaling assuming pKd 6.0 aligns to z-score 0.0 with 1.5 log-unit standard deviation
    return float(a * score + b)                                                                     # Execute the frozen linear transformation applying the slope and intercept


def main():
    """
    Command Line Interface entrypoint.
    Executes the calibration pipeline precisely once prior to active learning deployment.
    """
    # -------------------------------------------------------------------------------------
    # CLI Initialization
    # Parse user arguments, load YAML configs, and trigger the calibration routine.
    # -------------------------------------------------------------------------------------
    import yaml                                                                                     # Defer yaml import locally to prevent mandatory dependencies when loaded strictly as a python module
    ap = argparse.ArgumentParser(description="Freeze a docking->label calibration for the AL loop.")# Instantiate the argument parser framing the script's operational intent
    ap.add_argument("--config", default="../configs/kras_g12c.yaml")                                # Register an argument flag establishing the target biological setup configuration
    ap.add_argument("--base_csv", default="../data/surrogate_train.csv")                            # Register an argument flag indicating the location of the pre-labeled seed data
    ap.add_argument("--out", default="../artifacts/dock_calibration.json")                          # Register an argument flag detailing where to persist the frozen affine coefficients
    ap.add_argument("--n_ref", type=int, default=200)                                               # Register an argument flag dictating the sampling size budget for the regression pool
    ap.add_argument("--seed", type=int, default=0)                                                  # Register an argument flag locking the random number generator for reproducibility
    ap.add_argument("--cache", default="../artifacts/dock_cache.jsonl")                             # Register an argument flag defining the disk-based cache to bypass redundant docking
    a = ap.parse_args()                                                                             # Parse the provided console arguments resolving the variables
    from oracle.cache import DockCache                                                              # Defer the database import locally protecting against circular or heavy dependency loads
    cfg = yaml.safe_load(open(a.config))                                                            # Read and decode the YAML configuration establishing search boxes and target chains
    calibrate(a.base_csv, cfg, a.out, n_ref=a.n_ref, seed=a.seed, cache=DockCache(a.cache))         # Execute the primary calibration orchestration generating the disk artifact


if __name__ == "__main__":                                                                          # Shield routine from unintended execution when imported externally
    main()                                                                                          # Dispatch the CLI logic flow