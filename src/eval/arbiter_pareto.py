"""
================================================
Independent Physics Arbiter (Stage-3 Evaluation)
================================================

Evaluating reinforcement learning policies (e.g., PPO vs GFlowNet) using the exact same surrogate 
objective they were trained to optimize introduces severe circularity. This approach 
flaws the evaluation by simply rewarding whichever policy exploited the proxy the hardest. 
Previous failure modes demonstrated this: agents generated lipophilic macrocycles that artificially 
inflated Vina docking scores, while deep-learning pose confidence (gnina CNN) remained dismal (~0.25).

This arbiter resolves this by independently re-scoring each method's generated shortlist using 
stringent physics and mechanistic signals that NEITHER policy ever explicitly optimized:

  1. gnina CNNaffinity: A predicted binding affinity (pKd) derived from a Convolutional Neural 
     Network trained on empirical protein-ligand complexes.
  2. gnina CNNscore: A pose confidence metric strictly bounded in [0, 1]. Low values serve as 
     an "honesty check," flagging poses the CNN determines are physically unrealistic despite 
     high raw docking scores.
  3. PoseBusters: An algorithmic check ensuring the 3D pose is physically plausible (verifying 
     geometry, stereochemistry, and absence of steric clashes).
  4. ProLIF switch-II: A mechanistic topological filter confirming the pose successfully engages 
     the KRAS G12C switch-II selectivity groove (and optionally Cys12 for covalent binders).

Methodology:
Molecules containing warheads are dynamically routed to a covalent Cys12 docking protocol (gnina-only), 
while others undergo standard non-covalent docking. Crucially, candidates are ranked and shortlisted 
by their OWN training surrogate (ensuring fair representation of what the policy actually learned to propose), 
but judged solely by this independent physics suite. 

The output is a Pareto row per method capturing true physics-validated quality, covalent capability, 
and topological mode coverage. By tracking the exact fraction of physically valid poses, proxy 
exploitation is made explicitly visible rather than hidden behind raw scores.
"""
import argparse
import json
import os
import sys

# -----------------------------------------------------------------------------------------
# Environment & Path Setup
# Configure Python's module resolution to locate local repository imports and silence RDKit.
# -----------------------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))                     # Inject the parent 'src/' directory into the system path to allow local module imports

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")                                                                       # Mute RDKit's benign C++ backend chatter to keep the evaluation console output clean

# -----------------------------------------------------------------------------------------
# Local Oracle & Evaluation Imports
# Import the physics simulation routines, geometric calculators, and metric utilities.
# -----------------------------------------------------------------------------------------
from oracle.dock import dock_smiles, which_engine                                                    # Import single-molecule docking workflow and the engine resolution utility
from oracle.covalent_dock import covalent_dock_many                                                  # Import the specialized covalent docking orchestrator targeting Cys12
from oracle.prepare_receptor import ligand_com                                                       # Import utility to calculate the Center of Mass from a reference ligand for the docking box
from reward.warhead import has_warhead                                                               # Import the boolean checker that determines covalent vs non-covalent routing
from eval.compare_ppo_gfn import _n_modes                                                            # Reuse the structural mode calculation metric for consistency across evaluations


def judge(smiles_list, cfg, pose_dir, cache=None):
    """
    Evaluates a generated shortlist using independent gnina CNN scoring ("cnn_score", "cnn_affinity") 
    and physical/mechanistic filters (PoseBusters: "posebusters_valid", ProLIF: "switch_ii").

    Partitions the input SMILES list into covalent (warhead-bearing) and non-covalent subsets. 
    It executes the respective docking protocols, extracts raw affinity and CNN scores, and saves 
    the 3D poses. It then passes these poses through PoseBusters (for geometric sanity) and 
    ProLIF (for target-specific mechanistic engagement).

    Args:
        smiles_list (List[str]): The candidate molecules to judge, pre-sorted by their native surrogate.
        cfg (dict): The parsed YAML target configuration (receptor paths, box geometry, docking options).
        pose_dir (str): The directory where the 3D SDF poses will be physically written.
        cache (DockCache or None): An optional seed-locked caching mechanism to bypass redundant docking.

    Returns:
        pandas.DataFrame: A comprehensive dataframe containing one row per molecule, featuring:
        cnn_affinity, cnn_score, dock_affinity, posebusters_valid, switch_ii, and docking mode.
    """
    # -----------------------------------------------------------------------------------------
    # Target & Protocol Configuration
    # Extract the geometric box constraints, paths, and engine preferences from the config.
    # -----------------------------------------------------------------------------------------
    os.makedirs(pose_dir, exist_ok=True)                                                             # Ensure the destination directory for pose SDFs exists, as they are required by PoseBusters and ProLIF
    engine = which_engine(cfg["docking"].get("engine", "gnina")) or "smina"                          # Dynamically resolve the available docking engine, defaulting to 'smina' if 'gnina' is absent
    receptor = cfg["receptor_pdbqt"]                                                                 # Extract the path to the prepared receptor PDBQT file
    box = cfg["docking_box"]                                                                         # Extract the spatial boundary definitions for the docking search
    center = box["center"] or ligand_com(cfg["ref_ligand_sdf"])                                      # Define the explicit box center, falling back to the reference ligand's Center of Mass if omitted
    size = tuple(box["size_angstrom"])                                                               # Extract the rigid box dimensions in Angstroms
    cys = cfg.get("covalent", {}).get("cys_spec", "A:12:SG")                                         # Extract the precise atom specification for the Cys12 thiol anchor
    cnn_mode = cfg["docking"].get("cnn_scoring", "rescore")                                          # Extract the specific CNN scoring mode (e.g., 'rescore') to be used by gnina

    # -----------------------------------------------------------------------------------------
    # Shortlist Partitioning
    # Separate the molecules into covalent and non-covalent pools based on warhead presence.
    # -----------------------------------------------------------------------------------------
    cov = [s for s in smiles_list if has_warhead(Chem.MolFromSmiles(str(s)))]                        # Identify all warhead-bearing SMILES strings to route them to the specialized covalent Cys12 docking protocol
    noncov = [s for s in smiles_list if s not in set(cov)]                                           # Isolate the remaining standard SMILES strings for traditional non-covalent docking
    print(f"[arbiter] engine={engine}  n={len(smiles_list)} (covalent={len(cov)}, "
          f"noncovalent={len(noncov)})")                                                             # Report the split fraction to the console: covalent coverage represents a key KRAS G12C-specific capability claim

    rows = []                                                                                        # Initialize an empty list to accumulate the per-molecule physics judgements
    
    # -----------------------------------------------------------------------------------------
    # Covalent Docking Execution
    # -----------------------------------------------------------------------------------------
    if cov and which_engine("gnina") == "gnina":                                                     # Verify that covalent molecules exist and that gnina is installed, as covalent docking is gnina-only here
        for r in covalent_dock_many(cov, receptor, cys, engine="gnina", cnn_scoring=cnn_mode,
                                    keep_pose_dir=pose_dir, cache=cache):                            # Execute batch covalent docking against the defined Cys12 residue constraint
            rows.append({"smiles": r.smiles, "mode": "covalent", "dock_affinity": r.affinity,
                         "cnn_affinity": r.cnn_affinity, "cnn_score": r.cnn_score,
                         "pose_sdf": r.pose_sdf})                                                    # Record the raw physics signals and resulting pose path into the accumulator

    # -----------------------------------------------------------------------------------------
    # Non-Covalent Docking Execution
    # -----------------------------------------------------------------------------------------
    for s in noncov:                                                                                 # Iterate sequentially through the non-covalent candidate shortlist
        r = dock_smiles(str(s), receptor, center, size, engine=engine, cnn_scoring=cnn_mode,
                        gpu=cfg["docking"].get("gpu", True), keep_pose_dir=pose_dir)                 # Execute standard rigid-receptor docking with deep learning CNN rescoring enabled
        rows.append({"smiles": r.smiles, "mode": "noncovalent", "dock_affinity": r.affinity,
                     "cnn_affinity": r.cnn_affinity, "cnn_score": r.cnn_score,
                     "pose_sdf": r.pose_sdf})                                                        # Record the physical metrics and 3D pose path for the non-covalent candidate

    # Assemble the accumulated per-molecule dictionaries into a comprehensive pandas DataFrame
    df = pd.DataFrame(rows)                                                                          
    prot = cfg.get("receptor_pdb") or cfg.get("protein_pdb")                                         # Extract the standard PDB protein path, which is strictly required by PoseBusters and ProLIF analyzers

    # -----------------------------------------------------------------------------------------
    # PoseBusters Plausibility Check
    # Validates if the 3D pose is physically real (no severe clashes, sane geometry/stereochem).
    # -----------------------------------------------------------------------------------------
    pb = []                                                                                          # Initialize a list to store the per-molecule PoseBusters validity fraction
    try:
        from oracle.posebusters_check import pose_valid_rate                                         # Import the checker lazily to prevent crashing the script if the heavy dependency is absent
        for p in df.get("pose_sdf", []):                                                             # Iterate through every successfully written 3D SDF pose path
            if p and os.path.exists(p) and prot:                                                     # Ensure that both the pose file and the corresponding receptor PDB actively exist on disk
                v, _ = pose_valid_rate(p, prot)                                                      # Calculate the fraction of PoseBusters geometric/clash checks that the pose successfully passed
                pb.append(v)                                                                         # Append the computed validity fraction to the tracker
            else:
                pb.append(None)                                                                      # Append a null value if the pose generation failed entirely
    except Exception as e:                                                                           # Catch any fatal errors caused by missing dependencies or module failures
        print(f"[arbiter] PoseBusters skipped: {e}")                                                 # Alert the user that the geometric plausibility check was bypassed
        pb = [None] * len(df)                                                                        # Populate the tracker with null values so the DataFrame column remains structurally consistent
    df["posebusters_valid"] = pb                                                                     # Attach the resolved sequence of PoseBusters validations directly into the main DataFrame

    # -----------------------------------------------------------------------------------------
    # ProLIF Mechanistic Engagement Check
    # Validates if the pose engages the Switch-II groove (and Cys12 for covalent molecules).
    # -----------------------------------------------------------------------------------------
    sw = []                                                                                          # Initialize a list to store the per-molecule mechanistic interaction verdicts
    try:
        from oracle.interactions import engaged_switch_ii                                            # Import the ProLIF interaction checker lazily to fail-open if the dependency is missing
        for p, m in zip(df.get("pose_sdf", []), df.get("mode", [])):                                 # Iterate concurrently through the pose paths and their respective binding modes
            if p and os.path.exists(p) and prot:                                                     # Ensure that both the pose file and the receptor PDB actively exist
                sw.append(engaged_switch_ii(p, prot, need_cys12=(m == "covalent")))                  # Evaluate interaction constraints: Warhead-bearers dynamically trigger the strict Cys12 contact requirement
            else:
                sw.append(None)                                                                      # Append a null value if the pose does not exist to be analyzed
    except Exception as e:                                                                           # Catch underlying failures originating from MDAnalysis or ProLIF calculations
        print(f"[arbiter] ProLIF skipped: {e}")                                                      # Report the failure honestly to the console rather than silently returning an empty column
        sw = [None] * len(df)                                                                        # Populate the tracker with null values to maintain DataFrame integrity
    df["switch_ii"] = sw                                                                             # Attach the resolved sequence of mechanistic engagement flags directly into the main DataFrame
    
    return df                                                                                        # Return the fully compiled judgement table containing scores and validations


def summarise(df, label):
    """
    Condenses the raw per-molecule judgement table into a single Pareto summary row.

    Filters the DataFrame for valid CNN scores, strict PoseBusters compliance (>= 0.99), 
    and successful Switch-II engagement. It computes medians for affinities and fractions 
    for validities, exposing the true physics-grounded performance of the policy.

    Args:
        df (pandas.DataFrame): The detailed output DataFrame generated by `judge`.
        label (str): The method name identifier (e.g., "PPO", "GFN") for the report.

    Returns:
        dict: A dictionary containing the aggregated physics-validated quality, coverage, 
        and honesty statistics.
    """
    # -----------------------------------------------------------------------------------------
    # Metric Aggregation & Filtering: Filter the dataset by strict validity thresholds 
    # (>99% on Posebusters checks, and successful engagement with switch-II motif).
    # Return a dictionary that contains metrics for the filtered subset such as median pKd, 
    # fraction of valid poses, distinct modes, etc.
    # -----------------------------------------------------------------------------------------
    ok = df.dropna(subset=["cnn_affinity"])                                                             # Isolate the subset of molecules that successfully completed docking and produced a valid CNN score
    pb_ok = df[df.posebusters_valid.fillna(0) >= 0.99] if "posebusters_valid" in df else df.iloc[0:0]   # Isolate poses passing strictly >= 99% of PoseBusters checks; fallback to empty slice if missing
    sw_ok = df[df.switch_ii == True] if "switch_ii" in df else df.iloc[0:0]                             # Isolate the subset of poses that successfully engaged the target switch-II motif
    
    return {"label": label,                                                                          # Record the string identifier designating which policy method this summary row describes
            "n_judged": int(len(df)),                                                                # Record the total number of candidate molecules originally submitted in the shortlist
            "n_scored": int(len(ok)),                                                                # Record the absolute count of molecules that produced a usable CNN affinity score
            "cnn_affinity_median": (round(float(ok.cnn_affinity.median()), 3) if len(ok) else None), # Calculate the median predicted pKd across successful docks, rounding for readability
            "cnn_affinity_best": (round(float(ok.cnn_affinity.max()), 3) if len(ok) else None),      # Extract the absolute best predicted pKd value from the successfully scored subset
            "cnn_score_median": (round(float(ok.cnn_score.median()), 3) if len(ok) else None),       # Calculate median pose confidence: LOW values here explicitly expose proxy exploitation
            "posebusters_valid_frac": (round(float(len(pb_ok) / len(df)), 3) if len(df) else None),  # Calculate the fraction of the shortlist representing physically realistic 3D geometries
            "switch_ii_frac": (round(float(len(sw_ok) / len(df)), 3) if len(df) else None),          # Calculate the fraction of the shortlist successfully hitting the correct mechanistic target
            "n_covalent": int((df["mode"] == "covalent").sum()) if "mode" in df else 0,              # Tally the absolute count of covalent Cys12 binders successfully generated
            "n_modes_physics_valid": _n_modes(list(pb_ok.smiles)) if len(pb_ok) else 0}              # Calculate the distinct topological mode coverage specifically AMONG physically valid molecules


def main():
    """
    CLI Execution Entrypoint.
    
    Parses policy sample CSVs, shortlists the top-k molecules strictly by their *own* 
    training surrogate reward, and then executes the independent `judge` pipeline. 
    It prints a formatted Pareto table to the console and writes a JSON summary report.
    """
    # -----------------------------------------------------------------------------------------
    # CLI Argument Parsing & Setup
    # -----------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser(description="Independent physics arbiter for PPO vs GFlowNet.")
    ap.add_argument("--config", default="../configs/kras_g12c.yaml")                                 # Define the argument path for the master target YAML configuration
    ap.add_argument("--csv", nargs="+", required=True, help="LABEL=path/to/samples.csv (repeatable)")# Define repeatable argument to ingest generation CSVs in the form LABEL=path (e.g. PPO=../results/ppo_s0.csv)
    ap.add_argument("--topk", type=int, default=100, help="molecules judged per method")             # Define cap for how many molecules are judged per method, as physics simulations are expensive
    ap.add_argument("--out", default="../results/arbiter_pareto.json")                               # Define the destination path where the final aggregated arbiter JSON verdict lands
    ap.add_argument("--pose_dir", default="../results/arbiter_poses")                                # Define the destination directory where the 3D SDF poses are physically written
    ap.add_argument("--cache", default="../artifacts/dock_cache.jsonl")                              # Define the path for the shared docking cache to make subsequent re-judging computationally free
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))                                                             # Load and parse the targeted physical YAML configuration file
    from oracle.cache import DockCache                                                               # Import the caching module locally here to keep the global module import overhead light
    cache = DockCache(a.cache)                                                                       # Instantiate the seed-locked cache engine shared synchronously with the Active Learning loop

    # -----------------------------------------------------------------------------------------
    # Shortlist Generation & Judgement Execution: For every policy method in the CSV list,
    # sort the shortlist strictly by the policy's own rewards, slice top-k, run the independent 
    # physics arbiter and record the results to a summarized Pareto table dict.
    # -----------------------------------------------------------------------------------------
    out = {"topk": a.topk, "methods": []}                                                            # Initialize the master payload dictionary that will comprise the final arbiter report
    for spec in a.csv:                                                                               # Iterate sequentially over every provided policy shortlist CSV specification
        label, path = spec.split("=", 1)                                                             # Split the LABEL=path formatting to keep the internal parsing and final report self-describing
        df = pd.read_csv(path).dropna(subset=["reward"]).sort_values("reward", ascending=False)      # Rank strictly by the policy's OWN objective: shortlist selection stays perfectly fair to what the policy learned
        shortlist = df.drop_duplicates("smiles").head(a.topk).smiles.tolist()                        # Slice the top-K unique molecule strings this specific method would realistically propose to a chemist
        print(f"\n[arbiter] === {label}: judging top {len(shortlist)} from {path} ===")              # Stream progress to the console
        
        jdf = judge(shortlist, cfg, os.path.join(a.pose_dir, label), cache=cache)                    # Execute the heavy physics pipeline: Docking + CNN Rescoring + PoseBusters + ProLIF
        jdf.to_csv(os.path.join(os.path.dirname(a.out) or ".", f"arbiter_{label}.csv"), index=False) # Export the per-molecule judgement detail for granular auditing and transparency
        out["methods"].append(summarise(jdf, label))                                                 # Append the condensed Pareto summary row for this method into the master report

    # -----------------------------------------------------------------------------------------
    # Report Generation & Final Output: Write the final arbiter report to disk as a JSON file
    # -----------------------------------------------------------------------------------------
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)                                        # Ensure the physical destination directory for the JSON payload exists
    json.dump(out, open(a.out, "w"), indent=2)                                                       # Persist the fully assembled arbiter verdict to disk as a formatted JSON file

    print(f"\n{'method':14s}{'cnnAff_med':>12s}{'cnnScore':>10s}{'PB_valid':>10s}"
          f"{'switchII':>10s}{'modes':>7s}{'covalent':>10s}")                                        # Print the strictly formatted column header for the final console arbiter table
    for m in out["methods"]:                                                                         # Iterate over the calculated Pareto row for each judged method
        def f(k, d=2):                                                                               # Define a local helper function to format floating point metrics cleanly
            return f"{m[k]:.{d}f}" if m.get(k) is not None else "n/a"                                # Apply decimal formatting or return 'n/a' if the metric was missing or skipped
        print(f"{m['label']:14s}{f('cnn_affinity_median'):>12s}{f('cnn_score_median'):>10s}"
              f"{f('posebusters_valid_frac'):>10s}{f('switch_ii_frac'):>10s}"
              f"{m['n_modes_physics_valid']:>7d}{m['n_covalent']:>10d}")                             # Print the aligned data row representing the method's physical ground-truth performance
    print(f"\n[arbiter] wrote {a.out}")                                                              # Confirm the successful creation of the JSON artefact file
    print("[arbiter] cnn_score is the honesty check: a high cnn_affinity with a LOW cnn_score means\n"
          "          the docking score was exploited and the CNN does not believe the pose.")        # Print a final explicit warning making the exploitation signal impossible to misinterpret


if __name__ == "__main__":
    main()