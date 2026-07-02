#!/usr/bin/env python
"""
==============================
Stage 2 - ChEMBL Data Fetcher
==============================

This standalone script extracts real KRAS bioactivity data from the ChEMBL REST API 
to build a robust training dataset for the surrogate affinity model. It generates a 
clean CSV file containing `smiles` and `pchembl` pairs.

How it works:
1. Target Resolution: Maps a UniProt accession (default: human KRAS, P01116) to 
   its corresponding ChEMBL target IDs (filtering for SINGLE PROTEIN).
   
2. Data Extraction: Paginates through the ChEMBL API, extracting canonical SMILES 
   and valid pChEMBL values (where IC50/Ki/Kd are already -log10-scaled).
   
3. Data Cleaning: Filters out missing or zero values, and collapses duplicated 
   SMILES by computing their median pChEMBL activity.

Key Notes & Constraints:
- Minimal Dependencies: Operates entirely on the standard `requests` and `pandas` libraries, 
  requiring no heavy chemoinformatics installations (no PyTDC, no PhysDock).
  
- Wild-Type KRAS Profile: Most ChEMBL KRAS assays pool data predating the G12C series. 
  Thus, this script anchors the surrogate in real, albeit wild-type, experimental affinity. 
  The G12C-specificity/covalency caveat is acknowledged; the surrogate acts strictly as a 
  relative ranker, not an absolute binding free-energy (ΔG) model.
  
- Execution Environment: If local environments block outbound HTTP requests, this script 
  is designed to execute successfully on the SageMaker instance (which permits internet 
  access), allowing the resulting CSV to simply be committed back to a local repository.
"""

import argparse
import sys
import time

import pandas as pd

# -----------------------------------------------------------------------------------------
# Dependency Verification
# Ensure the external requests library is available, failing gracefully if missing.
# -----------------------------------------------------------------------------------------
try:
    import requests                                                                         # Import the standard requests library to handle HTTP GET calls to the ChEMBL API
except ImportError:
    print("This script needs `requests` (pip install requests).", file=sys.stderr)          # Log a fatal error to the standard error stream guiding the user to install the missing package
    raise                                                                                   # Re-raise the exception to strictly halt execution, as the script cannot function without API access

# -----------------------------------------------------------------------------------------
# Global API Constants
# Define the root endpoints and headers required by the EBI ChEMBL REST architecture.
# -----------------------------------------------------------------------------------------
BASE = "https://www.ebi.ac.uk/chembl/api/data"                                              # Define the base URL endpoint for all ChEMBL API data requests
HEADERS = {"Accept": "application/json"}                                                    # Define the HTTP headers to strictly enforce JSON-formatted responses from the server


def resolve_targets(uniprot: str) -> list:
    """
    Resolves a provided UniProt accession string (target's UniProt ID) into corresponding 
    ChEMBL target IDs (filtering for SINGLE PROTEIN).
    
    Queries the ChEMBL target endpoint matching the given UniProt accession. It filters 
    the returned payload to ensure only explicitly designated 'SINGLE PROTEIN' targets 
    are collected, discarding protein complexes or uncurated entries.
    
    Args:
        uniprot (str): The UniProt accession number (e.g., 'P01116' for human KRAS).
        
    Returns:
        list: A list of string ChEMBL target IDs matching the query.
        
    Example:
        >>> resolve_targets("P01116")
        ['CHEMBL4040', 'CHEMBL...']
    """
    # -------------------------------------------------------------------------------------
    # Target ID Resolution
    # Query the ChEMBL registry and filter for valid single-protein target identifiers.
    # -------------------------------------------------------------------------------------
    url = f"{BASE}/target.json?target_components__accession={uniprot}&limit=100"            # Construct the API URL querying targets by their associated UniProt accession component
    r = requests.get(url, headers=HEADERS, timeout=60); r.raise_for_status()                # Execute the GET request with a 60-second timeout, automatically raising an exception on HTTP error codes
    tids = []                                                                               # Initialize an empty list to securely accumulate the verified ChEMBL target ID strings
    for t in r.json().get("targets", []):                                                   # Parse the JSON response payload and iterate over the list of returned target dictionaries
        if t.get("target_type") == "SINGLE PROTEIN":                                        # Apply a strict filter to ensure the target is classified exactly as a 'SINGLE PROTEIN'
            tids.append(t["target_chembl_id"])                                              # Append the validated ChEMBL ID string to the tracking list
    return tids                                                                             # Return the fully populated list of resolved target IDs


def fetch_activities(target_id: str, sleep: float = 0.2) -> list:
    """
    Paginates through the ChEMBL API to retrieve bioactivity data for a specific target.
    
    Constructs a query for the `activity` endpoint, explicitly filtering for non-null 
    pChEMBL values. It iteratively processes 1000-record pages, extracting the canonical 
    SMILES string and parsing the pChEMBL scalar value for each small molecule (ligand), 
    pausing between pages to respect API rate limits.
    
    Args:
        target_id (str): The specific ChEMBL target ID to query (e.g., 'CHEMBL4040').
        sleep (float, optional): Seconds to pause between pagination requests. Defaults to 0.2.
        
    Returns:
        list: A list of tuples, each containing a (ligand/small molecule SMILES string, pChEMBL float).
        
    Example:
        >>> data = fetch_activities("CHEMBL4040")
        >>> len(data)
        1420
    """
    # -------------------------------------------------------------------------------------
    # Activity Data Pagination Loop
    # Sequentially fetch, parse, and filter bioactivity records across all result pages.
    # -------------------------------------------------------------------------------------
    rows = []                                                                               # Initialize an empty list to aggregate the valid (SMILES, pChEMBL) data tuples
    url = (f"{BASE}/activity.json?target_chembl_id={target_id}"                             # Construct the foundational API URL routing to the specific target ID's activity endpoint
           f"&pchembl_value__isnull=false&limit=1000")                                      # Append strict query parameters discarding null pChEMBLs and maximizing the page size to 1000
    while url:                                                                              # Initiate a while loop that persists as long as a valid pagination URL is dynamically resolved
        r = requests.get(url, headers=HEADERS, timeout=120); r.raise_for_status()           # Fire the HTTP GET request with a generous 120-second timeout, raising an exception upon failure
        js = r.json()                                                                       # Deserialize the raw JSON text response directly into a nested Python dictionary
        
        # ---------------------------------------------------------------------------------
        # Record Parsing & Type Casting
        # Extract desired fields and safely coerce pChEMBL strings to floating-point numbers.
        # ---------------------------------------------------------------------------------
        for a in js.get("activities", []):                                                  # Iterate sequentially over every individual activity dictionary in the current page's payload
            smi = a.get("canonical_smiles"); pv = a.get("pchembl_value")                    # Extract the canonical SMILES string alongside the pre-scaled pChEMBL value from the record
            if smi and pv:                                                                  # Verify that both the structural string and the numerical value actually exist (are truthy)
                try:                                                                        # Wrap the float cast inside a try block to intercept corrupted or non-numerical pChEMBL data
                    rows.append((smi, float(pv)))                                           # Cast the pChEMBL string into a float and append the resulting tuple pair to the rows list
                except (TypeError, ValueError):                                             # Catch specific casting exceptions raised if the API returns malformed numerical data
                    pass                                                                    # Silently ignore the problematic record and proceed to the next iteration safely
        
        # ---------------------------------------------------------------------------------
        # Pagination Handling
        # Resolve the next page cursor and enforce rate-limiting.
        # ---------------------------------------------------------------------------------
        nxt = js.get("page_meta", {}).get("next")                                           # Dig into the payload metadata dictionary to extract the relative URL routing to the next page
        url = ("https://www.ebi.ac.uk" + nxt) if nxt else None                              # Prepend the host domain to the relative path, or flip the URL to None if pagination has ended
        time.sleep(sleep)                                                                   # Pause the thread execution briefly to respect EBI server limits and prevent IP blacklisting
    return rows                                                                             # Return the master list containing every accumulated data tuple across all paginated queries


def main():
    """
    The main execution orchestrator for the data fetching script.
    
    How it works:
    1. Parses command-line arguments to resolve paths and target specifications.
    2. Resolves the UniProt accession into ChEMBL target IDs (or accepts overrides).
    3. Iterates over resolved targets, invoking the activity pagination fetcher.
    4. Aggregates data into a pandas DataFrame, filtering out <= 0 pChEMBL values.
    5. Deduplicates identical SMILES strings by collapsing them into a median pChEMBL score.
    6. Ensures output directories exist and writes the final clean dataset to a CSV file.
    
    Returns:
        None
    """
    # -------------------------------------------------------------------------------------
    # CLI Argument Parsing
    # Configure and parse user inputs defining output paths and target overrides.
    # -------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser()                                                          # Instantiate an ArgumentParser object to manage command-line interfaces and flags
    ap.add_argument("--out", default="data/chembl_kras.csv")                                # Define the output path argument, defaulting to a standard CSV file within the data directory
    ap.add_argument("--uniprot", default="P01116", help="KRAS human UniProt (default P01116)") # Define the target UniProt argument, providing the default human KRAS accession code
    ap.add_argument("--target_ids", default="", help="comma-separated ChEMBL target IDs to override resolution") # Define an optional string argument allowing direct ChEMBL ID overrides
    args = ap.parse_args()                                                                  # Execute the parser mapping provided CLI strings into a structured namespace object

    # -------------------------------------------------------------------------------------
    # Target Resolution & Validation
    # Determine the target IDs either through explicit overrides or UniProt API lookup.
    # -------------------------------------------------------------------------------------
    if args.target_ids:                                                                     # Evaluate if the user provided explicit comma-separated target IDs in the command line
        tids = [t.strip() for t in args.target_ids.split(",") if t.strip()]                 # Split the input string by commas, strip whitespace, and generate a clean list of IDs
    else:                                                                                   # Execute this branch if no direct overrides were provided by the user
        tids = resolve_targets(args.uniprot)                                                # Invoke the API resolution function mapping the provided UniProt string to ChEMBL IDs
    if not tids:                                                                            # Check if the list of target IDs remains empty after all resolution attempts
        print("No ChEMBL targets resolved; pass --target_ids explicitly.", file=sys.stderr) # Emit a fatal error to the stderr stream warning the user of the resolution failure
        sys.exit(1)                                                                         # Forcefully terminate the script with a non-zero exit status indicating an error state
    print(f"[chembl] targets: {tids}")                                                      # Log the fully resolved list of ChEMBL target IDs to the console for transparency

    # -------------------------------------------------------------------------------------
    # Master Data Aggregation
    # Sequentially fetch and combine activity records across all validated targets.
    # -------------------------------------------------------------------------------------
    all_rows = []                                                                           # Initialize an empty master list intended to aggregate rows from all queried target IDs
    for tid in tids:                                                                        # Iterate sequentially over every identified target ID in the resolved list
        rows = fetch_activities(tid)                                                        # Call the pagination fetcher to retrieve all (SMILES, pChEMBL) tuples for the current target
        print(f"[chembl] {tid}: {len(rows)} pChEMBL activities")                            # Print a status update logging the exact integer count of activities recovered for this target
        all_rows += rows                                                                    # Concatenate the current target's rows into the overarching master list via list addition

    if not all_rows:                                                                        # Verify if the master aggregation list remains entirely empty after traversing all targets
        print("No activities fetched.", file=sys.stderr); sys.exit(1)                       # Print a failure message to stderr and abort the script, as no data exists to process

    # -------------------------------------------------------------------------------------
    # Dataset Cleaning & Deduplication
    # Convert to DataFrame, filter invalid bounds, and collapse duplicates to median.
    # -------------------------------------------------------------------------------------
    df = pd.DataFrame(all_rows, columns=["smiles", "pchembl"])                              # Instantiate a pandas DataFrame from the master list assigning proper string column headers
    df = df[df["pchembl"] > 0].dropna()                                                     # Apply a strict boolean mask keeping only positive pChEMBL values and aggressively dropping NaNs
    df = df.groupby("smiles", as_index=False)["pchembl"].median()   # de-dup -> median activity # Group identical SMILES strings and mathematically collapse them by computing the median pChEMBL value
    
    # -------------------------------------------------------------------------------------
    # File Output & Serialization
    # Ensure the target directory structure exists and write the final tabular dataset.
    # -------------------------------------------------------------------------------------
    import os                                                                               # Import the standard os module locally to manipulate the operating system file paths
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)                            # Dynamically create the requisite parent directory tree for the output file, ignoring if it exists
    # Write in CSV the deduplicated pandas DataFrame which contains only unique SMILES strings
    # and their corresponding (median) pChEMBL values of all molecules (ligands) that were ever 
    # recorded to the ChEMBL database to bind against the target (human KRAS) and were successfully parsed 
    df.to_csv(args.out, index=False)                                                        # Serialize the cleaned, deduplicated pandas DataFrame to a standard CSV file without row indexes
    print(f"[done] wrote {len(df)} unique molecules to {args.out}")                         # Log a final success message denoting the exact count of unique molecules successfully written
    print(f"[done] pChEMBL range {df['pchembl'].min():.2f}–{df['pchembl'].max():.2f}, "     # Log detailed descriptive statistics highlighting the minimum and maximum bounds of the dataset
          f"median {df['pchembl'].median():.2f}")                                           # Log the overarching median pChEMBL value representing the central tendency of the entire batch


if __name__ == "__main__":                                                                  # Prevent automatic execution if the script is imported as a module elsewhere
    main()                                                                                  # Formally invoke the main execution orchestrator to begin the data extraction process