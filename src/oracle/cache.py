"""
======================================
Docking Result Cache (Stage-3 Oracle)
======================================

This module provides a small, dependency-free, disk-backed memoisation layer for the docking
oracle. Docking is by far the most computationally expensive operation in the Stage-3 active-learning 
loop. The exact same molecule is frequently re-encountered — across acquisition rounds, across 
PPO-vs-GFlowNet comparisons, and critically, across separate SageMaker sessions when a run is 
resumed on a fresh cloud instance. 

Re-docking an identical (molecule, receptor, box, engine) tuple is pure wasted GPU/CPU
time and, because every dock here is seeded deterministically (`--seed 0`), it also produces a bit-
for-bit identical result. Caching therefore accelerates the pipeline without changing any number the
oracle would have returned.

Design notes:
- The cache key is a SHA-1 of the CANONICAL SMILES plus every parameter that can change the score
  (receptor path, box, engine, covalent flag). Two different boxes or engines never collide.
- Values are stored as one JSON line per key in a single append-only file (`dock_cache.jsonl`),
  which is trivial to `git add`, to `tar`, and to carry between accounts as resumable state.
- The cache is read fully into memory once on construction (a few thousand small rows is nothing)
  and written through on every `put`, so a killed session never loses labels it already paid for.
"""
import hashlib
import json
import os
import threading
from typing import Dict, Optional


def _canon(smi: str) -> str:
    """
    Canonicalises a SMILES string so chemically-identical inputs map to one exact cache key.

    Uses RDKit when available (the normal case inside the docking environment) so that, 
    e.g., "OCC" and "CCO" resolve to the exact same key. If RDKit is somehow unavailable, 
    the raw string is returned unchanged — a strictly conservative fallback that can only 
    ever MISS the cache, never return a wrong molecule's score.

    Args:
        smi (str): The raw input SMILES string.

    Returns:
        str: The RDKit canonical SMILES, or the original string if RDKit is not importable.

    Example:
        >>> _canon("OCC")
        'CCO'
    """
    # ------------------------------------------------------------------------------------------
    # Chemical Standardisation
    # Attempt to normalise the molecular string, failing safely if chemoinformatics are missing.
    # ------------------------------------------------------------------------------------------
    try:                                                                                        # Attempt the RDKit canonicalisation path (the standard case in the docking environment)
        from rdkit import Chem                                                                  # Import RDKit lazily so this module stays importable in torch-free unit tests
        m = Chem.MolFromSmiles(smi)                                                             # Parse the input string into an RDKit molecule object for standardisation
        return Chem.MolToSmiles(m) if m is not None else smi                                    # Emit the canonical form, or fall back to the raw string if parsing failed
    except Exception:                                                                           # Catch a missing RDKit or any parser fault without ever crashing the cache
        return smi                                                                              # Conservative fallback: use the raw string (can only cause a benign cache miss)


def _key(smi: str, **parts) -> str:
    """
    Builds the deterministic cache key from the canonical SMILES and all score-affecting parameters (receptor, box, engine).

    Dumps the ligand string and all context parameters (like engine, receptor, box size) into 
    a sorted JSON payload, ensuring key order doesn't alter the result. It then hashes this 
    payload into a fixed-length SHA-1 digest to serve as a fast dictionary/file key.

    Args:
        smi (str): The (pre-canonicalised) ligand SMILES string.
        **parts: Any additional parameters that change the docking result (receptor, box, engine,
                 covalent flag). Order is irrelevant because the dict is sorted before hashing.

    Returns:
        str: A 40-character hexadecimal SHA-1 digest uniquely identifying this docking request.

    Example:
        >>> _key("CCO", receptor="6oim.pdbqt", engine="smina")  # doctest: +ELLIPSIS
        '...'
    """
    # --------------------------------------------------------------------------------------------
    # Cryptographic Key Generation: Serialise the canonical SMILES and all score-affecting 
    # parameters into a stable, order-independent JSON string, then hash it to a fixed-length key.
    # --------------------------------------------------------------------------------------------
    payload = json.dumps({"smi": smi, **parts}, sort_keys=True)                                 # Serialise the key parts to a canonical, order-independent JSON string
    return hashlib.sha1(payload.encode()).hexdigest()                                           # Hash that string into a compact, collision-resistant fixed-length key


class DockCache:
    """
    A minimal, thread-safe, append-only JSONL cache for docking results.

    Loads any pre-existing rows from a flat `.jsonl` file into an in-memory dictionary 
    on construction to serve instant O(1) lookups. It writes each new result straight 
    through to disk synchronously under a thread lock, so an interrupted SageMaker session 
    never loses labels it has already expended compute on. The backing file doubles as 
    highly portable resume state.
    """
    def __init__(self, path: str):
        """
        Opens (or creates) the cache file and loads existing rows directly into memory.

        Args:
            path (str): Filesystem path to the JSONL cache file (created if absent).

        Example:
            >>> cache = DockCache("state/dock_cache.jsonl")
        """
        # -----------------------------------------------------------------------------------------
        # Cache Initialisation
        # Materialise the backing directory, then stream any prior rows into the in-memory index.
        # -----------------------------------------------------------------------------------------
        self.path = path                                                                        # Persist the target file path for all subsequent write-through operations
        self._lock = threading.Lock()                                                           # Guard the in-memory dict and file appends against concurrent writers
        self._mem: Dict[str, dict] = {}                                                         # Allocate the in-memory key->result index that backs O(1) lookups
        d = os.path.dirname(path)                                                               # Resolve the parent directory of the cache file for creation
        if d:                                                                                   # Only attempt directory creation when a non-empty parent path exists
            os.makedirs(d, exist_ok=True)                                                       # Ensure the state directory exists so the first append cannot fail
        if os.path.exists(path):                                                                # Load previously-cached rows only if a cache file already exists on disk
            with open(path) as fh:                                                              # Open the append-only JSONL file for a single sequential read pass
                for line in fh:                                                                 # Iterate over each persisted row exactly once during warm-up
                    try:                                                                        # Defend each row so one corrupt line cannot abort the whole load
                        row = json.loads(line)                                                  # Parse the JSON row back into a python dict
                        self._mem[row["k"]] = row["v"]                                          # Register the stored value under its precomputed cache key
                    except Exception:                                                           # Silently skip malformed/partial lines (e.g., from a hard kill mid-write)
                        continue                                                                # Move on to the next row without disturbing the load

    def get(self, smi: str, **parts) -> Optional[dict]:
        """
        Retrieves the cached docking result (DockResult) for this molecule and parameter state.

        Args:
            smi (str): The ligand SMILES string (canonicalised internally).
            **parts: The same score-affecting parameters supplied to `put` (receptor, box, engine...).

        Returns:
            Optional[dict]: The previously stored result dict, or None on a cache miss.

        Example:
            >>> cache.get("CCO", receptor="6oim.pdbqt", engine="smina") is None
            True
        """
        # -----------------------------------------------------------------------------------------
        # State Retrieval
        # -----------------------------------------------------------------------------------------
        return self._mem.get(_key(_canon(smi), **parts))                                        # Canonicalise, hash, and look up — returning None cleanly on any miss

    def put(self, smi: str, result: dict, **parts) -> None:
        """
        Stores a docking result (DockResult) to RAM and writes it securely through to the JSONL disk file.

        Args:
            smi (str): The ligand SMILES string (canonicalised internally).
            result (dict): A JSON-serialisable result (typically DockResult.__dict__).
            **parts: The score-affecting parameters that, with `smi`, define the cache key.

        Returns:
            None

        Example:
            >>> cache.put("CCO", {"affinity": -6.1, "ok": True}, receptor="6oim.pdbqt", engine="smina")
        """
        # -----------------------------------------------------------------------------------------
        # Write-Through Persistence
        # Update the in-memory index and append one durable JSON row under a single lock.
        # -----------------------------------------------------------------------------------------
        k = _key(_canon(smi), **parts)                                                          # Compute the deterministic key from the canonical SMILES and parameters
        with self._lock:                                                                        # Serialise the memory update and file append to stay consistent under threads
            self._mem[k] = result                                                               # Update the hot in-memory index so the very next get() hits immediately
            with open(self.path, "a") as fh:                                                    # Open the backing file in append mode for a durable, crash-safe write
                fh.write(json.dumps({"k": k, "v": result}) + "\n")                              # Persist exactly one JSON line so a kill mid-run loses at most this row