"""
===============================
Stage-2 Pocket Encoding (ESM-2)
===============================

Encodes the target (KRAS) pocket with ESM-2, or with a fallback if ESM-2 is not installed.

-   Primary path: load an ESM-2 model, run the full target sequence (i.e. the whole protein), 
    and mean-pool the per-residue embeddings over the pocket residues. This is a genuine learned
    representation of the binding site; swapping it for a different pocket at eval is the ablation 
    that proves the policy is target-conditioned.

-   Fallback path (only if fair-esm is genuinely absent): a deterministic residue-identity 
    vector so the plumbing still runs. It is NOT biologically meaningful and prints a loud 
    warning; install fair-esm for the real run.

How it works:
The script provides a unified interface (`encode`) that attempts to leverage the primary path, 
gracefully defaulting to the fallback path upon import or execution failure. It can be utilized 
as a standalone CLI script configured via YAML or direct arguments, or imported as a module 
within the broader Stage-2 pipeline.

CLI:
    python -m pocket.encode_pocket --config ../configs/kras_g12c.yaml
  or
    python -m pocket.encode_pocket --fasta ../data/kras/kras_g12c.fasta \
        --residues 12,68,95,96,99,102 --model esm2_t33_650M_UR50D --out ../data/kras_g12c_pocket.npy
"""

import argparse
import hashlib

import numpy as np

# -----------------------------------------------------------------------------------------
# Global Constants
# Standard definitions for the 20 canonical amino acids used for histogram generation 
# within the fallback mechanism.
# -----------------------------------------------------------------------------------------
_AA = "ACDEFGHIKLMNPQRSTVWY"                                                                  # Define the single-letter codes for the 20 canonical amino acids as a string
_AA_IDX = {a: i for i, a in enumerate(_AA)}                                                   # Create a dictionary mapping each amino acid string character to its integer index


def _read_fasta(path: str) -> str:
    """
    Reads a FASTA file and extracts the raw continuous protein sequence.
    
    Opens the file, iterates line-by-line, ignores metadata lines (starting with '>'), 
    and concatenates the remaining sequence lines into a single continuous string with 
    no whitespace.
    
    Args:
        path (str): The file path to the target FASTA document.
        
    Returns:
        str: The fully assembled protein sequence.
        
    Example:
        >>> # Assuming 'dummy.fasta' contains ">Seq1\nMKWVT\nFGG"
        >>> _read_fasta("dummy.fasta")
        'MKWVTFGG'
    """
    # -------------------------------------------------------------------------------------
    # FASTA Parsing
    # Read the file and filter out structural formatting to isolate the sequence.
    # -------------------------------------------------------------------------------------
    seq = []                                                                                  # Initialize an empty list to accumulate segments of the protein sequence
    with open(path) as f:                                                                     # Open the specified file path in standard read mode using a context manager
        for line in f:                                                                        # Iterate sequentially over every individual line in the text file
            if line.startswith(">"):                                                          # Check if the current line is a FASTA header/metadata line (begins with '>')
                continue                                                                      # Skip the header line entirely and proceed to the next iteration
            seq.append(line.strip())                                                          # Strip trailing/leading whitespace and append the sequence fragment to the list
    return "".join(seq)                                                                       # Concatenate all collected string fragments into one continuous string and return


def esm_pocket_embedding(seq: str, residues, model_name: str = "esm2_t33_650M_UR50D",
                         device: str = None) -> np.ndarray:
    """
    Mean-pooled ESM-2 embedding over the pocket residues. Raises if fair-esm/torch unavailable.
    
    Dynamically imports PyTorch and fair-esm to avoid hard dependencies if they aren't installed. 
    Loads the requested pre-trained ESM-2 model and its alphabet. Tokenizes the input sequence, 
    runs a forward pass extracting the final layer's representation, and indexes into the 
    resulting tensor using 1-based biological residue indices (properly offset by the BOS token). 
    Finally, it averages these targeted residue vectors to form a single pocket embedding.
    
    Args:
        seq (str): The full continuous string of the protein sequence.
        residues (Iterable[int]): A list or iterable of 1-based residue indices defining the pocket.
        model_name (str, optional): The fair-esm model identifier. Defaults to "esm2_t33_650M_UR50D".
        device (str, optional): Compute device ("cuda" or "cpu"). Auto-detects if None.
        
    Returns:
        np.ndarray: A 1D numpy array (float32) representing the mean-pooled pocket embedding.
        
    Example:
        >>> seq = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY"
        >>> # Get embedding for residues 12 and 13
        >>> emb = esm_pocket_embedding(seq, [12, 13]) 
        >>> emb.shape
        (1280,)
    """
    # -----------------------------------------------------------------------------------------
    # Dynamic Model Loading
    # Import torch, fair-esm libraries locally and initialize the specified ESM-2 architecture.
    # -----------------------------------------------------------------------------------------
    import torch                                                                              # Dynamically import PyTorch inside the function to allow failure if missing
    import esm                                                                                # Dynamically import the fair-esm library for protein language modeling
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")                       # Resolve target device: use provided, fallback to CUDA if available, else CPU
    model, alphabet = esm.pretrained.__dict__[model_name]()                                   # Dynamically fetch and instantiate the specified ESM-2 model and its tokenizer
    model = model.to(device).eval()                                                           # Move model weights to the target device and freeze them in evaluation mode
    
    # -------------------------------------------------------------------------------------
    # Tokenization & Forward Pass: Instantiate the batch converter tool and convert the raw
    # sequence into padded integer token tensors.
    # -------------------------------------------------------------------------------------
    # 
    bc = alphabet.get_batch_converter()                                                       # Instantiate the batch converter tool to translate strings into token tensors
    _, _, toks = bc([("target", seq)])                                                        # Convert the raw sequence into padded integer tokens (ignoring labels/strings)
    toks = toks.to(device)                                                                    # Transfer the newly generated token tensor to the designated compute device
    with torch.no_grad():                                                                     # Suspend gradient tracking to conserve memory and accelerate inference
        out = model(toks, repr_layers=[model.num_layers])                                     # Execute the model forward pass, requesting explicit output from the final layer only
    
    # -------------------------------------------------------------------------------------------------------
    # Feature Extraction & Pooling
    # Extract only the targeted (pocket) residue vectors and average them to form the pocket representation.
    # -------------------------------------------------------------------------------------------------------
    # Extract the raw dense representations for the entire sequence
    rep = out["representations"][model.num_layers][0]                        # shape [SeqLen+2, HiddenDim] with BOS/EOS; 
    # Filter the requested residue indices to ensure they fall within the actual sequence bounds
    idx = [r for r in residues if 1 <= r <= len(seq)]                        # rep[r] is residue r (1-indexed, BOS at 0)
    # Stack targeted residue vectors
    picked = torch.stack([rep[r] for r in idx]) if idx else rep[1:-1]        # Fallback to stacking the whole sequence if pocket is empty
    # Average the stacked vectors
    return picked.mean(0).cpu().numpy().astype(np.float32)                   # Average the stacked vectors along the node dimension, cast to numpy float32, and return


def _deterministic_fallback(seq: str, residues, dim: int = 1280) -> np.ndarray:
    """
    Provides a non-biological, deterministic vector when ESM-2 is unavailable.
    
    Computes a normalized histogram of the amino acids explicitly present in the designated 
    pocket residues. It then creates a unique, deterministic hash key based on the sequence 
    and pocket structure. This hash seeds a random normal generator, which projects the 
    20-dimensional histogram into the required high-dimensional space (e.g., 1280). 
    Prints a loud warning.
    
    Args:
        seq (str): The full protein sequence.
        residues (Iterable[int]): 1-based indices of the pocket residues.
        dim (int, optional): The target output dimensionality to mock ESM. Defaults to 1280.
        
    Returns:
        np.ndarray: A 1D L2-normalized numpy array (float32) of size `dim`.
        
    Example:
        >>> emb = _deterministic_fallback("MKWVTFGG", [1, 2, 3])
        >>> emb.shape
        (1280,)
    """
    # -------------------------------------------------------------------------------------------------
    # Fallback Warning & Histogram: Alert the user to the failure state and compute a basic composition
    # profile: count the amino acids in the pocket, add them to a histogram and normalize it.
    # -------------------------------------------------------------------------------------------------
    print("[encode_pocket] WARNING: fair-esm not available -> deterministic fallback (NOT biological). " # Print a highly visible warning explaining that the generated vector lacks biological meaning
          "Install fair-esm for the real ablation.")                                          # Advise the user to install the proper dependencies to restore intended behavior
    hist = np.zeros(20, dtype=np.float64); picked = []                                        # Initialize a zeroed 20-bin histogram array and a tracking list for targeted amino acids
    for r in residues:                                                                        # Iterate through the requested 1-based biological residue indices
        if 1 <= r <= len(seq):                                                                # Validate that the current residue index safely falls within the sequence length
            aa = seq[r - 1].upper(); picked.append(aa)                                        # Extract the character (offsetting by -1 for zero-index), capitalize it, and log it
            if aa in _AA_IDX:                                                                 # Check if the extracted character is a recognized canonical amino acid
                hist[_AA_IDX[aa]] += 1.0                                                      # Increment the corresponding amino acid's specific bin in the histogram array
    if hist.sum() > 0:                                                                        # Check if any valid canonical amino acids were actually found and counted
        hist /= hist.sum()                                                                    # Normalize the histogram so all bin values sum cleanly to 1.0
        
    # ------------------------------------------------------------------------------------------------------
    # Deterministic Projection: Hash the inputs to seed a RNG, which would generate a pseudo-random 
    # projection matrix to expand the histogram to `dim` dimensions, and L2-normalize the resulting vector.
    # ------------------------------------------------------------------------------------------------------
    # Construct a unique byte-string key combining sequence prefix, residue indices, and matched amino acids
    key = (seq[:64] + "|" + ",".join(map(str, residues)) + "|" + "".join(picked)).encode()    
    # Hash the key via SHA-256 and use it to strictly seed a local numpy random number generator
    rng = np.random.RandomState(int(hashlib.sha256(key).hexdigest(), 16) % (2 ** 32))         
    # Generate a pseudo-random projection matrix and multiply it by the histogram to expand to `dim`
    vec = rng.normal(0, 1, size=(dim, 20)) @ hist                                             
    # L2-normalize the resulting vector to maintain stable magnitudes and cast to float32
    return (vec / (np.linalg.norm(vec) + 1e-8)).astype(np.float32)                            


def encode(seq, residues, model_name="esm2_t33_650M_UR50D", out=None):
    """
    High-level API for encoding a pocket, routing to a fallback on failure.
    
    Wraps `esm_pocket_embedding` in a try-except block. If successful, logs the embedding 
    dimension. If any exception occurs (e.g., missing dependencies, OOM, bad model string), 
    it catches the error, prints it, and routes execution to `_deterministic_fallback`. 
    If `out` path is provided, it doesn't only return the embedding but saves it as a `.npy` file.
    
    Args:
        seq (str): The protein sequence string.
        residues (Iterable[int]): 1-based pocket residue indices.
        model_name (str, optional): Target ESM-2 variant. Defaults to "esm2_t33_650M_UR50D".
        out (str, optional): Target filepath to save the resulting numpy array. Defaults to None.
        
    Returns:
        np.ndarray: The resulting 1D embedding vector.
        
    Example:
        >>> emb = encode("MKWVTFGG", [1, 2, 3])
        >>> type(emb)
        <class 'numpy.ndarray'>
    """
    # -------------------------------------------------------------------------------------
    # Encoding Router: Try the real network path, fallback to deterministic math if it 
    # crashes, then optionally save to disk as a .npy.
    # -------------------------------------------------------------------------------------
    try:                                                                                      # Initiate a try block to safely attempt heavy neural network execution
        emb = esm_pocket_embedding(seq, residues, model_name)                                 # Call the primary ESM-2 embedding function utilizing the specified parameters
        print(f"[encode_pocket] ESM-2 ({model_name}) embedding dim={emb.shape[0]}")           # Print a success confirmation identifying the loaded model and output vector dimension
    except Exception as e:                                                                    # Catch any exception thrown during import, tokenization, or model execution
        print(f"[encode_pocket] ESM path unavailable ({e}).")                                 # Print an error message detailing exactly why the primary ESM-2 path failed
        emb = _deterministic_fallback(seq, residues)                                          # Route execution to the fallback generator to ensure the pipeline doesn't crash completely
    if out:                                                                                   # Check if an explicit output destination path was provided as an argument
        np.save(out, emb); print(f"[encode_pocket] wrote {out}")                              # Serialize the numpy array directly to disk in binary format and confirm via print
    return emb                                                                                # Return the computed or fallback embedding vector back to the caller


def main():
    """
    Command Line Interface entry point.
    
    How it works:
    Initializes an `argparse` parser to accept CLI arguments. It natively supports loading 
    parameters directly from a YAML configuration file or via manual explicit flags. 
    It parses the FASTA target, parses the comma-separated residues, and triggers the 
    `encode` function, saving the output.
    
    Args:
        None (reads from `sys.argv`).
        
    Returns:
        None
        
    Example (Command Line):
        $ python encode_pocket.py --seq "MKWVTFGG" --residues "1,2,3" --out "test.npy"
    """
    # --------------------------------------------------------------------------------------------------------
    # CLI Argument Parsing
    # Bind flags for configuration file, sequence inputs, residue indices, model type, and output destination.
    # --------------------------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser()                                                            # Instantiate a standard python argument parser object for CLI interaction
    ap.add_argument("--config", default=None)                                                 # Add an optional argument to point to a centralized YAML configuration file
    ap.add_argument("--fasta", default=None)                                                  # Add an optional argument to specify a direct path to a FASTA sequence file
    ap.add_argument("--seq", default=None)                                                    # Add an optional argument to allow passing a raw sequence string directly via CLI
    ap.add_argument("--residues", default=None)                                               # Add an optional argument to accept a comma-separated string of target indices
    ap.add_argument("--model", default="esm2_t33_650M_UR50D")                                 # Add an optional argument to override the default ESM-2 model architecture variant
    # Add an optional argument to save the embedding vector to the 'out' path as a `.npy` vector
    ap.add_argument("--out", default=None)                                                    
    args = ap.parse_args()                                                                    # Parse the arguments provided by the user in the command line invocation

    # -------------------------------------------------------------------------------------
    # Data Resolution
    # Extract parameters from either the YAML config (preferred) or the direct CLI flags.
    # -------------------------------------------------------------------------------------
    if args.config:                                                                           # Check if the user opted to use a master YAML configuration file
        # Load the YAML configuration file, read the sequence, parse the residue indices, 
        # load the specific ESM-2 model, and determine the output path
        import yaml                                                                           # Dynamically import the YAML parsing library only if a config file is requested
        cfg = yaml.safe_load(open(args.config))                                               # Open the specified config file and safely parse it into a python dictionary
        seq = _read_fasta(cfg["sequence_fasta"])                                              # Extract the sequence by reading the FASTA path specified inside the YAML config
        residues = cfg["pocket_residues"]                                                     # Extract the pre-formatted list of integer residue indices from the YAML config
        model = cfg.get("esm_model", args.model)                                              # Fetch the model string from the YAML, falling back to the CLI argument if missing
        out = args.out or cfg["esm_pocket_embedding"]                                         # Determine the output path, preferring the CLI override, otherwise using the YAML
    else:     
        # If no YAML configuration file was provided via the CLI, resolve the sequence from 
        # either a direct string or a FASTA file, parse the residue indices, load the ESM-2 model,
        # and determine the output path
        seq = args.seq or _read_fasta(args.fasta)                                             # Resolve the sequence either from the direct string argument or by reading the FASTA file
        residues = [int(r) for r in args.residues.split(",") if r.strip()]                    # Parse the comma-separated string of indices into a strict python list of integers
        model = args.model                                                                    # Bind the targeted ESM-2 architecture string extracted directly from the CLI
        out = args.out                                                                        # Bind the explicit output file path string extracted directly from the CLI
    
    # -------------------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------------------
    encode(seq, residues, model_name=model, out=out)                                          # Execute the main high-level encoding pipeline with the fully resolved parameters


if __name__ == "__main__":                                                                    # Standard python guard to prevent execution if the file is imported as a module
    main()                                                                                    # Trigger the CLI parser and execution logic