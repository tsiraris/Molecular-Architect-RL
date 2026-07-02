"""
============================
Surrogate Featurizer Module
============================

This script serves as the ONE featurizer shared by both surrogate training pipelines 
(e.g., dataset generation) and RL inference loops (terminal active learning). 
The affinity surrogate must see the *exact same* graph representation whether it is being 
trained on ChEMBL SMILES in `surrogate/dataset.py` or being queried at RL terminal time 
on a freshly built molecule in `reward/affinity_reward.py`. This module acts as that 
single source of truth.

Representation (deliberately focus-free, unlike chem_env's 12-dim sequential observation):
    node features  = 11 dims  : 6 element one-hot (C,N,O,F,S,Cl) + 3 hybridisation (SP,SP2,SP3)
                                + 1 aromatic flag + 1 ring-membership flag
    edge features  =  4 dims  : one-hot bond order (single, double, triple, aromatic)

The element set strictly matches `chem_env.ActionSpec.atom_types`, so molecules the agent 
can build are always natively representable. Atoms outside the set still featurize 
(generating an all-zero element block) rather than crash, ensuring that ChEMBL molecules 
with exotic atoms degrade gracefully instead of being abruptly dropped from training batches.
"""

from typing import List, Optional, Tuple
from rdkit import Chem

# -----------------------------------------------------------------------------------------
# Global Feature Definitions
# Define the vocabulary and dimensions for the graph's node and edge representations.
# -----------------------------------------------------------------------------------------
ATOM_TYPES = ["C", "N", "O", "F", "S", "Cl"]                                                # Define the allowed atomic symbols, which strictly must match chem_env.ActionSpec.atom_types to maintain consistency.
NODE_DIM = len(ATOM_TYPES) + 3 + 1 + 1                                                      # Calculate total node feature dimensionality: 6 atoms + 3 hybridizations + 1 aromatic + 1 ring (total 11).
EDGE_DIM = 4                                                                                # Hardcode the edge feature dimensionality to 4, representing the four primary RDKit bond types.

_HYB = None                                                                                 # Initialize the global hybridization cache to None, to be filled lazily avoiding rdchem imports at module load time in odd environments.


def _hyb_types():
    """
    Lazily loads and caches RDKit's hybridization type enums.
    
    Checks if the global `_HYB` variable is None. If it is, it imports the specific 
    RDKit enum inside the function and assigns a tuple of SP, SP2, and SP3 types. 
    This prevents module import errors in environments where RDKit might not be 
    fully initialized or is heavily delayed.
    
    Args:
        None.
        
    Returns:
        tuple: A tuple containing RDKit HybridizationType enums for SP, SP2, and SP3.
        
    Example:
        >>> hybs = _hyb_types()
        >>> len(hybs)
        3
    """
    # -------------------------------------------------------------------------------------
    # Lazy Hybridization Loading
    # Ensure RDKit dependencies are resolved strictly at runtime.
    # -------------------------------------------------------------------------------------
    global _HYB                                                                             # Declare the intent to modify the module-level global _HYB variable
    if _HYB is None:                                                                        # Check if the hybridization cache is currently uninitialized
        from rdkit.Chem.rdchem import HybridizationType as H                                # Import the HybridizationType enum locally to avoid circular or early import issues
        _HYB = (H.SP, H.SP2, H.SP3)                                                         # Populate the global cache with a tuple of SP, SP2, and SP3 hybridization states
    return _HYB                                                                             # Return the now-guaranteed initialized tuple of RDKit hybridization enums


def atom_features(atom) -> List[float]:
    """
    Generates an 11-dimensional feature vector for a single RDKit atom (focus-free).
    
    Extracts the atom's chemical symbol and maps it to a 6-dim one-hot vector based on 
    ATOM_TYPES. Appends a 3-dim one-hot vector for hybridization, followed by two 
    binary flags representing aromaticity and ring membership.
    
    Args:
        atom (Chem.Atom): An active RDKit atom instance.
        
    Returns:
        List[float]: A complete 11-element floating point list representing the node.
        
    Example:
        >>> m = Chem.MolFromSmiles("c1ccccc1")
        >>> a = m.GetAtomWithIdx(0)
        >>> len(atom_features(a))
        11
    """
    # -------------------------------------------------------------------------------------
    # Node Feature Extraction
    # Map physical atomic properties to fixed-length numeric float representations.
    # -------------------------------------------------------------------------------------
    sym = atom.GetSymbol()                                                                  # Extract the string elemental symbol from the provided RDKit atom object
    feat = [1.0 if sym == k else 0.0 for k in ATOM_TYPES]                                   # Generate a 6-dimensional one-hot float list matching the atom to the predefined ATOM_TYPES list (fallback is all zeros)
    hyb = atom.GetHybridization()                                                           # Query RDKit for the atom's current topological hybridization state
    feat += [1.0 if hyb == t else 0.0 for t in _hyb_types()]                                # Append a 3-dimensional one-hot float list encoding the exact hybridization (SP, SP2, SP3)
    feat += [1.0 if atom.GetIsAromatic() else 0.0,                                          # Evaluate if the atom is part of an aromatic system and append as a binary float (1.0 or 0.0)
             1.0 if atom.IsInRing() else 0.0]                                               # Evaluate if the atom is contained within any topological ring structure and append as a binary float
    return feat                                                                             # Return the fully constructed 11-dimensional numerical feature list for the atom


def bond_features(bond) -> List[float]:
    """
    Generates a 4-dimensional one-hot bond-order vector.
    
    Queries the RDKit bond type and maps it directly against an expected tuple of 
    (SINGLE, DOUBLE, TRIPLE, AROMATIC), generating a mutually exclusive one-hot list.
    
    Args:
        bond (Chem.Bond): An active RDKit bond instance.
        
    Returns:
        List[float]: A 4-element floating point list describing the bond type.
        
    Example:
        >>> m = Chem.MolFromSmiles("C=C")
        >>> b = m.GetBondWithIdx(0)
        >>> bond_features(b)
        [0.0, 1.0, 0.0, 0.0]
    """
    # -------------------------------------------------------------------------------------
    # Edge Feature Extraction
    # Map topological connections to normalized numerical categories.
    # -------------------------------------------------------------------------------------
    bt = bond.GetBondType()                                                                 # Extract the specific RDKit bond type object from the provided graph connection
    from rdkit.Chem import BondType as B                                                    # Import the BondType enum locally to map the bond type into discrete comparative categories
    return [1.0 if bt == t else 0.0 for t in (B.SINGLE, B.DOUBLE, B.TRIPLE, B.AROMATIC)]    # Map the bond type against a tuple of SINGLE, DOUBLE, TRIPLE, and AROMATIC, returning a 4-dim one-hot list


def mol_to_graph(mol) -> Optional[Tuple[list, list, list]]:
    """
    Converts an RDKit Mol object into raw python lists of features and connectivity.
    
    Checks for molecule validity. Iterates over all atoms to compute the node feature 
    matrix `x`. Then iterates over all bonds to compute edge coordinates and edge 
    features. Since the target graph is undirected, every chemical bond yields two 
    bidirectional edge indices in the output.
    
    Args:
        mol (Chem.Mol): The RDKit molecule to be decomposed.
        
    Returns:
        Optional[Tuple[list, list, list]]: Returns None for an empty/None molecule. 
        Otherwise, returns a tuple of (node_features `x`, edge_index_pairs `rows`, 
        and edge_features `attr`).
        
    Example:
        >>> m = Chem.MolFromSmiles("CC")
        >>> x, rows, attr = mol_to_graph(m)
        >>> len(rows) # bidirectional connection for 1 bond
        2
    """
    # -------------------------------------------------------------------------------------
    # Pure Python Graph Construction
    # Deconstruct the RDKit molecule into primitive topological arrays.
    # -------------------------------------------------------------------------------------
    if mol is None or mol.GetNumAtoms() == 0:                                               # Check if the parsed molecule is chemically invalid or strictly empty (0 atoms)
        return None                                                                         # Return None to gracefully signal an empty or broken molecular graph preventing feature crashes
    x = [atom_features(a) for a in mol.GetAtoms()]                                          # Iterate sequentially over all atoms, generating the 11-dim feature list for each to form the node matrix 'x'
    rows, attr = [], []                                                                     # Initialize empty lists to dynamically store the raw directional edge indices and edge feature attributes
    for b in mol.GetBonds():                                                                # Iterate sequentially through all defined RDKit bonds existing in the global molecule
        s, e = b.GetBeginAtomIdx(), b.GetEndAtomIdx()                                       # Unpack the integer indices representing the abstract source and destination atoms of the current bond
        f = bond_features(b)                                                                # Generate the 4-dimensional one-hot edge feature list evaluating this specific connection
        rows.append([s, e]); attr.append(f)                                                 # Append the forward directional index pair [source, dest] and its corresponding feature list
        rows.append([e, s]); attr.append(f)                                                 # Append the reverse directional index pair [dest, source] and identical features forcing an undirected graph representation
    return x, rows, attr                                                                    # Return the finalized primitive tuple containing node features, edge coordinate pairs, and edge attributes


def smiles_to_graph(smi: str):
    """
    Parses a SMILES sequence and directly featurizes it into a raw python graph tuple.
    
    Delegates to `Chem.MolFromSmiles` and sequentially feeds the result into 
    `mol_to_graph`. Returns None if the string constitutes an invalid topology.
    
    Args:
        smi (str): The structural SMILES sequence.
        
    Returns:
        Optional[Tuple[list, list, list]]: The featurized tuple or None on parse failure.
        
    Example:
        >>> out = smiles_to_graph("INVALID_STRING")
        >>> type(out)
        <class 'NoneType'>
    """
    # -------------------------------------------------------------------------------------
    # String Integration
    # -------------------------------------------------------------------------------------
    return mol_to_graph(Chem.MolFromSmiles(smi))                                            # Parse the raw SMILES string into an RDKit Mol object and pass it directly to the graph featurizer, returning its output or None on parse failure


# -----------------------------------------------------------------------------------------
# Torch / PyG Helpers (Imported lazily so the pure-featurization above works without torch)
# Converts standard python lists into deep learning tensors dynamically.
# -----------------------------------------------------------------------------------------
def to_data(mol, y: Optional[float] = None):
    """
    Converts an RDKit Mol into a fully initialized `torch_geometric.data.Data` object.
    
    Leverages `mol_to_graph` to get pure python topologies. Dynamically imports PyTorch 
    and casts those lists into strictly formatted `torch.float32` and `torch.long` tensors. 
    Handles the extreme edge case of single-atom graphs (0 bonds) elegantly. Optionally 
    attaches a scalar target label `y`.
    
    Args:
        mol (Chem.Mol): The input RDKit molecule.
        y (Optional[float], optional): A scalar target variable (e.g., predicted affinity). Defaults to None.
        
    Returns:
        Optional[torch_geometric.data.Data]: The initialized PyG Data object or None.
        
    Example:
        >>> m = Chem.MolFromSmiles("C")
        >>> data = to_data(m)
        >>> data.edge_index.shape
        torch.Size([2, 0])
    """
    # -------------------------------------------------------------------------------------
    # Tensor Casting and Formulation
    # Bridge the gap between native Python topology and GPU-ready tensors.
    # -------------------------------------------------------------------------------------
    import torch                                                                            # Import PyTorch locally to avoid requiring it merely to load the surrounding script module in pure data pipelines
    from torch_geometric.data import Data                                                   # Import the PyG Data structure locally for standardized graph packaging
    g = mol_to_graph(mol)                                                                   # Convert the RDKit Mol object into raw python feature lists using the internal primary parser
    if g is None:                                                                           # Check if the primary featurization process failed (e.g., due to an empty molecule)
        return None                                                                         # Propagate the failure upward by returning None, bypassing all tensor allocations
    x, rows, attr = g                                                                       # Unpack the successfully computed sequence of node features, edge index pairs, and edge attributes
    # If the molecule is a single-atom graph, containing no valid bonds, the edge index tensor will be empty
    if len(rows) == 0:                                                                      # Check for the edge case where the molecule consists of a single atom without any bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)                                  # Instantiate an explicitly empty 2D coordinate tensor for the nonexistent edges mapped in long integers
        edge_attr = torch.empty((0, EDGE_DIM), dtype=torch.float32)                         # Instantiate an explicitly empty feature matrix representing 0 edges spanning the exact EDGE_DIM features
    else:                                                                                   # Handle standard multi-atomic molecules that contain at least one valid bond
        edge_index = torch.tensor(rows, dtype=torch.long).t().contiguous()                  # Convert the list of [src, dst] pairs to a transposed, contiguous [2, num_edges] PyTorch long integer tensor
        edge_attr = torch.tensor(attr, dtype=torch.float32)                                 # Convert the raw list of edge features into a dense, properly shaped PyTorch float32 attribute tensor
    data = Data(x=torch.tensor(x, dtype=torch.float32),                                     # Wrap the compiled node feature arrays into a dense float32 tensor mapped to 'x' inside the PyG Data container
                edge_index=edge_index, edge_attr=edge_attr)                                 # Attach the correctly formatted continuous edge index coordinates and edge attributes to complete the base Data object
    # If a target label (e.g., affinity) was provided by the user, attach it to the Data object
    if y is not None:                                                                       # Check if an external scalar target value (e.g., affinity regression label) was provided by the user
        data.y = torch.tensor([float(y)], dtype=torch.float32)                              # Wrap the scalar label into a strictly sized 1D float32 tensor and assign it securely to the Data object's 'y' attribute
    return data                                                                             # Return the fully constructed PyTorch Geometric Data instance ready for batching and neural inference


def batch_from_mols(mols):
    """
    Transforms a list of RDKit Mols into a unified PyTorch Geometric `Batch` graph.
    
    Iterates over the input list, casting each valid molecule into a `Data` object via 
    `to_data`. Records a boolean mask corresponding to which initial inputs survived the 
    featurization process without crashing. Combines surviving `Data` objects into a 
    single disconnected PyG `Batch`.
    
    Args:
        mols (List[Chem.Mol]): A list of active RDKit molecule instances.
        
    Returns:
        Tuple[Optional[torch_geometric.data.Batch], List[bool]]: A tuple containing 
        the compiled PyG Batch (or None) and a `keep_mask` array denoting valid parses.
        
    Example:
        >>> mols = [Chem.MolFromSmiles("C"), None, Chem.MolFromSmiles("CC")]
        >>> batch, keep = batch_from_mols(mols)
        >>> keep
        [True, False, True]
    """
    # -------------------------------------------------------------------------------------
    # Batch Accumulation
    # Collate multiple individual graph fragments into a large disconnected training batch.
    # -------------------------------------------------------------------------------------
    import torch                                                                            # Import PyTorch locally for executing deep batch tensor operations
    from torch_geometric.data import Batch                                                  # Import the PyG Batch object locally to consolidate multiple graphs into a single disconnected topology array
    datas, keep = [], []                                                                    # Initialize an empty list for valid Data objects alongside a boolean mask tracking which specific molecules survived
    for m in mols:                                                                          # Iterate sequentially over every RDKit molecule passed in the aggregated input list
        d = to_data(m)                                                                      # Attempt to reliably convert the individual RDKit molecule into a PyTorch Geometric Data object
        # keep is a boolean mask tracking (with True/False indicates) which molecules survived the featurization process
        keep.append(d is not None)                                                          # Append True to the ongoing keep mask if featurization succeeded without error, otherwise append False
        if d is not None:                                                                   # Check if the generated PyG Data object is chemically sound and structurally valid
            datas.append(d)                                                                 # Add the correctly formatted PyG Data object to the master batch accumulation list
    if not datas:                                                                           # Check if the ultimate accumulation list is completely empty (i.e., every single molecule failed featurization)
        return None, keep                                                                   # Return None in place of a fully structured batch, alongside the fully populated boolean failure mask
    return Batch.from_data_list(datas), keep                                                # Merge the individual valid graphs into one monolithic memory-efficient PyG Batch object and return alongside the tracking mask