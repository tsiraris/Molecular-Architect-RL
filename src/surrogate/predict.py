"""
=====================================
Target-Aware Surrogate Inference API
=====================================

This script defines the `AffinityScorer`, which serves as the crucial inference bridge 
between the reinforcement learning environment and the pre-trained target-affinity surrogate model.
It acts as the rapid inference API the RL reward function queries at terminal time to evaluate 
how strongly a generated molecule is expected to bind to the protein target.

The `AffinityScorer` loads a pre-trained `DeepEnsemble` model along with its corresponding 
normalization statistics exactly once upon initialization. It then exposes a robust API 
to score molecules individually or in efficient vectorized batches. 
The predicted binding affinity (`aff_hat_z`) is returned as a z-scored pChEMBL value: 
0 represents the dataset mean, and +1 represents a one-sigma stronger binder (higher is better). 
Furthermore, it calculates `aff_unc_z`, the ensemble's standard deviation across its independent 
neural network heads, representing the epistemic uncertainty of the prediction in the same units.

Exposed API:
    .score(mol)        -> (aff_hat_z, aff_unc_z)             # Evaluates a single RDKit Mol in normalized space
    .score_smiles(smi) -> (aff_hat_z, aff_unc_z)             # Parses and evaluates a SMILES string
    .score_mols(mols)  -> (np.ndarray[mu], np.ndarray[sd])   # Evaluates a batch of molecules (e.g., ranking top-k)
    .to_pchembl(z)     -> float                              # Inverts the normalization back to absolute pChEMBL units
"""
from typing import List, Tuple

import numpy as np

from .featurize import to_data, batch_from_mols
from .model import DeepEnsemble
from .dataset import load_norm


class AffinityScorer:
    """
    A high-level inference wrapper for a trained deep ensemble affinity predictor.
    
    Encapsulates the PyTorch model (`DeepEnsemble`), its structural featurization routines, 
    and the statistical normalization dictionary. It handles raw RDKit molecules, translates 
    them into graph data objects, passes them through the ensemble, and extracts both the 
    mean predicted affinity and the predictive uncertainty. Crucially, it manages edge cases 
    (like chemically invalid graphs that crash the featurizer) by assigning them safe, 
    highly penalized default values so the RL loop does not crash.
    """
    def __init__(self, surrogate_dir: str, device: str = "cpu", hid: int = 128, layers: int = 4):
        """
        Initializes the AffinityScorer by loading network weights and normalizations from disk.
        
        Args:
            surrogate_dir (str): Path to the directory containing model checkpoints and 'norm.json'.
            device (str, optional): Target compute hardware ("cpu" or "cuda"). Defaults to "cpu".
            hid (int, optional): The hidden dimensionality used during model training. Defaults to 128.
            layers (int, optional): The number of message-passing layers in the model. Defaults to 4.
            
        Example:
            >>> scorer = AffinityScorer(surrogate_dir="models/kras_ensemble_v1", device="cuda")
            >>> print(scorer.device)
            cuda
        """
        # -----------------------------------------------------------------------------------------
        # Ensemble Model Initialization
        # Load the pre-trained model checkpoint and the target normalization constants.
        # -----------------------------------------------------------------------------------------
        self.ens = DeepEnsemble.load(surrogate_dir, hid=hid, layers=layers, device=device)          # Instantiate and load the pre-trained DeepEnsemble PyTorch model to the target device.
        self.norm = load_norm(f"{surrogate_dir}/norm.json")                                         # Load the dataset normalization statistics (mean and std) from the JSON configuration.
        self.device = device                                                                        # Store the requested compute device string for internal tensor routing.

    def score(self, mol) -> Tuple[float, float]:
        """
        Calculates the predicted affinity and uncertainty for a single RDKit molecule.
        
        Converts the RDKit `Mol` into a PyTorch Geometric `Data` object. If featurization fails 
        (e.g., empty graph, impossible valence), it catches the null return and provides a safe 
        default fallback: (0.0 mean, 1.0 uncertainty). This essentially tells the RL agent that 
        the molecule is "average but highly uncertain," so the reward's uncertainty penalty 
        (beta * aff_unc_z) heavily down-weights it.
        
        Args:
            mol (rdkit.Chem.rdchem.Mol): The fully constructed RDKit molecule object to score.
            
        Returns:
            Tuple[float, float]: A tuple containing the `mean_z` (z-scored predicted affinity) 
            and `std_z` (epistemic uncertainty scalar).
            
        Example:
            >>> m = Chem.MolFromSmiles("CC1=CC=CC=C1")
            >>> mu, unc = scorer.score(m)
        """
        # ------------------------------------------------------------------------------------------------------
        # Single Molecule Featurization & Inference
        # Convert the chemical graph to tensors, pass through the ensemble models, and extract the mean and std.
        # ------------------------------------------------------------------------------------------------------
        import torch                                                                                # Locally import torch to prevent circular dependencies or global namespace pollution.
        data = to_data(mol)                                                                         # Convert the raw RDKit molecule object into a PyTorch Geometric data graph.
        # If the featurization failed, catch the null return and provide a safe default (0.0 mean, 1.0 uncertainty).
        if data is None:                                                                            # Check if the featurization completely failed due to un-embeddable or invalid geometry.
            return 0.0, 1.0                                                                         # Fallback to returning a zero mean (average) and high uncertainty (1.0) to safely penalize un-embeddable graphs.
        from torch_geometric.data import Batch                                                      # Locally import the PyG Batch utility required for single-item formatting.
        batch = Batch.from_data_list([data])                                                        # Wrap the single isolated PyG data object into a standardized batch structure for the network.
        with torch.no_grad():                                                                       # Temporarily disable the autograd engine to optimize memory and speed during pure inference.
            mu, sd = self.ens.predict_batch(batch)                                                  # Pass the batch through the Deep Ensemble to yield mean binding affinity and epistemic uncertainty tensors.
        return float(mu.reshape(-1)[0]), float(sd.reshape(-1)[0])                                   # Extract the scalar values from the 1D tensors, cast to python floats, and return the final tuple.

    def score_smiles(self, smi: str) -> Tuple[float, float]:
        """
        A convenience wrapper to score a molecule directly from its SMILES string.
        
        Parses the string into an RDKit object and relays it directly to the primary `score` method.
        
        Args:
            smi (str): The SMILES string representation of the target molecule.
            
        Returns:
            Tuple[float, float]: The `mean_z` and `std_z` predictions.
            
        Example:
            >>> mu, unc = scorer.score_smiles("CCO")
        """
        # -----------------------------------------------------------------------------------------
        # SMILES String Parsing
        # Process raw string representations before handing off to the graph evaluator.
        # -----------------------------------------------------------------------------------------
        from rdkit import Chem                                                                      # Locally import the RDKit chemistry module specifically for string parsing operations.
        return self.score(Chem.MolFromSmiles(smi))                                                  # Parse the SMILES string into a Mol object and directly delegate evaluation to the standard graph scoring function.

    def score_mols(self, mols: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes highly efficient batched scoring for a list of RDKit molecules.
        
        Processes the entire list of molecules into a unified PyTorch Geometric block-diagonal 
        batch graph, tracking any invalid/failed structures using a boolean mask (`keep`). 
        It initializes baseline Numpy arrays with pessimistic defaults (0.0 mean, 1.0 std). 
        After evaluating the valid subgraph, it maps the dense network predictions back into 
        the correct sparse indices of the output arrays.
        
        Args:
            mols (List[rdkit.Chem.rdchem.Mol]): A list containing RDKit molecule instances.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Two 1D numpy arrays parallel to the input list, 
            containing the sequence of predicted means and uncertainties respectively.
            
        Example:
            >>> batch = [Chem.MolFromSmiles("C"), Chem.MolFromSmiles("CC")]
            >>> mu_arr, sd_arr = scorer.score_mols(batch)
        """
        # -----------------------------------------------------------------------------------------
        # Batched Initialization & Default Fallbacks
        # Form the batch graph and initialize the baseline arrays for the mean and std predictions.
        # -----------------------------------------------------------------------------------------
        import torch                                                                                # Locally import torch for tensor manipulation within the batched processing loop.
        batch, keep = batch_from_mols(mols)                                                         # Convert the list of RDKit molecules into a unified PyG batch and a boolean masking list indicating successful featurizations.
        mu_out = np.zeros(len(mols), dtype=np.float64)                                              # Allocate a zeroed Numpy array for the mean predictions, inherently defaulting invalid molecules to 0.0 (average).
        sd_out = np.ones(len(mols), dtype=np.float64)                                               # Allocate an ones Numpy array for the uncertainty predictions, defaulting invalid molecules to maximum uncertainty (1.0).
        if batch is None:                                                                           # Safely check if the entire provided batch failed to featurize entirely, resulting in a null batch.
            return mu_out, sd_out                                                                   # Return the default initialized arrays immediately to prevent catastrophic network crashes.
        
        # -----------------------------------------------------------------------------------------
        # Batched Network Inference
        # Propagate the unified graph through the network rapidly.
        # -----------------------------------------------------------------------------------------
        with torch.no_grad():                                                                       # Disable PyTorch gradient tracking context to vastly accelerate pure batch evaluation.
            mu, sd = self.ens.predict_batch(batch)                                                  # Feed the sanitized geometric batch through the ensemble to produce dense tensor predictions strictly for valid graphs.
        mu = mu.cpu().numpy().reshape(-1); sd = sd.cpu().numpy().reshape(-1)                        # Move the resulting tensors to CPU RAM, convert them to Numpy arrays, and flatten them into dense 1D vectors.
        
        # ----------------------------------------------------------------------------------------------
        # Result Mapping & Output Formatting
        # Re-align dense network outputs with the original sparse (due to 'keep' masking) input indices.
        # ----------------------------------------------------------------------------------------------
        j = 0                                                                                       # Initialize an independent counter 'j' to track the dense index within the valid-only network output arrays.
        for i, k in enumerate(keep):                                                                # Loop simultaneously over the absolute original batch indices 'i' and the boolean validity flags 'k'.
            if k:                                                                                   # Check if the specific molecule at the current absolute index was successfully evaluated by the neural network.
                mu_out[i], sd_out[i] = mu[j], sd[j]; j += 1                                         # Map the dense network predictions back to their original sparse positions in the output arrays and increment the dense counter.
        return mu_out, sd_out                                                                       # Return the fully populated Numpy arrays containing both the successful predictions and the fallback defaults.

    def to_pchembl(self, z: float) -> float:
        """
        Inverts the Z-score normalization back into absolute physical units (pChEMBL).
        
        Uses the internally stored standard deviation and mean loaded from the dataset's 
        JSON configuration to mathematically reverse the (x - u) / std scaling. This is 
        used to translate the network's abstract values into chemically intuitive reporting metrics.
        
        Args:
            z (float): The network's normalized Z-score prediction.
            
        Returns:
            float: The estimated pChEMBL value (e.g., -log10 of IC50 or Kd).
            
        Example:
            >>> scorer.to_pchembl(1.0) # Network predicts +1 sigma
            7.52
        """
        # -----------------------------------------------------------------------------------------
        # Normalization Inversion
        # Translate dimensionless network outputs into physical affinity units.
        # -----------------------------------------------------------------------------------------
        return float(z) * self.norm["std"] + self.norm["mean"]                                      # Scale the normalized Z-score by the dataset standard deviation and add the mean to recover absolute pChEMBL units.