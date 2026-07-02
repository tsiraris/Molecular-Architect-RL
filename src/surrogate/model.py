"""
==================================================
AfinityGNN: Fast, Uncertainty-Aware Affinity Proxy
==================================================

This script defines the Stage-2 surrogate model (`AffinityGNN`) and its ensemble 
wrapper (`DeepEnsemble`). It serves as a rapid, uncertainty-aware proxy for 
predicting protein-ligand binding affinity (pChEMBL).

How it works:
The `AffinityGNN` is a graph neural network built with GINEConv message-passing layers 
over the shared 11-dimensional node and 4-dimensional edge graph representation. It uses 
a mean-pool readout and a scalar head to predict normalized pChEMBL values.

The `DeepEnsemble` encapsulates K independently initialized AffinityGNNs. The ensemble 
mean acts as the robust prediction, while the standard deviation provides a cheap 
estimate of epistemic uncertainty. In Stage 2, this uncertainty serves two critical roles:
1. It acts as an RL reward penalty, discouraging the agent from exploiting the proxy 
   in off-distribution chemical spaces.
2. It is designed to rank newly generated candidates for the active-learning oracle
   (Stage-3; not wired in this build), flagging high-uncertainty points that would be
   maximally informative for retraining.
   
Note: All predictions are in NORMALIZED label space (z-scored pChEMBL). They must be 
converted back to raw values using saved normalization statistics.
"""
import glob
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool

from .featurize import NODE_DIM, EDGE_DIM


class AffinityGNN(nn.Module):
    """
    A Graph Neural Network regressor for predicting normalized binding affinity.
    
    Projects initial input node features into a hidden representational space, passes them 
    through a configurable number of GINEConv message-passing layers (with residual connections 
    and layer normalization), aggregates the graph globally via mean pooling, and projects 
    the result to a single scalar prediction representing the normalized pChEMBL score.
    
    Args:
        in_dim (int, optional): Dimensionality of input node features. Defaults to NODE_DIM.
        edge_dim (int, optional): Dimensionality of input edge features. Defaults to EDGE_DIM.
        hid (int, optional): Hidden dimension size for embeddings. Defaults to 128.
        layers (int, optional): Number of message-passing layers. Defaults to 4.
        
    Example:
        >>> model = AffinityGNN(in_dim=11, edge_dim=4, hid=128, layers=4)
        >>> out = model(x, edge_index, edge_attr, batch)
    """
    def __init__(self, in_dim: int = NODE_DIM, edge_dim: int = EDGE_DIM, hid: int = 128, layers: int = 4):
        # -----------------------------------------------------------------------------------------
        # Module Initialization & Encoding
        # Bind the parent class and define the structural GNN backbone.
        # -----------------------------------------------------------------------------------------
        super().__init__()                                                                                  # Initialize the base nn.Module to properly register network parameters
        self.lin = nn.Linear(in_dim, hid)                                                                   # Create an initial linear projection layer mapping raw node features to the hidden dimension
        self.convs = nn.ModuleList([                                                                        # Initialize a container list to hold multiple GINE convolutional message-passing layers
            GINEConv(nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid)), edge_dim=edge_dim) # Instantiate GINEConv with a 2-layer MLP to integrate node and bond (edge) features
            for _ in range(layers)                                                                          # Duplicate the convolutional layer architecture for the specified network depth
        ])                                                                                                  # Close the convolutional module list
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])                              # Create corresponding LayerNorm modules for each convolutional layer to stabilize training
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hid, 1))       # Define the final prediction MLP head, utilizing dropout for regularization and outputting a scalar

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Executes a forward pass through the affinity graph neural network.
        
        Projects the nodes, iterates through the spatial GINE convolutions applying ReLU, 
        layer normalization, and residual additions. Pools the nodes to a graph vector, 
        and extracts the scalar prediction.
        
        Args:
            x (torch.Tensor): The input node feature matrix.
            edge_index (torch.Tensor): The graph connectivity matrix.
            edge_attr (torch.Tensor): The edge feature matrix.
            batch (torch.Tensor): The batch assignment vector mapping nodes to graphs.
            
        Returns:
            torch.Tensor: A 1D tensor of predicted scalar affinity values for the batch.
        """
        # -----------------------------------------------------------------------------------------
        # Forward Inference Pass
        # Route graph tensors through the message passing architecture and readout head.
        # -----------------------------------------------------------------------------------------
        h = self.lin(x)                                                                             # Apply the initial linear projection to the raw input node features
        # Sequentially iterate over pairs of convolutional and normalization layers
        for conv, norm in zip(self.convs, self.norms):                                              
            # Apply convolution, ReLU, and layernorm, then add to the input as a residual connection
            h = h + norm(torch.relu(conv(h, edge_index, edge_attr)))                                
        # Pool node embeddings into graph embeddings, pass through the head, and flatten the output dimension
        return self.head(global_mean_pool(h, batch)).squeeze(-1)                                    

    def predict_data(self, data, device="cpu"):
        """
        Convenience wrapper to forward a single PyG Data or Batch object, performing inference.
        
        Sets the model to evaluation mode, moves data to the target device, handles missing 
        batch assignments (treating the input as a single graph), and computes predictions 
        without tracking gradients.
        
        Args:
            data (torch_geometric.data.Data): A single molecular graph or batch object.
            device (str, optional): The compute device to run inference on. Defaults to "cpu".
            
        Returns:
            torch.Tensor: The predicted scalar affinity value(s).
        """
        # -----------------------------------------------------------------------------------------
        # Single Data Inference
        # Prepare execution context and safely handle batch indexing for standalone molecules.
        # -----------------------------------------------------------------------------------------
        self.eval()                                                                                 # Explicitly lock the network into evaluation mode, disabling dropout and batch norm tracking
        data = data.to(device)                                                                      # Transfer the PyG data structure to the requested computational hardware device
        # If the batch attribute is missing, treat the input as a single graph and construct a zeroed batch vector
        b = data.batch if hasattr(data, "batch") and data.batch is not None \
            else torch.zeros(data.x.size(0), dtype=torch.long, device=device)                       # Extract existing batch indices or construct a zeroed vector assuming a single graph topology
        with torch.no_grad():                                                                       # Suspend PyTorch's automatic differentiation engine to save memory and accelerate inference
            return self.forward(data.x, data.edge_index, data.edge_attr, b)                         # Execute the standard forward pass using the unpacked graph tensors


class DeepEnsemble:
    """
    A unified wrapper managing a list of trained AffinityGNNs queried together for epistemic uncertainty.
    
    Loads multiple independently trained AffinityGNN models. During inference, it routes the 
    batch through every model and calculates the mean (prediction) and standard deviation 
    (uncertainty) across the ensemble's outputs.
    
    Args:
        models (List[AffinityGNN]): A pre-instantiated list of AffinityGNN models.
        device (str, optional): The target compute device. Defaults to "cpu".
        
    Example:
        >>> ensemble = DeepEnsemble([model1, model2, model3], device="cuda")
        >>> mean, std = ensemble.predict_batch(batch)
    """

    def __init__(self, models: List[AffinityGNN], device: str = "cpu"):
        # -----------------------------------------------------------------------------------------
        # Ensemble Initialization
        # Prepare the collection of independent models for joint inference.
        # -----------------------------------------------------------------------------------------
        self.models = [m.to(device).eval() for m in models]                                         # Move each individual model to the target device and permanently lock into evaluation mode
        self.device = device                                                                        # Store the string representing the active computational hardware device

    @torch.no_grad()                                                                                # Suspend PyTorch's automatic differentiation engine to save memory and accelerate inference
    def predict_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates a batch of molecules across the entire ensemble.
        
        Passes the batch through every model sequentially, stacks their predictions, and 
        computes the statistical mean and standard deviation along the model dimension.
        
        Args:
            batch (torch_geometric.data.Batch): A PyG batch of molecular graphs.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean predictions 
            and the standard deviation (uncertainty) estimates in normalized label space.
        """
        # -----------------------------------------------------------------------------------------
        # Ensemble Inference & Uncertainty: Sequentially route the batch through each model, stack 
        # the predictions, and compute the ensemble mean and epistemic standard deviation
        # -----------------------------------------------------------------------------------------
        batch = batch.to(self.device)                                                               # Relocate the entire composite batch payload to the correct processing device
        # If the batch attribute is missing, treat the input as a single graph and construct a zeroed batch vector
        b = batch.batch if getattr(batch, "batch", None) is not None \
            else torch.zeros(batch.x.size(0), dtype=torch.long, device=self.device)                 # Safely resolve the node-to-graph mapping vector, creating one if entirely absent
        # Execute inference on every model in the ensemble
        preds = torch.stack([m(batch.x, batch.edge_index, batch.edge_attr, b) for m in self.models])# Execute inference on every model in the ensemble and stack the resulting 1D tensors
        if preds.dim() == 1:                                                                        # Check if the output collapsed due to processing a solitary single-graph input
            preds = preds.unsqueeze(-1)                                                             # Manually expand the tensor dimension to consistently retain a shape of [K, 1]
        return preds.mean(0), preds.std(0)                                                          # Collapse the model dimension to compute the prediction mean and epistemic standard deviation

    def save(self, out_dir: str):
        """
        Serializes the entire ensemble of models to disk.
        
        Iterates over the member models and saves their state dictionaries as individual 
        files in the specified directory.
        
        Args:
            out_dir (str): The target directory to write the state dictionaries into.
        """
        # -----------------------------------------------------------------------------------------
        # Ensemble Serialization
        # Write each member model's state dictionary to a localized .pt file
        # -----------------------------------------------------------------------------------------
        os.makedirs(out_dir, exist_ok=True)                                                         # Ensure the output directory structure exists on the filesystem, bypassing errors if it does
        for i, m in enumerate(self.models):                                                         # Iterate over the active model list alongside an enumerated integer index
            torch.save(m.state_dict(), os.path.join(out_dir, f"member_{i}.pt"))                     # Extract the raw parameter dictionary and physically write it to a localized .pt file

    # As a static method to be able to load a DeepEnsemble from disk, without needing a pre-existing object
    @staticmethod                                                                                   # Define a static method, i.e., a method that is not bound to a specific instance, allowing the creation and configuration of a brand-new DeepEnsemble object directly from files on the hard drive, without needing a pre-existing object first.
    def load(out_dir: str, in_dim: int = NODE_DIM, edge_dim: int = EDGE_DIM,
             hid: int = 128, layers: int = 4, device: str = "cpu") -> "DeepEnsemble":
        """
        Instantiates a DeepEnsemble by loading multiple state dictionaries from disk.
        
        Scans the directory for member files, instantiates a new AffinityGNN for each, 
        loads the corresponding weights, and wraps them in a DeepEnsemble.
        
        Args:
            out_dir (str): The directory containing the saved member files.
            in_dim (int, optional): Node feature dimension. Defaults to NODE_DIM.
            edge_dim (int, optional): Edge feature dimension. Defaults to EDGE_DIM.
            hid (int, optional): Hidden dimension. Defaults to 128.
            layers (int, optional): GNN layer count. Defaults to 4.
            device (str, optional): The target compute device. Defaults to "cpu".
            
        Returns:
            DeepEnsemble: A fully initialized and loaded ensemble ready for inference.
        """
        # -----------------------------------------------------------------------------------------
        # Ensemble Deserialization: Detect member model's state checkpoint files, spawn network 
        # architectures, load the serialized weights, and assemble the ensemble.
        # -----------------------------------------------------------------------------------------
        paths = sorted(glob.glob(os.path.join(out_dir, "member_*.pt")))                             # Scan the target directory for all files matching the member naming convention and sort them
        if not paths:                                                                               # Check if the glob operation failed to locate any valid checkpoint files
            raise FileNotFoundError(f"no surrogate members found in {out_dir}")                     # Immediately crash execution, explicitly notifying the user of the missing assets
        models = []                                                                                 # Initialize an empty array to collect the dynamically reconstructed neural networks
        for p in paths:                                                                             # Sequentially iterate over each discovered physical file path
            m = AffinityGNN(in_dim, edge_dim, hid, layers)                                          # Instantiate a blank AffinityGNN architecture conforming to the provided shape arguments
            # Read the serialized parameters from disk and strictly map them into the active architecture
            m.load_state_dict(torch.load(p, map_location=device))                                   
            models.append(m)                                                                        # Append the fully populated network into the running model collector array
        return DeepEnsemble(models, device=device)                                                  # Construct and return the finalized DeepEnsemble wrapper class encapsulating the loaded models