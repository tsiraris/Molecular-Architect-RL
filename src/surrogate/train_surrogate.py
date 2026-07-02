"""
=================================
Deep Ensemble Surrogate Training
=================================

Train the deep-ensemble affinity proxy and report a leakage-safe scaffold-split result (Spearman rho + RMSE),
saving the ensemble, the label-normalisation stats, and a calibration scatter for the report.

This script orchestrates the training of multiple independent Graph Neural Networks (AffinityGNN) 
on molecular graph data to predict binding affinity (pChEMBL). It uses a scaffold split to ensure 
robust generalization to novel chemotypes. The trained models form a DeepEnsemble, which yields 
both mean predictions and epistemic uncertainty estimates. These outputs guide the RL generator.

Note: This can be considered as a self-contained QSAR result on its own, independent of the RL generator.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .dataset import build_datasets, save_norm
from .model import AffinityGNN, DeepEnsemble


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the Spearman rank correlation coefficient between two arrays without relying on SciPy.
    
    Spearman's rho is mathematically equivalent to the Pearson correlation coefficient 
    applied to the rank variables. This function uses double `np.argsort` to convert raw 
    values into ranks, mean-centers them, and computes the normalized covariance.
    
    Args:
        a (np.ndarray): First array of numerical values (e.g., ground truth labels).
        b (np.ndarray): Second array of numerical values (e.g., model predictions).
        
    Returns:
        float: The Spearman correlation coefficient, bounded between -1.0 and 1.0.
        
    Example:
        >>> y_true = np.array([1.2, 2.5, 3.1])
        >>> y_pred = np.array([1.0, 2.8, 2.9])
        >>> rho = _spearman(y_true, y_pred)
        >>> round(rho, 2)
        1.0
    """
    # -----------------------------------------------------------------------------------------
    # Ranking and Centering
    # Convert raw continuous values to ranks and mean-center them for Pearson correlation logic.
    # -----------------------------------------------------------------------------------------
    ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))                              # Double argsort yields the exact integer rank of each element within the respective arrays
    ar = ar - ar.mean(); br = br - br.mean()                                                    # Mean-center both rank arrays to prepare for variance and covariance calculations
    
    # -----------------------------------------------------------------------------------------
    # Correlation Calculation
    # Compute the Pearson correlation coefficient on the centered ranks.
    # -----------------------------------------------------------------------------------------
    denom = (np.sqrt((ar ** 2).sum()) * np.sqrt((br ** 2).sum())) + 1e-12                       # Calculate the denominator (product of standard deviations) adding epsilon to prevent division by zero
    return float((ar * br).sum() / denom)                                                       # Divide the covariance by the denominator to return the bounded [-1, 1] Spearman rho


def _evaluate(model, loader, device):
    """
    Evaluates a given model over a dataset loader, calculating RMSE and Spearman correlation.
    
    Sets the model to evaluation mode, disables gradient tracking, and iterates through 
    the provided PyG DataLoader. It aggregates all predictions and true labels, then 
    computes the Root Mean Squared Error (RMSE) and Spearman rho to gauge performance.
    
    Args:
        model (torch.nn.Module): The PyTorch geometric model to evaluate.
        loader (torch_geometric.loader.DataLoader): The data loader providing batches.
        device (torch.device): The hardware device on which to perform inference.
        
    Returns:
        tuple: A 4-tuple containing (RMSE float, Spearman rho float, true labels array, predictions array).
        
    Example:
        >>> # Assuming pre-initialized model, loader, and device
        >>> rmse, rho, y_true, y_pred = _evaluate(gnn_model, val_loader, torch.device('cpu'))
    """
    # -----------------------------------------------------------------------------------------------
    # Inference Setup: Prepare the model, execute inference and accumulate true labels (ys - pChEMBL 
    # values from a reference dataset) and predictions (ps - predicted pChEMBL values from the model)
    # -----------------------------------------------------------------------------------------------
    model.eval(); ys, ps = [], []                                                               # Lock dropout/batchnorm layers and initialize empty lists to accumulate true labels (ys) and predictions (ps)
    with torch.no_grad():                                                                       # Suspend PyTorch's autograd engine to save memory and accelerate inference calculations
        for batch in loader:                                                                    # Iterate sequentially through all graph batches provided by the PyG DataLoader
            batch = batch.to(device)                                                            # Migrate the current batch of graphs and labels to the active compute device (CPU/GPU)
            p = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)                  # Execute a forward pass through the GNN model using node, edge, and batch assignment tensors
            ps.append(p.cpu().numpy()); ys.append(batch.y.cpu().numpy())                        # Detach predictions and labels, move to CPU memory, convert to NumPy arrays, and append to tracking lists
            
    # --------------------------------------------------------------------------------------------------------
    # Metric Computation
    # Compute the Root Mean Squared Error (RMSE) and Spearman correlation between true labels and predictions
    # --------------------------------------------------------------------------------------------------------
    y = np.concatenate(ys); p = np.concatenate(ps)                                              # Flatten the list of batched arrays into single contiguous 1D NumPy arrays for true labels and predictions
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))                                                # Calculate the Root Mean Squared Error (RMSE) representing the standard deviation of the prediction residuals
    return rmse, _spearman(y, p), y, p                                                          # Yield the computed RMSE, Spearman correlation, and the raw aggregated arrays for further analysis


def train_one(train_list, val_list, device, epochs, lr, hid, layers, seed):
    """
    Initializes and trains a single graph neural network member of the deep ensemble.
    
    Sets specific random seeds for diversity across the ensemble. Initializes an `AffinityGNN` 
    and an Adam optimizer. Runs a standard training loop utilizing a Smooth L1 (Huber) loss 
    function to maintain robustness against binding affinity outliers. Evaluates the model on 
    the validation set every epoch and implements early stopping based on the best Spearman rho.
    
    Args:
        train_list (list): A list of PyG Data objects representing the training split.
        val_list (list): A list of PyG Data objects representing the validation split.
        device (torch.device): Compute device for training (e.g., 'cuda' or 'cpu').
        epochs (int): The maximum number of training epochs to execute.
        lr (float): Learning rate for the Adam optimizer.
        hid (int): Hidden dimension size for the GNN layers.
        layers (int): Number of message passing layers in the GNN.
        seed (int): The random seed ensuring unique initialization and shuffling for this member.
        
    Returns:
        AffinityGNN: The trained model, restored to its best validation state.
        
    Example:
        >>> # Assuming populated PyG lists
        >>> model = train_one(tr_graphs, va_graphs, torch.device('cuda'), 100, 1e-3, 128, 4, 42)
    """
    # -----------------------------------------------------------------------------------------
    # Initialization and Setup
    # Lock seeds for individual member variance and initialize model, optimizer, and loaders.
    # -----------------------------------------------------------------------------------------
    torch.manual_seed(seed); np.random.seed(seed)                                               # Explicitly isolate the random seed to ensure this specific ensemble member learns a unique trajectory
    model = AffinityGNN(hid=hid, layers=layers).to(device)                                      # Instantiate the base GNN architecture with target dimensions and push weights to the compute device
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)                        # Initialize the Adam optimizer with the provided learning rate and minor L2 regularization
    tr = DataLoader(train_list, batch_size=128, shuffle=True)                                   # Wrap the training graph list in a PyG DataLoader, shuffling to randomize batch distributions
    va = DataLoader(val_list, batch_size=256)                                                   # Wrap the validation graph list in a PyG DataLoader with a larger static batch size for faster evaluation
    best_rho, best_state = -1.0, None                                                           # Initialize tracking variables to monitor validation progress and preserve the best network weights
    
    # --------------------------------------------------------------------------------------------
    # Epoch Loop and Forward Pass
    # Iterate over datasets in each epoch, calculate regression loss, and backpropagate gradients.
    # --------------------------------------------------------------------------------------------
    for ep in range(epochs):                                                                    # Iterate chronologically over the specified maximum number of training epochs
        model.train()                                                                           # Set the network to training mode to activate dropout and batch normalization tracking
        for batch in tr:                                                                        # Iterate sequentially over every mini-batch yielded by the training DataLoader
            batch = batch.to(device)                                                            # Migrate the current batched training graph structures and labels to the compute device
            opt.zero_grad()                                                                     # Flush historical gradients from the optimizer to prevent compounding during backpropagation
            # Execute forward pass, compute Huber loss, and backpropagate gradients
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)               # Execute a forward pass predicting the scaled affinity scores for the batch
            loss = F.smooth_l1_loss(pred, batch.y)                                              # Huber: robust to activity outliers - calculates a loss that is L1 for large errors and L2 for small ones
            loss.backward()                                                                     # Trigger backpropagation to compute analytical gradients across the network graph
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)                             # Clip the absolute magnitude of gradient vectors to prevent unstable parameter updates (exploding gradients)
            opt.step()                                                                          # Apply the computed and clipped gradients to update the model parameters
            
        # -----------------------------------------------------------------------------------------------------
        # Validation and Early Stopping
        # -----------------------------------------------------------------------------------------------------
        # Run the evaluation helper on the validation set to extract the current Spearman rank correlation coefficient (rho).
        _, rho, _, _ = _evaluate(model, va, device)                                             
        if rho > best_rho:                                                                      # Check if the newly computed validation correlation exceeds the historical maximum
            # Update the best score and clone all model parameters strictly to CPU memory for safekeeping
            best_rho, best_state = rho, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()} 
    
    # Early stopping: If at least one successful validation improvement occurred, 
    # restore the network to the best state (that achieved peak validation performance)     
    if best_state is not None:                                                                  # Verify that at least one successful validation improvement occurred during training
        model.load_state_dict(best_state)                                                       # early-stop on val Spearman - restore the network to the exact state that achieved peak validation performance
    return model                                                                                # Return the fully trained and restored GNN member


def main():
    """
    Main CLI entry point for training and saving a deep ensemble surrogate model.
    
    Steps:
    1. Parses command-line hyperparameters (data path, ensemble size, epochs, architecture).
    2. Builds scaffold-split datasets ensuring disjoint chemotypes between train/val/test.
    3. Loops `members` times, sequentially training independent `AffinityGNN` models.
    4. Aggregates trained models into a `DeepEnsemble` and serializes weights and norm stats.
    5. Performs a final holistic evaluation on the held-out test set to log ensemble metrics.
    6. Attempts to render and save a Matplotlib calibration scatter plot for reporting.
    
    Args:
        None (relies on sys.argv via argparse).
        
    Returns:
        None
    """
    # -----------------------------------------------------------------------------------------
    # Argument Parsing
    # Configure CLI flags and define architectural hyperparameters.
    # -----------------------------------------------------------------------------------------
    ap = argparse.ArgumentParser()                                                              # Initialize the argument parser for handling command line configurations
    ap.add_argument("--csv", required=True, help="ChEMBL CSV with columns smiles,pchembl")      # Define the strictly required path parameter pointing to the raw training dataset
    ap.add_argument("--out", required=True, help="output dir for the ensemble + stats")         # Define the strictly required path parameter dictating where artifacts will be saved
    ap.add_argument("--members", type=int, default=4)                                           # Define the number of independent models to train for the uncertainty-aware ensemble
    ap.add_argument("--epochs", type=int, default=120)                                          # Define the maximum iteration limit per ensemble member
    ap.add_argument("--lr", type=float, default=1e-3)                                           # Define the global learning rate used across all Adam optimizers
    ap.add_argument("--hid", type=int, default=128)                                             # Define the hidden embedding dimensionality for the graph convolutions
    ap.add_argument("--layers", type=int, default=4)                                            # Define the depth (number of message passing steps) of the network
    ap.add_argument("--seed", type=int, default=42)                                             # Define the global master random seed for reproducible dataset splitting
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")         # Automatically select hardware acceleration if available, otherwise fallback to CPU
    args = ap.parse_args()                                                                      # Process and extract all provided system arguments into the args namespace

    # -----------------------------------------------------------------------------------------
    # Dataset Construction
    # Split raw CSV into unique scaffold-separated PyG lists and extract normalization logic.
    # -----------------------------------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)                                                        # Generate the required output directory structure, ignoring errors if it already exists
    train_list, val_list, test_list, stats, df = build_datasets(args.csv, seed=args.seed)       # Ingest the CSV, perform rigorous Bemis-Murcko scaffold splitting, and calculate label mean/std
    print(f"[data] {len(df)} unique molecules -> train {len(train_list)} / val {len(val_list)} / test {len(test_list)}") # Output diagnostic split sizes to the standard output console
    print(f"[data] label stats (pChEMBL): mean={stats['mean']:.3f} std={stats['std']:.3f}")     # Output the computed target normalizers that will be applied to the neural network targets

    # --------------------------------------------------------------------------------------------
    # Ensemble Training
    # Iteratively train and validate (RMSE, rho) multiple independent networks (ensemble members).
    # --------------------------------------------------------------------------------------------
    members = []                                                                                # Initialize an empty list to gather the fully trained PyTorch models
    for k in range(args.members):                                                               # Loop linearly up to the requested ensemble capacity
        model = train_one(train_list, val_list, args.device, args.epochs, args.lr,              # Invoke the training sequence for a single network passing required hyperparameters
                          args.hid, args.layers, seed=args.seed + k)                            # Offset the random seed by the member index 'k' to guarantee diverse initializations
        rmse, rho, _, _ = _evaluate(model, DataLoader(test_list, batch_size=256), args.device)  # Assess the newly minted member against the strictly held-out test data
        print(f"[member {k}] scaffold-test  RMSE(z)={rmse:.3f}  Spearman={rho:.3f}")            # Print the isolated generalization metrics for this specific network
        members.append(model)                                                                   # Add the validated model to the final ensemble collection array

    # -----------------------------------------------------------------------------------------
    # Ensemble Aggregation & Serialization: Wrap models into the queryable DeepEnsemble class 
    # and save all the ensemble member models state dicts and normalization stats (mean/std) to disk.
    # -----------------------------------------------------------------------------------------
    ens = DeepEnsemble(members, device=args.device)                                             # Wrap the collection of independently trained models in the unifying DeepEnsemble class
    ens.save(args.out)                                                                          # Trigger the internal save mechanism to dump all state dicts to the target directory
    save_norm(stats, os.path.join(args.out, "norm.json"))                                       # Serialize the label statistics dict to JSON to ensure inference scaling matches training

    # -----------------------------------------------------------------------------------------
    # Validation & Metric Export
    # Perform a holistic ensemble inference pass to capture uncertainty, .
    # -----------------------------------------------------------------------------------------
    te = DataLoader(test_list, batch_size=256)                                                  # Wrap the held-out test graphs in a DataLoader for the final holistic pass
    ys, mus, sds = [], [], []                                                                   # Initialize empty tracking arrays for true labels, mean predictions, and ensemble standard deviations
    for batch in te:                                                                            # Iterate sequentially through the final test set
        mu, sd = ens.predict_batch(batch)                                                       # Request both the consensus prediction (mu) and epistemic uncertainty (sd) from the ensemble
        mus.append(mu.cpu().numpy()); sds.append(sd.cpu().numpy()); ys.append(batch.y.cpu().numpy()) # Download tensor results to CPU memory, convert to NumPy, and append to trackers
    # y are the true labels, mu are the predicted labels (mean prediction across all 
    # independent models in the ensemble for each molecule), and sd is the epistemic 
    # uncertainty (std predictions across the ensemble members).
    y = np.concatenate(ys); mu = np.concatenate(mus); sd = np.concatenate(sds)                  # Flatten the list of batched arrays into contiguous 1D structures for statistical evaluation
    
    metrics = {                                                                                 # Begin construction of the comprehensive output metrics dictionary
        # Number of test molecules
        "n_test": int(len(y)),                                                                  # Record the absolute count of evaluated test molecules
        # RMSE between true pChEMBL values and ensemble's mean prediction, averaged over all test molecules
        "ensemble_rmse_z": float(np.sqrt(np.mean((y - mu) ** 2))),                              # Compute the consensus RMSE over standardized z-score labels
        # Spearman rank correlation between true pChEMBL values and ensemble's mean prediction
        "ensemble_spearman": _spearman(y, mu),                                                  # Compute the consensus Spearman rank correlation
        # Average ensemble member's predictions std (epistemic uncertainty), averaged over all test molecules
        "mean_uncertainty_z": float(np.mean(sd)),                                               # Compute the average scale of model disagreement (epistemic uncertainty proxy)
        # Ensemble members (indpendent networks) and training epochs
        "members": args.members, "epochs": args.epochs,                                         # Stamp the architectural hyperparameters directly into the report
        # Mean and std of the true pChEMBL labels
        "label_mean": stats["mean"], "label_std": stats["std"],                                 # Stamp the absolute physical scaling factors into the report
    }                                                                                           # Close dictionary definition
    with open(os.path.join(args.out, "metrics.json"), "w") as f:                                # Open a target JSON file in write mode within the output directory
        json.dump(metrics, f, indent=2)                                                         # Serialize the dictionary to text with pretty-print indentation
    print("[ensemble] scaffold-test  RMSE(z)={ensemble_rmse_z:.3f}  Spearman={ensemble_spearman:.3f}" # Print a heavily formatted string detailing the final ensemble capability
          "  mean_unc={mean_uncertainty_z:.3f}".format(**metrics))                              # Unpack the metrics dictionary natively into the format string

    # -----------------------------------------------------------------------------------------
    # Plotting & Visualization
    # Attempt to render a visual diagnostic scatter plot detailing calibration accuracy.
    # -----------------------------------------------------------------------------------------
    try:                                                                                        # Wrap the plotting block in a try-except to gracefully handle missing display libraries
        import matplotlib                                                                       # Import the root matplotlib namespace dynamically
        matplotlib.use("Agg")                                                                   # Force the non-interactive Agg backend to support headless server generation
        import matplotlib.pyplot as plt                                                         # Import the standard pyplot interface for rendering commands
        # back to real pChEMBL units for an interpretable plot
        yt = y * stats["std"] + stats["mean"]; pt = mu * stats["std"] + stats["mean"]           # Multiply by standard deviation and add mean to revert z-scores back to raw physical pChEMBL values
        plt.figure(figsize=(5, 5))                                                              # Initialize a new square canvas figure object
        plt.errorbar(yt, pt, yerr=sd * stats["std"], fmt="o", ms=3, alpha=0.5, lw=0.5)          # Draw a scatter plot with error bars representing the unscaled model uncertainty boundaries
        lo, hi = min(yt.min(), pt.min()), max(yt.max(), pt.max())                               # Calculate absolute minimum and maximum data bounds to frame the chart accurately
        plt.plot([lo, hi], [lo, hi], "k--", lw=1)                                               # Draw a dashed black unity line (y=x) representing the threshold of perfect theoretical prediction
        plt.xlabel("true pChEMBL"); plt.ylabel("predicted pChEMBL")                             # Append descriptive axis labels specifying the physical unit space
        plt.title("Surrogate calibration (scaffold-held-out)\n"                                 # Generate a multi-line title header detailing the split criteria
                  f"rho={metrics['ensemble_spearman']:.2f}  RMSE(z)={metrics['ensemble_rmse_z']:.2f}") # Inject the primary performance metrics natively into the title
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "calibration.png"), dpi=140)     # Strip excess border padding and write the image out to disk at moderate resolution
        print(f"[plot] wrote {os.path.join(args.out, 'calibration.png')}")                      # Output a confirmation string to the console indicating a successful plot render
    except Exception as e:                                                                      # Intercept any system errors thrown by missing graphical dependencies or formatting bugs
        print(f"[plot] skipped calibration plot ({e})")                                         # Issue a benign warning print and proceed safely without crashing

    print(f"[done] surrogate ensemble saved to {args.out}")                                     # Issue final execution completion print terminating the script natively


if __name__ == "__main__":                                                                      # Shield main execution logic from triggering on arbitrary module imports
    main()                                                                                      # Formally invoke the master training procedure