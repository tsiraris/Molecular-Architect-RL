# 🧬 Molecular-Architect-RL (2026 Baseline)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyG-Graph_Neural_Networks-3C2179)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-blue)
![RL](https://img.shields.io/badge/RL-PPO-009688)

## 📌 Project Overview

**Molecular-Architect-RL** is a Reinforcement Learning baseline for constrained, *de novo* molecular generation. It utilizes Proximal Policy Optimization (PPO) paired with Graph Neural Networks (GNNs) to construct molecules atom-by-atom and bond-by-bond. 

Unlike naive generation models prone to "reward hacking" (e.g., infinite sulfur stacking or unrealistic halogen combinations), this environment is strictly governed by RDKit-driven medicinal chemistry rules, structural alert penalties, and Multi-Parameter Objective (MPO) reward shaping.

## 🧠 Architecture & Methodology

### 1. The GNN Agent (`src/gnn_agent.py`)
The actor-critic architecture operates directly on molecular graphs:
*   **GINEConv:** Captures local chemical environments by injecting bond (edge) features before neighbor aggregation.
*   **GATv2Conv:** Multi-head attention layers capture long-range structural dependencies and prioritize highly relevant neighbors.
*   **Attentional Aggregation:** A learned global pooling mechanism identifies the most "interesting" nodes to form the final molecule embedding.

### 2. The RL Algorithm (`src/ppo.py` & `src/train_rl.py`)
A heavily stabilized PPO implementation designed for non-stationary graph environments:
*   **Vectorized Environments:** Supports parallel collection with proper terminal SMILES capture.
*   **Stabilizers:** Incorporates per-env Generalized Advantage Estimation (GAE), advantage normalization, value function clipping, and gradient clipping.
*   **KL Control:** Epoch-level KL divergence monitoring with early stopping to prevent catastrophic policy updates.
*   **Curriculum Learning:** Linearly shifts the reward signal from pure QED (easy) to the full MPO function (hard) across training epochs.

### 3. MedChem-Aware Environment (`src/chem_env.py`)
The `MoleculeEnvironment` acts as a strict chemical judge:
*   **Action Masking:** Prevents invalid valences, blocks highly strained rings (< 5 members), and dynamically limits halogens (≤1) and sulfurs (≤2).
*   **MPO Reward Shaping:** Combines Drug-likeness (QED), desirability windows (MW, cLogP, TPSA, HBD/HBA), Bertz complexity, and Synthetic Accessibility (SA) proxies.
*   **Structural Penalties:** Rapidly penalizes unwanted structural alerts (e.g., azides, hydrazines).

## 📁 Repository Layout
```text
.
├── experiments/          # Tracked plain-text training logs and SOTA metrics
├── src/                  
│   ├── chem_env.py       # RDKit environment, MPO rewards, action masking
│   ├── vec_env.py        # Vectorized environment wrapper
│   ├── gnn_agent.py      # Actor-Critic GNN architecture (GINE + GATv2)
│   ├── ppo.py            # PPO buffer, GAE calculations, and batching
│   └── train_rl.py       # Main training loop, W&B integration, and Top-K tracking
├── requirements.txt      # Python dependencies
└── README.md
```

## 🚀 Quickstart

### Installation

Install the required dependencies (PyTorch, PyG, RDKit, W&B, etc.):
```bash
pip install -r requirements.txt
```

### Training the Agent

Execute the main training loop. The script automatically handles device assignment (CUDA/CPU) and initializes the curriculum.

```bash
python src/train_rl.py
```

### Logging & Artifacts

*   **Console & W&B:** Real-time metrics including Mean Reward, Validity, Uniqueness, Tanimoto Diversity, and Explained Variance are printed to the console and synced to Weights & Biases.
*   **Top-K Molecules:** The system persistently tracks the top 50 highest-reward molecules (SMILES + calculated properties + Bemis-Murcko scaffolds) and saves them to `artifacts/<run_name>/topk_final.txt`.
*   **Research Logs:** Human-readable training progress is dumped to `experiments/Run_<timestamp>.txt`.
```
