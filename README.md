# Molecular-Architect-RL (2026 baseline)

Graph RL baseline for constrained molecular generation using **PPO + GNNs** with RDKit-based chemistry validity and multi-objective reward shaping.

## Highlights
- PPO training loop with stabilizers (GAE, clipping, entropy regularization, KL monitoring)
- Vectorized molecule environment (`VectorMoleculeEnv`) with **correct terminal SMILES capture**
- RDKit-driven reward shaping (drug-likeness proxies + constraints)
- Plain-text experiment logs in `experiments/` suitable for GitHub review
- W&B integration for tracking runs

## Repo layout
- `src/chem_env.py` — molecule-building environment + reward + action masking
- `src/vec_env.py` — vectorized wrapper with per-env `infos["terminal_smiles"]`
- `src/gnn_agent.py` — policy/value GNN
- `src/ppo.py` — PPO buffer + update utilities
- `src/train_rl.py` — training entrypoint
- `experiments/` — TXT run logs (tracked)
- `wandb/` — W&B local artifacts (ignored)

## Installation
```bash
pip install -r requirements.txt