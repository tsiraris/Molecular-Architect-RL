
# Molecular RL (2026-ready baseline)

This repo is an upgraded, research-grade baseline for **constrained molecular design with PPO** using **graph neural networks** (PyTorch Geometric) and RDKit.

It is **not** a full structure-based drug discovery system (no protein pocket / 3D binding), but it is engineered to look and behave like an industry-quality baseline:
- correct vectorized-environment PPO (per-env GAE bootstrapping)
- proper action masking (invalid logits set to large negative before sampling)
- KL early stopping
- entropy scheduling
- value function clipping
- expanded action space: **stop**, **add atom**, **focus atom**, **ring closure**
- expanded atom vocabulary: C, N, O, F, S, P, Cl, Br
- medchem-style terminal reward proxy (QED + smooth property windows + structural alert penalty)
- top-K artifact export (CSV with properties + scaffolds)
- checkpoints

## Install

You'll need:
- PyTorch + PyTorch Geometric
- RDKit
- wandb (optional)

## Run

```bash
python train_rl.py
```

Artifacts are saved under `artifacts/<run_name>/`.

## Files

- `chem_env.py` – molecule environment + action space + reward proxy
- `gnn_agent.py` – GNN policy/value network
- `ppo.py` – vectorized PPO buffer with correct GAE
- `vec_env.py` – vectorized environment wrapper
- `train_rl.py` – training script
