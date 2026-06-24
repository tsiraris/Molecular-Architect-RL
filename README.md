# Molecular-Architect-RL

**De novo small-molecule design with Graph Reinforcement Learning.** An actor–critic GNN builds 2D
molecular graphs atom-by-atom under a chemically-constrained action space, shaped by a
multi-objective RDKit reward and — critically — a **synthesizability gate** that closes the
descriptor reward-hacking loophole most QED-driven generators fall into.

> **Stage 1 status:** a de-hacked, benchmarked, honestly-reported generative baseline. The reward no
> longer rewards valence-legal-but-unsynthesizable chemistry, and every run is scored with
> GuacaMol/MOSES-style metrics. This is the launch point for the target-aware program (Stage 2).

---

## Why this project is different

Most "QED-maximising" RL generators quietly **reward-hack**: they discover valence-legal motifs
(carbon–sulfur triple bonds `S#C`, cumulated chains `C#S#S#C`) that score QED 0.8–0.9 yet are
physically impossible to synthesise (SA score 6–8; real drugs sit ~1.5–3.5). This repo treats that
failure as a first-class problem:

- **A synthesizability gate** (`reward/synth_gate.py`) hard-bans exploit SMARTS, enforces an SA
  ceiling, and soft-penalises PAINS/BRENK structural alerts — applied to the *entire* terminal
  reward so it constrains both the easy (QED) and hard (MPO) regimes.
- **Honest reporting**: policy mean reward (`reward/terminal_mean`) is logged *separately* from the
  best-of-archive sample (`best/reward`), and live `gen/sa_mean` tracks synthesizability as it falls.
- **A standard evaluation harness** (`eval/metrics.py`): validity, uniqueness, novelty, internal &
  scaffold diversity, SA distribution, and FCD — plus a before/after **gallery** that renders each
  molecule with its `QED | SA | gate-verdict`.

## Architecture (as built)

A three-layer message-passing encoder with attentional readout and an actor–critic head — **not** a
plain GCN:

- **GINEConv** (edge-aware Graph Isomorphism Network) — establishes local chemical environments by
  folding bond features into aggregation.
- **GATv2Conv ×2** (4 heads → 2 heads) — graph attention for long-range topological context.
- **Attentional Aggregation** — a learned gated pool (weights atoms by salience, not a plain mean).
- **Actor / Critic heads** — 119-way action logits + a scalar state value.

Trained with **GAE-PPO**: clipped surrogate, clipped value loss, entropy regularisation annealed in
lockstep with a **QED→MPO reward curriculum**, KL early-stopping, gradient clipping, batched PyG
forward passes, and AMP. The molecule builder is a **vectorised environment** with valence-aware
action masking (`finfo(dtype).min` masking — AMP-safe) and correct per-env terminal-SMILES capture.

**Action space (119):** Stop (1) · Add atom+bond (18) · Focus-shift (25) · Ring-closure (75), over
elements {C, N, O, F, S, Cl} and bond orders {single, double, triple}, with valence, heavy-atom-cap,
and ring-size (≥5) masking.

## Highlights

- PPO with the full stabiliser stack: GAE, ratio + value clipping, entropy schedule, KL monitoring,
  grad-norm clipping, AMP mixed precision.
- Vectorised molecule environment (`VectorMoleculeEnv`) with correct terminal-SMILES capture.
- RDKit multi-objective reward (QED + property-window MPO) **gated by a synthesizability auditor**.
- GuacaMol/MOSES benchmark suite + annotated before/after molecule gallery.
- Plain-text experiment logs in `experiments/` (GitHub-reviewable) and full W&B integration.
- Comprehensive engineering reference (`STAGE1_MASTER_REFERENCE.md`) covering theory, formulas,
  pipeline, phase history, and a per-script analysis.

## Repo layout

```
src/
  chem_env.py        — molecule-building MDP: ActionSpec, masking, MPO reward, terminal gate hook
  vec_env.py         — vectorised wrapper with per-env infos["terminal_smiles"]
  gnn_agent.py       — GINE→GATv2→GATv2 + attentional-pool actor–critic
  ppo.py             — PPO buffer + per-env GAE + minibatching
  train_rl.py        — training entrypoint (CONFIG, curriculum, W&B, TopK, checkpoints)
  reward/
    synth_gate.py    — synthesizability & realism gate (banned motifs, SA ceiling, PAINS/BRENK)
  eval/
    metrics.py       — validity / uniqueness / novelty / diversity / SA / FCD
  gallery.py         — annotated molecule grid (QED | SA | verdict) for before/after figures
  make_reference.py  — build a MOSES/ZINC/ChEMBL reference set (PyTDC) for novelty/FCD
experiments/         — TXT run logs (tracked)
wandb/               — W&B local artifacts (ignored)
```

## Installation

```bash
pip install -r requirements.txt
```
Core dependencies: PyTorch, PyTorch-Geometric, RDKit, NumPy, W&B, tqdm, pandas, Pillow; optional
`fcd_torch` (FCD metric) and `PyTDC` (reference-set download).

## Quickstart

```bash
cd src

# 0) sanity-check the synthesizability gate (known hacks fail, real drugs pass)
python reward/synth_gate.py

# 1) train (logs to W&B + experiments/; checkpoints + top-k to experiments/)
python train_rl.py

# 2) benchmark a checkpoint's molecules (core metrics need no reference set)
python eval/metrics.py --gen ../experiments/<run>/topk_final.txt

# 3) render the annotated gallery (the before/after "money figure")
python gallery.py --gen ../experiments/<run>/topk_final.txt --out ../results/gallery.png --title "Gated reward"

# 4) (optional) add novelty + FCD against a drug-like reference set
python make_reference.py --out ../data/ref/moses.csv --n 30000 --name MOSES
python eval/metrics.py --gen ../experiments/<run>/topk_final.txt --ref ../data/ref/moses.csv --device cuda
```

## Results & honest framing

With the full 119-action space, ring closures, and the QED→MPO curriculum, best archived reward rises
into the ~10.8–11.3 band. Two caveats are reported openly rather than hidden:

1. **Best ≠ policy.** The single best archived molecule is not the policy's typical output; mean
   terminal reward, validity, and diversity are tracked alongside it.
2. **The gate matters.** Pre-gate "elite" molecules were reward-hacks (`S#C` / `C#S#S#C`); post-gate,
   such motifs are eliminated and `gen/sa_mean` falls toward the synthesizable range (~3).

This is a **strong, modern RL baseline**, not a SOTA drug-discovery tool — by design.

## Roadmap (Stage 2+)

Replace the hackable 2D reward with a **target-conditioned affinity signal**, unifying with a physics
oracle into a generate → dock/score → reward → active-learning loop on a real oncology target
(KRAS G12C): pocket-conditioned generation, a fast learned affinity surrogate calibrated by docking
(gnina/Vina) + Boltz-2 + OpenMM, pose validity via PoseBusters, and PPO-vs-GFlowNet diversity studies.
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
