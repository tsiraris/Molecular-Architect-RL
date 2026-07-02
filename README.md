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
- Plain-text training logs + top-k tables under `artifacts/<run>/` (GitHub-reviewable) and full W&B integration.
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
artifacts/<run>/      — per-run outputs: TXT training log, top-k tables, checkpoints
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

# 1) train (run logs, checkpoints, and top-k are written to src/artifacts/<run>/)
python train_rl.py

# 2) benchmark the run's molecules (core metrics need no reference set)
TOPK=$(find artifacts -name topk_final.txt | head -1)
python eval/metrics.py --gen "$TOPK"

# 3) render the annotated gallery (the before/after "money figure")
#    note: gallery.py needs system X libs once: sudo apt-get install -y libxrender1 libxext6 libsm6
python gallery.py --gen "$TOPK" --out ../results/gallery.png --title "Gated reward"

# 4) (optional) add novelty + FCD against a drug-like reference set
#    PyTDC pulls a heavy dependency tree — install it last / in a separate env so it can't downgrade rdkit
python make_reference.py --out ../data/ref/moses.csv --n 30000 --name MOSES
python eval/metrics.py --gen "$TOPK" --ref ../data/ref/moses.csv --device cuda
```

## Results & honest framing

**Reference run `molrl_20260624_165347` (A10G, 800 updates).** The synthesizability gate did its
job, and the benchmark harness exposed the baseline's real limitation — both reported openly.

What worked:
- **Validity 1.0** throughout (action masking is airtight).
- **SA collapsed into synthesizable territory**: final `gen/sa_mean ≈ 3.8`, top-k `sa_mean 3.09`,
  `sa_frac_le_4 = 1.0`, and **zero banned motifs** in the gallery — the `S#C` / `C#S#S#C`
  reward-hacks of the pre-gate baseline are gone. Best molecule: `CCC(CN)C1CCCCCCCCC1` (a real,
  synthesizable aminomethyl-cyclodecane).

What the metrics exposed (the honest part):
- **Mode collapse.** Final archive `uniqueness 0.02`, `internal_diversity 0.0`,
  `scaffold_diversity 0.02`; in the late run `ppo/kl → 0.000` and `ppo/clip_frac → 0.000` (the
  policy stopped changing for ~250 updates) and the best reward was frozen for ~365 updates.
- **Best ≠ policy.** The archived best (9.39) is a lucky sample, ~15–20× the policy's mean reward;
  mean/terminal reward, validity, and diversity are tracked alongside it rather than headlining
  the single best.

This is a **strong, modern RL baseline with a characterised failure mode** — not a SOTA
drug-discovery tool. The collapse is the expected behaviour of single-objective on-policy RL on a
sparse reward, and it is exactly what motivates the diversity/uncertainty pressure in Stage 2. See
`STAGE1_MASTER_REFERENCE.md` §VI for the full run analysis.

## Roadmap (Stage 2+)

Replace the hackable 2D reward with a **target-conditioned affinity signal**, unifying with a physics
oracle into a generate → dock/score → reward → active-learning loop on a real oncology target
(KRAS G12C): pocket-conditioned generation, a fast learned affinity surrogate calibrated by docking
(gnina/Vina) + Boltz-2 + OpenMM, pose validity via PoseBusters, and PPO-vs-GFlowNet diversity studies.
