# 🧬 Molecular-Architect-RL

**De novo small-molecule design with Graph Reinforcement Learning.** An actor–critic GNN builds 2D
molecular graphs atom-by-atom under a chemically-constrained action space, shaped by a multi-objective
RDKit reward, a **synthesizability gate** (Stage 1), and — for a real oncology target (KRAS G12C) — a
learned, uncertainty-aware **affinity surrogate**, a **diversity penalty**, and **ESM-2 pocket
conditioning** (Stage 2).

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![PyG](https://img.shields.io/badge/PyG-Graph_Neural_Networks-3C2179)
![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-blue)
![RL](https://img.shields.io/badge/RL-PPO-009688)

- **Stage 1 — complete.** A de-hacked, benchmarked, honestly-reported generative baseline; the reward no
  longer rewards valence-legal-but-unsynthesizable chemistry, and the mode collapse of single-objective
  RL is characterised openly.
- **Stage 2 — complete.** Target-aware generation for KRAS G12C. A scaffold-split deep-ensemble affinity
  surrogate + uncertainty-penalised reward + Tanimoto diversity + ESM-2 pocket FiLM. **Fixes the Stage-1
  collapse (top-k uniqueness 0.02 → 0.50) while keeping affinity engaged, and validates the top-k with
  independent docking + PoseBusters.** All Stage-2 machinery is off by default and reduces exactly to the
  Stage-1 reward.

See `STAGE2_RESULTS.md` for the full Stage-2 analysis and `STAGE1_MASTER_REFERENCE.md` §VI for Stage 1.

---

## Why this project is different

Most "QED-maximising" RL generators quietly **reward-hack**: they find valence-legal motifs (carbon–
sulfur triple bonds `S#C`, cumulated chains `C#S#S#C`) that score QED 0.8–0.9 yet are impossible to
synthesise (SA 6–8; real drugs sit ~1.5–3.5). This repo treats that as a first-class problem, then goes
further and makes generation **target-aware** with an explicit anti-proxy-hacking guard:

- **Synthesizability gate** (`reward/synth_gate.py`) — hard-bans exploit SMARTS, enforces an SA ceiling,
  soft-penalises PAINS/BRENK alerts, applied to the *entire* terminal reward.
- **Uncertainty-penalised affinity** (`reward/affinity_reward.py`) — `A = sigmoid(pred_z) − β·uncertainty`
  from a deep ensemble; the β·uncertainty term zeroes affinity credit off-distribution, so the agent
  can't farm reward by exploiting the surrogate outside its applicability domain.
- **Diversity penalty** (`reward/composite.py`) — a rolling-archive Tanimoto term that directly counters
  mode collapse.
- **Honest reporting** — policy mean reward logged separately from best-of-archive; live `gen/sa_mean`
  and per-term affinity/uncertainty/diversity tracked throughout.

## Architecture (as built)

GINEConv → GATv2 (×2) → Attentional-Aggregation actor–critic — **not** a plain GCN.

- **GINEConv** — edge-aware message passing folds bond features into aggregation.
- **GATv2Conv ×2** (4→2 heads) — attention for long-range topological context.
- **Attentional Aggregation** — learned gated pool over atoms.
- **Actor / Critic** — 119-way action logits + scalar value.
- **Pocket FiLM (Stage 2)** — optional ESM-2 pocket vector modulates the graph embedding;
  identity-initialised so a conditioned agent starts identical to baseline.

Trained with **GAE-PPO**: clipped surrogate + value loss, entropy annealed in lockstep with a QED→MPO
reward curriculum, KL early-stopping, grad-norm clipping, AMP, batched PyG forward passes, and a
**vectorised** builder with valence-aware masking (`finfo(dtype).min`, AMP-safe).

**Action space (119):** Stop (1) · Add atom+bond (18) · Focus-shift (25) · Ring-closure (75), over
{C, N, O, F, S, Cl} and bond orders {single, double, triple}, with valence, heavy-atom-cap, and
ring-size (≥5) masking.

## Repo layout

```
src/
  chem_env.py        — molecule-building MDP: ActionSpec, masking, MPO reward, Stage-2 terminal hook
  vec_env.py         — vectorised wrapper; per-env terminal SMILES + reward infos
  gnn_agent.py       — GINE→GATv2→GATv2 + attentional-pool actor–critic (+ optional pocket FiLM)
  ppo.py             — PPO buffer + per-env GAE + minibatching
  train_rl.py        — training entrypoint (CONFIG, curriculum, W&B, TopK, checkpoints)
  reward/            — synth_gate · property_reward · affinity_reward · composite
  surrogate/         — featurize · dataset · model · train_surrogate · predict
  pocket/            — conditioning (FiLM) · encode_pocket (ESM-2)
  oracle/            — dock (gnina/smina/vina) · prepare_receptor · posebusters_check · physdock_bridge
  eval/              — metrics · validate_topk
  gallery.py · make_reference.py
configs/kras_g12c.yaml — target: receptor, ref ligand, pocket residues, box, ESM model, gnina knobs
data/                  — fetch_chembl_kras · build_docking_labels · build_training_set
artifacts/<run>/       — TXT log, top-k tables, checkpoints
setup_docking.sh · requirements_stage1.txt · requirements_stage2.txt
```

## Installation

```bash
pip install -r requirements_stage1.txt          # PyTorch, PyG, RDKit, NumPy, W&B, tqdm, pandas, Pillow
pip install -r requirements_stage2.txt          # requests, pyyaml, matplotlib, posebusters, fair-esm, meeko, vina
bash setup_docking.sh                            # optional path: smina/gnina + Meeko
```

## Quickstart

### Stage 1 — gated generator
```bash
cd src
python reward/synth_gate.py                                  # sanity: known hacks fail, real drugs pass
python train_rl.py                                           # logs/checkpoints/top-k -> src/artifacts/<run>/
TOPK=$(find artifacts -name topk_final.txt | head -1)
python eval/metrics.py --gen "$TOPK"
python gallery.py --gen "$TOPK" --out ../results/gallery.png --title "Gated reward"
```

### Stage 2 — target-aware (KRAS G12C)
```bash
cd src
python -m oracle.prepare_receptor --config ../configs/kras_g12c.yaml     # receptor box
python -m pocket.encode_pocket   --config ../configs/kras_g12c.yaml     # ESM-2 pocket vector
python ../data/fetch_chembl_kras.py --out ../data/chembl_kras.csv
python ../data/build_training_set.py --chembl ../data/chembl_kras.csv --out ../data/surrogate_train.csv
python -m surrogate.train_surrogate --csv ../data/surrogate_train.csv --out ../artifacts/surrogate_kras --members 5
# CONFIG: use_affinity/use_diversity/use_pocket=True, w_affinity=0.7, w_diversity=1.0, entropy_coef_end=0.02
python train_rl.py
TOPK=$(ls -td artifacts/molrl_* | head -1)/topk_final.txt
python eval/metrics.py --gen "$TOPK"
python -m eval.validate_topk --config ../configs/kras_g12c.yaml --topk "$TOPK" --surrogate ../artifacts/surrogate_kras
```
> gallery.py needs system X libs once: `sudo apt-get install -y libxrender1 libxext6 libsm6`.
> Optional docking labels + DiffDock-L/OpenMM validation: see `STAGE2_RESULTS.md` and `oracle/`.

## Results

### Stage 1 (`molrl_20260624_165347`, A10G, 800 updates)
Gate works: **validity 1.0**, `gen/sa_mean ≈ 3.8`, zero banned motifs — the `S#C`/`C#S#S#C` hacks are
gone. Metrics exposed the honest limitation: **mode collapse** (uniqueness 0.02, internal_diversity 0.0;
`ppo/kl → 0`), the expected behaviour of single-objective on-policy RL and the motivation for Stage 2.

### Stage 2 (`molrl_20260702_191404`, A10G, 300 updates) — target-aware
| Metric | Stage 1 | **Stage 2** |
|---|---:|---:|
| validity | 1.00 | **1.00** |
| top-k uniqueness | 0.02 | **0.50** |
| scaffold diversity | 0.02 | **0.40** |
| internal diversity | 0.00 | **0.55** |
| SA mean (all ≤4) | 3.09 | **3.39** |
| affinity term (engaged) | — | **0.70** |
| surrogate uncertainty | — | **0.39** |
| PoseBusters valid | — | **1.00** |
| docking mean / best (kcal/mol) | — | **−6.13 / −7.23** |

Surrogate: 5-member GINEConv ensemble, **scaffold-split Spearman 0.564**. Collapse solved while affinity
stayed engaged; **25 unique, 100% pose-valid, synthesizable** candidates. Docking is modest and
non-covalent (G12C binders are covalent), and the surrogate is confident only near wild-type ChEMBL —
the characterised limitation that motivates Stage 3. Full analysis in `STAGE2_RESULTS.md`.

## Honest caveats (Stage 2)

Docking is a **relative** proxy, not absolute ΔG; KRAS G12C binders are **covalent** (Cys12) while smina
docks non-covalently; the surrogate trains on **wild-type ChEMBL**. Two anti-hacking guards by design:
the synthesizability gate and the surrogate uncertainty penalty. Top-k in-silico hits are validated by
independent physics (PoseBusters + smina; gnina/CNN + DiffDock-L/OpenMM via the PhysDock env).

## Roadmap (Stage 3)

Close the proxy↔oracle **active-learning loop**: dock the generator's own top-k each round, add those
labels, retrain the surrogate, and watch calibration tighten and "confident" migrate toward "confidently
potent" — progressively extending the applicability domain into the chemistry the generator explores.
Plus PPO-vs-GFlowNet diversity studies and a value-function fix (`explained_var ≈ 0`).
