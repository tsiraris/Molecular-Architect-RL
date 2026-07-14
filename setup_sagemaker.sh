#!/usr/bin/env bash
# ============================================================================================
# setup_sagemaker.sh — one-shot environment bring-up for the full-capabilities Stage-3 run
# on a SageMaker Studio JupyterLab space (ml.g5.4xlarge, A10G 24 GB, eu-central-1).
#
# Sets up TWO isolated environments (their deep-learning stacks conflict, by design):
#   (A) the RL generator env   — torch + PyG + rdkit + docking oracle (this repo)
#   (B) the PhysDock env        — DiffDock-L + Boltz-2 + OpenMM (physdock/env_physdock.yml)
#
# Run it once, from the repo root, in a terminal (NOT pasted line-by-line — see note on `set -e`):
#     bash setup_sagemaker.sh
# ============================================================================================
# NOTE: `set -e` belongs in a SCRIPT, never pasted into an interactive shell (one failed command
# would close your terminal). This file is meant to be *run* as `bash setup_sagemaker.sh`.
set -uo pipefail                                                                # undefined vars + pipe failures are errors; not -e so one optional failure won't abort the whole bring-up

REPO="$(cd "$(dirname "$0")" && pwd)"                                          # Absolute path to the repo root (works regardless of where it's launched)
export SM_BASE="${SM_BASE:-/home/sagemaker-user}"                              # SageMaker Studio home (NOT /home/ec2-user/SageMaker — that's classic notebooks)
export PATH="$HOME/bin:$PATH"                                                  # Ensure locally-installed docking binaries are visible immediately

# Keep parallel workers from oversubscribing the 16 vCPUs during docking (each proc -> 1 math thread).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false        # Pin BLAS/OpenMP threads; silence HF tokenizer fork noise

echo "==================================================================="
echo " (A) RL generator env — pip stack + docking oracle"
echo "==================================================================="
cd "$REPO"
python -m pip install -q --upgrade pip
# Stage 1 + 2 + 3 python deps (torch/PyG/rdkit are assumed present in the SageMaker DL image;
# if not, install the CUDA-matching torch/torch_geometric wheels first).
for req in requirements_stage1.txt requirements_stage2.txt requirements_stage3.txt; do
  [ -f "$req" ] && { echo "[pip] $req"; python -m pip install -q -r "$req" || echo "[pip] some deps in $req need manual attention"; }
done

# Docking binaries (gnina GPU best-effort; smina CPU is the guaranteed, now-parallelised fallback).
echo "[docking] installing gnina (GPU, best-effort) + smina (CPU, guaranteed)…"
bash setup_docking.sh || echo "[docking] setup_docking.sh reported an issue — smina CPU path still works"
echo "[docking] engine check:"
gnina --version 2>/dev/null | head -1 || echo "  gnina: not runnable on this image -> using smina CPU (parallelised across all 16 vCPUs)"
smina --version 2>/dev/null | head -1 || echo "  smina: install from setup_docking.sh"

# ESM-2 pocket weights download on first import; pre-warm so the run doesn't stall mid-loop.
echo "[esm] pre-warming ESM-2 (downloads ~2.5 GB once)…"
python - <<'PY' || echo "[esm] pre-warm skipped (will download on first use)"
try:
    import esm
    esm.pretrained.esm2_t33_650M_UR50D()
    print("[esm] esm2_t33_650M_UR50D ready")
except Exception as e:
    print("[esm] deferred:", e)
PY

echo "==================================================================="
echo " (B) PhysDock env — DiffDock-L / Boltz-2 / OpenMM (isolated conda env)"
echo "==================================================================="
if command -v conda >/dev/null 2>&1; then
  if [ -f "$REPO/physdock/env_physdock.yml" ]; then
    echo "[physdock] creating conda env from physdock/env_physdock.yml (one-time, several minutes)…"
    conda env create -f "$REPO/physdock/env_physdock.yml" 2>/dev/null || echo "[physdock] env 'physdock' may already exist — skipping"
    echo "[physdock] installing DiffDock-L weights/deps via physdock/setup_diffdock.sh…"
    ( cd "$REPO/physdock" && conda run -n physdock bash setup_diffdock.sh ) || echo "[physdock] setup_diffdock.sh needs a look; Boltz-2 weights download on first run"
  else
    echo "[physdock] env_physdock.yml not found — see physdock/README.md for env setup"
  fi
else
  echo "[physdock] conda not found. On SageMaker DL images use the provided conda; otherwise install micromamba."
fi

echo "==================================================================="
echo " DONE. Next:  cd src && python run_stage3.py --help"
echo " Budget/order-of-operations: AWS_BUDGET_AND_RESUME_eu-central-1.md"
echo " Remember to STOP the space when you're just reading results."
echo "==================================================================="
