#!/usr/bin/env bash
# ============================================================================================
# save_state.sh — package the resumable state into a single downloadable tarball.
#
# Run this before Stopping the SageMaker space (or before the account closes). It tars exactly
# the files needed to CONTINUE the run on a fresh AWS account: the active-learning output dir
# (docking cache + retrained surrogates + per-round CSVs/metrics/poses) and PhysDock's prepared
# target + results. Restoring the tarball over the repo root and re-running the identical command
# resumes with ZERO recompute (see AWS_BUDGET_AND_RESUME_eu-central-1.md).
#
# Usage:
#   bash save_state.sh                      # defaults: OUT_DIR=artifacts/active_learning
#   OUT_DIR=artifacts/active_learning bash save_state.sh
# ============================================================================================
set -uo pipefail                                                                # Safe-ish: undefined vars + pipe failures are errors (NOT -e, so a missing optional dir won't abort)

OUT_DIR="${OUT_DIR:-artifacts/active_learning}"                                 # The stable, resumable Stage-3 output directory
TS="$(date -u +%Y%m%d_%H%M%S)"                                                  # UTC timestamp so successive snapshots never collide or overwrite
DEST_DIR="${DEST_DIR:-/mnt/user-data/outputs}"                                  # Where SageMaker/Jupyter exposes downloadable files (falls back to CWD if absent)
[ -d "$DEST_DIR" ] || DEST_DIR="."                                             # If the outputs mount is not present, drop the tarball in the current directory
TARBALL="${DEST_DIR}/RESUME_${TS}.tar.gz"                                       # Timestamped resume archive name

# Collect only paths that actually exist (a partial run may not have every artifact yet).
PATHS=()                                                                        # Accumulate existing paths to include
for p in \
    "$OUT_DIR" \
    "artifacts/surrogate_kras" \
    "physdock/data/processed" \
    "physdock/results"; do
  [ -e "$p" ] && PATHS+=("$p")                                                  # Include the path only if it is present on disk
done

if [ "${#PATHS[@]}" -eq 0 ]; then                                              # Nothing to save yet -> tell the user rather than writing an empty archive
  echo "[save_state] nothing to archive yet (no $OUT_DIR / physdock outputs found)."; exit 0
fi

echo "[save_state] archiving: ${PATHS[*]}"                                      # Show exactly what is being captured
tar -czf "$TARBALL" "${PATHS[@]}"                                              # Create the compressed resume tarball
echo "[save_state] wrote $TARBALL ($(du -h "$TARBALL" | cut -f1))"             # Confirm the output path + size for download
echo "[save_state] download this file, then Stop the space. Restore on the new account with:"
echo "             tar -xzf RESUME_${TS}.tar.gz   # from the repo root"
