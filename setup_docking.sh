#!/usr/bin/env bash

# ====================================================================================================
# Docking Oracle Setup Script
#
# Role: This script configures the local environment (specifically targeted at an A10G GPU box) 
# to act as a target-aware "docking oracle". It securely downloads and installs the necessary 
# bioinformatics binaries and Python bindings required to evaluate the binding affinity of 
# generated 3D molecules.
#
# Core Components:
# - Meeko: Prepares receptor and ligand structural files into the required PDBQT format.
# - AutoDock Vina: Installed via Python bindings as a robust, native CPU-driven scoring fallback.
# - gnina: The frontier choice; a GPU-accelerated fork of smina/vina that uses Convolutional Neural 
#          Networks (CNNs) to rescore docking poses for significantly higher accuracy.
# - smina: A static, single-file, highly portable CPU fallback executable if the GPU environment fails.
# ====================================================================================================

set -e                                                                                          # Instruct the bash interpreter to immediately exit if any subsequent command returns a non-zero status code (fail-fast protection).

# ----------------------------------------------------------------------------------------------------
# Python Dependencies
# Install the necessary Python packages for ligand preparation and baseline CPU docking.
# ----------------------------------------------------------------------------------------------------
pip install "meeko>=0.6,<0.7" vina                                                              # Install Meeko (strictly v0.6.x for reliable PDBQT preparation) and AutoDock Vina python bindings (as a reliable CPU fallback).

# ----------------------------------------------------------------------------------------------------
# Primary Oracle (gnina) Installation
# Fetch the pre-compiled, CNN-rescored, GPU-accelerated gnina executable.
# Preferring the static binary avoids complex source builds on the target box.
# ----------------------------------------------------------------------------------------------------
mkdir -p "$HOME/bin" && cd "$HOME/bin"                                                          # Safely create a local user binary directory if it doesn't exist, then immediately navigate into it.
if [ ! -x "$HOME/bin/gnina" ]; then                                                             # Check if the 'gnina' executable file is currently missing or lacks execution permissions in the bin directory.
  wget -q https://github.com/gnina/gnina/releases/latest/download/gnina -O gnina && chmod +x gnina # Quietly download the latest static gnina binary directly from GitHub, save it natively as 'gnina', and grant it execution rights.
fi                                                                                              # Close the conditional check for the gnina installation.

# ----------------------------------------------------------------------------------------------------
# Fallback Oracle (smina) Installation
# Fetch the highly portable, single-file smina executable for CPU-only environments.
# ----------------------------------------------------------------------------------------------------
if [ ! -x "$HOME/bin/smina" ]; then                                                             # Check if the 'smina' fallback executable file is currently missing or lacks execution permissions.
  wget -q https://sourceforge.net/projects/smina/files/smina.static/download -O smina && chmod +x smina || true # Quietly download the static smina binary from SourceForge, make it executable, and gracefully swallow any network failures (|| true) so the script doesn't crash.
fi                                                                                              # Close the conditional check for the smina installation.

# ----------------------------------------------------------------------------------------------------
# Path Configuration
# Ensure the newly created local binary directory is globally accessible within the system PATH.
# ----------------------------------------------------------------------------------------------------
grep -q 'export PATH=$HOME/bin:$PATH' "$HOME/.bashrc" || echo 'export PATH=$HOME/bin:$PATH' >> "$HOME/.bashrc" # Silently search .bashrc for the path export; if absent, append the export command to persist across future terminal sessions.
export PATH="$HOME/bin:$PATH"                                                                   # Immediately export the local bin directory to the current active session's PATH so the newly downloaded binaries can be used right away.

# ----------------------------------------------------------------------------------------------------
# Installation Verification & Information
# Print diagnostic info to verify successful installation and provide future steps.
# ----------------------------------------------------------------------------------------------------
echo "---- versions ----"                                                                       # Print a visual string separator to the console to clearly denote the beginning of the diagnostic version output.
gnina --version 2>/dev/null | head -1 || echo "gnina: not runnable on this box -> fall back to smina/vina" # Attempt to print gnina's version (hiding error streams); if it fails (e.g., driver issues), print a warning indicating reliance on the fallbacks.
smina --version 2>/dev/null | head -1 || true                                                   # Attempt to safely print smina's version (hiding error streams), forcefully ignoring any failures (|| true) so the script succeeds cleanly.
echo "PoseBusters (optional): pip install posebusters"                                          # Print a final informational message reminding the user about the optional PoseBusters package for post-docking pose validity checking.