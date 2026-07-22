#!/usr/bin/env bash
# Rebuilds the base conda env after a SageMaker space stop/restart (packages in /opt/conda are wiped;
# your files, ~/bin/gnina, and ~/.conda envs persist). Run once per fresh session: bash rebuild_env.sh
set -uo pipefail
export PATH="$HOME/bin:$PATH"
echo "== docking engines =="
conda install -y -c conda-forge -c bioconda smina openbabel
echo "== RL stack =="
python -c "import torch,torch_geometric,rdkit" 2>/dev/null || pip install rdkit torch_geometric
echo "== gnina CUDA libs (the wrapper ~/bin/gnina already bakes in the lib path) =="
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
            nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12
echo "== validation deps =="
pip install posebusters prolif MDAnalysis
echo "== checks =="; gnina --version; python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
