#!/bin/bash
#SBATCH --job-name=xrdmlip
#SBATCH --account=hmt03
#SBATCH --partition=intelgpu80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/intern004/fastMD-opt3-verify-20260926/results/opt3-fastmd-%j.out
#SBATCH --error=/data/home/intern004/fastMD-opt3-verify-20260926/results/opt3-fastmd-%j.err

set -euo pipefail
BASE=/data/home/intern004/mlip-geoopt-20260904
ROOT=/data/home/intern004/fastMD-opt3-verify-20260926
PYTHON=$BASE/env/bin/python
export PYTHONPATH="$ROOT/src:$BASE/src/deepmd-kit-master:$BASE/src/SevenNet-main:$BASE/src/orb-models:$BASE/src/tace:$BASE/src/nequip:$BASE/md-opt-integration-main"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0

hostname
nvidia-smi -L
"$PYTHON" -u -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count(), flush=True)'

"$PYTHON" -u "$ROOT/scripts/verify_opt3_cuda.py" \
  --device cuda:0 \
  --steps 8 \
  --warmup-steps 2 \
  --output "$ROOT/results/opt3-fastmd-$SLURM_JOB_ID.json" \
  --checkpoint dpa4="$BASE/checkpoints/dpa4-retry.pt" \
  --checkpoint nequip="$BASE/checkpoints/NequIP-OAM-L-0.1.nequip.zip" \
  --checkpoint orbv3="$BASE/checkpoints/orb-v3-conservative-inf-mpa-20250404.ckpt" \
  --checkpoint sevennet="$BASE/src/SevenNet-main/sevenn/pretrained_potentials/SevenNet_0__22May2024/checkpoint_sevennet_0.pth" \
  --checkpoint tace="$BASE/checkpoints/TACE-OAM-L.pt"
