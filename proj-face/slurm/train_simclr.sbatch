#!/bin/bash
#SBATCH --job-name=rn50_simclr_28way
#SBATCH --partition=issa
#SBATCH --gres=gpu:gtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail  # NOT -u: conda.sh references unset $PS1 in a non-interactive shell

source $(conda info --base)/etc/profile.d/conda.sh
conda activate faceproj
export PYTHONNOUSERSITE=1

OUTDIR=/mnt/smb/locker/issa-locker/users/Seojin/data/saved_models
mkdir -p "$OUTDIR"
cd "$OUTDIR"

python "$HOME/Lee-et-al-Face-Geometry/proj-face/train_simclr.py"
