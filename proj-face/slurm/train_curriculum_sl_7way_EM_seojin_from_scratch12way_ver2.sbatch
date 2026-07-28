#!/bin/bash
#SBATCH --job-name=rn50_curriculum_sl_7way_seojin_from12way_ver2
#SBATCH --partition=issa
#SBATCH --gres=gpu:gtx2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail  # NOT -u: conda.sh references unset $PS1 in a non-interactive shell

module load anaconda3-2019.03  # ensure `conda` is on PATH regardless of the submitting shell's state
source $(conda info --base)/etc/profile.d/conda.sh
conda activate faceproj
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1  # otherwise stdout buffers fully when redirected to a file, and .out stays empty for a long time

OUTDIR=/mnt/smb/locker/issa-locker/users/Seojin/data/saved_models
mkdir -p "$OUTDIR"
cd "$OUTDIR"

python "$HOME/Lee-et-al-Face-Geometry/proj-face/train_curriculum_sl_7way_EM_seojin_from_scratch12way_ver2.py"
