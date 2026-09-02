#!/bin/bash
#SBATCH --job-name=pairwise_svm
#SBATCH --partition=issa
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Pairwise SVM decoding + live HTML dashboard, on a compute node (the login
# node OOMs on the torch import). Edit TASKS / MODELS / OUTDIR below, then:
#     cd ~/Lee-et-al-Face-Geometry/proj-face/pairwise_svm
#     sbatch run_pairwise_svm.sbatch
# Watch:  tail -f pairwise_svm_<jobid>.out   (and open OUTDIR/dashboard.html)

set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py3            # same env you use interactively; change if needed
export PYTHONUNBUFFERED=1

cd "$HOME/Lee-et-al-Face-Geometry/proj-face/pairwise_svm"

TASKS="basel_GFR basel_CTFR GER_AppleMesh00 CTER_AppleMesh00 lfw"
MODELS="SL_resnet50_finetune_28way_IDEM_combined_seed777_model_best"
OUTDIR="./results_combined_misc"

mkdir -p "$OUTDIR"
python run_pairwise_svm.py \
    --tasks $TASKS \
    --models $MODELS \
    --outdir "$OUTDIR" \
    --redo-errors          # retry pairs that failed on the login node
