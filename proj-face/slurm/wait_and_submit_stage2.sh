#!/bin/bash
# Polls for the item-12 stage-1 checkpoint
# (train_from_scratch_7way_EM_seojin_resnet18.py) and, once it exists,
# automatically submits both stage-2 curriculum jobs. Meant to be backgrounded so
# you don't have to babysit a terminal:
#   nohup bash wait_and_submit_stage2.sh > wait_and_submit_stage2.log 2>&1 &
#   disown
# Then check progress any time with:
#   cat ~/Lee-et-al-Face-Geometry/proj-face/slurm/wait_and_submit_stage2.log
#   squeue -u sl5700

set -eo pipefail

CKPT=/mnt/smb/locker/issa-locker/users/Seojin/data/saved_models/resnet18_scratch_7way_EM_seojin_seed777_model_best.pth.tar
SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SLURM_DIR"

echo "$(date): waiting for $CKPT"
until [ -f "$CKPT" ]; do
  sleep 30
done
echo "$(date): checkpoint found, submitting stage-2 jobs"

sbatch train_curriculum_sl_12way_from_scratch7way_EM_seojin_resnet18.sbatch
sbatch train_curriculum_sl_12way_colorbg_from_scratch7way_EM_seojin_resnet18.sbatch

echo "$(date): stage-2 jobs submitted -- done"
