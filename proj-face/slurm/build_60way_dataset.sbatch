#!/bin/bash
#SBATCH --job-name=build_60way_dataset
#SBATCH --partition=issa
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -eo pipefail

bash "$HOME/Lee-et-al-Face-Geometry/proj-face/slurm/build_60way_dataset.sh"
