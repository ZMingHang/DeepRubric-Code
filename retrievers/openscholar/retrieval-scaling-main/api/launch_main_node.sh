#!/bin/bash
#SBATCH --job-name=main-api
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --hint=multithread
#SBATCH --account comem
#SBATCH --qos comem_high
#SBATCH --mem 400G
#SBATCH --time 120:00:00
#SBATCH --requeue
#SBATCH --chdir=.
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --array=0


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ -n "${CONDA_EXE:-}" ]; then
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
    conda activate "${OPENSCHOLAR_CONDA_ENV:-scaling}"
fi


PYTHONPATH=.  python api/serve_main_node.py
