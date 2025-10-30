#!/bin/bash
# OzSTAR SLURM batch script for running docs/GW150914/GW150914.py

#SBATCH --job-name=GW150914
#SBATCH --account=<your_ozstar_account>
#SBATCH --partition=skylake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/GW150914_%j.out
#SBATCH --error=logs/GW150914_%j.err

set -euo pipefail

# Adjust this path to the repository location on OzSTAR
PROJECT_DIR="${PROJECT_DIR:-$HOME/virgo_ad_pvalues_analysis}"

# Optionally set VENV_DIR to use a pre-existing Python virtual environment
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"

echo "Job started at $(date)"
echo "Running on ${SLURMD_NODENAME:-unknown node}"
echo "Project directory: $PROJECT_DIR"
echo "Virtual environment: $VENV_DIR"

module purge
module load python/3.10.11

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: PROJECT_DIR '$PROJECT_DIR' does not exist." >&2
    exit 1
fi

cd "$PROJECT_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at $VENV_DIR; creating one."
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
elif [ -f "pyproject.toml" ]; then
    pip install --upgrade pip
    pip install .
fi

mkdir -p logs outdir data_cache

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python docs/GW150914/GW150914.py

echo "Job finished at $(date)"
