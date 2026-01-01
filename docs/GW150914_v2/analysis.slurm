#!/bin/bash
#SBATCH --job-name=gw150914_array
#SBATCH --output=logs/gw150914_%A_%a.out
#SBATCH --error=logs/gw150914_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1G
#SBATCH --array=0-2  # This defines 3 array tasks with indices 0, 1, and 2

module purge
module load python-scientific/3.10.4-foss-2022a
source /fred/oz303/jliu/virtual_evns/sgvb_venv/bin/activate

# Define the array of PSD names
PSD_NAMES=('bayeswave' 'welch' 'ar_sgvb')

# Get the PSD name based on the array task ID
PSD_NAME_TO_USE=${PSD_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Starting GW150914 job ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Using PSD Name: ${PSD_NAME_TO_USE}"
date

# Run the python script with the selected PSD name
srun python GW150914_analysis.py --psd ${PSD_NAME_TO_USE}

rc=$?
date
echo "Finished GW150914 job ${SLURM_JOB_ID} with exit code ${rc}"
exit ${rc}