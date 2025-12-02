#!/bin/bash
#SBATCH --job-name=simulate_noise_pe_study       
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=32              
#SBATCH --mem=12G                      
#SBATCH --time=24:00:00                
#SBATCH --partition=skylake
#SBATCH --array=1-100
#SBATCH --output=logs/pe_%A_%a.out
#SBATCH --error=logs/pe_%A_%a.err
        

module load python-scientific/3.10.4-foss-2022a
source /fred/oz303/jliu/virtual_evns/sgvb_venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python simulate_noise_pe_study.py $SLURM_ARRAY_TASK_ID
