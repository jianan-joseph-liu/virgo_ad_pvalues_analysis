#!/bin/bash
#SBATCH --job-name=GW150914       
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=32              
#SBATCH --mem=16G                     
#SBATCH --time=10:00:00                
#SBATCH --partition=skylake
#SBATCH --output=logs/pe_%A_%a.out
#SBATCH --error=logs/pe_%A_%a.err


module load python-scientific/3.10.4-foss-2022a
source /fred/oz303/jliu/virtual_evns/sgvb_venv/bin/activate

python GW150914.py
