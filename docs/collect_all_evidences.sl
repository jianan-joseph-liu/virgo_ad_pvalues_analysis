#!/bin/bash
#SBATCH --job-name=collect_all_evidences       
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=1              
#SBATCH --mem=1G                      
#SBATCH --time=00:10:00                
#SBATCH --partition=skylake            

module load python-scientific/3.10.4-foss-2022a
source /fred/oz303/jliu/virtual_evns/sgvb_venv/bin/activate

python collect_all_evidences.py
