#!/bin/bash
#SBATCH --job-name=GW150914       
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=8              
#SBATCH --mem=4G                      
#SBATCH --time=03:00:00                
#SBATCH --partition=skylake            

module load python-scientific/3.10.4-foss-2022a
source /fred/oz303/jliu/virtual_evns/sgvb_venv/bin/activate

python GW150914.py
