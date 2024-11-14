#!/bin/bash
#SBATCH --job-name=cosine_sim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --output=cosine_sim_%j.out
#SBATCH --error=cosine_sim_%j.err
#SBATCH --partition=cpu_med

module purge
module load anaconda3/2021.05/gcc-9.2.0

cd /workdir/himmian/ADA/

source activate mPYTHON
echo "all loaded"

python cosine_similarity_calculator.py