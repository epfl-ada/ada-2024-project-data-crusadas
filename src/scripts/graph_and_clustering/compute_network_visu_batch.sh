#!/bin/bash
#SBATCH --job-name=network_visu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=80
#SBATCH --mem=1460GB
#SBATCH --time=24:00:00
#SBATCH --output=network_visu_%j.out
#SBATCH --error=network_visu_%j.err
#SBATCH --partition=mem

# Load required modules
module purge
module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/gcc-9.2.0
module load gcc/9.2.0/gcc-4.8.5
module load openmpi/4.0.2/gcc-9.2.0

cd /workdir/himmian/ADA/

source activate mPYTHON
echo "all loaded"

# Set environment variables for memory efficiency
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_MMAP_THRESHOLD_=100000

# Run the script
srun python compute_network_visu.py