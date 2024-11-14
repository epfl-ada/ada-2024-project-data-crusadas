#!/bin/bash
#SBATCH --job-name=similarity_clustering
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu_long
#SBATCH --output=clustering_%j.log
#SBATCH --error=clustering_%j.err

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
srun python compute_louvain_cluster.py \
    --output-dir outputs/ \
    --threshold 0.7 \
    --batch-size 100000 \
    --resolution 1.0