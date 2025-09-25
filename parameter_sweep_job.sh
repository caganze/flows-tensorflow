#!/bin/bash
#SBATCH --job-name=param_sweep
#SBATCH --output=param_sweep_%j.out
#SBATCH --error=param_sweep_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=owners
#SBATCH --qos=owners
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load modules and activate environment
module load python/3.11
source activate bosque

# Set working directory
cd /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow

# Run the parameter sweep
echo "🚀 Starting parameter sweep job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo ""

# Choose which sweep to run
# Uncomment one of the following:

# Quick sweep (faster, fewer parameters)
./quick_particle_sweep.sh

# Full sweep (comprehensive, takes longer)
# ./parameter_sweep_particle_sizes.sh

echo ""
echo "🎉 Parameter sweep job completed!"
echo "End time: $(date)"
