#!/bin/bash
#SBATCH --job-name=tfp_single
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --output=logs/single_%j.out
#SBATCH --error=logs/single_%j.err
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Single GPU Training Script for TensorFlow Probability Flows
# Usage: sbatch train_single_gpu.sh [PID] [N_SAMPLES] [EPOCHS]

set -e

echo "🚀 Single GPU TFP Flow Training"
echo "==============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo

# Load required modules
module purge
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0

# Activate environment
source ~/.bashrc
conda activate bosque

# Parameters (can be overridden)
PID="${1:-1}"
N_SAMPLES="${2:-100000}"
EPOCHS="${3:-50}"
H5_FILE="${H5_FILE:-/scratch/groups/aganze/christianaganze/symphony_mocks/all_in_one.h5}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/groups/aganze/christianaganze/tfp_flows_output}"

echo "📋 Parameters:"
echo "  Particle PID: $PID"
echo "  Samples: $N_SAMPLES"
echo "  Epochs: $EPOCHS"
echo "  H5 file: $H5_FILE"
echo "  Output: $OUTPUT_DIR"
echo

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Train the flow
python train_tfp_flows.py \
    --data_path "$H5_FILE" \
    --particle_pid $PID \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size 1024 \
    --learning_rate 1e-3 \
    --n_layers 4 \
    --hidden_units 64

echo "🎉 Training completed for PID $PID"
