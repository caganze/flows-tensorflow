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
H5_FILE="${H5_FILE:-/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/all_in_one.h5}"
# Extract data source and halo ID from H5 file
FILENAME=$(basename "$H5_FILE")
HALO_ID=$(echo "$FILENAME" | sed 's/.*Halo\([0-9][0-9]*\).*/\1/')

# Determine data source from filename
if [[ "$FILENAME" == *"eden_scaled"* ]]; then
    DATA_SOURCE="eden"
elif [[ "$FILENAME" == *"symphonyHR_scaled"* ]]; then
    DATA_SOURCE="symphony-hr"
elif [[ "$FILENAME" == *"symphony_scaled"* ]]; then
    DATA_SOURCE="symphony"
else
    DATA_SOURCE="unknown"
fi

# Handle fallback file (all_in_one.h5) - use default structure
if [[ "$FILENAME" == "all_in_one.h5" ]] || [[ "$HALO_ID" == "$FILENAME" ]]; then
    echo "⚠️  Using fallback file, setting default halo structure"
    HALO_ID="000"
    DATA_SOURCE="symphony"
fi

# Output directories - save in same parent directory as H5 file with halo/PID structure
H5_PARENT_DIR=$(dirname "$H5_FILE")
OUTPUT_BASE_DIR="$H5_PARENT_DIR/tfp_output"
MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/${DATA_SOURCE}/halo${HALO_ID}"

echo "📋 Parameters:"
echo "  Particle PID: $PID"
echo "  Samples: $N_SAMPLES"
echo "  Epochs: $EPOCHS"
echo "  H5 file: $H5_FILE"
echo "  H5 parent: $H5_PARENT_DIR"
echo "  Data source: $DATA_SOURCE"
echo "  Halo ID: $HALO_ID"
echo "  Model dir: $MODEL_DIR"
echo

# Create directories
mkdir -p "$MODEL_DIR"
mkdir -p logs

# Train the flow
python train_tfp_flows.py \
    --data_path "$H5_FILE" \
    --particle_pid $PID \
    --output_dir "$MODEL_DIR" \
    --epochs $EPOCHS \
    --batch_size 1024 \
    --learning_rate 1e-3 \
    --n_layers 4 \
    --hidden_units 64

echo "🎉 Training completed for PID $PID"
