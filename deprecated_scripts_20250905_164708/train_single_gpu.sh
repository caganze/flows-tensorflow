#!/bin/bash
#SBATCH --job-name=tfp_single
#SBATCH --partition=owners
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

# Get particle count for sophisticated parameter selection
echo "🔍 Determining optimal parameters for PID $PID..."
if command -v python >/dev/null 2>&1 && [[ -f "$H5_FILE" ]]; then
    OBJECT_COUNT=$(python -c "
import h5py, sys
try:
    with h5py.File('$H5_FILE', 'r') as f:
        pid = int(sys.argv[1])
        keys = [k for k in f.keys() if k.startswith('pid')]
        if f'pid{pid}' in keys:
            print(f[f'pid{pid}'].shape[0])
        else:
            print(50000)  # Default fallback
except: 
    print(50000)  # Default fallback
" $PID 2>/dev/null || echo 50000)
else
    OBJECT_COUNT=50000  # Default fallback
fi

echo "📊 PID $PID has $OBJECT_COUNT particles"

# SOPHISTICATED parameter selection based on particle count
if [[ $OBJECT_COUNT -gt 100000 ]]; then
    # Large particles (100k+): Need capacity with low learning rate
    EPOCHS=60
    BATCH_SIZE=1024
    N_LAYERS=4
    HIDDEN_UNITS=512
    LEARNING_RATE=2e-4
    echo "🐋 Large particle (>100k): epochs=60, layers=4, units=512, lr=2e-4"
elif [[ $OBJECT_COUNT -gt 50000 ]]; then
    # Medium-large particles
    EPOCHS=45
    BATCH_SIZE=1024
    N_LAYERS=3
    HIDDEN_UNITS=384
    LEARNING_RATE=3e-4
    echo "🐟 Medium-large (50k-100k): epochs=45, layers=3, units=384, lr=3e-4"
elif [[ $OBJECT_COUNT -lt 5000 ]]; then
    # Small particles
    EPOCHS=35
    BATCH_SIZE=512
    N_LAYERS=3
    HIDDEN_UNITS=256
    LEARNING_RATE=5e-4
    echo "🐭 Small particle (<5k): epochs=35, layers=3, units=256, lr=5e-4"
else
    # Medium particles
    EPOCHS=40
    BATCH_SIZE=1024
    N_LAYERS=3
    HIDDEN_UNITS=320
    LEARNING_RATE=4e-4
    echo "🐟 Medium particle (5k-50k): epochs=40, layers=3, units=320, lr=4e-4"
fi

# Train the flow
python train_tfp_flows.py \
    --data_path "$H5_FILE" \
    --particle_pid $PID \
    --output_dir "$MODEL_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --n_layers $N_LAYERS \
    --hidden_units $HIDDEN_UNITS \
    --generate-samples \
    --use_kroupa_imf

echo "🎉 Training completed for PID $PID"
