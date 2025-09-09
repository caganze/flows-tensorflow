#!/bin/bash
#SBATCH --job-name=tfp_test
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --output=logs/test_%A_%a.out
#SBATCH --error=logs/test_%A_%a.err
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-10%5

# Small test job: 10 tasks (50 particles total)
# Use this to test if your submission limits allow smaller arrays

set -e

echo "🧪 TFP Flows Small Test Job"
echo "=========================="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo

# Load modules
module purge
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0
export CUDA_VISIBLE_DEVICES=0

# Activate environment
source ~/.bashrc
conda activate bosque

# Find H5 file
find_h5_file() {
    local eden_files=$(find /oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/ -name "eden_scaled_Halo*" -type f 2>/dev/null | head -1)
    if [[ -n "$eden_files" ]]; then
        echo "$eden_files"
        return 0
    fi
    echo "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/all_in_one.h5"
}

H5_FILE=$(find_h5_file)
OUTPUT_BASE_DIR="/oak/stanford/orgs/kipac/users/caganze/tfp_flows_test_output"
PARTICLES_PER_JOB=5

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

mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p logs

echo "📋 Test Configuration:"
echo "  H5 file: $H5_FILE"
echo "  Output base: $OUTPUT_BASE_DIR"
echo "  Data source: $DATA_SOURCE"
echo "  Halo ID: $HALO_ID"
echo "  Testing with reduced parameters for speed"
echo

# Calculate particle range
START_PID=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PARTICLES_PER_JOB + 1 ))
END_PID=$(( $SLURM_ARRAY_TASK_ID * $PARTICLES_PER_JOB ))

echo "🎯 Processing PIDs: $START_PID to $END_PID"

# Process each particle with reduced parameters for testing
for PID in $(seq $START_PID $END_PID); do
    echo "--- Testing PID $PID ---"
    
    # Create hierarchical output directory for each PID
    MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/${DATA_SOURCE}/halo${HALO_ID}"
    mkdir -p "$MODEL_DIR"
    
    python train_tfp_flows.py \
        --data_path "$H5_FILE" \
        --particle_pid $PID \
        --output_dir "$MODEL_DIR" \
        --epochs 10 \
        --batch_size 512 \
        --learning_rate 1e-3 \
        --n_layers 2 \
        --hidden_units 32
    
    if [ $? -eq 0 ]; then
        echo "✅ PID $PID test completed successfully"
    else
        echo "❌ PID $PID test failed"
    fi
done

echo "🏁 Test task $SLURM_ARRAY_TASK_ID completed"
