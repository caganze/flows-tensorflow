#!/bin/bash
#SBATCH --job-name=tfp_flows
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-1000%20

# TensorFlow Probability Flows Training Array Job
# Uses the new TFP-based pipeline with Kroupa IMF and optimized I/O
# NO JAX DEPENDENCIES

set -e

echo "🚀 TFP Flows Array Job"
echo "====================="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo

# Load required modules (compatible versions)
module purge
module load math
module load devel  
module load nvhpc/24.7
module load cuda/12.2.0
module load cudnn/8.9.0.131

# Set CUDA environment for TensorFlow
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0
export CUDA_VISIBLE_DEVICES=0

# Activate conda environment
source ~/.bashrc
conda activate bosque

# Configuration (adjust as needed)
# Smart H5 file discovery
find_h5_file() {
    if [[ -n "$H5_FILE" && -f "$H5_FILE" ]]; then
        echo "$H5_FILE"
        return 0
    fi
    
    # Look for eden_scaled files first (newer format)
    local eden_files=$(find /oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/ -name "eden_scaled_Halo*" -type f 2>/dev/null | head -1)
    if [[ -n "$eden_files" ]]; then
        echo "$eden_files"
        return 0
    fi
    
    # Fallback to other locations in your oak directory
    local search_paths=(
        "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/"
        "/oak/stanford/orgs/kipac/users/caganze/milkyway-hr-mocks/"
        "/oak/stanford/orgs/kipac/users/caganze/milkywaymocks/"
        "/oak/stanford/orgs/kipac/users/caganze/"
    )
    
    for path in "${search_paths[@]}"; do
        if [[ -d "$path" ]]; then
            h5_file=$(find "$path" -name "*.h5" -type f 2>/dev/null | head -1)
            if [[ -n "$h5_file" ]]; then
                echo "$h5_file"  # Only the file path goes to stdout
                return 0
            fi
        fi
    done
    echo "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/all_in_one.h5"
}

H5_FILE=$(find_h5_file)
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/oak/stanford/orgs/kipac/users/caganze/tfp_flows_output}"
PARTICLES_PER_JOB="${PARTICLES_PER_JOB:-5}"
N_SAMPLES="${N_SAMPLES:-100000}"
EPOCHS="${EPOCHS:-50}"

# Create output directories
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p logs

echo "📋 Configuration:"
echo "  H5 file: $H5_FILE"
echo "  Output dir: $OUTPUT_BASE_DIR"
echo "  Particles per job: $PARTICLES_PER_JOB"
echo "  Samples per particle: $N_SAMPLES"
echo "  Training epochs: $EPOCHS"
echo

# Calculate particle range for this array task
START_PID=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PARTICLES_PER_JOB + 1 ))
END_PID=$(( $SLURM_ARRAY_TASK_ID * $PARTICLES_PER_JOB ))

echo "🎯 Processing PIDs: $START_PID to $END_PID"
echo

# Extract data source and halo ID from H5 file
FILENAME=$(basename "$H5_FILE")
HALO_ID=$(echo "$FILENAME" | sed 's/.*Halo\([0-9]\+\).*/\1/')

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

echo "📁 Data source: $DATA_SOURCE"
echo "📁 Halo ID: $HALO_ID"

# Process each particle in this job's range
for PID in $(seq $START_PID $END_PID); do
    echo "--- Processing PID $PID ---"
    
    # Check if already completed in new hierarchical structure
    MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/${DATA_SOURCE}/halo${HALO_ID}"
    SAMPLES_DIR="$OUTPUT_BASE_DIR/samples/${DATA_SOURCE}/halo${HALO_ID}"
    
    # Check for both model file and results file to ensure complete training
    if [[ -f "$MODEL_DIR/model_pid${PID}.npz" && -f "$MODEL_DIR/model_pid${PID}_results.json" ]]; then
        echo "✅ PID $PID already completed, skipping"
        continue
    fi
    
    # Create directories
    mkdir -p "$MODEL_DIR" "$SAMPLES_DIR"
    
    # Train the flow for this particle
    python train_tfp_flows.py \
        --data_path "$H5_FILE" \
        --particle_pid $PID \
        --output_dir "$MODEL_DIR" \
        --epochs $EPOCHS \
        --batch_size 1024 \
        --learning_rate 1e-3 \
        --n_layers 4 \
        --hidden_units 64
    
    TRAIN_EXIT_CODE=$?
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        echo "✅ PID $PID completed successfully"
    else
        echo "❌ PID $PID failed with exit code $TRAIN_EXIT_CODE"
        # Continue with next PID rather than failing entire job
    fi
    
    echo
done

echo "🏁 Array task $SLURM_ARRAY_TASK_ID completed"
echo "Processed PIDs: $START_PID to $END_PID"
echo "Total runtime: $SECONDS seconds"
