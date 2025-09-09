#!/bin/bash
#SBATCH --job-name="brute_force_all_halos_pids"
#SBATCH --partition=owners
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --output=logs/brute_force_%j.out
#SBATCH --error=logs/brute_force_%j.err
# Sequential job - no array

set -e

echo "🚀 BRUTE FORCE GPU JOB - ARRAY TASK ${SLURM_ARRAY_TASK_ID:-1}"
echo "Started: $(date)"
echo "Node: ${SLURM_NODELIST:-local}"
echo "Job ID: ${SLURM_JOB_ID:-test}"

# Environment setup
module --force purge
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0
export CUDA_VISIBLE_DEVICES=0

source ~/.bashrc
conda activate bosque

# Create directories
mkdir -p logs success_logs failed_jobs

# Find halo files
echo "🔍 Finding halo files..."
H5_FILES=($(find /oak/stanford/orgs/kipac/users/caganze -name '*Halo*.h5' -type f 2>/dev/null | sort))
echo "Found ${#H5_FILES[@]} halo files"

if [[ ${#H5_FILES[@]} -eq 0 ]]; then
    echo "❌ No halo files found"
    exit 1
fi

# PIDs to test
PIDS=(1 2 3 4 5 23 88 188 268 327 364 415 440 469 530 570 641 718 800 852 939)
echo "Testing ${#PIDS[@]} PIDs"

# Calculate which halo and PID for this task
ARRAY_ID=${SLURM_ARRAY_TASK_ID:-1}
TOTAL_PIDS=${#PIDS[@]}
FILE_INDEX=$(( (ARRAY_ID - 1) / TOTAL_PIDS ))
PID_INDEX=$(( (ARRAY_ID - 1) % TOTAL_PIDS ))

# Check bounds
if [[ $FILE_INDEX -ge ${#H5_FILES[@]} ]]; then
    echo "Array task $ARRAY_ID exceeds available combinations"
    exit 0
fi

SELECTED_FILE="${H5_FILES[$FILE_INDEX]}"
SELECTED_PID="${PIDS[$PID_INDEX]}"

# Extract halo ID and data source
FILENAME=$(basename "$SELECTED_FILE")
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

# Handle fallback/non-standard files
if [[ "$FILENAME" == "all_in_one.h5" ]] || [[ "$HALO_ID" == "$FILENAME" ]]; then
    echo "⚠️  Non-standard filename detected, using fallback structure"
    HALO_ID="000"
    if [[ "$DATA_SOURCE" == "unknown" ]]; then
        DATA_SOURCE="symphony"
    fi
fi

echo "🎯 Processing: $(basename $SELECTED_FILE) PID $SELECTED_PID"
echo "   Data source: $DATA_SOURCE"
echo "   Halo ID: $HALO_ID"
echo "   Output: $DATA_SOURCE/halo$HALO_ID/"

# Output directories - save in same parent directory as H5 file with halo/PID structure
H5_PARENT_DIR=$(dirname "$SELECTED_FILE")
OUTPUT_BASE_DIR="$H5_PARENT_DIR/tfp_output"
MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/${DATA_SOURCE}/halo${HALO_ID}"
SAMPLES_DIR="$OUTPUT_BASE_DIR/samples/${DATA_SOURCE}/halo${HALO_ID}"

echo "📁 H5 file parent: $H5_PARENT_DIR"
echo "📁 Output base: $OUTPUT_BASE_DIR"
echo "📁 Data source: $DATA_SOURCE, Halo ID: $HALO_ID"

mkdir -p "$MODEL_DIR" "$SAMPLES_DIR"

# Check if already completed
if [[ -f "$MODEL_DIR/model_pid${SELECTED_PID}.npz" ]] && [[ -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.npz" || -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.h5" ]]; then
    echo "✅ Already completed: Halo $HALO_ID PID $SELECTED_PID"
    echo "$(date) SUCCESS halo:$HALO_ID pid:$SELECTED_PID already_completed" >> success_logs/brute_force_success.log
    exit 0
fi

# Run training
echo "🧠 Training Halo $HALO_ID PID $SELECTED_PID..."
python train_tfp_flows.py \
    --data_path "$SELECTED_FILE" \
    --particle_pid "$SELECTED_PID" \
    --output_dir "$MODEL_DIR" \
    --epochs 100 \
    --batch_size 512 \
    --learning_rate 1e-3 \
    --n_layers 6 \
    --hidden_units 1024

TRAIN_EXIT=$?

if [[ $TRAIN_EXIT -eq 0 ]]; then
    # Check success
    if [[ -f "$MODEL_DIR/model_pid${SELECTED_PID}.npz" ]] && [[ -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.npz" || -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.h5" ]]; then
        echo "✅ SUCCESS: Halo $HALO_ID PID $SELECTED_PID"
        echo "$(date) SUCCESS halo:$HALO_ID pid:$SELECTED_PID training_completed" >> success_logs/brute_force_success.log
    else
        echo "❌ FAILED: Missing output files"
        echo "$(date) FAILED halo:$HALO_ID pid:$SELECTED_PID missing_outputs" >> failed_jobs/brute_force_failures.log
        exit 1
    fi
else
    echo "❌ FAILED: Training exit code $TRAIN_EXIT"
    echo "$(date) FAILED halo:$HALO_ID pid:$SELECTED_PID training_failed_$TRAIN_EXIT" >> failed_jobs/brute_force_failures.log
    exit 1
fi

echo "🏁 Completed: $(date)"
