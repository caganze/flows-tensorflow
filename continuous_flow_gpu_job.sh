#!/bin/bash
#SBATCH --job-name="continuous_flow_particle_list"
#SBATCH --partition=owners
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/continuous_flow_%A_%a.out
#SBATCH --error=logs/continuous_flow_%A_%a.err
#SBATCH --array=1-22713%10

set -e

echo "🚀 CONTINUOUS FLOW GPU JOB - ARRAY TASK ${SLURM_ARRAY_TASK_ID:-1}"
echo "🔧 Mode: Conditional Continuous Flow Training"
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

# Use particle list for array processing
PARTICLE_LIST_FILE="${PARTICLE_LIST_FILE:-particle_list.txt}"
echo "📋 Using particle list: $PARTICLE_LIST_FILE"

if [[ ! -f "$PARTICLE_LIST_FILE" ]]; then
    echo "❌ Particle list not found: $PARTICLE_LIST_FILE"
    exit 1
fi

# Get total number of particles
TOTAL_PARTICLES=$(wc -l < "$PARTICLE_LIST_FILE")
echo "📊 Found $TOTAL_PARTICLES particles in list"

# Get array task ID
ARRAY_ID=${SLURM_ARRAY_TASK_ID:-1}

# Check bounds
if [[ $ARRAY_ID -gt $TOTAL_PARTICLES ]]; then
    echo "Array task $ARRAY_ID exceeds available particles ($TOTAL_PARTICLES)"
    exit 0
fi

# Get the specific line for this array task
PARTICLE_LINE=$(sed -n "${ARRAY_ID}p" "$PARTICLE_LIST_FILE")

if [[ -z "$PARTICLE_LINE" ]]; then
    echo "❌ No particle found for array task $ARRAY_ID"
    exit 1
fi

# Parse particle list line: PID,HALO_ID,SUITE,OBJECT_COUNT,SIZE_CATEGORY (symlib format)
IFS=',' read -r SELECTED_PID HALO_ID SUITE OBJECT_COUNT SIZE_CATEGORY <<< "$PARTICLE_LINE"

echo "🎯 Processing: PID $SELECTED_PID from $HALO_ID (suite: $SUITE)"
echo "   Objects: $OBJECT_COUNT ($SIZE_CATEGORY)"

# Output directories - use conditional continuous flow directory
OUTPUT_BASE_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output_conditional"
MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/${SUITE}/halo${HALO_ID#Halo}"
SAMPLES_DIR="$OUTPUT_BASE_DIR/samples/${SUITE}/halo${HALO_ID#Halo}"

echo "📁 Output base: $OUTPUT_BASE_DIR"
echo "📁 Suite: $SUITE, Halo ID: $HALO_ID"

mkdir -p "$MODEL_DIR" "$SAMPLES_DIR"

# Check if already completed
if [[ -f "$MODEL_DIR/model_pid${SELECTED_PID}.npz" ]] && [[ -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.npz" || -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.h5" ]]; then
    echo "✅ Already completed: Halo $HALO_ID PID $SELECTED_PID"
    echo "$(date) SUCCESS halo:$HALO_ID pid:$SELECTED_PID already_completed" >> success_logs/continuous_flow_success.log
    exit 0
fi

# Determine optimal parameters based on particle size
if [[ $OBJECT_COUNT -gt 100000 ]]; then
    # Large particles (100k+): Need capacity with low learning rate
    EPOCHS=60
    BATCH_SIZE=512
    N_LAYERS=6
    HIDDEN_UNITS=512
    LEARNING_RATE=1e-4
    echo "🧠 Training Large particle (>100k): epochs=60, layers=6, units=512, lr=1e-4"
elif [[ $OBJECT_COUNT -gt 50000 ]]; then
    # Medium-large particles
    EPOCHS=50
    BATCH_SIZE=512
    N_LAYERS=5
    HIDDEN_UNITS=384
    LEARNING_RATE=2e-4
    echo "🧠 Training Medium-large (50k-100k): epochs=50, layers=5, units=384, lr=2e-4"
elif [[ $OBJECT_COUNT -lt 5000 ]]; then
    # Small particles
    EPOCHS=40
    BATCH_SIZE=256
    N_LAYERS=4
    HIDDEN_UNITS=256
    LEARNING_RATE=3e-4
    echo "🧠 Training Small particle (<5k): epochs=40, layers=4, units=256, lr=3e-4"
else
    # Medium particles
    EPOCHS=45
    BATCH_SIZE=512
    N_LAYERS=4
    HIDDEN_UNITS=320
    LEARNING_RATE=2e-4
    echo "🧠 Training Medium particle (5k-50k): epochs=45, layers=4, units=320, lr=2e-4"
fi

echo "🧠 Training Conditional Continuous Flow: Halo $HALO_ID PID $SELECTED_PID..."
python train_tfp_flows_conditional.py \
    --halo_id "$HALO_ID" \
    --particle_pid "$SELECTED_PID" \
    --suite "$SUITE" \
    --output_dir "$MODEL_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --n_layers $N_LAYERS \
    --hidden_units $HIDDEN_UNITS

TRAIN_EXIT=$?

if [[ $TRAIN_EXIT -eq 0 ]]; then
    # Check success
    if [[ -f "$MODEL_DIR/model_pid${SELECTED_PID}.npz" ]] && [[ -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.npz" || -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.h5" ]]; then
        echo "✅ SUCCESS: Continuous Flow Halo $HALO_ID PID $SELECTED_PID"
        echo "$(date) SUCCESS halo:$HALO_ID pid:$SELECTED_PID training_completed" >> success_logs/continuous_flow_success.log
    else
        echo "❌ FAILED: Missing output files"
        echo "$(date) FAILED halo:$HALO_ID pid:$SELECTED_PID missing_outputs" >> failed_jobs/continuous_flow_failures.log
        exit 1
    fi
else
    echo "❌ FAILED: Training exit code $TRAIN_EXIT"
    echo "$(date) FAILED halo:$HALO_ID pid:$SELECTED_PID training_failed_$TRAIN_EXIT" >> failed_jobs/continuous_flow_failures.log
    exit 1
fi

echo "🏁 Completed: $(date)"
