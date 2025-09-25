#!/bin/bash
#SBATCH --job-name="brute_force_particle_list"
#SBATCH --partition=owners
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --output=logs/brute_force_%A_%a.out
#SBATCH --error=logs/brute_force_%A_%a.err
#SBATCH --array=1-22713%10

set -e

echo "🚀 BRUTE FORCE GPU JOB - ARRAY TASK ${SLURM_ARRAY_TASK_ID:-1}"
echo "🔧 Mode: Particle List Processing"
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

# Output directories - use consistent tfp_output directory
OUTPUT_BASE_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output"
MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/${SUITE}/halo${HALO_ID#Halo}"
SAMPLES_DIR="$OUTPUT_BASE_DIR/samples/${SUITE}/halo${HALO_ID#Halo}"

echo "📁 Output base: $OUTPUT_BASE_DIR"
echo "📁 Suite: $SUITE, Halo ID: $HALO_ID"

mkdir -p "$MODEL_DIR" "$SAMPLES_DIR"

# Check if already completed
if [[ -f "$MODEL_DIR/model_pid${SELECTED_PID}.npz" ]] && [[ -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.npz" || -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.h5" ]]; then
    echo "✅ Already completed: Halo $HALO_ID PID $SELECTED_PID"
    echo "$(date) SUCCESS halo:$HALO_ID pid:$SELECTED_PID already_completed" >> success_logs/brute_force_success.log
    exit 0
fi


# Determine optimal parameters based on particle size (with stability options)
VALID_FREQ=5
EARLY_STOP=25
REDUCE_PAT=12
CLIP_OUT=3.0
WEIGHT_DECAY=2e-5
NOISE_STD=0.005
USE_BN_FLAG="--use_batchnorm"

if [[ "$SELECTED_PID" == "5" ]]; then
    # Special stabilization for PID 5
    EPOCHS=200
    BATCH_SIZE=1024
    N_LAYERS=8
    HIDDEN_UNITS=768
    LEARNING_RATE=1e-4
    VALID_FREQ=2
    EARLY_STOP=30
    REDUCE_PAT=15
    WEIGHT_DECAY=1e-5
    NOISE_STD=0.0
    USE_BN_FLAG=""  # disable batchnorm for stability
    echo "🧠 PID 5 stabilization: epochs=$EPOCHS, layers=$N_LAYERS, units=$HIDDEN_UNITS, lr=$LEARNING_RATE, BN=off"
elif [[ $OBJECT_COUNT -gt 100000 ]]; then
    # Large particles (100k+): lower LR, enable regularization, BN off
    EPOCHS=60
    BATCH_SIZE=512
    N_LAYERS=4
    HIDDEN_UNITS=512
    LEARNING_RATE=2e-4
    EARLY_STOP=30
    REDUCE_PAT=15
    WEIGHT_DECAY=1e-5
    NOISE_STD=0.0
    USE_BN_FLAG=""  # BN off for very large
    echo "🧠 Large (>100k): epochs=$EPOCHS, layers=$N_LAYERS, units=$HIDDEN_UNITS, lr=$LEARNING_RATE, BN=off"
elif [[ $OBJECT_COUNT -gt 50000 ]]; then
    # Medium-large particles
    EPOCHS=45
    BATCH_SIZE=512
    N_LAYERS=3
    HIDDEN_UNITS=384
    LEARNING_RATE=3e-4
    echo "🧠 Medium-large (50k-100k): epochs=$EPOCHS, layers=$N_LAYERS, units=$HIDDEN_UNITS, lr=$LEARNING_RATE"
elif [[ $OBJECT_COUNT -lt 5000 ]]; then
    # Small particles
    EPOCHS=35
    BATCH_SIZE=256
    N_LAYERS=3
    HIDDEN_UNITS=256
    LEARNING_RATE=5e-4
    WEIGHT_DECAY=1e-5
    NOISE_STD=0.0
    echo "🧠 Small (<5k): epochs=$EPOCHS, layers=$N_LAYERS, units=$HIDDEN_UNITS, lr=$LEARNING_RATE"
else
    # Medium particles
    EPOCHS=40
    BATCH_SIZE=512
    N_LAYERS=3
    HIDDEN_UNITS=320
    LEARNING_RATE=4e-4
    echo "🧠 Medium (5k-50k): epochs=$EPOCHS, layers=$N_LAYERS, units=$HIDDEN_UNITS, lr=$LEARNING_RATE"
fi

echo "🧠 Training Halo $HALO_ID PID $SELECTED_PID..."
    python train_tfp_flows.py \
        --halo_id "$HALO_ID" \
        --particle_pid "$SELECTED_PID" \
        --suite "$SUITE" \
        --output_dir "$MODEL_DIR" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --n_layers $N_LAYERS \
        --hidden_units $HIDDEN_UNITS \
        --validation_freq $VALID_FREQ \
        --early_stopping_patience $EARLY_STOP \
        --reduce_lr_patience $REDUCE_PAT \
        --clip_outliers $CLIP_OUT \
        --weight_decay $WEIGHT_DECAY \
        --noise_std $NOISE_STD \
        $USE_BN_FLAG \
        --generate-samples \

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