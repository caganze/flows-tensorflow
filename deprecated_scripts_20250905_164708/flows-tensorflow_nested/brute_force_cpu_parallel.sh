#!/bin/bash
#SBATCH --job-name="cpu_flows_parallel"
#SBATCH --partition=kipac
#SBATCH --time=12:00:00
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/cpu_flows_%A_%a.out
#SBATCH --error=logs/cpu_flows_%A_%a.err
#SBATCH --array=1-100%10

# CPU-optimized TensorFlow Probability Flow Training
# Similar to GPU version but optimized for CPU resources

set -e

echo "🖥️  CPU PARALLEL FLOWS - ARRAY TASK ${SLURM_ARRAY_TASK_ID:-1}"
echo "Started: $(date)"
echo "Node: ${SLURM_NODELIST:-local}"
echo "Job ID: ${SLURM_JOB_ID:-test}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-64}"

# Environment setup - CPU optimized
module --force purge
module load math devel python/3.9.0

# CPU-specific TensorFlow optimizations
export TF_CPP_MIN_LOG_LEVEL=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export TF_NUM_INTEROP_THREADS=${SLURM_CPUS_PER_TASK:-64}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-64}

# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""

source ~/.bashrc
conda activate bosque

echo "✅ Environment loaded (CPU-optimized with ${OMP_NUM_THREADS} threads)"

# Create directories
mkdir -p logs success_logs failed_jobs

# Use custom particle list file if provided via environment variable
PARTICLE_LIST_FILE=${PARTICLE_LIST_FILE:-"particle_list.txt"}

# Generate or use existing particle list
if [[ ! -f "$PARTICLE_LIST_FILE" ]]; then
    if [[ "$PARTICLE_LIST_FILE" == "particle_list.txt" ]]; then
        echo "📋 Generating particle list..."
        ./generate_particle_list.sh
    else
        echo "❌ Custom particle list file not found: $PARTICLE_LIST_FILE"
        exit 1
    fi
fi

if [[ ! -f "$PARTICLE_LIST_FILE" || ! -s "$PARTICLE_LIST_FILE" ]]; then
    echo "❌ No particle list found or empty: $PARTICLE_LIST_FILE"
    exit 1
fi

echo "📋 Using particle list: $PARTICLE_LIST_FILE"

# Read particle list and calculate array task assignment
mapfile -t PARTICLE_ENTRIES < "$PARTICLE_LIST_FILE"
TOTAL_PARTICLES=${#PARTICLE_ENTRIES[@]}

echo "📊 Found $TOTAL_PARTICLES particles in list"

ARRAY_ID=${SLURM_ARRAY_TASK_ID:-1}

# Check bounds
if [[ $ARRAY_ID -gt $TOTAL_PARTICLES ]]; then
    echo "Array task $ARRAY_ID exceeds available particles ($TOTAL_PARTICLES)"
    exit 0
fi

# Get particle entry for this array task
PARTICLE_ENTRY="${PARTICLE_ENTRIES[$((ARRAY_ID-1))]}"

# Parse particle entry: PID,H5_FILE,OBJECT_COUNT,SIZE_CATEGORY
IFS=',' read -r SELECTED_PID H5_FILE OBJECT_COUNT SIZE_CATEGORY <<< "$PARTICLE_ENTRY"

echo "🎯 Processing: PID $SELECTED_PID from $(basename $H5_FILE)"
echo "   Objects: $OBJECT_COUNT ($SIZE_CATEGORY)"

# Extract halo ID and data source from filename
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

# Handle fallback/non-standard files
if [[ "$FILENAME" == "all_in_one.h5" ]] || [[ "$HALO_ID" == "$FILENAME" ]]; then
    echo "⚠️  Non-standard filename detected, using fallback structure"
    HALO_ID="000"
    if [[ "$DATA_SOURCE" == "unknown" ]]; then
        DATA_SOURCE="symphony"
    fi
fi

echo "🎯 Processing: $(basename $H5_FILE) PID $SELECTED_PID"
echo "   Data source: $DATA_SOURCE"
echo "   Halo ID: $HALO_ID"
echo "   Output: $DATA_SOURCE/halo$HALO_ID/"

# Output directories - save in same parent directory as H5 file with halo/PID structure
H5_PARENT_DIR=$(dirname "$H5_FILE")
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
    echo "$(date) SUCCESS halo:$HALO_ID pid:$SELECTED_PID already_completed" >> success_logs/cpu_flows_success.log
    exit 0
fi

# CPU-optimized training parameters based on particle size
if [[ "$SIZE_CATEGORY" == "Large" ]]; then
    # Large particles: optimize for CPU efficiency
    EPOCHS=80
    BATCH_SIZE=768
    N_LAYERS=4
    HIDDEN_UNITS=512
    echo "🐋 Large particle optimization: epochs=80"
elif [[ $OBJECT_COUNT -lt 5000 ]]; then
    # Very small particles: more epochs for better training
    EPOCHS=150
    BATCH_SIZE=256
    N_LAYERS=6
    HIDDEN_UNITS=1024
    echo "🐭 Small particle optimization: epochs=150"
else
    # Medium particles: balanced approach
    EPOCHS=100
    BATCH_SIZE=512
    N_LAYERS=5
    HIDDEN_UNITS=768
    echo "🐟 Medium particle optimization: epochs=100"
fi

# Run training
echo "🧠 Training Halo $HALO_ID PID $SELECTED_PID..."
python train_tfp_flows.py \
    --data_path "$H5_FILE" \
    --particle_pid "$SELECTED_PID" \
    --output_dir "$MODEL_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 1e-3 \
    --n_layers $N_LAYERS \
    --hidden_units $HIDDEN_UNITS

TRAIN_EXIT=$?

if [[ $TRAIN_EXIT -eq 0 ]]; then
    # Check success
    if [[ -f "$MODEL_DIR/model_pid${SELECTED_PID}.npz" ]] && [[ -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.npz" || -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.h5" ]]; then
        echo "✅ SUCCESS: Halo $HALO_ID PID $SELECTED_PID"
        echo "$(date) SUCCESS halo:$HALO_ID pid:$SELECTED_PID training_completed size:$SIZE_CATEGORY" >> success_logs/cpu_flows_success.log
    else
        echo "❌ FAILED: Missing output files"
        echo "$(date) FAILED halo:$HALO_ID pid:$SELECTED_PID missing_outputs size:$SIZE_CATEGORY" >> failed_jobs/cpu_flows_failures.log
        exit 1
    fi
else
    echo "❌ FAILED: Training exit code $TRAIN_EXIT"
    echo "$(date) FAILED halo:$HALO_ID pid:$SELECTED_PID training_failed_$TRAIN_EXIT size:$SIZE_CATEGORY" >> failed_jobs/cpu_flows_failures.log
    exit 1
fi

echo "🏁 Completed: $(date)"
