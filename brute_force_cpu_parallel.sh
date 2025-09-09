#!/bin/bash
#SBATCH --job-name="cpu_flows_parallel"
#SBATCH --partition=kipac
#SBATCH --time=24:00:00
#SBATCH --mem=512GB
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

# Load required modules (compatible versions) - from your working script
module purge
module load math
module load devel
module load python/3.9.0

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
        echo "❌ Symlib particle list required. Run: ./generate_all_priority_halos.sh"
        exit 1
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

# Parse particle entry: PID,HALO_ID,SUITE,OBJECT_COUNT,SIZE_CATEGORY (symlib format)
IFS=',' read -r SELECTED_PID HALO_ID SUITE OBJECT_COUNT SIZE_CATEGORY <<< "$PARTICLE_ENTRY"

echo "🎯 Processing: PID $SELECTED_PID from $HALO_ID (suite: $SUITE)"
echo "   Objects: $OBJECT_COUNT ($SIZE_CATEGORY)"

# Output directories - symlib structure
OUTPUT_BASE_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output"
MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/${SUITE}/halo${HALO_ID#Halo}"
SAMPLES_DIR="$OUTPUT_BASE_DIR/samples/${SUITE}/halo${HALO_ID#Halo}"

echo "📁 Output base: $OUTPUT_BASE_DIR"
echo "📁 Data source: $DATA_SOURCE, Halo ID: $HALO_ID"

mkdir -p "$MODEL_DIR" "$SAMPLES_DIR"

# Check if already completed
if [[ -f "$MODEL_DIR/model_pid${SELECTED_PID}.npz" ]] && [[ -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.npz"  ]]; then
    echo "✅ Already completed: Halo $HALO_ID PID $SELECTED_PID"
    echo "$(date) SUCCESS halo:$HALO_ID pid:$SELECTED_PID already_completed" >> success_logs/cpu_flows_success.log
    exit 0
fi

# SOPHISTICATED anti-overfitting parameters based on particle size
if [[ $OBJECT_COUNT -gt 100000 ]]; then
    # Large particles (100k+): Need capacity but with strong regularization
    EPOCHS=50
    BATCH_SIZE=768
    N_LAYERS=4
    HIDDEN_UNITS=512
    LEARNING_RATE=2e-4
    echo "🐋 Large particle (>100k): epochs=50, layers=4, units=512, lr=2e-4"
elif [[ $OBJECT_COUNT -gt 50000 ]]; then
    # Medium-large particles: Moderate capacity  
    EPOCHS=40
    BATCH_SIZE=512
    N_LAYERS=3
    HIDDEN_UNITS=384
    LEARNING_RATE=3e-4
    echo "🐟 Medium-large (50k-100k): epochs=40, layers=3, units=384, lr=3e-4"
elif [[ $OBJECT_COUNT -lt 5000 ]]; then
    # Small particles: Simple models work fine
    EPOCHS=30
    BATCH_SIZE=256
    N_LAYERS=3
    HIDDEN_UNITS=256
    LEARNING_RATE=5e-4
    echo "🐭 Small particle (<5k): epochs=30, layers=3, units=256, lr=5e-4"
else
    # Medium particles: Balanced approach
    EPOCHS=35
    BATCH_SIZE=512
    N_LAYERS=3
    HIDDEN_UNITS=320
    LEARNING_RATE=4e-4
    echo "🐟 Medium particle (5k-50k): epochs=35, layers=3, units=320, lr=4e-4"
fi

# Run training
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
    --generate-samples \

TRAIN_EXIT=$?

if [[ $TRAIN_EXIT -eq 0 ]]; then
    # Check success
    if [[ -f "$MODEL_DIR/model_pid${SELECTED_PID}.npz" ]] && [[ -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.npz"  ]]; then
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
