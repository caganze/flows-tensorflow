#!/bin/bash

# Smart CPU Job Submission for Conditional Coupling Flows Training
# Automatically filters completed particles and submits only incomplete ones

set -e

# Default configuration
CHUNK_SIZE=50
CONCURRENT=5
PARTITION="kipac"
TIME_LIMIT="12:00:00"
MEMORY="64GB"
CPUS_PER_TASK=8
VERBOSE=false
DRY_RUN=false

# Base training parameters (will be scaled based on particle size)
# These are the proven working parameters from your successful run
BASE_EPOCHS=200
BASE_BATCH_SIZE=256
N_LAYERS=8
HIDDEN_UNITS=256
BASE_LEARNING_RATE=1e-4
N_MASS_BINS=200
EMBEDDING_DIM=16

# Function to scale parameters based on particle size
scale_parameters_by_size() {
    local particle_count=$1
    local size_category=$2
    
    # Define scaling factors based on size category
    case $size_category in
        "Large")
            LR_FACTOR=0.5
            BATCH_FACTOR=2.0
            EPOCH_FACTOR=0.8
            ;;
        "Medium-Large")
            LR_FACTOR=0.7
            BATCH_FACTOR=1.5
            EPOCH_FACTOR=0.9
            ;;
        "Medium")
            LR_FACTOR=1.0
            BATCH_FACTOR=1.0
            EPOCH_FACTOR=1.0
            ;;
        "Small")
            LR_FACTOR=1.5
            BATCH_FACTOR=0.5
            EPOCH_FACTOR=1.2
            ;;
        *)
            # Default to Medium if category not found
            LR_FACTOR=1.0
            BATCH_FACTOR=1.0
            EPOCH_FACTOR=1.0
            ;;
    esac
    
    # Additional scaling based on particle count
    if [ $particle_count -gt 100000 ]; then
        COUNT_FACTOR=0.8  # Large datasets need more conservative parameters
    elif [ $particle_count -gt 50000 ]; then
        COUNT_FACTOR=0.9
    elif [ $particle_count -gt 10000 ]; then
        COUNT_FACTOR=1.0
    else
        COUNT_FACTOR=1.1  # Small datasets can use more aggressive parameters
    fi
    
    # Calculate scaled parameters
    EPOCHS=$(echo "$BASE_EPOCHS * $EPOCH_FACTOR * $COUNT_FACTOR" | bc -l | cut -d. -f1)
    BATCH_SIZE=$(echo "$BASE_BATCH_SIZE * $BATCH_FACTOR * $COUNT_FACTOR" | bc -l | cut -d. -f1)
    LEARNING_RATE=$(echo "$BASE_LEARNING_RATE * $LR_FACTOR * $COUNT_FACTOR" | bc -l)
    
    # Ensure minimum values
    [ $EPOCHS -lt 50 ] && EPOCHS=50
    [ $BATCH_SIZE -lt 32 ] && BATCH_SIZE=32
    [ $BATCH_SIZE -gt 1024 ] && BATCH_SIZE=1024
    
    echo "📊 Scaled parameters for $size_category (count: $particle_count):"
    echo "   Epochs: $EPOCHS (base: $BASE_EPOCHS)"
    echo "   Batch size: $BATCH_SIZE (base: $BASE_BATCH_SIZE)"
    echo "   Learning rate: $LEARNING_RATE (base: $BASE_LEARNING_RATE)"
}

# Output directory
OUTPUT_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/coupling_output"

show_usage() {
    echo "🚀 Smart CPU Submission for Conditional Coupling Flows"
    echo "======================================================"
    echo ""
    echo "USAGE:"
    echo "  $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --chunk-size N        Particles per chunk (default: $CHUNK_SIZE)"
    echo "  --concurrent N        Max concurrent jobs per chunk (default: $CONCURRENT)"
    echo "  --partition NAME      SLURM partition (default: $PARTITION)"
    echo "  --time-limit TIME     Time limit per job (default: $TIME_LIMIT)"
    echo "  --memory SIZE         Memory per job (default: $MEMORY)"
    echo "  --cpus N              CPUs per job (default: $CPUS_PER_TASK)"
    echo "  --epochs N            Training epochs (default: $EPOCHS)"
    echo "  --learning-rate RATE  Learning rate (default: $LEARNING_RATE)"
    echo "  --batch-size N        Batch size (default: $BATCH_SIZE)"
    echo "  --n-mass-bins N       Number of mass bins (default: $N_MASS_BINS)"
    echo "  --output-dir DIR      Output directory (default: $OUTPUT_DIR)"
    echo "  --verbose             Show detailed output"
    echo "  --dry-run             Preview without submitting"
    echo "  --help                Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                                    # Smart submission with defaults"
    echo "  $0 --chunk-size 100 --concurrent 3   # Larger chunks, fewer concurrent"
    echo "  $0 --epochs 200 --learning-rate 1e-3 # Custom training parameters"
    echo "  $0 --verbose --dry-run               # Preview with detailed info"
    echo ""
    echo "PROCESS:"
    echo "  1. Scans for completed coupling flow models"
    echo "  2. Creates filtered list of incomplete particles"
    echo "  3. Submits in manageable chunks to avoid QOS limits"
    echo "  4. Uses CPU-optimized training with proper resource allocation"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --concurrent)
            CONCURRENT="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --time-limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --cpus)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --n-mass-bins)
            N_MASS_BINS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

echo "🚀 SMART CPU SUBMISSION FOR CONDITIONAL COUPLING FLOWS"
echo "======================================================"
echo ""
echo "📋 Configuration:"
echo "   Chunk size: $CHUNK_SIZE particles"
echo "   Concurrent: $CONCURRENT jobs"
echo "   Partition: $PARTITION"
echo "   Time limit: $TIME_LIMIT"
echo "   Memory: $MEMORY"
echo "   CPUs: $CPUS_PER_TASK"
echo "   Epochs: $EPOCHS"
echo "   Learning rate: $LEARNING_RATE"
echo "   Batch size: $BATCH_SIZE"
echo "   Mass bins: $N_MASS_BINS"
echo "   Output dir: $OUTPUT_DIR"
echo ""

# Check required files
required_files=(
    "train_coupling_flows_conditional.py"
    "filter_completed_coupling.sh"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Required file missing: $file"
        echo "💡 Make sure you're in the flows-tensorflow directory"
        exit 1
    fi
done

echo "✅ All required files found"

# Create filter script if it doesn't exist
if [[ ! -f "filter_completed_coupling.sh" ]]; then
    echo "📝 Creating filter script for coupling flows..."
    cat > filter_completed_coupling.sh << 'EOF'
#!/bin/bash

# Filter completed coupling flow models
# Looks for successful model files in the output directory

set -e

OUTPUT_DIR="${OUTPUT_DIR:-/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/coupling_output}"
PARTICLE_LIST_FILE="${PARTICLE_LIST_FILE:-particle_list.txt}"
VERBOSE="${VERBOSE:-false}"
DRY_RUN="${DRY_RUN:-false}"

echo "🔍 Filtering completed coupling flow models..."
echo "   Output directory: $OUTPUT_DIR"
echo "   Particle list: $PARTICLE_LIST_FILE"

if [[ ! -f "$PARTICLE_LIST_FILE" ]]; then
    echo "❌ Particle list not found: $PARTICLE_LIST_FILE"
    echo "💡 Run ./generate_all_priority_halos.sh first"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create incomplete particle list
INCOMPLETE_FILE="particle_list_coupling_incomplete.txt"
> "$INCOMPLETE_FILE"

TOTAL_PARTICLES=0
COMPLETED_PARTICLES=0
INCOMPLETE_PARTICLES=0

while IFS=',' read -r pid halo_id suite object_count size_category; do
    TOTAL_PARTICLES=$((TOTAL_PARTICLES + 1))
    
    # Check for completed model files (organized by suite)
    model_dir="$OUTPUT_DIR/${suite}/${halo_id,,}/pid${pid}"
    weights_file="$model_dir/flow_weights_${halo_id}_${pid}"
    config_file="$model_dir/flow_config_${halo_id}_${pid}.pkl"
    preprocessing_file="$model_dir/coupling_flow_pid${pid}_preprocessing.npz"
    results_file="$model_dir/coupling_flow_pid${pid}_results.json"
    
    # Check if all required files exist (checkpoint files have .index extension)
    if [[ -f "$weights_file.index" && -f "$config_file" && -f "$preprocessing_file" && -f "$results_file" ]]; then
        COMPLETED_PARTICLES=$((COMPLETED_PARTICLES + 1))
        if [[ "$VERBOSE" == "true" ]]; then
            echo "✅ Completed: $halo_id PID $pid"
        fi
    else
        INCOMPLETE_PARTICLES=$((INCOMPLETE_PARTICLES + 1))
        echo "$pid,$halo_id,$suite,$object_count,$size_category" >> "$INCOMPLETE_FILE"
        if [[ "$VERBOSE" == "true" ]]; then
            echo "❌ Incomplete: $halo_id PID $pid"
        fi
    fi
done < "$PARTICLE_LIST_FILE"

echo ""
echo "📊 Filtering Results:"
echo "   Total particles: $TOTAL_PARTICLES"
echo "   Completed: $COMPLETED_PARTICLES"
echo "   Incomplete: $INCOMPLETE_PARTICLES"
echo "   Incomplete list: $INCOMPLETE_FILE"

if [[ $INCOMPLETE_PARTICLES -eq 0 ]]; then
    echo ""
    echo "🎉 ALL COUPLING FLOW MODELS COMPLETED!"
    echo "======================================"
    echo "✅ No incomplete models found - all work is done!"
    exit 0
fi

echo ""
echo "📋 Next: Submit incomplete particles with submit_coupling_flows_cpu_chunked.sh"
EOF
    chmod +x filter_completed_coupling.sh
    echo "✅ Filter script created"
fi

# Step 1: Filter completed particles
echo ""
echo "🔍 STEP 1: Filtering completed coupling flow models"
echo "==================================================="

FILTER_ARGS="--output-dir $OUTPUT_DIR"
if [[ "$VERBOSE" == "true" ]]; then
    FILTER_ARGS="$FILTER_ARGS --verbose"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    FILTER_ARGS="$FILTER_ARGS --dry-run"
fi

if ! ./filter_completed_coupling.sh $FILTER_ARGS; then
    echo "❌ Failed to filter particles"
    exit 1
fi

# Check if we have any incomplete particles (only if not dry run)
if [[ "$DRY_RUN" != "true" ]]; then
    if [[ ! -f "particle_list_coupling_incomplete.txt" || ! -s "particle_list_coupling_incomplete.txt" ]]; then
        echo ""
        echo "🎉 ALL COUPLING FLOW MODELS COMPLETED!"
        echo "======================================"
        echo "✅ No incomplete models found - all work is done!"
        exit 0
    fi
    
    INCOMPLETE_COUNT=$(wc -l < particle_list_coupling_incomplete.txt)
    echo ""
    echo "📊 Found $INCOMPLETE_COUNT incomplete coupling flow models to train"
fi

# Create chunked submission script if it doesn't exist
if [[ ! -f "submit_coupling_flows_cpu_chunked.sh" ]]; then
    echo "📝 Creating chunked submission script..."
    cat > submit_coupling_flows_cpu_chunked.sh << EOF
#!/bin/bash

# Chunked CPU submission for coupling flows
# Submits particles in manageable chunks

set -e

# Default configuration
CHUNK_SIZE=50
CONCURRENT=5
PARTITION="kipac"
TIME_LIMIT="12:00:00"
MEMORY="64GB"
CPUS_PER_TASK=8
PARTICLE_LIST="particle_list_coupling_incomplete.txt"
OUTPUT_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/coupling_output"
EPOCHS=150
LEARNING_RATE=5e-3
BATCH_SIZE=256
N_MASS_BINS=100
DRY_RUN=false

# Parse arguments
while [[ \$# -gt 0 ]]; do
    case \$1 in
        --chunk-size)
            CHUNK_SIZE="\$2"
            shift 2
            ;;
        --concurrent)
            CONCURRENT="\$2"
            shift 2
            ;;
        --partition)
            PARTITION="\$2"
            shift 2
            ;;
        --time-limit)
            TIME_LIMIT="\$2"
            shift 2
            ;;
        --memory)
            MEMORY="\$2"
            shift 2
            ;;
        --cpus)
            CPUS_PER_TASK="\$2"
            shift 2
            ;;
        --particle-list)
            PARTICLE_LIST="\$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="\$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="\$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="\$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="\$2"
            shift 2
            ;;
        --n-mass-bins)
            N_MASS_BINS="\$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "❌ Unknown option: \$1"
            exit 1
            ;;
    esac
done

# Check particle list
if [[ ! -f "\$PARTICLE_LIST" ]]; then
    echo "❌ Particle list not found: \$PARTICLE_LIST"
    exit 1
fi

TOTAL_PARTICLES=\$(wc -l < "\$PARTICLE_LIST")
TOTAL_CHUNKS=\$(( (TOTAL_PARTICLES + CHUNK_SIZE - 1) / CHUNK_SIZE ))

echo "🚀 COUPLING FLOWS CPU CHUNKED SUBMISSION"
echo "========================================"
echo "📋 Particle list: \$PARTICLE_LIST"
echo "📊 Total particles: \$TOTAL_PARTICLES"
echo "📦 Chunk size: \$CHUNK_SIZE"
echo "🔢 Total chunks: \$TOTAL_CHUNKS"
echo "🎛️  Concurrent: \$CONCURRENT"
echo "⏰ Time limit: \$TIME_LIMIT"
echo "💾 Memory: \$MEMORY"
echo "🧮 CPUs: \$CPUS_PER_TASK"
echo "🎯 Epochs: \$EPOCHS"
echo "📈 Learning rate: \$LEARNING_RATE"
echo "📦 Batch size: \$BATCH_SIZE"
echo "📊 Mass bins: \$N_MASS_BINS"
echo "📁 Output: \$OUTPUT_DIR"
echo ""

if [[ "\$DRY_RUN" == "true" ]]; then
    echo "🧪 DRY RUN - Commands that would be executed:"
    echo ""
fi

# Submit chunks
SUBMITTED_JOBS=()

for (( chunk=1; chunk<=TOTAL_CHUNKS; chunk++ )); do
    start_particle=\$(( (chunk - 1) * CHUNK_SIZE + 1 ))
    end_particle=\$(( chunk * CHUNK_SIZE ))
    
    if [[ \$end_particle -gt \$TOTAL_PARTICLES ]]; then
        end_particle=\$TOTAL_PARTICLES
    fi
    
    chunk_size_actual=\$((end_particle - start_particle + 1))
    
    echo "📦 Chunk \$chunk: particles \$start_particle-\$end_particle (\$chunk_size_actual particles)"
    
    if [[ "\$DRY_RUN" == "true" ]]; then
        echo "  sbatch --array=\$start_particle-\$end_particle%\$CONCURRENT --partition=\$PARTITION --time=\$TIME_LIMIT --mem=\$MEMORY --cpus-per-task=\$CPUS_PER_TASK coupling_flows_cpu_job.sh"
    else
        JOB_ID=\$(sbatch \\
            --array=\$start_particle-\$end_particle%\$CONCURRENT \\
            --partition=\$PARTITION \\
            --time=\$TIME_LIMIT \\
            --mem=\$MEMORY \\
            --cpus-per-task=\$CPUS_PER_TASK \\
            --job-name="coupling_chunk_\$chunk" \\
            --export=PARTICLE_LIST_FILE="\$PARTICLE_LIST",OUTPUT_DIR="\$OUTPUT_DIR",EPOCHS="\$EPOCHS",LEARNING_RATE="\$LEARNING_RATE",BATCH_SIZE="\$BATCH_SIZE",N_MASS_BINS="\$N_MASS_BINS" \\
            coupling_flows_cpu_job.sh | grep -o '[0-9]\+')
        
        if [[ -n "\$JOB_ID" ]]; then
            echo "  ✅ Submitted: Job ID \$JOB_ID"
            SUBMITTED_JOBS+=("\$JOB_ID")
        else
            echo "  ❌ Failed to submit chunk \$chunk"
        fi
        
        sleep 2
    fi
done

echo ""
if [[ "\$DRY_RUN" == "true" ]]; then
    echo "🧪 This was a dry run - no jobs were actually submitted"
else
    echo "✅ Chunked submission completed"
    echo "📊 Submitted \${#SUBMITTED_JOBS[@]} job chunks"
    echo ""
    echo "📊 Monitor progress:"
    echo "   squeue -u \\\$USER"
    echo "   ./monitor_coupling_flows.sh"
fi
EOF
    chmod +x submit_coupling_flows_cpu_chunked.sh
    echo "✅ Chunked submission script created"
fi

# Create CPU job script if it doesn't exist
if [[ ! -f "coupling_flows_cpu_job.sh" ]]; then
    echo "📝 Creating CPU job script..."
    cat > coupling_flows_cpu_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name="coupling_flows_cpu"
#SBATCH --partition=kipac
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/coupling_flows_cpu_%A_%a.out
#SBATCH --error=logs/coupling_flows_cpu_%A_%a.err

# CPU-optimized Conditional Coupling Flow Training

set -e

echo "🖥️  COUPLING FLOWS CPU JOB - ARRAY TASK ${SLURM_ARRAY_TASK_ID:-1}"
echo "Started: $(date)"
echo "Node: ${SLURM_NODELIST:-local}"
echo "Job ID: ${SLURM_JOB_ID:-test}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-8}"

# CPU-specific TensorFlow optimizations
export TF_CPP_MIN_LOG_LEVEL=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TF_NUM_INTEROP_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Force CPU-only mode
export CUDA_VISIBLE_DEVICES=""

source ~/.bashrc
conda activate bosque

echo "✅ Environment loaded (CPU-optimized with ${OMP_NUM_THREADS} threads)"

# Create directories
mkdir -p logs success_logs failed_jobs

# Use particle list for array processing
PARTICLE_LIST_FILE="${PARTICLE_LIST_FILE:-particle_list_coupling_incomplete.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/coupling_output}"
BASE_EPOCHS="${EPOCHS:-200}"
BASE_LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BASE_BATCH_SIZE="${BATCH_SIZE:-256}"
N_MASS_BINS="${N_MASS_BINS:-200}"

# Function to scale parameters based on particle size
scale_parameters_by_size() {
    local particle_count=$1
    local size_category=$2
    
    # Define scaling factors based on size category
    case $size_category in
        "Large")
            LR_FACTOR=0.5
            BATCH_FACTOR=2.0
            EPOCH_FACTOR=0.8
            ;;
        "Medium-Large")
            LR_FACTOR=0.7
            BATCH_FACTOR=1.5
            EPOCH_FACTOR=0.9
            ;;
        "Medium")
            LR_FACTOR=1.0
            BATCH_FACTOR=1.0
            EPOCH_FACTOR=1.0
            ;;
        "Small")
            LR_FACTOR=1.5
            BATCH_FACTOR=0.5
            EPOCH_FACTOR=1.2
            ;;
        *)
            # Default to Medium if category not found
            LR_FACTOR=1.0
            BATCH_FACTOR=1.0
            EPOCH_FACTOR=1.0
            ;;
    esac
    
    # Additional scaling based on particle count
    if [ $particle_count -gt 100000 ]; then
        COUNT_FACTOR=0.8  # Large datasets need more conservative parameters
    elif [ $particle_count -gt 50000 ]; then
        COUNT_FACTOR=0.9
    elif [ $particle_count -gt 10000 ]; then
        COUNT_FACTOR=1.0
    else
        COUNT_FACTOR=1.1  # Small datasets can use more aggressive parameters
    fi
    
    # Calculate scaled parameters
    EPOCHS=$(echo "$BASE_EPOCHS * $EPOCH_FACTOR * $COUNT_FACTOR" | bc -l | cut -d. -f1)
    BATCH_SIZE=$(echo "$BASE_BATCH_SIZE * $BATCH_FACTOR * $COUNT_FACTOR" | bc -l | cut -d. -f1)
    LEARNING_RATE=$(echo "$BASE_LEARNING_RATE * $LR_FACTOR * $COUNT_FACTOR" | bc -l)
    
    # Ensure minimum values
    [ $EPOCHS -lt 50 ] && EPOCHS=50
    [ $BATCH_SIZE -lt 32 ] && BATCH_SIZE=32
    [ $BATCH_SIZE -gt 1024 ] && BATCH_SIZE=1024
    
    echo "📊 Scaled parameters for $size_category (count: $particle_count):"
    echo "   Epochs: $EPOCHS (base: $BASE_EPOCHS)"
    echo "   Batch size: $BATCH_SIZE (base: $BASE_BATCH_SIZE)"
    echo "   Learning rate: $LEARNING_RATE (base: $BASE_LEARNING_RATE)"
}

echo "📋 Using particle list: $PARTICLE_LIST_FILE"
echo "📁 Output directory: $OUTPUT_DIR"
echo "🎯 Training epochs: $EPOCHS"
echo "📈 Learning rate: $LEARNING_RATE"
echo "📦 Batch size: $BATCH_SIZE"
echo "📊 Mass bins: $N_MASS_BINS"

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

# Parse particle list line: PID,HALO_ID,SUITE,OBJECT_COUNT,SIZE_CATEGORY
IFS=',' read -r SELECTED_PID HALO_ID SUITE OBJECT_COUNT SIZE_CATEGORY <<< "$PARTICLE_LINE"

echo "🎯 Processing: PID $SELECTED_PID from $HALO_ID (suite: $SUITE)"
echo "   Objects: $OBJECT_COUNT ($SIZE_CATEGORY)"

# Scale training parameters based on particle size
echo ""
echo "🔧 Scaling training parameters based on particle size..."
scale_parameters_by_size $OBJECT_COUNT "$SIZE_CATEGORY"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Training parameters (architecture remains constant)
N_LAYERS=8
HIDDEN_UNITS=256
EMBEDDING_DIM=16

echo ""
echo "🚀 Starting Conditional Coupling Flow Training"
echo "=============================================="
echo "📋 Training parameters:"
echo "   Epochs: $EPOCHS"
echo "   Learning rate: $LEARNING_RATE"
echo "   Layers: $N_LAYERS"
echo "   Hidden units: $HIDDEN_UNITS"
echo "   Mass bins: $N_MASS_BINS"
echo "   Embedding dim: $EMBEDDING_DIM"
echo ""

# Run training
cd /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow

python train_coupling_flows_conditional.py \
    --halo_id "$HALO_ID" \
    --particle_pid "$SELECTED_PID" \
    --suite "$SUITE" \
    --n_layers "$N_LAYERS" \
    --hidden_units "$HIDDEN_UNITS" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --n_mass_bins "$N_MASS_BINS" \
    --embedding_dim "$EMBEDDING_DIM" \
    --output_dir "$OUTPUT_DIR"

TRAINING_EXIT_CODE=$?

echo ""
echo "📋 Training completed at: $(date)"
echo "   Exit code: $TRAINING_EXIT_CODE"

if [[ $TRAINING_EXIT_CODE -eq 0 ]]; then
    echo "✅ SUCCESS: Coupling flow training completed for $HALO_ID PID $SELECTED_PID"
    echo "$PARTICLE_LINE" >> success_logs/coupling_flows_success_$(date +%Y%m%d).log
else
    echo "❌ FAILED: Coupling flow training failed for $HALO_ID PID $SELECTED_PID"
    echo "$PARTICLE_LINE" >> failed_jobs/coupling_flows_failed_$(date +%Y%m%d).log
    exit $TRAINING_EXIT_CODE
fi
EOF
    chmod +x coupling_flows_cpu_job.sh
    echo "✅ CPU job script created"
fi

# Step 2: Submit in chunks
echo ""
echo "🚀 STEP 2: Submitting coupling flow training jobs"
echo "================================================="

SUBMIT_ARGS="--chunk-size $CHUNK_SIZE --concurrent $CONCURRENT --partition $PARTITION --time-limit $TIME_LIMIT --memory $MEMORY --cpus $CPUS_PER_TASK --particle-list particle_list_coupling_incomplete.txt --output-dir $OUTPUT_DIR --epochs $EPOCHS --learning-rate $LEARNING_RATE --batch-size $BATCH_SIZE --n-mass-bins $N_MASS_BINS"

if [[ "$DRY_RUN" == "true" ]]; then
    SUBMIT_ARGS="$SUBMIT_ARGS --dry-run"
    echo "🧪 DRY RUN MODE - Preview of submission:"
    echo ""
fi

if ! ./submit_coupling_flows_cpu_chunked.sh $SUBMIT_ARGS; then
    echo "❌ Failed to submit chunks"
    exit 1
fi

echo ""
echo "✅ SMART COUPLING FLOWS SUBMISSION COMPLETED"
echo "============================================"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "🧪 This was a dry run - no jobs were actually submitted"
    echo "💡 Run without --dry-run to actually submit jobs"
else
    echo "🎯 Only incomplete coupling flow models were submitted"
    echo "🔄 Jobs should start running soon (CPU queue is typically active)"
    echo ""
    echo "📊 Monitor progress:"
    echo "   squeue -u \$USER"
    echo "   ./monitor_coupling_flows.sh"
    echo ""
    echo "🔍 Check job status:"
    echo "   squeue -u \$USER --format=\"%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R\""
    echo ""
    echo "📁 Output files will be saved to: $OUTPUT_DIR"
fi

