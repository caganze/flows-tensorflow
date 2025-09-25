#!/bin/bash
#SBATCH --job-name="kde_batch"
#SBATCH --partition=kipac
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/kde_batch_%j.out
#SBATCH --error=logs/kde_batch_%j.err

set -e

echo "🚀 KDE BATCH JOB - Job ID ${SLURM_JOB_ID:-test}"
echo "🔧 Mode: Batch KDE Processing (For Loop)"
echo "Started: $(date)"
echo "Node: ${SLURM_NODELIST:-local}"

# Get line range arguments
START_LINE=$1
END_LINE=$2

if [[ -z "$START_LINE" || -z "$END_LINE" ]]; then
    echo "❌ Usage: $0 <start_line> <end_line>"
    exit 1
fi

echo "📋 Processing lines $START_LINE to $END_LINE from particle_list_kde_incomplete.txt"

# Environment setup
module --force purge

source ~/.bashrc
conda activate bosque

# Disable TensorFlow GPU to avoid CUDA conflicts on CPU nodes
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2

# Create directories
mkdir -p logs success_logs failed_jobs

# Check if incomplete particle list exists
PARTICLE_LIST_FILE="particle_list_kde_incomplete.txt"
if [[ ! -f "$PARTICLE_LIST_FILE" ]]; then
    echo "❌ Incomplete particle list not found: $PARTICLE_LIST_FILE"
    exit 1
fi

TOTAL_LINES=$(wc -l < "$PARTICLE_LIST_FILE")
echo "📊 Total incomplete particles: $TOTAL_LINES"

if [[ $END_LINE -gt $TOTAL_LINES ]]; then
    END_LINE=$TOTAL_LINES
    echo "🔧 Adjusted end line to $END_LINE (file limit)"
fi

ACTUAL_PARTICLES=$((END_LINE - START_LINE + 1))
echo "🎯 Will process $ACTUAL_PARTICLES particles"
echo ""

# Process particles in for loop
PROCESSED=0
SUCCEEDED=0
FAILED=0

for (( line_num=START_LINE; line_num<=END_LINE; line_num++ )); do
    # Get the specific line
    PARTICLE_LINE=$(sed -n "${line_num}p" "$PARTICLE_LIST_FILE")
    
    if [[ -z "$PARTICLE_LINE" ]]; then
        echo "⚠️  Empty line $line_num, skipping"
        continue
    fi
    
    # Parse particle list line: PID,HALO_ID,SUITE,OBJECT_COUNT,SIZE_CATEGORY
    IFS=',' read -r SELECTED_PID HALO_ID SUITE OBJECT_COUNT SIZE_CATEGORY <<< "$PARTICLE_LINE"
    
    echo "🎯 Processing [$((++PROCESSED))/$ACTUAL_PARTICLES]: PID $SELECTED_PID from $HALO_ID (suite: $SUITE, objects: $OBJECT_COUNT)"
    
    # Output directories
    OUTPUT_BASE_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/kde_output"
    SAMPLES_DIR="$OUTPUT_BASE_DIR/kde_samples/${SUITE}/halo${HALO_ID#Halo}"
    
    mkdir -p "$SAMPLES_DIR"
    
    # Check if already completed (safety check)
    if [[ -f "$SAMPLES_DIR/kde_samples_${HALO_ID}_pid${SELECTED_PID}.h5" ]]; then
        echo "  ✅ Already completed: $HALO_ID PID $SELECTED_PID"
        echo "$(date) SUCCESS halo:$HALO_ID pid:$SELECTED_PID already_completed_batch" >> success_logs/kde_success.log
        SUCCEEDED=$((SUCCEEDED + 1))
        continue
    fi
    
    # Determine parameters based on particle size
    if [[ $OBJECT_COUNT -gt 100000 ]]; then
        N_NEIGHBORS=64
        SAMPLE_FRACTION=0.3
        MASS_RANGE_MIN=0.08
        MASS_RANGE_MAX=100
        echo "  🧠 Large particle: neighbors=64, fraction=0.3"
    elif [[ $OBJECT_COUNT -gt 50000 ]]; then
        N_NEIGHBORS=64
        SAMPLE_FRACTION=0.5
        MASS_RANGE_MIN=0.08
        MASS_RANGE_MAX=100
        echo "  🧠 Medium-large: neighbors=64, fraction=0.5"
    elif [[ $OBJECT_COUNT -lt 5000 ]]; then
        N_NEIGHBORS=32
        SAMPLE_FRACTION=2.0
        MASS_RANGE_MIN=0.08
        MASS_RANGE_MAX=50
        echo "  🧠 Small particle: neighbors=32, fraction=2.0"
    else
        N_NEIGHBORS=48
        SAMPLE_FRACTION=1.0
        MASS_RANGE_MIN=0.08
        MASS_RANGE_MAX=100
        echo "  🧠 Medium particle: neighbors=48, fraction=1.0"
    fi
    
    # Run KDE training
    echo "  🚀 Running KDE sampling..."
    
    KDE_EXIT=0
    python train_kde_conditional.py \
        --halo_id "$HALO_ID" \
        --parent_id "$SELECTED_PID" \
        --suite "$SUITE" \
        --output_dir "$SAMPLES_DIR" \
        --n_neighbors $N_NEIGHBORS \
        --sample_fraction $SAMPLE_FRACTION \
        --mass_range $MASS_RANGE_MIN $MASS_RANGE_MAX \
        --seed 42 || KDE_EXIT=$?
    
    if [[ $KDE_EXIT -eq 0 ]]; then
        # Check success
        if [[ -f "$SAMPLES_DIR/kde_samples_${HALO_ID}_pid${SELECTED_PID}.h5" ]]; then
            echo "  ✅ SUCCESS: KDE $HALO_ID PID $SELECTED_PID"
            echo "$(date) SUCCESS halo:$HALO_ID pid:$SELECTED_PID kde_completed_batch" >> success_logs/kde_success.log
            SUCCEEDED=$((SUCCEEDED + 1))
        else
            echo "  ❌ FAILED: Missing output files"
            echo "$(date) FAILED halo:$HALO_ID pid:$SELECTED_PID missing_outputs_batch" >> failed_jobs/kde_failures.log
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  ❌ FAILED: KDE exit code $KDE_EXIT"
        echo "$(date) FAILED halo:$HALO_ID pid:$SELECTED_PID kde_failed_${KDE_EXIT}_batch" >> failed_jobs/kde_failures.log
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
done

echo "📊 BATCH PROCESSING SUMMARY"
echo "==========================="
echo "🎯 Processed: $PROCESSED particles"
echo "✅ Succeeded: $SUCCEEDED"
echo "❌ Failed: $FAILED"
echo "📈 Success rate: $(( SUCCEEDED * 100 / PROCESSED ))%"
echo ""
echo "🏁 Batch completed: $(date)"


