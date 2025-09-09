#!/bin/bash
#SBATCH --job-name="kde_particle_list"
#SBATCH --partition=kipac
#SBATCH --time=6:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/kde_%A_%a.out
#SBATCH --error=logs/kde_%A_%a.err
#SBATCH --array=1-22713%20

set -e

echo "🚀 KDE CPU JOB - ARRAY TASK ${SLURM_ARRAY_TASK_ID:-1}"
echo "🔧 Mode: Kernel Density Estimation Training"
echo "Started: $(date)"
echo "Node: ${SLURM_NODELIST:-local}"
echo "Job ID: ${SLURM_JOB_ID:-test}"

# Environment setup
module --force purge
module load python/3.9.0

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

# Output directories - use KDE output directory
OUTPUT_BASE_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/kde_output"
MODEL_DIR="$OUTPUT_BASE_DIR/kde_models/${SUITE}/halo${HALO_ID#Halo}"
SAMPLES_DIR="$OUTPUT_BASE_DIR/kde_samples/${SUITE}/halo${HALO_ID#Halo}"

echo "📁 Output base: $OUTPUT_BASE_DIR"
echo "📁 Suite: $SUITE, Halo ID: $HALO_ID"

mkdir -p "$MODEL_DIR" "$SAMPLES_DIR"

# Check if already completed
if [[ -f "$SAMPLES_DIR/kde_samples_pid${SELECTED_PID}.h5" ]]; then
    echo "✅ Already completed: Halo $HALO_ID PID $SELECTED_PID"
    echo "$(date) SUCCESS halo:$HALO_ID pid:$SELECTED_PID already_completed" >> success_logs/kde_success.log
    exit 0
fi

# Determine optimal parameters based on particle size
if [[ $OBJECT_COUNT -gt 100000 ]]; then
    # Large particles (100k+): Need more neighbors for smoothing
    N_NEIGHBORS=128
    SAMPLE_FRACTION=0.8
    MASS_RANGE_MIN=0.1
    MASS_RANGE_MAX=120
    echo "🧠 KDE Large particle (>100k): neighbors=128, fraction=0.8"
elif [[ $OBJECT_COUNT -gt 50000 ]]; then
    # Medium-large particles
    N_NEIGHBORS=96
    SAMPLE_FRACTION=1.0
    MASS_RANGE_MIN=0.1
    MASS_RANGE_MAX=120
    echo "🧠 KDE Medium-large (50k-100k): neighbors=96, fraction=1.0"
elif [[ $OBJECT_COUNT -lt 5000 ]]; then
    # Small particles: need oversampling
    N_NEIGHBORS=64
    SAMPLE_FRACTION=2.0
    MASS_RANGE_MIN=0.08
    MASS_RANGE_MAX=50
    echo "🧠 KDE Small particle (<5k): neighbors=64, fraction=2.0, focused mass range"
else
    # Medium particles
    N_NEIGHBORS=64
    SAMPLE_FRACTION=1.2
    MASS_RANGE_MIN=0.1
    MASS_RANGE_MAX=100
    echo "🧠 KDE Medium particle (5k-50k): neighbors=64, fraction=1.2"
fi

echo "🧠 Running KDE Sampling: Halo $HALO_ID PID $SELECTED_PID..."
python train_kde_conditional.py \
    --halo_id "$HALO_ID" \
    --parent_id "$SELECTED_PID" \
    --suite "$SUITE" \
    --output_dir "$SAMPLES_DIR" \
    --n_neighbors $N_NEIGHBORS \
    --sample_fraction $SAMPLE_FRACTION \
    --mass_range $MASS_RANGE_MIN $MASS_RANGE_MAX \
    --seed 42

KDE_EXIT=$?

if [[ $KDE_EXIT -eq 0 ]]; then
    # Check success
    if [[ -f "$SAMPLES_DIR/kde_samples_${HALO_ID}_pid${SELECTED_PID}.h5" ]]; then
        echo "✅ SUCCESS: KDE Halo $HALO_ID PID $SELECTED_PID"
        echo "$(date) SUCCESS halo:$HALO_ID pid:$SELECTED_PID kde_completed" >> success_logs/kde_success.log
        
        # Optional: Run a quick quality check
        echo "📊 KDE Quality Check:"
        python -c "
import h5py
import numpy as np
with h5py.File('$SAMPLES_DIR/kde_samples_${HALO_ID}_pid${SELECTED_PID}.h5', 'r') as f:
    pos = f['positions'][:]
    vel = f['velocities'][:]
    print(f'   Samples: {len(pos)}')
    print(f'   Position range: [{pos.min():.2f}, {pos.max():.2f}]')
    print(f'   Velocity range: [{vel.min():.2f}, {vel.max():.2f}]')
    print(f'   Position std: {pos.std():.2f}')
    print(f'   Velocity std: {vel.std():.2f}')
"
    else
        echo "❌ FAILED: Missing KDE output files"
        echo "$(date) FAILED halo:$HALO_ID pid:$SELECTED_PID missing_outputs" >> failed_jobs/kde_failures.log
        exit 1
    fi
else
    echo "❌ FAILED: KDE exit code $KDE_EXIT"
    echo "$(date) FAILED halo:$HALO_ID pid:$SELECTED_PID kde_failed_$KDE_EXIT" >> failed_jobs/kde_failures.log
    exit 1
fi

echo "🏁 Completed: $(date)"
