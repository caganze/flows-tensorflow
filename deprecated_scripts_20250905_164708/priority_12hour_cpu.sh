#!/bin/bash
#SBATCH --job-name="priority_12h_cpu"
#SBATCH --partition=kipac
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/priority_cpu_%A_%a.out
#SBATCH --error=logs/priority_cpu_%A_%a.err
#SBATCH --array=1-8%8

# 🚀 PRIORITY 12-HOUR CPU JOB FOR CRITICAL HALOS (CPU NODE OPTIMIZED)
# Target: symphony halos 239, 718, 270, 925 + eden halos 239, 718, 270, 925
# Optimized for CPU compute nodes with fast training parameters

set -e

echo "🚀 PRIORITY 12-HOUR CPU HALO COMPLETION"
echo "======================================="
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-1}"
echo "Job ID: ${SLURM_JOB_ID:-test}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "CPU cores: ${SLURM_CPUS_PER_TASK:-16}"
echo "Started: $(date)"
echo "Priority halos: 239, 718, 270, 925 (symphony + eden)"

# Environment setup for CPU
module --force purge
module load math devel python/3.9.0

# Force CPU-only TensorFlow
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""
export TF_FORCE_GPU_ALLOW_GROWTH=false

source ~/.bashrc
conda activate bosque

# Create directories
mkdir -p logs success_logs failed_jobs

# Define priority halos and suites
declare -a PRIORITY_HALOS=("239" "718" "270" "925")
declare -a SUITES=("symphony" "eden")

# Calculate which halo+suite combination this array task should handle
ARRAY_ID=${SLURM_ARRAY_TASK_ID:-1}

# Array mapping: 1-4 = symphony halos, 5-8 = eden halos
if [[ $ARRAY_ID -le 4 ]]; then
    SUITE="symphony"
    HALO_INDEX=$((ARRAY_ID - 1))
else
    SUITE="eden"
    HALO_INDEX=$((ARRAY_ID - 5))
fi

HALO_ID=${PRIORITY_HALOS[$HALO_INDEX]}

echo "🎯 Array task $ARRAY_ID processing: $SUITE Halo$HALO_ID"

# Find the H5 file for this halo+suite combination
if [[ "$SUITE" == "symphony" ]]; then
    H5_PATTERN="/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/*Halo${HALO_ID}_*.h5"
elif [[ "$SUITE" == "eden" ]]; then
    H5_PATTERN="/oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/*Halo${HALO_ID}_*.h5"
else
    echo "❌ Unknown suite: $SUITE"
    exit 1
fi

# Find the actual H5 file
H5_FILES=($(ls $H5_PATTERN 2>/dev/null || true))

if [[ ${#H5_FILES[@]} -eq 0 ]]; then
    echo "❌ No H5 files found for $SUITE Halo$HALO_ID"
    echo "   Searched: $H5_PATTERN"
    exit 1
elif [[ ${#H5_FILES[@]} -gt 1 ]]; then
    echo "📁 Multiple files found for $SUITE Halo$HALO_ID, using first:"
    for file in "${H5_FILES[@]}"; do
        echo "   - $(basename $file)"
    done
fi

SELECTED_H5="${H5_FILES[0]}"
echo "✅ Selected H5 file: $(basename $SELECTED_H5)"

# Get list of particles in this halo (use the existing train_tfp_flows.py that works)
echo "🔍 Extracting particle list from H5 file..."

# Use a simple approach that doesn't require symlib
PARTICLE_LIST=$(python -c "
import h5py
import numpy as np
import sys
import os

# Set CPU-only mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    with h5py.File('$SELECTED_H5', 'r') as f:
        if 'PartType1' in f and 'ParticleIDs' in f['PartType1']:
            pids = np.unique(f['PartType1']['ParticleIDs'][:])
            # Filter to reasonable particle range and remove zeros
            pids = pids[(pids > 0) & (pids < 100000)]
            # Limit to 30 particles for 12h CPU completion
            pids = pids[:30]  
            print(' '.join(map(str, pids)))
        else:
            print('')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
")

if [[ -z "$PARTICLE_LIST" ]]; then
    echo "❌ Could not extract particle list from H5 file"
    exit 1
fi

PARTICLES=($PARTICLE_LIST)
echo "📊 Found ${#PARTICLES[@]} particles to process (CPU-optimized limit)"

# Output directories  
OUTPUT_BASE_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output"
MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/${SUITE}/halo${HALO_ID}"
SAMPLES_DIR="$OUTPUT_BASE_DIR/samples/${SUITE}/halo${HALO_ID}"

mkdir -p "$MODEL_DIR" "$SAMPLES_DIR"

echo "📁 Output directories:"
echo "   Models: $MODEL_DIR"
echo "   Samples: $SAMPLES_DIR"

# Process each particle with ultra-fast CPU parameters
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

for PID in "${PARTICLES[@]}"; do
    # Check if already completed
    if [[ -f "$MODEL_DIR/model_pid${PID}.npz" ]] && [[ -f "$SAMPLES_DIR/model_pid${PID}_samples.npz" || -f "$SAMPLES_DIR/model_pid${PID}_samples.h5" ]]; then
        echo "✅ PID $PID already completed, skipping"
        ((COMPLETED++))
        continue
    fi
    
    # Check time constraint (leave 45 min buffer for CPU)
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    TIME_REMAINING=$((43200 - ELAPSED))  # 12 hours = 43200 seconds
    
    if [[ $TIME_REMAINING -lt 2700 ]]; then  # Less than 45 minutes remaining
        echo "⏰ Time constraint reached, stopping processing"
        break
    fi
    
    echo "🔧 Processing PID $PID (${TIME_REMAINING}s remaining)..."
    
    # Get particle size for adaptive parameters
    OBJECT_COUNT=$(python -c "
import h5py
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
try:
    with h5py.File('$SELECTED_H5', 'r') as f:
        mask = f['PartType1']['ParticleIDs'][:] == $PID
        print(np.sum(mask))
except:
    print(10000)  # Default fallback
")
    
    # Ultra-fast CPU parameters for 12-hour completion
    if [[ $OBJECT_COUNT -gt 100000 ]]; then
        # Very large particles - minimal epochs for CPU speed
        EPOCHS=15
        BATCH_SIZE=1024
        N_LAYERS=2
        HIDDEN_UNITS=128
        LEARNING_RATE=2e-3
        echo "🚀 CPU Ultra-fast Large (>100k): epochs=15, layers=2, units=128, lr=2e-3"
    elif [[ $OBJECT_COUNT -gt 50000 ]]; then
        # Large particles
        EPOCHS=20
        BATCH_SIZE=768
        N_LAYERS=2
        HIDDEN_UNITS=256
        LEARNING_RATE=1e-3
        echo "🚀 CPU Ultra-fast Medium-Large (50k-100k): epochs=20, layers=2, units=256, lr=1e-3"
    elif [[ $OBJECT_COUNT -lt 5000 ]]; then
        # Small particles
        EPOCHS=15
        BATCH_SIZE=512
        N_LAYERS=2
        HIDDEN_UNITS=128
        LEARNING_RATE=2e-3
        echo "🚀 CPU Ultra-fast Small (<5k): epochs=15, layers=2, units=128, lr=2e-3"
    else
        # Medium particles
        EPOCHS=18
        BATCH_SIZE=512
        N_LAYERS=2
        HIDDEN_UNITS=192
        LEARNING_RATE=1.5e-3
        echo "🚀 CPU Ultra-fast Medium (5k-50k): epochs=18, layers=2, units=192, lr=1.5e-3"
    fi
    
    # Train with timeout for time management (shorter for CPU)
    timeout 1200 python train_tfp_flows.py \
        --data_path "$SELECTED_H5" \
        --particle_pid "$PID" \
        --output_dir "$MODEL_DIR" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --n_layers $N_LAYERS \
        --hidden_units $HIDDEN_UNITS \
        --generate-samples \
        --use_kroupa_imf \
        --model_name "model_pid${PID}" 2>&1 | tee "logs/priority_cpu_${SUITE}_halo${HALO_ID}_pid${PID}.log"
    
    TRAIN_EXIT=$?
    
    if [[ $TRAIN_EXIT -eq 0 ]]; then
        echo "✅ PID $PID completed successfully"
        ((COMPLETED++))
        echo "PID_${PID}" >> "success_logs/${SUITE}_halo${HALO_ID}_cpu_success.txt"
    else
        echo "❌ PID $PID failed (exit code: $TRAIN_EXIT)"
        ((FAILED++))
        echo "PID_${PID}_EXIT_${TRAIN_EXIT}" >> "failed_jobs/${SUITE}_halo${HALO_ID}_cpu_failed.txt"
    fi
    
    # Quick status
    echo "📊 Progress: ${COMPLETED} completed, ${FAILED} failed"
done

# Final summary
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "🏁 PRIORITY CPU JOB SUMMARY for $SUITE Halo$HALO_ID"
echo "================================================="
echo "⏱️  Total time: ${TOTAL_TIME}s ($(($TOTAL_TIME / 60)) minutes)"
echo "✅ Completed: $COMPLETED particles"
echo "❌ Failed: $FAILED particles"
echo "📊 Total processed: $((COMPLETED + FAILED)) particles"
echo "🎯 Success rate: $(( COMPLETED * 100 / (COMPLETED + FAILED + 1) ))%"
echo "📁 Outputs saved to: $MODEL_DIR"
echo "Finished: $(date)"

if [[ $COMPLETED -gt 0 ]]; then
    echo "🎉 SUCCESS: $SUITE Halo$HALO_ID has $COMPLETED completed models!"
else
    echo "⚠️  WARNING: No particles completed for $SUITE Halo$HALO_ID"
fi
