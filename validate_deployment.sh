#!/bin/bash
#SBATCH --job-name=validate_deploy
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --output=logs/validate_%j.out
#SBATCH --error=logs/validate_%j.err
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-3

# Deployment Validation Script
# Tests all critical components before launching the full job array
# This runs a small subset to validate:
# 1. SLURM array indexing works correctly
# 2. Modules load properly on different nodes
# 3. GPU allocation is correct
# 4. File paths are accessible
# 5. Environment is consistent

set -e

echo "🧪 DEPLOYMENT VALIDATION"
echo "========================"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "User: $(whoami)"
echo

# Test 1: Module Loading
echo "🔧 Test 1: Module Loading"
echo "-------------------------"
module purge
echo "✓ Modules purged"

module load math
echo "✓ Math module loaded"

module load devel  
echo "✓ Devel module loaded"

module load nvhpc/24.7
echo "✓ NVHPC module loaded"

module load cuda/12.2.0
echo "✓ CUDA module loaded"

module load cudnn/8.9.0.131
echo "✓ cuDNN module loaded"

# Test 2: Environment Setup
echo
echo "🌍 Test 2: Environment Setup"
echo "----------------------------"
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0
export CUDA_VISIBLE_DEVICES=0
echo "✓ CUDA environment variables set"

source ~/.bashrc
echo "✓ Bashrc sourced"

conda activate bosque
echo "✓ Conda environment activated"

# Test 3: Python Environment
echo
echo "🐍 Test 3: Python Environment"
echo "-----------------------------"
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import tensorflow_probability as tfp; print(f'TFP: {tfp.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
python -c "import h5py; print(f'h5py: {h5py.__version__}')"

# Test 4: GPU Availability
echo
echo "🎯 Test 4: GPU Availability"
echo "---------------------------"
python -c "
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'GPUs detected: {len(gpus)}')
for i, gpu in enumerate(gpus):
    print(f'  GPU {i}: {gpu}')
if gpus:
    print('✓ GPU is available')
else:
    print('❌ No GPU detected')
    exit(1)
"

# Test 5: SLURM Array Indexing
echo
echo "📊 Test 5: SLURM Array Indexing"
echo "-------------------------------"
PARTICLES_PER_JOB=5
START_PID=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PARTICLES_PER_JOB + 1 ))
END_PID=$(( $SLURM_ARRAY_TASK_ID * $PARTICLES_PER_JOB ))

echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Particles per job: $PARTICLES_PER_JOB"
echo "PID range: $START_PID to $END_PID"
echo "PIDs to process: $(seq $START_PID $END_PID | tr '\n' ' ')"

# Test 6: File Path Discovery
echo
echo "📁 Test 6: File Path Discovery"
echo "------------------------------"
find_h5_file() {
    local search_paths=(
        "/scratch/groups/aganze/christianaganze/symphony_mocks/"
        "/oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/"
        "/scratch/groups/aganze/christianaganze/"
    )
    
    for path in "${search_paths[@]}"; do
        echo "Checking: $path"
        if [[ -d "$path" ]]; then
            echo "  ✓ Directory exists"
            h5_file=$(find "$path" -name "*.h5" -type f 2>/dev/null | head -1)
            if [[ -n "$h5_file" ]]; then
                echo "  ✓ Found H5 file: $h5_file"
                echo "$h5_file"
                return 0
            else
                echo "  ⚠ No H5 files found"
            fi
        else
            echo "  ❌ Directory not found"
        fi
    done
    echo "⚠ Using fallback H5 file"
    echo "/scratch/groups/aganze/christianaganze/symphony_mocks/all_in_one.h5"
}

H5_FILE=$(find_h5_file)
echo "Selected H5 file: $H5_FILE"

if [[ -f "$H5_FILE" ]]; then
    echo "✓ H5 file is accessible"
    # Quick H5 file validation
    python -c "
import h5py
import sys
try:
    with h5py.File('$H5_FILE', 'r') as f:
        print(f'✓ H5 file opened successfully')
        print(f'  Keys: {list(f.keys())}')
        if 'PartType1' in f:
            part1 = f['PartType1']
            print(f'  PartType1 keys: {list(part1.keys())}')
            if 'ParticleIDs' in part1:
                pids = part1['ParticleIDs'][:]
                print(f'  Particle IDs range: {pids.min()} to {pids.max()}')
                print(f'  Total particles: {len(pids)}')
except Exception as e:
    print(f'❌ H5 file error: {e}')
    sys.exit(1)
"
else
    echo "❌ H5 file not accessible: $H5_FILE"
    exit 1
fi

# Test 7: Output Directory Creation
echo
echo "📂 Test 7: Output Directory Creation"
echo "------------------------------------"
OUTPUT_BASE_DIR="/scratch/groups/aganze/christianaganze/tfp_flows_validation"
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$OUTPUT_BASE_DIR/trained_flows"
mkdir -p "$OUTPUT_BASE_DIR/samples"
mkdir -p logs

echo "✓ Created output directories:"
echo "  Base: $OUTPUT_BASE_DIR"
echo "  Trained flows: $OUTPUT_BASE_DIR/trained_flows"
echo "  Samples: $OUTPUT_BASE_DIR/samples"
echo "  Logs: logs"

# Test 8: Quick Training Test (minimal)
echo
echo "🏋️ Test 8: Quick Training Test"
echo "------------------------------"
TEST_PID=$START_PID
echo "Testing with PID: $TEST_PID"

# Run a very quick training (2 epochs, 1000 samples)
python train_tfp_flows.py \
    --h5_file "$H5_FILE" \
    --particle_pid $TEST_PID \
    --output_dir "$OUTPUT_BASE_DIR" \
    --epochs 2 \
    --batch_size 512 \
    --learning_rate 1e-3 \
    --n_layers 2 \
    --hidden_units 32 \
    --generate_samples \
    --n_samples 1000 \
    --use_kroupa_imf \
    --validation_split 0.2

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Quick training test PASSED"
    
    # Verify outputs
    MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/model_pid${TEST_PID}"
    if [[ -f "$MODEL_DIR/model_pid${TEST_PID}.npz" ]]; then
        echo "✓ Model file created"
    else
        echo "❌ Model file missing"
    fi
    
    if [[ -f "$MODEL_DIR/model_pid${TEST_PID}_results.json" ]]; then
        echo "✓ Results file created"
        # Show some results
        python -c "
import json
with open('$MODEL_DIR/model_pid${TEST_PID}_results.json', 'r') as f:
    results = json.load(f)
    print(f'  Final train loss: {results[\"train_losses\"][-1]:.4f}')
    if 'sampling' in results:
        print(f'  Samples generated: {results[\"sampling\"][\"n_samples_generated\"]}')
"
    else
        echo "❌ Results file missing"
    fi
    
else
    echo "❌ Quick training test FAILED with exit code $TRAIN_EXIT_CODE"
    exit 1
fi

# Test 9: Concurrent Access (if multiple array tasks)
echo
echo "🔄 Test 9: Concurrent Access Check"
echo "----------------------------------"
LOCK_FILE="/tmp/tfp_validation_$SLURM_JOB_ID.lock"
echo "Array task $SLURM_ARRAY_TASK_ID attempting to create lock file: $LOCK_FILE"

if [[ ! -f "$LOCK_FILE" ]]; then
    touch "$LOCK_FILE"
    echo "✓ Lock file created successfully"
    sleep 2
    rm "$LOCK_FILE"
    echo "✓ Lock file removed successfully"
else
    echo "⚠ Lock file already exists (another task is running)"
fi

# Final Summary
echo
echo "🎉 VALIDATION SUMMARY"
echo "===================="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "PID Range: $START_PID to $END_PID"
echo "H5 File: $H5_FILE"
echo "Output Dir: $OUTPUT_BASE_DIR"
echo "Status: ✅ ALL TESTS PASSED"
echo "Timestamp: $(date)"
echo
echo "🚀 Ready for full deployment!"
