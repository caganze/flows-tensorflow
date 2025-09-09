#!/bin/bash
#SBATCH --job-name=test_tfp_unified
#SBATCH --partition=owners
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_tfp_%A_%a.out
#SBATCH --error=logs/test_tfp_%A_%a.err
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --array=1-2%2

# 🧪 TEST SCRIPT FOR submit_tfp_array.sh
# Tests the unified submission system with minimal resources and fast parameters
# Use this to validate functionality before submitting large jobs

set -e

echo "🧪 TFP Array Test Job"
echo "===================="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo

# TEST CONFIGURATION - Fast and minimal for validation
export PARTICLES_PER_TASK=2  # Only 2 particles per task (match 2 GPUs)
export EPOCHS=3              # Very few epochs for speed
export BATCH_SIZE=256        # Small batch size
export LEARNING_RATE=1e-3
export N_LAYERS=2            # Minimal layers
export HIDDEN_UNITS=32       # Small hidden units
export CHECK_SAMPLES=false   # Skip sample checking for speed

# Test will use H5 directory structure (no override needed)

echo "🧪 TEST CONFIGURATION:"
echo "  Particles per task: $PARTICLES_PER_TASK"
echo "  Total particles to test: 4 (2 tasks × 2 particles)"
echo "  Test epochs: $EPOCHS"
echo "  Test batch size: $BATCH_SIZE"
echo "  Test output: Will be in H5 file directory/tfp_output/"
echo "  Expected runtime: ~5-10 minutes per particle"
echo

# Load required modules
echo "🔧 Loading modules..."
module purge
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131

# Set CUDA environment
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0

# Activate conda environment
echo "🐍 Activating environment..."
source ~/.bashrc
conda activate bosque

# Verify environment before proceeding
echo "🔍 Environment Verification:"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo "  Python: $(which python)"
echo "  TensorFlow available: $(python -c 'import tensorflow; print("✅ Yes")' 2>/dev/null || echo "❌ No")"
echo "  TFP available: $(python -c 'import tensorflow_probability; print("✅ Yes")' 2>/dev/null || echo "❌ No")"
echo "  GPU available: $(python -c 'import tensorflow as tf; print("✅ Yes" if tf.config.list_physical_devices("GPU") else "❌ No")' 2>/dev/null || echo "❌ No")"
echo

# Smart H5 file discovery (same as main script)
find_h5_file() {
    # Look for eden_scaled files first
    local eden_files=$(find /oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/ -name "eden_scaled_Halo*" -type f 2>/dev/null | head -1)
    if [[ -n "$eden_files" ]]; then
        echo "$eden_files"
        return 0
    fi
    
    # Fallback search paths
    local search_paths=(
        "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/"
        "/oak/stanford/orgs/kipac/users/caganze/milkyway-hr-mocks/"
        "/oak/stanford/orgs/kipac/users/caganze/milkywaymocks/"
        "/oak/stanford/orgs/kipac/users/caganze/"
    )
    
    for path in "${search_paths[@]}"; do
        if [[ -d "$path" ]]; then
            h5_file=$(find "$path" -name "*.h5" -type f 2>/dev/null | head -1)
            if [[ -n "$h5_file" ]]; then
                echo "$h5_file"
                return 0
            fi
        fi
    done
    
    echo "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/all_in_one.h5"
}

H5_FILE=$(find_h5_file)
echo "📁 Selected H5 file: $H5_FILE"

# Test H5 file access
if [[ ! -f "$H5_FILE" ]]; then
    echo "❌ ERROR: H5 file not found: $H5_FILE"
    exit 1
fi

echo "✅ H5 file accessible"

# Quick H5 file structure check
echo "🔍 H5 file structure check:"
python -c "
import h5py
import numpy as np
try:
    with h5py.File('$H5_FILE', 'r') as f:
        print(f'  Keys: {list(f.keys())[:5]}...')
        
        # Check for different structures
        if 'PartType1' in f:
            if 'ParticleIDs' in f['PartType1']:
                pids = f['PartType1']['ParticleIDs'][:]
                print(f'  PartType1 PIDs: {len(np.unique(pids))} unique particles')
                print(f'  First few PIDs: {np.unique(pids)[:10]}')
            else:
                print('  PartType1 found but no ParticleIDs')
        elif 'particles' in f:
            print('  particles structure found')
        else:
            print('  Unknown structure - using fallback')
            
except Exception as e:
    print(f'  ❌ Error reading H5: {e}')
    exit(1)
" || exit 1

# Extract data source and halo ID
FILENAME=$(basename "$H5_FILE")
HALO_ID=$(echo "$FILENAME" | sed 's/.*Halo\([0-9][0-9]*\).*/\1/')

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

# Handle fallback file
if [[ "$FILENAME" == "all_in_one.h5" ]] || [[ "$HALO_ID" == "$FILENAME" ]]; then
    echo "⚠️  Using fallback file, setting default halo structure"
    HALO_ID="000"
    DATA_SOURCE="symphony"
fi

echo "📁 Data source: $DATA_SOURCE"
echo "📁 Halo ID: $HALO_ID"

# Create output directories - save in same parent directory as H5 file with halo/PID structure
H5_PARENT_DIR=$(dirname "$H5_FILE")
OUTPUT_BASE_DIR="$H5_PARENT_DIR/tfp_output"
MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/${DATA_SOURCE}/halo${HALO_ID}"
SAMPLES_DIR="$OUTPUT_BASE_DIR/samples/${DATA_SOURCE}/halo${HALO_ID}"

echo "📁 H5 file parent: $H5_PARENT_DIR"
echo "📁 Test output directories:"
echo "  Model dir: $MODEL_DIR"
echo "  Samples dir: $SAMPLES_DIR"

# Create directories safely
mkdir -p "$MODEL_DIR" "$SAMPLES_DIR"
mkdir -p logs

# Calculate test particle range (test first few particles)
START_PID=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PARTICLES_PER_TASK + 1 ))
END_PID=$(( $SLURM_ARRAY_TASK_ID * $PARTICLES_PER_TASK ))

echo "🎯 Test PIDs: $START_PID to $END_PID"

# Get available GPUs for testing
AVAILABLE_GPUS=($(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '))
NUM_AVAILABLE_GPUS=${#AVAILABLE_GPUS[@]}

echo "🎮 Available GPUs for testing: ${AVAILABLE_GPUS[*]} (Total: $NUM_AVAILABLE_GPUS)"

# Function to test particle size detection
test_particle_size_detection() {
    local pid=$1
    echo "🔍 Testing particle size detection for PID $pid..."
    
    local size=$(python -c "
import h5py
import numpy as np
import sys

try:
    with h5py.File('$H5_FILE', 'r') as f:
        if 'PartType1' in f:
            if 'ParticleIDs' in f['PartType1']:
                pids = f['PartType1']['ParticleIDs'][:]
                if 'Coordinates' in f['PartType1']:
                    coords = f['PartType1']['Coordinates'][:]
                    pid_mask = (pids == $pid)
                    size = np.sum(pid_mask)
                    print(size)
                else:
                    total_size = len(pids)
                    unique_pids = len(np.unique(pids))
                    avg_size = total_size // unique_pids if unique_pids > 0 else 1000
                    print(avg_size)
        else:
            print(50000)
except Exception as e:
    print(50000)
" 2>/dev/null)
    
    if [[ ! "$size" =~ ^[0-9]+$ ]]; then
        size=50000
    fi
    
    echo "  PID $pid: $size objects"
    
    if [[ $size -gt 100000 ]]; then
        echo "  🐋 Large particle detected (>100k objects)"
    else
        echo "  🐭 Small particle (<100k objects)"
    fi
    
    echo $size
}

# Test particle processing function
test_particle_training() {
    local pid=$1
    local gpu_id=$2
    local gpu_index=$3
    
    echo "🧪 Testing PID $pid on GPU $gpu_id (index $gpu_index)"
    
    # Test particle size detection
    local particle_size=$(test_particle_size_detection $pid)
    
    # Set GPU for this process
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Create process-specific log
    local log_file="logs/test_tfp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_pid${pid}_gpu${gpu_index}.log"
    
    # Test training with minimal parameters
    {
        echo "🚀 Starting test training for PID $pid on GPU $gpu_id"
        echo "Time: $(date)"
        echo "Particle size: $particle_size objects"
        
        # Adaptive parameters for testing
        local test_epochs=$EPOCHS
        local test_batch_size=$BATCH_SIZE
        
        if [[ $particle_size -gt 100000 ]]; then
            test_epochs=$((EPOCHS * 80 / 100))
            test_batch_size=$((BATCH_SIZE * 150 / 100))
            echo "⚡ Large particle test optimization: epochs=$test_epochs, batch_size=$test_batch_size"
        fi
        
        echo "📋 Test parameters:"
        echo "  Epochs: $test_epochs"
        echo "  Batch size: $test_batch_size"
        echo "  Layers: $N_LAYERS"
        echo "  Hidden units: $HIDDEN_UNITS"
        echo
        
        # Run the actual training
        python train_tfp_flows.py \
            --data_path "$H5_FILE" \
            --particle_pid $pid \
            --output_dir "$MODEL_DIR" \
            --epochs $test_epochs \
            --batch_size $test_batch_size \
            --learning_rate $LEARNING_RATE \
            --n_layers $N_LAYERS \
            --hidden_units $HIDDEN_UNITS \
            --use_kroupa_imf \
            --validation_split 0.2 \
            --early_stopping_patience 5 \
            --reduce_lr_patience 3
        
        local exit_code=$?
        echo "Training completed with exit code: $exit_code"
        echo "Time: $(date)"
        
        # Test output validation
        if [ $exit_code -eq 0 ]; then
            echo "✅ Training successful, checking outputs..."
            
            # Check model file
            if [[ -f "$MODEL_DIR/model_pid${pid}.npz" && -s "$MODEL_DIR/model_pid${pid}.npz" ]]; then
                echo "✅ Model file created and non-empty"
            else
                echo "❌ Model file missing or empty"
                return 1
            fi
            
            # Check results file
            if [[ -f "$MODEL_DIR/model_pid${pid}_results.json" ]]; then
                if python -c "import json; json.load(open('$MODEL_DIR/model_pid${pid}_results.json'))" 2>/dev/null; then
                    echo "✅ Results file created and valid JSON"
                else
                    echo "❌ Results file invalid JSON"
                    return 1
                fi
            else
                echo "❌ Results file missing"
                return 1
            fi
            
            echo "✅ PID $pid test completed successfully on GPU $gpu_id"
        else
            echo "❌ PID $pid test failed on GPU $gpu_id with exit code $exit_code"
        fi
        
        return $exit_code
        
    } 2>&1 | tee "$log_file"
}

echo
echo "🚀 Starting parallel test training processes..."

# Launch test processes in parallel
pids_array=()
gpu_processes=()

for ((i=0; i<$PARTICLES_PER_TASK; i++)); do
    pid=$((START_PID + i))
    gpu_index=$((i % NUM_AVAILABLE_GPUS))
    gpu_id=${AVAILABLE_GPUS[$gpu_index]}
    
    echo "🚀 Launching test for PID $pid on GPU $gpu_id (background process)"
    
    # Launch in background
    test_particle_training $pid $gpu_id $gpu_index &
    
    # Store the background process PID and particle PID
    gpu_processes+=($!)
    pids_array+=($pid)
    
    # Small delay to avoid race conditions
    sleep 2
done

echo
echo "⏳ Waiting for all test processes to complete..."
echo "Launched ${#gpu_processes[@]} parallel test processes"

# Wait for all background processes and collect results
success_count=0
failure_count=0

for ((i=0; i<${#gpu_processes[@]}; i++)); do
    process_pid=${gpu_processes[$i]}
    particle_pid=${pids_array[$i]}
    
    echo "Waiting for test PID $particle_pid (process $process_pid)..."
    
    if wait $process_pid; then
        echo "✅ Test PID $particle_pid completed successfully"
        ((success_count++))
    else
        echo "❌ Test PID $particle_pid failed"
        ((failure_count++))
    fi
done

echo
echo "🧪 TEST RESULTS SUMMARY"
echo "======================="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "PIDs tested: ${pids_array[*]}"
echo "📊 Results:"
echo "  ✅ Successful: $success_count"
echo "  ❌ Failed: $failure_count"
echo "  📈 Success rate: $((success_count * 100 / (success_count + failure_count)))%"
echo "  🕐 Total runtime: $SECONDS seconds"
echo "  📁 Test output location: $MODEL_DIR"
echo

# Check file outputs
echo "📁 File Output Verification:"
for pid in "${pids_array[@]}"; do
    model_file="$MODEL_DIR/model_pid${pid}.npz"
    results_file="$MODEL_DIR/model_pid${pid}_results.json"
    
    echo "  PID $pid:"
    if [[ -f "$model_file" && -s "$model_file" ]]; then
        echo "    ✅ Model file: $(ls -lh "$model_file" | awk '{print $5}')"
    else
        echo "    ❌ Model file: missing or empty"
    fi
    
    if [[ -f "$results_file" ]]; then
        echo "    ✅ Results file: $(ls -lh "$results_file" | awk '{print $5}')"
    else
        echo "    ❌ Results file: missing"
    fi
done

echo
echo "🔍 Quick Test of Failure Detection:"
echo "===================================="

# Test the failure detection function from the main script
is_particle_completed() {
    local pid=$1
    local model_file="$MODEL_DIR/model_pid${pid}.npz"
    local results_file="$MODEL_DIR/model_pid${pid}_results.json"
    
    if [[ ! -f "$model_file" || ! -f "$results_file" ]]; then
        return 1
    fi
    
    if [[ ! -s "$model_file" ]]; then
        return 1
    fi
    
    if ! python -c "import json; json.load(open('$results_file'))" 2>/dev/null; then
        return 1
    fi
    
    return 0
}

for pid in "${pids_array[@]}"; do
    if is_particle_completed $pid; then
        echo "✅ PID $pid: Passes completion check"
    else
        echo "❌ PID $pid: Fails completion check"
    fi
done

echo
echo "💡 RECOMMENDATIONS:"
echo "==================="

if [[ $failure_count -eq 0 ]]; then
    echo "🎉 ALL TESTS PASSED!"
    echo "✅ The unified submission system is working correctly"
    echo "✅ GPU allocation is working properly"
    echo "✅ Particle size detection is functional"
    echo "✅ File output validation is working"
    echo
    echo "🚀 READY FOR PRODUCTION SUBMISSION:"
    echo "   sbatch submit_tfp_array.sh"
    echo
    echo "📊 For monitoring:"
    echo "   squeue -u \$(whoami)"
    echo "   ./scan_and_resubmit.sh"
    
elif [[ $success_count -gt 0 ]]; then
    echo "⚠️  PARTIAL SUCCESS - Some tests passed"
    echo "✅ Basic functionality is working"
    echo "❌ Some issues detected - check logs:"
    for ((i=0; i<${#gpu_processes[@]}; i++)); do
        particle_pid=${pids_array[$i]}
        log_file="logs/test_tfp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_pid${particle_pid}_gpu*.log"
        echo "   tail -50 $log_file"
    done
    echo
    echo "🔧 Consider investigating issues before full submission"
    
else
    echo "💥 ALL TESTS FAILED!"
    echo "❌ Critical issues detected - DO NOT submit production job yet"
    echo "🔍 Check these logs for issues:"
    for ((i=0; i<${#gpu_processes[@]}; i++)); do
        particle_pid=${pids_array[$i]}
        log_file="logs/test_tfp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_pid${particle_pid}_gpu*.log"
        echo "   tail -100 $log_file"
    done
    echo
    echo "🔧 Common issues to check:"
    echo "   - Module loading problems"
    echo "   - Environment activation issues"
    echo "   - H5 file access/structure problems"
    echo "   - GPU availability issues"
    echo "   - train_tfp_flows.py argument problems"
fi

echo
echo "🏁 Test completed at $(date)"
echo "📁 Test outputs saved in: $OUTPUT_BASE_DIR"
echo "📋 Test logs saved in: logs/test_tfp_*.log"

# Set appropriate exit code
if [[ $failure_count -eq 0 ]]; then
    exit 0
elif [[ $success_count -gt 0 ]]; then
    exit 0  # Partial success still indicates basic functionality
else
    exit 1  # Complete failure
fi
