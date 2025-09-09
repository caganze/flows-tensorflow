#!/bin/bash
#SBATCH --job-name=tfp_flows_unified
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --output=logs/tfp_%A_%a.out
#SBATCH --error=logs/tfp_%A_%a.err
#SBATCH --mem=128GB
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --array=1-1000%10

# 🚀 UNIFIED TensorFlow Probability Flows Array Job
# Strategic 8 GPU allocation for maximum efficiency and QOS optimization
# Each array task processes multiple particles using different GPUs

set -e

echo "🚀 TFP Flows Unified Array Job"
echo "=============================="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo

# Configuration - can be overridden via environment variables
PARTICLES_PER_TASK="${PARTICLES_PER_TASK:-8}"  # Match 8 GPUs
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
N_LAYERS="${N_LAYERS:-4}"
HIDDEN_UNITS="${HIDDEN_UNITS:-64}"

# Resubmission mode - only process failed PIDs
RESUBMIT_MODE="${RESUBMIT_MODE:-false}"
FAILED_PIDS_LIST="${FAILED_PIDS_LIST:-}"
CHECK_SAMPLES="${CHECK_SAMPLES:-true}"  # Also check for sample files

# Output configuration - will be set based on H5 file location
H5_FILE_OVERRIDE="${H5_FILE_OVERRIDE:-}"

echo "📋 Configuration:"
echo "  Particles per task: $PARTICLES_PER_TASK"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Resubmit mode: $RESUBMIT_MODE"
if [[ "$RESUBMIT_MODE" == "true" && -n "$FAILED_PIDS_LIST" ]]; then
    echo "  Failed PIDs to retry: $FAILED_PIDS_LIST"
fi
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

# Smart H5 file discovery
find_h5_file() {
    if [[ -n "$H5_FILE_OVERRIDE" && -f "$H5_FILE_OVERRIDE" ]]; then
        echo "$H5_FILE_OVERRIDE"
        return 0
    fi
    
    # Look for eden_scaled files first (newer format)
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
    
    # Final fallback
    echo "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/all_in_one.h5"
}

H5_FILE=$(find_h5_file)
echo "📁 Selected H5 file: $H5_FILE"

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

# Function to get particle size from H5 file
get_particle_size() {
    local pid=$1
    local size=$(python -c "
import h5py
import numpy as np
import sys

try:
    with h5py.File('$H5_FILE', 'r') as f:
        # Try different possible structures
        if 'PartType1' in f:
            if 'ParticleIDs' in f['PartType1']:
                pids = f['PartType1']['ParticleIDs'][:]
                if 'Coordinates' in f['PartType1']:
                    coords = f['PartType1']['Coordinates'][:]
                    # Count objects for this specific PID
                    pid_mask = (pids == $pid)
                    size = np.sum(pid_mask)
                    print(size)
                else:
                    # Fallback: estimate from total size
                    total_size = len(pids)
                    unique_pids = len(np.unique(pids))
                    avg_size = total_size // unique_pids if unique_pids > 0 else 1000
                    print(avg_size)
        elif 'particles' in f:
            if 'ParticleIDs' in f['particles']:
                pids = f['particles']['ParticleIDs'][:]
                pid_mask = (pids == $pid)
                size = np.sum(pid_mask)
                print(size)
            else:
                print(50000)  # Default estimate
        else:
            # Fallback for unknown structure
            print(50000)
except Exception as e:
    print(50000, file=sys.stderr)  # Default on error
    print(50000)
" 2>/dev/null)
    
    # Ensure we have a valid number
    if [[ ! "$size" =~ ^[0-9]+$ ]]; then
        size=50000  # Default fallback
    fi
    
    echo $size
}

# Function to check if any particle in the task has >100k objects
check_large_particles() {
    local has_large=false
    local max_size=0
    
    echo "🔍 Checking particle sizes for time allocation..."
    
    for pid in "${TASK_PIDS[@]}"; do
        local size=$(get_particle_size $pid)
        echo "  PID $pid: $size objects"
        
        if [[ $size -gt $max_size ]]; then
            max_size=$size
        fi
        
        if [[ $size -gt 100000 ]]; then
            has_large=true
        fi
    done
    
    echo "📊 Largest particle: $max_size objects"
    
    if [[ "$has_large" == "true" ]]; then
        echo "⏰ Large particles detected (>100k objects) - will need extended runtime"
        echo "💡 Recommendation: Use 12+ hour time allocation for this job"
    else
        echo "⚡ All particles are small (<100k objects) - standard runtime sufficient"
    fi
    
    echo $has_large
}

# Create output directories - save in same parent directory as H5 file with halo/PID structure
H5_PARENT_DIR=$(dirname "$H5_FILE")
OUTPUT_BASE_DIR="$H5_PARENT_DIR/tfp_output"
MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/${DATA_SOURCE}/halo${HALO_ID}"
SAMPLES_DIR="$OUTPUT_BASE_DIR/samples/${DATA_SOURCE}/halo${HALO_ID}"

echo "📁 H5 file parent: $H5_PARENT_DIR"
echo "📁 Output base: $OUTPUT_BASE_DIR"
echo "📁 Data source: $DATA_SOURCE, Halo ID: $HALO_ID"

# Use file locking for directory creation
create_directories_safely() {
    local lock_file="/tmp/tfp_mkdir_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.lock"
    
    # Try to acquire lock for 30 seconds
    local attempts=0
    while ! mkdir "$lock_file" 2>/dev/null && [ $attempts -lt 30 ]; do
        sleep 1
        ((attempts++))
    done
    
    if [ $attempts -ge 30 ]; then
        echo "❌ Failed to acquire directory creation lock"
        exit 1
    fi
    
    # Create directories
    mkdir -p "$MODEL_DIR" "$SAMPLES_DIR"
    mkdir -p logs
    
    # Release lock
    rmdir "$lock_file" 2>/dev/null || true
}

create_directories_safely

# Function to check if a particle is truly completed
is_particle_completed() {
    local pid=$1
    local model_file="$MODEL_DIR/model_pid${pid}.npz"
    local results_file="$MODEL_DIR/model_pid${pid}_results.json"
    
    # Check for required model files
    if [[ ! -f "$model_file" || ! -f "$results_file" ]]; then
        return 1  # Not completed
    fi
    
    # Check model file is not empty
    if [[ ! -s "$model_file" ]]; then
        echo "⚠️ Model file exists but is empty: $model_file"
        return 1
    fi
    
    # Check results file is valid JSON and not empty
    if ! python -c "import json; json.load(open('$results_file'))" 2>/dev/null; then
        echo "⚠️ Results file exists but is invalid JSON: $results_file"
        return 1
    fi
    
    # Optionally check for sample files if requested
    if [[ "$CHECK_SAMPLES" == "true" ]]; then
        local sample_file_npz="$SAMPLES_DIR/model_pid${pid}_samples.npz"
        local sample_file_h5="$SAMPLES_DIR/model_pid${pid}_samples.h5"
        
        if [[ ! -f "$sample_file_npz" && ! -f "$sample_file_h5" ]]; then
            echo "⚠️ Sample files missing for PID $pid"
            return 1
        fi
        
        # Check sample file is not empty
        if [[ -f "$sample_file_npz" && ! -s "$sample_file_npz" ]]; then
            echo "⚠️ Sample file exists but is empty: $sample_file_npz"
            return 1
        fi
        if [[ -f "$sample_file_h5" && ! -s "$sample_file_h5" ]]; then
            echo "⚠️ Sample file exists but is empty: $sample_file_h5"
            return 1
        fi
    fi
    
    return 0  # Completed successfully
}

# Function to scan for failed particles and generate resubmission commands
scan_and_generate_resubmit() {
    echo "🔍 Scanning for failed particles..."
    
    local total_expected=$1
    local failed_pids=()
    local completed_count=0
    
    for ((pid=1; pid<=total_expected; pid++)); do
        if is_particle_completed $pid; then
            ((completed_count++))
        else
            failed_pids+=($pid)
        fi
    done
    
    echo "📊 Scan Results:"
    echo "  ✅ Completed: $completed_count"
    echo "  ❌ Failed/Missing: ${#failed_pids[@]}"
    
    if [[ ${#failed_pids[@]} -gt 0 ]]; then
        echo "🚨 Failed PIDs: ${failed_pids[*]}"
        
        # Generate resubmission command
        local failed_list=$(IFS=','; echo "${failed_pids[*]}")
        local resubmit_file="resubmit_failed_$(date +%Y%m%d_%H%M%S).sh"
        
        cat > "$resubmit_file" << EOF
#!/bin/bash
# Auto-generated resubmission script for failed particles
# Generated: $(date)
# Failed PIDs: ${failed_pids[*]}

echo "🔄 Resubmitting ${#failed_pids[@]} failed particles..."

# Calculate how many array tasks needed
failed_count=${#failed_pids[@]}
particles_per_task=$PARTICLES_PER_TASK
array_tasks=\$(( (failed_count + particles_per_task - 1) / particles_per_task ))

echo "Array tasks needed: \$array_tasks"

# Submit with failed PIDs list
FAILED_PIDS_LIST="$failed_list" RESUBMIT_MODE=true sbatch --array=1-\$array_tasks%5 submit_tfp_array.sh

echo "✅ Resubmission job submitted!"
EOF
        
        chmod +x "$resubmit_file"
        echo "📝 Generated resubmission script: $resubmit_file"
        echo "🚀 To resubmit failed particles, run: ./$resubmit_file"
    else
        echo "🎉 All particles completed successfully!"
    fi
}

# Determine PIDs to process based on mode
if [[ "$RESUBMIT_MODE" == "true" && -n "$FAILED_PIDS_LIST" ]]; then
    # Resubmission mode: process specific failed PIDs
    echo "🔄 RESUBMISSION MODE: Processing failed PIDs"
    
    # Convert comma-separated list to array
    IFS=',' read -ra ALL_FAILED_PIDS <<< "$FAILED_PIDS_LIST"
    
    # Calculate which PIDs this array task should handle
    START_INDEX=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PARTICLES_PER_TASK ))
    END_INDEX=$(( $START_INDEX + $PARTICLES_PER_TASK - 1 ))
    
    # Get subset of failed PIDs for this task
    TASK_PIDS=()
    for ((i=START_INDEX; i<=END_INDEX && i<${#ALL_FAILED_PIDS[@]}; i++)); do
        TASK_PIDS+=("${ALL_FAILED_PIDS[$i]}")
    done
    
    echo "🎯 This task will process PIDs: ${TASK_PIDS[*]}"
    
    # Check particle sizes for resubmitted PIDs
    HAS_LARGE_PARTICLES=$(check_large_particles)
else
    # Normal mode: calculate particle range for this array task
    START_PID=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PARTICLES_PER_TASK + 1 ))
    END_PID=$(( $SLURM_ARRAY_TASK_ID * $PARTICLES_PER_TASK ))
    
    # Create array of PIDs for this task
    TASK_PIDS=()
    for ((pid=START_PID; pid<=END_PID; pid++)); do
        TASK_PIDS+=($pid)
    done
    
    echo "🎯 Processing PIDs: $START_PID to $END_PID"
    
    # Check particle sizes for this task
    HAS_LARGE_PARTICLES=$(check_large_particles)
fi

echo "🎮 Using up to $PARTICLES_PER_TASK GPUs for parallel processing"
echo "📝 PIDs assigned to this task: ${TASK_PIDS[*]}"
echo

# Get available GPUs
AVAILABLE_GPUS=($(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '))
NUM_AVAILABLE_GPUS=${#AVAILABLE_GPUS[@]}

echo "Available GPUs: ${AVAILABLE_GPUS[*]} (Total: $NUM_AVAILABLE_GPUS)"

# Process particles in parallel using different GPUs
process_particle() {
    local pid=$1
    local gpu_id=$2
    local gpu_index=$3
    
    echo "--- Processing PID $pid on GPU $gpu_id (index $gpu_index) ---"
    
    # Check if already completed using comprehensive check
    if is_particle_completed $pid; then
        echo "✅ PID $pid already completed (verified), skipping"
        return 0
    fi
    
    # Clean up any partial/corrupted files
    echo "🧹 Cleaning up any partial files for PID $pid..."
    rm -f "$MODEL_DIR/model_pid${pid}.npz" 2>/dev/null || true
    rm -f "$MODEL_DIR/model_pid${pid}_results.json" 2>/dev/null || true
    rm -f "$SAMPLES_DIR/model_pid${pid}_samples.npz" 2>/dev/null || true
    rm -f "$SAMPLES_DIR/model_pid${pid}_samples.h5" 2>/dev/null || true
    
    # Get particle size for adaptive parameters
    local particle_size=$(get_particle_size $pid)
    echo "📊 PID $pid has $particle_size objects"
    
    # Adjust training parameters based on particle size
    local adaptive_epochs=$EPOCHS
    local adaptive_batch_size=$BATCH_SIZE
    
    if [[ $particle_size -gt 100000 ]]; then
        # Large particles: potentially reduce epochs and increase batch size for efficiency
        adaptive_epochs=$((EPOCHS * 80 / 100))  # 20% fewer epochs
        adaptive_batch_size=$((BATCH_SIZE * 150 / 100))  # 50% larger batches
        echo "⚡ Large particle optimization: epochs=$adaptive_epochs, batch_size=$adaptive_batch_size"
    elif [[ $particle_size -lt 10000 ]]; then
        # Small particles: can afford more epochs
        adaptive_epochs=$((EPOCHS * 120 / 100))  # 20% more epochs
        echo "🔬 Small particle optimization: epochs=$adaptive_epochs"
    fi
    
    # Set GPU for this process
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Create process-specific log
    local log_file="logs/tfp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_pid${pid}_gpu${gpu_index}.log"
    
    # Train the flow
    {
        echo "🚀 Starting training for PID $pid on GPU $gpu_id"
        echo "Time: $(date)"
        
        python train_tfp_flows.py \
            --data_path "$H5_FILE" \
            --particle_pid $pid \
            --output_dir "$MODEL_DIR" \
            --epochs $adaptive_epochs \
            --batch_size $adaptive_batch_size \
            --learning_rate $LEARNING_RATE \
            --n_layers $N_LAYERS \
            --hidden_units $HIDDEN_UNITS \
            --use_kroupa_imf \
            --validation_split 0.2 \
            --early_stopping_patience 20 \
            --reduce_lr_patience 10
        
        local exit_code=$?
        echo "Training completed with exit code: $exit_code"
        echo "Time: $(date)"
        
        if [ $exit_code -eq 0 ]; then
            echo "✅ PID $pid completed successfully on GPU $gpu_id"
        else
            echo "❌ PID $pid failed on GPU $gpu_id with exit code $exit_code"
        fi
        
        return $exit_code
        
    } 2>&1 | tee "$log_file"
}

# Special handling for scan mode (when array task ID is 0 or scan_failures is set)
if [[ "$SLURM_ARRAY_TASK_ID" == "0" || "${SCAN_FAILURES:-false}" == "true" ]]; then
    echo "🔍 SCAN MODE: Checking all particles and generating resubmission commands"
    
    # Determine total expected particles (could be parameterized)
    TOTAL_EXPECTED="${TOTAL_PARTICLES:-1000}"
    scan_and_generate_resubmit $TOTAL_EXPECTED
    exit 0
fi

# Launch parallel training processes
pids_array=()
gpu_processes=()

# Process the PIDs assigned to this task
for ((i=0; i<${#TASK_PIDS[@]}; i++)); do
    particle_pid=${TASK_PIDS[$i]}
    gpu_index=$((i % NUM_AVAILABLE_GPUS))
    gpu_id=${AVAILABLE_GPUS[$gpu_index]}
    
    echo "🚀 Launching PID $particle_pid on GPU $gpu_id (background process)"
    
    # Launch in background
    process_particle $particle_pid $gpu_id $gpu_index &
    
    # Store the background process PID and particle PID
    gpu_processes+=($!)
    pids_array+=($particle_pid)
    
    # Small delay to avoid race conditions
    sleep 2
done

echo
echo "🔄 Waiting for all training processes to complete..."
echo "Launched ${#gpu_processes[@]} parallel training processes"

# Wait for all background processes and collect results
success_count=0
failure_count=0

for ((i=0; i<${#gpu_processes[@]}; i++)); do
    process_pid=${gpu_processes[$i]}
    particle_pid=${pids_array[$i]}
    
    echo "Waiting for PID $particle_pid (process $process_pid)..."
    
    if wait $process_pid; then
        echo "✅ PID $particle_pid completed successfully"
        ((success_count++))
    else
        echo "❌ PID $particle_pid failed"
        ((failure_count++))
    fi
done

echo
echo "🏁 Array task $SLURM_ARRAY_TASK_ID completed"
echo "📊 Results:"
echo "  ✅ Successful: $success_count"
echo "  ❌ Failed: $failure_count"
echo "  📈 Success rate: $((success_count * 100 / (success_count + failure_count)))%"
echo "  🕐 Total runtime: $SECONDS seconds"
echo "  📁 Output location: $MODEL_DIR"
echo

# Generate failure summary for easy resubmission tracking
if [[ $failure_count -gt 0 ]]; then
    failed_in_this_task=()
    for ((i=0; i<${#gpu_processes[@]}; i++)); do
        process_pid=${gpu_processes[$i]}
        particle_pid=${pids_array[$i]}
        
        # Check if this process failed (we already waited for it above)
        if ! wait $process_pid 2>/dev/null; then
            failed_in_this_task+=($particle_pid)
        fi
    done
    
    echo "🚨 Failed PIDs in this task: ${failed_in_this_task[*]}"
    
    # Append to global failure log
    failure_log="$OUTPUT_BASE_DIR/failed_particles.log"
    for failed_pid in "${failed_in_this_task[@]}"; do
        echo "$(date '+%Y-%m-%d %H:%M:%S') Task-$SLURM_ARRAY_TASK_ID PID-$failed_pid" >> "$failure_log"
    done
    
    echo "📝 Failures logged to: $failure_log"
    echo "💡 To scan all failures and generate resubmission: SCAN_FAILURES=true sbatch --array=0 submit_tfp_array.sh"
fi
echo

# Set exit code based on results
if [ $failure_count -eq 0 ]; then
    echo "🎉 All particles completed successfully!"
    exit 0
elif [ $success_count -gt 0 ]; then
    echo "⚠️ Partial success: $success_count/$((success_count + failure_count)) completed"
    exit 0
else
    echo "💥 All particles failed!"
    exit 1
fi
