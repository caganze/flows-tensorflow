#!/bin/bash
#SBATCH --job-name="brute_force_particle_list"
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4
#SBATCH --output=logs/brute_force_%j.out
#SBATCH --error=logs/brute_force_%j.err

set -e

echo "🚀 BRUTE FORCE GPU JOB - PARTICLE LIST MODE"
echo "Started: $(date)"
echo "Node: ${SLURM_NODELIST:-local}"
echo "Job ID: ${SLURM_JOB_ID:-test}"
echo "Time limit: 12 hours"

# Environment setup
module --force purge
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0
export CUDA_VISIBLE_DEVICES=0

source ~/.bashrc
conda activate bosque

# Create directories
mkdir -p logs success_logs failed_jobs

# Check for particle list file
PARTICLE_LIST_FILE="particle_list.txt"

if [[ ! -f "$PARTICLE_LIST_FILE" ]]; then
    echo "❌ ERROR: Particle list file not found: $PARTICLE_LIST_FILE"
    echo "💡 Run ./generate_particle_list.sh first to create the particle list"
    exit 1
fi

TOTAL_PARTICLES=$(wc -l < "$PARTICLE_LIST_FILE")
echo "📋 Found $TOTAL_PARTICLES particles in particle list"
echo

# Brute force configuration
echo "⚙️ BRUTE FORCE CONFIGURATION"
echo "============================"
echo "🕐 Time limit: 12 hours"
echo "🔄 Mode: Sequential processing"
echo "📊 Total particles: $TOTAL_PARTICLES"
echo "🎮 GPUs available: $CUDA_VISIBLE_DEVICES"
echo

# Function to check if particle is completed
is_particle_completed() {
    local pid=$1
    local h5_file="$2"
    
    # Get output directories for this specific H5 file
    local h5_parent_dir=$(dirname "$h5_file")
    local output_base_dir="$h5_parent_dir/tfp_output"
    
    # Extract data source and halo ID
    local filename=$(basename "$h5_file")
    local halo_id=$(echo "$filename" | sed 's/.*Halo\([0-9][0-9]*\).*/\1/')
    
    if [[ "$filename" == *"eden_scaled"* ]]; then
        local data_source="eden"
    elif [[ "$filename" == *"symphonyHR_scaled"* ]]; then
        local data_source="symphony-hr"
    elif [[ "$filename" == *"symphony_scaled"* ]]; then
        local data_source="symphony"
    else
        local data_source="unknown"
    fi
    
    # Handle fallback file
    if [[ "$filename" == "all_in_one.h5" ]] || [[ "$halo_id" == "$filename" ]]; then
        halo_id="000"
        data_source="symphony"
    fi
    
    local model_dir="$output_base_dir/trained_flows/${data_source}/halo${halo_id}"
    local samples_dir="$output_base_dir/samples/${data_source}/halo${halo_id}"
    
    # Check if model exists and has reasonable size
    if [[ ! -f "$model_dir/model_pid${pid}.npz" ]] || [[ ! -s "$model_dir/model_pid${pid}.npz" ]]; then
        return 1  # Not completed
    fi
    
    # Check if sample file exists
    if [[ ! -f "$samples_dir/model_pid${pid}_samples.npz" ]] && [[ ! -f "$samples_dir/model_pid${pid}_samples.h5" ]]; then
        return 1  # Not completed
    fi
    
    return 0  # Completed
}

# Function to process a single particle
process_particle() {
    local pid=$1
    local h5_file="$2"
    local count=$3
    local category="$4"
    
    echo
    echo "🎯 Processing PID $pid ($count objects, $category)"
    echo "📁 H5 file: $(basename "$h5_file")"
    echo "⏰ Time: $(date)"
    
    # Check if already completed
    if is_particle_completed $pid "$h5_file"; then
        echo "✅ PID $pid already completed, skipping"
        echo "$(date) SUCCESS pid:$pid already_completed" >> success_logs/brute_force_success.log
        return 0
    fi
    
    # Get output directories for this specific H5 file
    local h5_parent_dir=$(dirname "$h5_file")
    local output_base_dir="$h5_parent_dir/tfp_output"
    
    # Extract data source and halo ID
    local filename=$(basename "$h5_file")
    local halo_id=$(echo "$filename" | sed 's/.*Halo\([0-9][0-9]*\).*/\1/')
    
    if [[ "$filename" == *"eden_scaled"* ]]; then
        local data_source="eden"
    elif [[ "$filename" == *"symphonyHR_scaled"* ]]; then
        local data_source="symphony-hr"
    elif [[ "$filename" == *"symphony_scaled"* ]]; then
        local data_source="symphony"
    else
        local data_source="unknown"
    fi
    
    # Handle fallback file
    if [[ "$filename" == "all_in_one.h5" ]] || [[ "$halo_id" == "$filename" ]]; then
        halo_id="000"
        data_source="symphony"
    fi
    
    local model_dir="$output_base_dir/trained_flows/${data_source}/halo${halo_id}"
    local samples_dir="$output_base_dir/samples/${data_source}/halo${halo_id}"
    
    # Create directories
    mkdir -p "$model_dir" "$samples_dir"
    
    # Adaptive parameters based on particle size
    local epochs=100
    local batch_size=512
    
    if [[ "$category" == "Large" ]]; then
        epochs=80    # Fewer epochs for large particles
        batch_size=768  # Larger batches for efficiency
        echo "⚡ Large particle optimization: epochs=$epochs, batch_size=$batch_size"
    elif [[ $count -lt 10000 ]]; then
        epochs=120   # More epochs for very small particles
        echo "🔬 Small particle optimization: epochs=$epochs"
    fi
    
    # Run training
    echo "🚀 Starting training..."
    local start_time=$(date +%s)
    
    if python train_tfp_flows.py \
        --data_path "$h5_file" \
        --particle_pid "$pid" \
        --output_dir "$model_dir" \
        --epochs $epochs \
        --batch_size $batch_size \
        --learning_rate 1e-3 \
        --n_layers 4 \
        --hidden_units 512; then
        
        local end_time=$(date +%s)
        local runtime=$((end_time - start_time))
        
        echo "✅ SUCCESS: PID $pid completed in ${runtime}s"
        echo "$(date) SUCCESS pid:$pid runtime:${runtime}s file:$(basename "$h5_file")" >> success_logs/brute_force_success.log
        
        # Verify completion
        if is_particle_completed $pid "$h5_file"; then
            echo "✅ Verification passed: All output files created"
            return 0
        else
            echo "⚠️ Warning: Training reported success but files incomplete"
            return 1
        fi
    else
        local end_time=$(date +%s)
        local runtime=$((end_time - start_time))
        
        echo "❌ FAILED: PID $pid failed after ${runtime}s"
        echo "$(date) FAILED pid:$pid runtime:${runtime}s file:$(basename "$h5_file")" >> failed_jobs/brute_force_failures.log
        return 1
    fi
}

# MAIN BRUTE FORCE LOOP - Process particles sequentially for 12 hours
echo
echo "🔄 STARTING BRUTE FORCE PARTICLE PROCESSING"
echo "==========================================="
echo "⏰ Start time: $(date)"
echo "⏱️ Will run until time limit (12 hours)"
echo

# Track statistics
particles_processed=0
particles_completed=0
particles_failed=0
particles_skipped=0
start_time=$(date +%s)
job_start_time=$start_time

# Function to check if we're approaching time limit (leave 30 minutes buffer)
check_time_limit() {
    local current_time=$(date +%s)
    local elapsed=$((current_time - job_start_time))
    local time_limit=$((12 * 3600 - 30 * 60))  # 12 hours - 30 minutes buffer
    
    if [[ $elapsed -gt $time_limit ]]; then
        echo "⏰ Approaching 12-hour time limit, stopping gracefully..."
        return 1
    fi
    return 0
}

# Main processing loop - go through particle list sequentially
while IFS=',' read -r pid h5_file count category; do
    particles_processed=$((particles_processed + 1))
    
    # Check time limit before starting each particle
    if ! check_time_limit; then
        echo "⏰ Time limit reached after processing $particles_processed particles"
        break
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Progress: $particles_processed/$TOTAL_PARTICLES particles processed"
    
    # Process the particle
    if process_particle "$pid" "$h5_file" "$count" "$category"; then
        particles_completed=$((particles_completed + 1))
    else
        # Check if it was skipped or failed
        if is_particle_completed "$pid" "$h5_file"; then
            particles_skipped=$((particles_skipped + 1))
        else
            particles_failed=$((particles_failed + 1))
        fi
    fi
    
    # Show running statistics every 10 particles
    if (( particles_processed % 10 == 0 )); then
        current_time=$(date +%s)
        elapsed=$((current_time - job_start_time))
        echo
        echo "📊 PROGRESS REPORT ($(date))"
        echo "  ⏱️ Elapsed time: $((elapsed / 3600))h $((elapsed % 3600 / 60))m"
        echo "  📈 Processed: $particles_processed/$TOTAL_PARTICLES"
        echo "  ✅ Completed: $particles_completed"
        echo "  ⏭️ Skipped: $particles_skipped"
        echo "  ❌ Failed: $particles_failed"
        echo
    fi
    
done < "$PARTICLE_LIST_FILE"

# Final statistics
end_time=$(date +%s)
total_runtime=$((end_time - job_start_time))

echo
echo "🏁 BRUTE FORCE JOB COMPLETED"
echo "============================"
echo "⏰ End time: $(date)"
echo "⏱️ Total runtime: $((total_runtime / 3600))h $((total_runtime % 3600 / 60))m $((total_runtime % 60))s"
echo "📊 Final Statistics:"
echo "  📈 Total processed: $particles_processed/$TOTAL_PARTICLES"
echo "  ✅ Successfully completed: $particles_completed"
echo "  ⏭️ Already completed (skipped): $particles_skipped"
echo "  ❌ Failed: $particles_failed"
echo "  📈 Success rate: $(( particles_completed * 100 / (particles_processed > 0 ? particles_processed : 1) ))%"
echo

if [[ $particles_failed -gt 0 ]]; then
    echo "💡 To reprocess failed particles, check: failed_jobs/brute_force_failures.log"
fi

echo "📁 Success log: success_logs/brute_force_success.log"
echo "🎉 Brute force processing completed!"
