#!/bin/bash
#SBATCH --job-name="auto_submit_flows"
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/auto_submit_%j.out
#SBATCH --error=logs/auto_submit_%j.err

# Intelligent auto-submission script for TensorFlow Probability flows
# Automatically detects unprocessed particles and submits them efficiently

set -e

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    🤖 INTELLIGENT FLOW SUBMISSION SYSTEM                    ║"
echo "╠══════════════════════════════════════════════════════════════════════════════╣"
echo "║  🎯 Automatically detecting unprocessed particles                           ║"
echo "║  📊 Managing batch submissions with queue monitoring                         ║"
echo "║  ⏰ Started: $(date)                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo

# Configuration
MAX_CONCURRENT=5  # Based on your JobArrayTaskLimit
BATCH_SIZE=20     # Tasks per submission batch
PARTICLES_PER_TASK=50
# WAIT_TIME removed - no sleep/wait commands allowed on Sherlock
OUTPUT_BASE_DIR="/oak/stanford/orgs/kipac/users/caganze/tfp_flows_output"

# Find H5 files and detect available particles
find_h5_files_and_particles() {
    echo "┌─────────────────────────────────────────────────────────────────────────────┐"
    echo "│ 🔍 SCANNING FOR H5 FILES AND AVAILABLE PARTICLES                           │"
    echo "└─────────────────────────────────────────────────────────────────────────────┘"
    
    # Look for eden_scaled files
    local h5_files=($(find /oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/ -name "eden_scaled_Halo*" -type f 2>/dev/null))
    
    if [[ ${#h5_files[@]} -eq 0 ]]; then
        echo "⚠️  No eden_scaled files found, using fallback"
        h5_files=("/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/all_in_one.h5")
    fi
    
    echo "📁 Found ${#h5_files[@]} H5 files:"
    for file in "${h5_files[@]}"; do
        echo "   └── $(basename "$file")"
    done
    
    # For now, use the first available file
    H5_FILE="${h5_files[0]}"
    echo "✅ Selected: $(basename "$H5_FILE")"
    echo
}

# Check which particles are already completed
check_completed_particles() {
    echo "┌─────────────────────────────────────────────────────────────────────────────┐" >&2
    echo "│ 🔍 CHECKING FOR COMPLETED PARTICLES                                        │" >&2
    echo "└─────────────────────────────────────────────────────────────────────────────┘" >&2
    
    local completed_pids=()
    if [[ -d "$OUTPUT_BASE_DIR/trained_flows" ]]; then
        # Search in new hierarchical structure: source/haloXXX/
        for source_dir in "$OUTPUT_BASE_DIR/trained_flows"/*; do
            if [[ -d "$source_dir" ]]; then
                for halo_dir in "$source_dir"/halo*/; do
                    if [[ -d "$halo_dir" ]]; then
                        for model_file in "$halo_dir"/model_pid*.npz; do
                            if [[ -f "$model_file" ]]; then
                                local pid=$(basename "$model_file" | sed 's/model_pid\([0-9]*\)\.npz/\1/')
                                local results_file="${model_file%.npz}_results.json"
                                
                                if [[ -f "$model_file" && -f "$results_file" ]]; then
                                    completed_pids+=($pid)
                                fi
                            fi
                        done
                    fi
                done
            fi
        done
    fi
    
    echo "📊 Found ${#completed_pids[@]} completed particles" >&2
    if [[ ${#completed_pids[@]} -gt 0 ]]; then
        echo "   📈 Range: ${completed_pids[0]} → ${completed_pids[-1]}" >&2
    fi
    
    # Return as space-separated string
    echo "${completed_pids[@]}"
}

# Detect available particles from H5 file
detect_available_particles() {
    echo "┌─────────────────────────────────────────────────────────────────────────────┐" >&2
    echo "│ 🔬 ANALYZING H5 FILE STRUCTURE                                             │" >&2
    echo "└─────────────────────────────────────────────────────────────────────────────┘" >&2
    
    python3 -c "
import h5py
import numpy as np
import sys

try:
    with h5py.File('$H5_FILE', 'r') as f:
        # Try different possible structures
        if 'parentid' in f:
            pids = f['parentid'][:]
        elif 'PartType1' in f and 'ParticleIDs' in f['PartType1']:
            pids = f['PartType1']['ParticleIDs'][:]
        else:
            # Fallback: assume PIDs 1-1000
            pids = np.arange(1, 1001)
        
        unique_pids = np.unique(pids)
        print(f'📊 Available PIDs: {len(unique_pids)} particles', file=sys.stderr)
        print(f'📈 Range: {unique_pids.min()} → {unique_pids.max()}', file=sys.stderr)
        
        # Output all PIDs for processing
        print(' '.join(map(str, unique_pids)))
        
except Exception as e:
    print(f'❌ Error reading H5 file: {e}', file=sys.stderr)
    # Fallback: use PIDs 1-200
    print(' '.join(map(str, range(1, 201))))
"
}

# Calculate particles needing processing
calculate_needed_particles() {
    local available_pids_str="$1"
    local completed_pids_str="$2"
    
    # Use stdin to avoid string interpolation issues
    python3 << EOF
import sys

# Read from environment variables to avoid string escaping issues
available_str = """$available_pids_str"""
completed_str = """$completed_pids_str"""

# Clean the strings by taking only the last line (the actual PID list)
available_lines = available_str.strip().split('\n')
completed_lines = completed_str.strip().split('\n')

# Get the actual PID lists (last non-empty line)
available_pids_line = ""
for line in reversed(available_lines):
    if line.strip() and not any(word in line.lower() for word in ['detecting', 'available', 'range', 'checking', 'found']):
        available_pids_line = line.strip()
        break

completed_pids_line = ""
for line in reversed(completed_lines):
    if line.strip() and not any(word in line.lower() for word in ['detecting', 'available', 'range', 'checking', 'found']):
        completed_pids_line = line.strip()
        break

# Parse the PIDs
available = set()
if available_pids_line:
    try:
        available = set(map(int, available_pids_line.split()))
    except ValueError:
        pass

completed = set()
if completed_pids_line:
    try:
        completed = set(map(int, completed_pids_line.split()))
    except ValueError:
        pass

needed = sorted(available - completed)
print('┌─────────────────────────────────────────────────────────────────────────────┐')
print('│ 📊 PROCESSING STATUS SUMMARY                                               │')
print('└─────────────────────────────────────────────────────────────────────────────┘')
print(f'📈 Available particles: {len(available)}')
print(f'✅ Completed particles: {len(completed)}')
print(f'🎯 Needed particles: {len(needed)}')

if needed:
    next_pids = needed[:10]
    if len(needed) > 10:
        print(f'🚀 Next PIDs to process: {next_pids}...')
    else:
        print(f'🚀 PIDs to process: {needed}')
    print(' '.join(map(str, needed)))
else:
    print('🎉 All particles completed!')
EOF
}

# Check current job queue status
check_queue_status() {
    local running=$(squeue -u $USER -t running | wc -l)
    local pending=$(squeue -u $USER -t pending | wc -l)
    
    # Subtract 1 for header, but ensure we don't go negative
    local running_count=$((running > 1 ? running - 1 : 0))
    local pending_count=$((pending > 1 ? pending - 1 : 0))
    local total=$((running_count + pending_count))
    
    echo "┌─────────────────────────────────────────────────────────────────────────────┐" >&2
    echo "│ 📊 CURRENT QUEUE STATUS                                                    │" >&2
    echo "└─────────────────────────────────────────────────────────────────────────────┘" >&2
    echo "🏃 Running jobs: $running_count" >&2
    echo "⏳ Pending jobs: $pending_count" >&2
    echo "📈 Total jobs: $total" >&2
    
    echo $total
}

# Submit a batch of particles
submit_particle_batch() {
    local start_pid=$1
    local end_pid=$2
    local batch_name=$3
    
    echo "┌─────────────────────────────────────────────────────────────────────────────┐"
    echo "│ 🚀 SUBMITTING BATCH JOB                                                    │"
    echo "└─────────────────────────────────────────────────────────────────────────────┘"
    echo "🎯 Job name: $batch_name"
    echo "📊 PIDs: $start_pid → $end_pid"
    
    # Calculate task range (each task processes PARTICLES_PER_TASK particles)
    local start_task=$(( (start_pid - 1) / PARTICLES_PER_TASK + 1 ))
    local end_task=$(( (end_pid - 1) / PARTICLES_PER_TASK + 1 ))
    
    echo "📈 Task range: $start_task → $end_task"
    echo "🔢 Max concurrent: $MAX_CONCURRENT"
    
    local job_output=$(sbatch --job-name="$batch_name" \
                              --array=${start_task}-${end_task}%${MAX_CONCURRENT} \
                              submit_flows_array.sh 2>&1)
    
    if [[ $? -eq 0 ]]; then
        local job_id=$(echo "$job_output" | grep -o '[0-9]\+')
        echo "✅ Success! Job ID: $job_id"
        return 0
    else
        echo "❌ Submission failed: $job_output"
        return 1
    fi
}

# Main submission loop
main_submission_loop() {
    local needed_pids_str="$1"
    local needed_pids=($needed_pids_str)
    
    if [[ ${#needed_pids[@]} -eq 0 ]]; then
        echo "╔══════════════════════════════════════════════════════════════════════════════╗"
        echo "║                           🎉 ALL PARTICLES COMPLETED!                       ║"
        echo "╚══════════════════════════════════════════════════════════════════════════════╝"
        return 0
    fi
    
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                        🚀 STARTING INTELLIGENT SUBMISSION                   ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    echo "║  🎯 Total particles to process: ${#needed_pids[@]}                                    ║"
    echo "║  📊 Batch size: $((BATCH_SIZE * PARTICLES_PER_TASK)) particles per batch                             ║"
    echo "║  ⏰ Queue monitoring: Immediate check (no wait)                               ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo
    
    local batch_num=1
    local pids_per_batch=$((BATCH_SIZE * PARTICLES_PER_TASK))
    local num_needed=${#needed_pids[@]}

    for ((i=0; i<$num_needed; i+=pids_per_batch)); do
        echo "┌─────────────────────────────────────────────────────────────────────────────┐"
        echo "│ 🔄 BATCH $batch_num PROCESSING                                               │"
        echo "└─────────────────────────────────────────────────────────────────────────────┘"
        
        # Check queue status and get total jobs
        local current_jobs=$(check_queue_status)
        
        # Check if we have too many jobs (no sleep, just immediate check)
        if [[ $current_jobs -ge 50 ]]; then  # Conservative limit
            echo "⚠️  Queue full ($current_jobs jobs), skipping submission for now"
            echo "💡 Run this script again later when queue has space"
            return 1
        fi
        
        # Calculate batch start and end PIDs from the array
        local batch_start_pid=${needed_pids[$i]}
        
        # Get the index of the last PID in the batch
        local last_idx=$((i + pids_per_batch - 1))
        
        # Ensure the last index doesn't go beyond the array
        if [[ $last_idx -ge $num_needed ]]; then
            last_idx=$((num_needed - 1))
        fi
        
        local batch_end_pid=${needed_pids[$last_idx]}
        
        # Submit batch
        submit_particle_batch "$batch_start_pid" "$batch_end_pid" "tfp_auto_batch${batch_num}"
        
        if [[ $? -eq 0 ]]; then
            echo "✅ Batch $batch_num submitted successfully"
            ((batch_num++))
            
            # No delay needed - immediate submission
            echo "🚀 Ready for next batch submission..."
        else
            echo "❌ Batch submission failed, stopping"
            break
        fi
        
        echo
    done
    
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                           🎯 SUBMISSION COMPLETE!                           ║"
    echo "╠══════════════════════════════════════════════════════════════════════════════╣"
    echo "║  📊 Monitor progress: watch 'squeue -u \$USER'                               ║"
    echo "║  📁 Check outputs: ls /oak/stanford/orgs/kipac/users/caganze/tfp_flows_output/trained_flows/ ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
}

# Main execution
main() {
    find_h5_files_and_particles
    
    local completed_pids=$(check_completed_particles)
    echo
    
    local available_output=$(detect_available_particles)
    local available_pids=$(echo "$available_output" | tail -1)
    echo "$available_output" | head -n -1  # Print all but last line
    echo
    
    local needed_result=$(calculate_needed_particles "$available_pids" "$completed_pids")
    local needed_pids=$(echo "$needed_result" | tail -1)
    echo "$needed_result" | head -n -1  # Print all but last line
    echo
    
    main_submission_loop "$needed_pids"
}

# Run the script
main "$@"
