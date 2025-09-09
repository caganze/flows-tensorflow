#!/bin/bash

# 🔍 Scan and Resubmit Failed TFP Flows
# Comprehensive failure detection and intelligent resubmission

set -e

echo "🔍 TFP Flows Failure Scanner & Resubmitter"
echo "=========================================="
echo "Time: $(date)"
echo

# Configuration
PARTICLE_LIST_FILE="${PARTICLE_LIST_FILE:-particle_list.txt}"
CHECK_SAMPLES="${CHECK_SAMPLES:-true}"
DRY_RUN="${DRY_RUN:-false}"

echo "📋 Configuration:"
echo "  Particle list file: $PARTICLE_LIST_FILE"
echo "  Check sample files: $CHECK_SAMPLES"
echo "  Dry run mode: $DRY_RUN"
echo

# Check if particle list exists
if [[ ! -f "$PARTICLE_LIST_FILE" ]]; then
    echo "❌ ERROR: Particle list file not found: $PARTICLE_LIST_FILE"
    echo "💡 Run ./generate_particle_list.sh first to create the particle list"
    exit 1
fi

TOTAL_PARTICLES=$(wc -l < "$PARTICLE_LIST_FILE")
echo "📊 Found $TOTAL_PARTICLES particles in particle list"
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
FILENAME=$(basename "$H5_FILE")
HALO_ID=$(echo "$FILENAME" | sed 's/.*Halo\([0-9][0-9]*\).*/\1/')

# Determine data source
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
    HALO_ID="000"
    DATA_SOURCE="symphony"
fi

# Output directories - use consistent tfp_output directory
OUTPUT_BASE_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output"
MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/${DATA_SOURCE}/halo${HALO_ID}"
SAMPLES_DIR="$OUTPUT_BASE_DIR/samples/${DATA_SOURCE}/halo${HALO_ID}"

echo "📁 Output base: $OUTPUT_BASE_DIR"
echo "📁 Using data source: $DATA_SOURCE, Halo ID: $HALO_ID"
echo "📁 Model directory: $MODEL_DIR"
echo "📁 Samples directory: $SAMPLES_DIR"
echo

# Function to get particle size from H5 file (robust detection)
get_particle_size() {
    local pid=$1
    local size=$(python -c "
import h5py
import numpy as np
import sys

def get_particle_size_robust(h5_file_path, pid):
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Method 1: PartType1 with ParticleIDs (most common)
            if 'PartType1' in f and 'ParticleIDs' in f['PartType1']:
                pids = f['PartType1']['ParticleIDs'][:]
                pid_mask = (pids == pid)
                size = np.sum(pid_mask)
                if size > 0:
                    return size
                    
            # Method 2: Check for parentid field
            elif 'parentid' in f:
                pids = f['parentid'][:]
                pid_mask = (pids == pid)
                size = np.sum(pid_mask)
                if size > 0:
                    return size
                    
            # Method 3: PartType1 with other ID fields
            elif 'PartType1' in f:
                part1 = f['PartType1']
                id_fields = ['IDs', 'HaloID', 'SubhaloID', 'ParentID', 'parentid']
                for id_field in id_fields:
                    if id_field in part1:
                        pids = part1[id_field][:]
                        pid_mask = (pids == pid)
                        size = np.sum(pid_mask)
                        if size > 0:
                            return size
                            
            # Method 4: particles structure
            elif 'particles' in f:
                particles = f['particles']
                id_fields = ['ParticleIDs', 'IDs', 'HaloID', 'parentid']
                for id_field in id_fields:
                    if id_field in particles:
                        pids = particles[id_field][:]
                        pid_mask = (pids == pid)
                        size = np.sum(pid_mask)
                        if size > 0:
                            return size
                            
            # Method 5: Estimate based on file structure and PID
            total_particles = 0
            if 'PartType1' in f and 'Coordinates' in f['PartType1']:
                total_particles = len(f['PartType1']['Coordinates'])
            
            if total_particles > 0:
                # Realistic estimates based on PID
                if pid <= 10:
                    return min(500000, total_particles // 2)  # Large halos
                elif pid <= 100:
                    return min(200000, total_particles // 5)  # Medium halos
                elif pid <= 500:
                    return min(50000, total_particles // 20)  # Small halos
                else:
                    return min(10000, total_particles // 100) # Very small halos
                    
    except Exception as e:
        pass
        
    # Ultimate fallback - use PID-based estimation
    if pid <= 10:
        return 300000  # Assume large halos for low PIDs
    elif pid <= 100:
        return 150000  # Medium halos
    elif pid <= 500:
        return 75000   # Small halos
    else:
        return 25000   # Very small halos

print(get_particle_size_robust('$H5_FILE', $pid))
" 2>/dev/null)
    
    # Ensure we have a valid number
    if [[ ! "$size" =~ ^[0-9]+$ ]] || [[ $size -eq 0 ]]; then
        # PID-based fallback estimation
        if [[ $pid -le 10 ]]; then
            size=300000  # Large halos
        elif [[ $pid -le 100 ]]; then
            size=150000  # Medium halos
        elif [[ $pid -le 500 ]]; then
            size=75000   # Small halos
        else
            size=25000   # Very small halos
        fi
    fi
    
    echo $size
}

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
        return 1
    fi
    
    # Check results file is valid JSON
    if ! python -c "import json; json.load(open('$results_file'))" 2>/dev/null; then
        return 1
    fi
    
    # Optionally check for sample files
    if [[ "$CHECK_SAMPLES" == "true" ]]; then
        local sample_file_npz="$SAMPLES_DIR/model_pid${pid}_samples.npz"
        local sample_file_h5="$SAMPLES_DIR/model_pid${pid}_samples.h5"
        
        if [[ ! -f "$sample_file_npz" && ! -f "$sample_file_h5" ]]; then
            return 1
        fi
        
        # Check sample file is not empty
        if [[ -f "$sample_file_npz" && ! -s "$sample_file_npz" ]]; then
            return 1
        fi
        if [[ -f "$sample_file_h5" && ! -s "$sample_file_h5" ]]; then
            return 1
        fi
    fi
    
    return 0  # Completed successfully
}

# Scan all particles
echo "🔍 Scanning particles 1 to $TOTAL_PARTICLES..."
echo

failed_pids=()
completed_pids=()
partial_pids=()
large_particles=()
small_particles=()

# Progress tracking
last_progress=0

# Arrays to track H5 files for failed/partial particles
failed_h5_files=()
partial_h5_files=()
line_count=0

while IFS=',' read -r pid h5_file count category; do
    line_count=$((line_count + 1))
    
    # Show progress
    progress=$((line_count * 100 / TOTAL_PARTICLES))
    if (( progress >= last_progress + 10 )); then
        echo "📊 Progress: $progress% (PID $pid from $(basename "$h5_file"))"
        last_progress=$progress
    fi
    
    # Categorize by size (from particle list)
    if [[ "$category" == "Large" ]]; then
        large_particles+=($pid)
    else
        small_particles+=($pid)
    fi
    
    if is_particle_completed $pid; then
        completed_pids+=($pid)
    else
        # Check if partially completed
        model_file="$MODEL_DIR/model_pid${pid}.npz"
        results_file="$MODEL_DIR/model_pid${pid}_results.json"
        
        if [[ -f "$model_file" || -f "$results_file" ]]; then
            partial_pids+=($pid)
            partial_h5_files+=("$h5_file")
        else
            failed_pids+=($pid)
            failed_h5_files+=("$h5_file")
        fi
    fi
done < "$PARTICLE_LIST_FILE"

echo
echo "📊 SCAN RESULTS"
echo "==============="
echo "✅ Completed: ${#completed_pids[@]}"
echo "⚠️  Partial: ${#partial_pids[@]}"
echo "❌ Failed/Missing: ${#failed_pids[@]}"
echo "📈 Success rate: $((${#completed_pids[@]} * 100 / TOTAL_PARTICLES))%"
echo
echo "📊 PARTICLE SIZE ANALYSIS"
echo "======================="
echo "🐋 Large particles (>100k objects): ${#large_particles[@]}"
echo "🐜 Small particles (<100k objects): ${#small_particles[@]}"
echo

# Show some examples
if [[ ${#completed_pids[@]} -gt 0 ]]; then
    echo "✅ Example completed PIDs: ${completed_pids[@]:0:10}..."
fi

if [[ ${#partial_pids[@]} -gt 0 ]]; then
    echo "⚠️ Example partial PIDs: ${partial_pids[@]:0:10}..."
fi

if [[ ${#failed_pids[@]} -gt 0 ]]; then
    echo "❌ Example failed PIDs: ${failed_pids[@]:0:10}..."
fi

echo

# Generate resubmission for all incomplete (partial + failed)
incomplete_pids=("${partial_pids[@]}" "${failed_pids[@]}")

if [[ ${#incomplete_pids[@]} -gt 0 ]]; then
    echo "🚨 RESUBMISSION REQUIRED"
    echo "========================"
    echo "Incomplete particles: ${#incomplete_pids[@]}"
    
    # Sort the incomplete PIDs
    IFS=$'\n' incomplete_sorted=($(sort -n <<<"${incomplete_pids[*]}"))
    unset IFS
    
    # Generate resubmission command
    failed_list=$(IFS=','; echo "${incomplete_sorted[*]}")
    
    # Separate large and small incomplete particles
    large_incomplete=()
    small_incomplete=()
    
    for pid in "${incomplete_sorted[@]}"; do
        particle_size=$(get_particle_size $pid)
        if [[ $particle_size -gt 100000 ]]; then
            large_incomplete+=($pid)
        else
            small_incomplete+=($pid)
        fi
    done
    
    echo "🔍 Incomplete particle breakdown:"
    echo "  🐋 Large incomplete: ${#large_incomplete[@]}"
    echo "  🐜 Small incomplete: ${#small_incomplete[@]}"
    
    # Calculate array parameters
    particles_per_task=8
    array_tasks=$(( (${#incomplete_pids[@]} + particles_per_task - 1) / particles_per_task ))
    
    # Recommend different time allocations
    if [[ ${#large_incomplete[@]} -gt 0 ]]; then
        recommended_time="12:00:00"  # 12 hours for large particles
        echo "⏰ Recommended time allocation: $recommended_time (contains large particles)"
    else
        recommended_time="04:00:00"  # 4 hours for small particles only
        echo "⏰ Recommended time allocation: $recommended_time (small particles only)"
    fi
    
    echo "📝 Array tasks needed: $array_tasks"
    echo "📝 Particles per task: $particles_per_task"
    
    # Create resubmission script
    resubmit_file="resubmit_failed_$(date +%Y%m%d_%H%M%S).sh"
    
    cat > "$resubmit_file" << EOF
#!/bin/bash
# Auto-generated resubmission script for incomplete particles
# Generated: $(date)
# Incomplete PIDs: ${#incomplete_pids[@]}
# Large particles (>100k): ${#large_incomplete[@]}
# Small particles (<100k): ${#small_incomplete[@]}
# PIDs: ${incomplete_sorted[*]:0:20}...

echo "🔄 Resubmitting ${#incomplete_pids[@]} incomplete particles..."
echo "📊 Array tasks: $array_tasks"
echo "⏰ Recommended time: $recommended_time"

# Submit resubmission job with appropriate time allocation
FAILED_PIDS_LIST="$failed_list" \\
RESUBMIT_MODE=true \\
CHECK_SAMPLES=$CHECK_SAMPLES \\
sbatch --time=$recommended_time --array=1-$array_tasks%5 submit_tfp_array.sh

echo "✅ Resubmission job submitted with $recommended_time time limit!"
echo "📊 Monitor with: squeue -u \$(whoami)"
echo "📝 Check logs in: logs/tfp_*.out"
EOF
    
    chmod +x "$resubmit_file"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "🧪 DRY RUN MODE - Would create: $resubmit_file"
        echo "📝 Resubmission command would be:"
        cat "$resubmit_file"
    else
        echo "📝 Generated resubmission script: $resubmit_file"
        echo
        echo "🚀 NEXT STEPS:"
        echo "1. Review the resubmission script: cat $resubmit_file"
        echo "2. Submit failed particles: ./$resubmit_file"
        echo "3. Monitor progress: squeue -u \$(whoami)"
        echo "4. Re-run this scanner after completion"
        echo
        echo "💡 Quick submit: ./$resubmit_file"
    fi
    
    # Save detailed failure report
    failure_report="failure_report_$(date +%Y%m%d_%H%M%S).txt"
    {
        echo "TFP Flows Failure Report"
        echo "Generated: $(date)"
        echo "========================"
        echo
        echo "Configuration:"
        echo "  Total particles checked: $TOTAL_PARTICLES"
        echo "  Output directory: $OUTPUT_BASE_DIR"
        echo "  Data source: $DATA_SOURCE"
        echo "  Halo ID: $HALO_ID"
        echo "  Check samples: $CHECK_SAMPLES"
        echo
        echo "Results:"
        echo "  ✅ Completed: ${#completed_pids[@]}"
        echo "  ⚠️ Partial: ${#partial_pids[@]}"
        echo "  ❌ Failed: ${#failed_pids[@]}"
        echo "  📈 Success rate: $((${#completed_pids[@]} * 100 / TOTAL_PARTICLES))%"
        echo
        echo "Incomplete PIDs (${#incomplete_pids[@]} total):"
        for pid in "${incomplete_sorted[@]}"; do
            echo "  $pid"
        done
        echo
        echo "Resubmission command:"
        echo "  ./$resubmit_file"
    } > "$failure_report"
    
    echo "📋 Detailed report saved: $failure_report"
    
else
    echo "🎉 ALL PARTICLES COMPLETED SUCCESSFULLY!"
    echo "No resubmission needed."
fi

echo
echo "🏁 Scan completed at $(date)"
