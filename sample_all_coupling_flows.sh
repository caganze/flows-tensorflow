#!/bin/bash

# Script to sample from all trained coupling flow models
# Reads from particle_list.txt and runs sampling for each entry

set -e  # Exit on any error

# Configuration
PARTICLE_LIST_FILE="particle_list.txt"
BASE_OUTPUT_DIR="coupling_output"
LOG_FILE="sampling_log_$(date +%Y%m%d_%H%M%S).txt"
MAX_PARALLEL_JOBS=4  # Number of parallel sampling jobs

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Function to check if model files exist
check_model_exists() {
    local base_dir="$1"
    local halo_id="$2"
    local particle_pid="$3"
    
    local config_file="${base_dir}/flow_config_${halo_id}_${particle_pid}.pkl"
    local weights_file="${base_dir}/flow_weights_${halo_id}_${particle_pid}.index"
    local preproc_file="${base_dir}/coupling_flow_pid${particle_pid}_preprocessing.npz"
    
    if [[ -f "$config_file" && -f "$weights_file" && -f "$preproc_file" ]]; then
        return 0  # All files exist
    else
        return 1  # Missing files
    fi
}

# Function to run sampling for a single particle
run_sampling() {
    local particle_pid="$1"
    local halo_id="$2"
    local suite="$3"
    local count="$4"
    local size="$5"
    
    local base_dir="${BASE_OUTPUT_DIR}/${suite}/${halo_id,,}/pid${particle_pid}"
    
    print_status "Processing PID ${particle_pid} (${halo_id}, ${suite}, ${count} particles, ${size})"
    
    # Check if model files exist
    if ! check_model_exists "$base_dir" "$halo_id" "$particle_pid"; then
        print_warning "Model files not found for PID ${particle_pid}, skipping..."
        return 1
    fi
    
    # Check if samples already exist
    local output_file="${base_dir}/samples_${halo_id}_${particle_pid}.npz"
    if [[ -f "$output_file" ]]; then
        print_warning "Samples already exist for PID ${particle_pid}, skipping..."
        return 0
    fi
    
    # Run sampling
    print_status "Running sampling for PID ${particle_pid}..."
    
    if python sample_coupling_flow.py \
        --base_dir "$base_dir" \
        --halo_id "$halo_id" \
        --particle_pid "$particle_pid" \
        --suite "$suite" \
        --use_kroupa \
        --seed 42; then
        
        print_success "Successfully sampled PID ${particle_pid}"
        return 0
    else
        print_error "Failed to sample PID ${particle_pid}"
        return 1
    fi
}

# Main function
main() {
    print_status "Starting coupling flow sampling for all models..."
    print_status "Reading particle list from: ${PARTICLE_LIST_FILE}"
    print_status "Log file: ${LOG_FILE}"
    
    # Check if particle list file exists
    if [[ ! -f "$PARTICLE_LIST_FILE" ]]; then
        print_error "Particle list file not found: ${PARTICLE_LIST_FILE}"
        exit 1
    fi
    
    # Initialize counters
    local total_count=0
    local success_count=0
    local skip_count=0
    local error_count=0
    
    # Read particle list and process each entry
    while IFS=',' read -r particle_pid halo_id suite count size; do
        # Skip empty lines and comments
        if [[ -z "$particle_pid" || "$particle_pid" =~ ^# ]]; then
            continue
        fi
        
        # Remove any whitespace
        particle_pid=$(echo "$particle_pid" | xargs)
        halo_id=$(echo "$halo_id" | xargs)
        suite=$(echo "$suite" | xargs)
        count=$(echo "$count" | xargs)
        size=$(echo "$size" | xargs)
        
        total_count=$((total_count + 1))
        
        # Run sampling (with parallel job control if needed)
        if run_sampling "$particle_pid" "$halo_id" "$suite" "$count" "$size"; then
            success_count=$((success_count + 1))
        else
            # Check if it was skipped or failed
            local base_dir="${BASE_OUTPUT_DIR}/${suite}/${halo_id,,}/pid${particle_pid}"
            local output_file="${base_dir}/samples_${halo_id}_${particle_pid}.npz"
            if [[ -f "$output_file" ]]; then
                skip_count=$((skip_count + 1))
            else
                error_count=$((error_count + 1))
            fi
        fi
        
        # Log progress every 10 items
        if [[ $((total_count % 10)) -eq 0 ]]; then
            print_status "Progress: ${total_count} processed, ${success_count} successful, ${skip_count} skipped, ${error_count} errors"
        fi
        
    done < "$PARTICLE_LIST_FILE"
    
    # Final summary
    print_status "Sampling completed!"
    print_success "Total processed: ${total_count}"
    print_success "Successful: ${success_count}"
    print_warning "Skipped: ${skip_count}"
    if [[ $error_count -gt 0 ]]; then
        print_error "Errors: ${error_count}"
    fi
    
    # Save summary to log file
    {
        echo "Coupling Flow Sampling Summary - $(date)"
        echo "=========================================="
        echo "Total processed: ${total_count}"
        echo "Successful: ${success_count}"
        echo "Skipped: ${skip_count}"
        echo "Errors: ${error_count}"
        echo "Success rate: $(( (success_count * 100) / total_count ))%"
    } > "$LOG_FILE"
    
    print_status "Summary saved to: ${LOG_FILE}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -f, --file FILE     Particle list file (default: particle_list.txt)"
    echo "  -o, --output DIR    Base output directory (default: coupling_output)"
    echo "  -j, --jobs NUM      Maximum parallel jobs (default: 4)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Particle list format:"
    echo "  pid,halo_id,suite,count,size"
    echo "  Example: 1,Halo718,eden,270306,Large"
    echo ""
    echo "The script will:"
    echo "  1. Read particle list from file"
    echo "  2. Check if trained models exist for each PID"
    echo "  3. Run sampling with Kroupa IMF for each model"
    echo "  4. Skip if samples already exist"
    echo "  5. Generate summary report"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            PARTICLE_LIST_FILE="$2"
            shift 2
            ;;
        -o|--output)
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        -j|--jobs)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Run main function
main "$@"







