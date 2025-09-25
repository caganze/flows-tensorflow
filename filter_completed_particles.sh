#!/bin/bash

# 🔍 Filter Completed Particles
# Creates a filtered particle list excluding already processed particles

# Removed set -e to prevent early exit on edge cases

INPUT_FILE="particle_list.txt"
OUTPUT_FILE="particle_list_incomplete.txt"
VERBOSE=false
DRY_RUN=false

show_usage() {
    echo "🔍 Filter Completed Particles"
    echo "=========================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --input FILE        Input particle list (default: particle_list.txt)"
    echo "  --output FILE       Output filtered list (default: particle_list_incomplete.txt)"
    echo "  --verbose           Show detailed progress"
    echo "  --dry-run           Show what would be filtered without creating output"
    echo "  --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                                    # Filter with defaults"
    echo "  $0 --verbose                         # Show detailed progress"
    echo "  $0 --dry-run                         # Preview filtering"
    echo "  $0 --output remaining_particles.txt  # Custom output file"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check input file
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "❌ Input file not found: $INPUT_FILE"
    exit 1
fi

echo "🔍 Filtering completed particles from $INPUT_FILE"
echo "📝 Output will be saved to: $OUTPUT_FILE"
echo ""

# Function to check if particle is completed
is_particle_completed() {
    local particle_entry="$1"
    
    # Parse particle entry: PID,HALO_ID,SUITE,OBJECT_COUNT,SIZE_CATEGORY (symlib format)
    IFS=',' read -r pid halo_id suite object_count size_category <<< "$particle_entry"
    
    # Debug output for troubleshooting
    if [[ "$VERBOSE" == "true" ]]; then
        echo "  🔍 Raw entry: $particle_entry"
        echo "  📊 Parsed: pid=$pid, halo_id=$halo_id, suite=$suite, count=$object_count, size=$size_category"
    fi
    
    # Validate parsed data
    if [[ -z "$pid" || -z "$halo_id" || -z "$suite" ]]; then
        if [[ "$VERBOSE" == "true" ]]; then
            echo "  ⚠️ Skipping malformed entry: $particle_entry"
        fi
        return 1  # Treat as incomplete
    fi
    
    # Use symlib format data directly (no filename parsing needed)
    # halo_id and suite are already parsed from symlib format
    
    # Construct expected output paths - use symlib structure
    local output_base_dir="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output"
    local model_dir="$output_base_dir/trained_flows/${suite}/halo${halo_id#Halo}"
    local samples_dir="$output_base_dir/samples/${suite}/halo${halo_id#Halo}"
    
    # Check if completion files exist
    local model_file="$model_dir/model_pid${pid}.npz"
    local samples_file_npz="$samples_dir/model_pid${pid}_samples.npz"
    
    # Check if both model and samples exist
    if [[ -f "$model_file" && -f "$samples_file_npz" ]]; then
        if [[ "$VERBOSE" == "true" ]]; then
            echo "  ✅ Completed: PID $pid (Halo $halo_id)"
        fi
        return 0  # Completed
    else
        if [[ "$VERBOSE" == "true" ]]; then
            echo "  ⏳ Incomplete: PID $pid (Halo $halo_id)"
        fi
        return 1  # Not completed
    fi
}

# Read input file and filter
total_particles=0
completed_particles=0
incomplete_particles=0

# Backup existing output file if it exists
if [[ -f "$OUTPUT_FILE" && "$DRY_RUN" != "true" ]]; then
    backup_file="${OUTPUT_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$OUTPUT_FILE" "$backup_file"
    echo "📁 Backed up existing output to: $backup_file"
fi

# Clear output file (unless dry run)
if [[ "$DRY_RUN" != "true" ]]; then
    > "$OUTPUT_FILE"
fi

echo "🔍 Scanning particles for completion status..."
echo ""

# Process each line
while IFS= read -r line; do
    if [[ -n "$line" ]]; then
        ((total_particles++))
        
        # Debug first few particles
        if [[ $total_particles -le 3 && "$VERBOSE" == "true" ]]; then
            echo "🔍 Debug particle $total_particles: $line"
        fi
        
        if is_particle_completed "$line"; then
            ((completed_particles++))
        else
            ((incomplete_particles++))
            # Add to output file (unless dry run)
            if [[ "$DRY_RUN" != "true" ]]; then
                echo "$line" >> "$OUTPUT_FILE"
            fi
        fi
        
        # Progress indicator (every 100 particles)
        if [[ $((total_particles % 100)) -eq 0 ]]; then
            echo "📊 Processed $total_particles particles..."
        fi
        
        # Early exit for testing (disabled for production)
        # if [[ $total_particles -ge 5 && "$VERBOSE" == "true" ]]; then
        #     echo "🧪 Early exit for debugging (processed 5 particles)"
        #     break
        # fi
    fi
done < "$INPUT_FILE"

echo ""
echo "📊 FILTERING RESULTS"
echo "==================="
echo "🔢 Total particles: $total_particles"
echo "✅ Already completed: $completed_particles"
echo "⏳ Incomplete (need processing): $incomplete_particles"
echo "📈 Completion rate: $(( completed_particles * 100 / total_particles ))%"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "🧪 DRY RUN - No output file created"
    echo "💡 Run without --dry-run to create filtered list"
else
    echo "📝 Filtered list saved to: $OUTPUT_FILE"
    echo ""
    echo "🚀 NEXT STEPS"
    echo "============"
    echo "Use the filtered list for chunked submission:"
    echo "  ./submit_cpu_chunked.sh --particle-list $OUTPUT_FILE"
    echo ""
    echo "Or use with regular submission:"
    echo "  cp $OUTPUT_FILE particle_list.txt"
    echo "  ./submit_cpu_parallel.sh"
fi
