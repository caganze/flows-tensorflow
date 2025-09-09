#!/bin/bash

# 🔍 Filter Completed Particles
# Creates a filtered particle list excluding already processed particles

set -e

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
    
    # Parse particle entry: PID,H5_FILE,OBJECT_COUNT,SIZE_CATEGORY
    IFS=',' read -r pid h5_file object_count size_category <<< "$particle_entry"
    
    # Extract halo ID and data source from filename
    local filename=$(basename "$h5_file")
    local halo_id=$(echo "$filename" | sed 's/.*Halo\([0-9]\+\).*/\1/')
    
    # Determine data source
    local data_source="unknown"
    if [[ "$filename" == *"eden_scaled"* ]]; then
        data_source="eden"
    elif [[ "$filename" == *"symphonyHR_scaled"* ]]; then
        data_source="symphony-hr"
    elif [[ "$filename" == *"symphony_scaled"* ]]; then
        data_source="symphony"
    fi
    
    # Handle fallback/non-standard files
    if [[ "$filename" == "all_in_one.h5" ]] || [[ "$halo_id" == "$filename" ]]; then
        halo_id="000"
        if [[ "$data_source" == "unknown" ]]; then
            data_source="symphony"
        fi
    fi
    
    # Construct expected output paths
    local h5_parent_dir=$(dirname "$h5_file")
    local output_base_dir="$h5_parent_dir/tfp_output"
    local model_dir="$output_base_dir/trained_flows/${data_source}/halo${halo_id}"
    local samples_dir="$output_base_dir/samples/${data_source}/halo${halo_id}"
    
    # Check if completion files exist
    local model_file="$model_dir/model_pid${pid}.npz"
    local samples_file_npz="$samples_dir/model_pid${pid}_samples.npz"
    local samples_file_h5="$samples_dir/model_pid${pid}_samples.h5"
    
    if [[ -f "$model_file" ]] && [[ -f "$samples_file_npz" || -f "$samples_file_h5" ]]; then
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
