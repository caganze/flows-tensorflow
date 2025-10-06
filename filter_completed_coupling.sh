#!/bin/bash

# Filter completed coupling flow models
# Looks for successful model files in the output directory

set -e

OUTPUT_DIR="${OUTPUT_DIR:-/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/coupling_output}"
PARTICLE_LIST_FILE="${PARTICLE_LIST_FILE:-particle_list.txt}"
VERBOSE="${VERBOSE:-false}"
DRY_RUN="${DRY_RUN:-false}"

echo "🔍 Filtering completed coupling flow models..."
echo "   Output directory: $OUTPUT_DIR"
echo "   Particle list: $PARTICLE_LIST_FILE"

if [[ ! -f "$PARTICLE_LIST_FILE" ]]; then
    echo "❌ Particle list not found: $PARTICLE_LIST_FILE"
    echo "💡 Run ./generate_all_priority_halos.sh first"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create incomplete particle list
INCOMPLETE_FILE="particle_list_coupling_incomplete.txt"
> "$INCOMPLETE_FILE"

TOTAL_PARTICLES=0
COMPLETED_PARTICLES=0
INCOMPLETE_PARTICLES=0

while IFS=',' read -r pid halo_id suite object_count size_category; do
    TOTAL_PARTICLES=$((TOTAL_PARTICLES + 1))
    
    # Check for completed model files (organized by suite)
    model_dir="$OUTPUT_DIR/${suite}/${halo_id,,}/pid${pid}"
    weights_file="$model_dir/flow_weights_${halo_id}_${pid}"
    config_file="$model_dir/flow_config_${halo_id}_${pid}.pkl"
    preprocessing_file="$model_dir/coupling_flow_pid${pid}_preprocessing.npz"
    results_file="$model_dir/coupling_flow_pid${pid}_results.json"
    
    # Check if all required files exist
    if [[ -f "$weights_file.index" && -f "$config_file" && -f "$preprocessing_file" && -f "$results_file" ]]; then
        COMPLETED_PARTICLES=$((COMPLETED_PARTICLES + 1))
        if [[ "$VERBOSE" == "true" ]]; then
            echo "✅ Completed: $halo_id PID $pid"
        fi
    else
        INCOMPLETE_PARTICLES=$((INCOMPLETE_PARTICLES + 1))
        echo "$pid,$halo_id,$suite,$object_count,$size_category" >> "$INCOMPLETE_FILE"
        if [[ "$VERBOSE" == "true" ]]; then
            echo "❌ Incomplete: $halo_id PID $pid"
        fi
    fi
done < "$PARTICLE_LIST_FILE"

echo ""
echo "📊 Filtering Results:"
echo "   Total particles: $TOTAL_PARTICLES"
echo "   Completed: $COMPLETED_PARTICLES"
echo "   Incomplete: $INCOMPLETE_PARTICLES"
echo "   Incomplete list: $INCOMPLETE_FILE"

if [[ $INCOMPLETE_PARTICLES -eq 0 ]]; then
    echo ""
    echo "🎉 ALL COUPLING FLOW MODELS COMPLETED!"
    echo "======================================"
    echo "✅ No incomplete models found - all work is done!"
    exit 0
fi

echo ""
echo "📋 Next: Submit incomplete particles with submit_coupling_flows_cpu_chunked.sh"

