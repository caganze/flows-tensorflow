#!/bin/bash

#=============================================================================
# MIGRATE OUTPUT STRUCTURE
# 
# This script migrates existing flat output structure to the new hierarchical
# structure organized by data source and halo ID.
#
# Usage: ./migrate_output_structure.sh
#=============================================================================

set -e

OUTPUT_BASE="/oak/stanford/orgs/kipac/users/caganze/tfp_flows_output"

echo "🔄 MIGRATING OUTPUT STRUCTURE"
echo "============================="
echo "Output base: $OUTPUT_BASE"
echo ""

# Check if migration is needed
if [[ ! -d "$OUTPUT_BASE" ]]; then
    echo "❌ Output directory not found: $OUTPUT_BASE"
    exit 1
fi

echo "🔍 Analyzing current structure..."

# Find old-style model directories (model_pidX)
OLD_MODEL_DIRS=($(find "$OUTPUT_BASE/trained_flows" -maxdepth 1 -name "model_pid*" -type d 2>/dev/null))
echo "Found ${#OLD_MODEL_DIRS[@]} old-style model directories"

# Find old-style sample files in root samples directory
OLD_SAMPLE_FILES=($(find "$OUTPUT_BASE/samples" -maxdepth 1 -name "model_pid*" -type f 2>/dev/null))
echo "Found ${#OLD_SAMPLE_FILES[@]} old-style sample files"

if [[ ${#OLD_MODEL_DIRS[@]} -eq 0 && ${#OLD_SAMPLE_FILES[@]} -eq 0 ]]; then
    echo "✅ No migration needed - structure already organized"
    exit 0
fi

echo ""
echo "🚀 Starting migration..."

# Function to determine data source from model files
determine_data_source_from_model() {
    local model_dir="$1"
    local results_file="$model_dir/model_pid*_results.json"
    
    # Try to extract data source from results file
    if ls $results_file 1> /dev/null 2>&1; then
        local filepath=$(grep -o '"filepath"[^,]*' $results_file 2>/dev/null | head -1 | cut -d'"' -f4)
        if [[ "$filepath" == *"eden_scaled"* ]]; then
            echo "eden"
        elif [[ "$filepath" == *"symphonyHR_scaled"* ]]; then
            echo "symphony-hr"
        elif [[ "$filepath" == *"symphony_scaled"* ]]; then
            echo "symphony"
        else
            echo "unknown"
        fi
    else
        echo "unknown"
    fi
}

# Function to extract halo ID from model directory or file
extract_halo_from_path() {
    local path="$1"
    # Try to extract from results file first
    local results_file="$path/model_pid*_results.json"
    if ls $results_file 1> /dev/null 2>&1; then
        local halo_id=$(grep -o '"halo_id"[^,]*' $results_file 2>/dev/null | head -1 | cut -d'"' -f4)
        if [[ -n "$halo_id" && "$halo_id" != "null" ]]; then
            echo "$halo_id"
            return
        fi
    fi
    
    # Fallback: extract from directory name
    echo "$(basename $path)" | sed 's/.*pid\([0-9]\+\).*/\1/'
}

# Migrate model directories
for old_dir in "${OLD_MODEL_DIRS[@]}"; do
    echo "📁 Processing model directory: $(basename $old_dir)"
    
    # Determine data source and halo
    data_source=$(determine_data_source_from_model "$old_dir")
    halo_id=$(extract_halo_from_path "$old_dir")
    
    # Create new directory structure
    new_dir="$OUTPUT_BASE/trained_flows/${data_source}/halo${halo_id}"
    mkdir -p "$new_dir"
    
    echo "   Moving to: $data_source/halo$halo_id/"
    
    # Move all files from old directory to new directory
    if [[ -d "$old_dir" && "$old_dir" != "$new_dir" ]]; then
        mv "$old_dir"/* "$new_dir/" 2>/dev/null || true
        rmdir "$old_dir" 2>/dev/null || true
    fi
done

# Migrate sample files
for old_file in "${OLD_SAMPLE_FILES[@]}"; do
    filename=$(basename "$old_file")
    echo "📄 Processing sample file: $filename"
    
    # Extract PID from filename
    pid=$(echo "$filename" | sed 's/.*pid\([0-9]\+\).*/\1/')
    
    # Try to find corresponding model directory to determine data source and halo
    data_source="unknown"
    halo_id="$pid"
    
    # Search for matching model in new structure
    for source in eden symphony symphony-hr unknown; do
        if [[ -d "$OUTPUT_BASE/trained_flows/$source" ]]; then
            for halo_dir in "$OUTPUT_BASE/trained_flows/$source"/halo*/; do
                if [[ -d "$halo_dir" && -f "$halo_dir/model_pid${pid}.npz" ]]; then
                    data_source="$source"
                    halo_id=$(basename "$halo_dir" | sed 's/halo//')
                    break 2
                fi
            done
        fi
    done
    
    # Create new directory structure for samples
    new_dir="$OUTPUT_BASE/samples/${data_source}/halo${halo_id}"
    mkdir -p "$new_dir"
    
    echo "   Moving to: $data_source/halo$halo_id/"
    
    # Move file to new location
    mv "$old_file" "$new_dir/"
done

echo ""
echo "✅ Migration completed!"
echo ""
echo "📊 New structure summary:"

# Show new structure
for source in eden symphony symphony-hr unknown; do
    if [[ -d "$OUTPUT_BASE/trained_flows/$source" ]]; then
        model_count=$(find "$OUTPUT_BASE/trained_flows/$source" -name "*.npz" 2>/dev/null | wc -l)
        if [[ $model_count -gt 0 ]]; then
            echo "  $source: $model_count models"
        fi
    fi
    if [[ -d "$OUTPUT_BASE/samples/$source" ]]; then
        sample_count=$(find "$OUTPUT_BASE/samples/$source" -name "*samples*" 2>/dev/null | wc -l)
        if [[ $sample_count -gt 0 ]]; then
            echo "  $source: $sample_count sample files"
        fi
    fi
done

echo ""
echo "🎯 Structure now organized by:"
echo "   📁 Data source: eden, symphony, symphony-hr"
echo "   📁 Halo ID: halo023, halo088, etc."
echo "   📄 Files: model_pidX.npz, model_pidX_samples.npz"
