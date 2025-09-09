#!/bin/bash

# 📊 Quick Completion Status Checker
# Check how many particles are completed vs incomplete

set -e

echo "📊 PARTICLE COMPLETION STATUS"
echo "============================="

# Check if particle list exists
if [[ ! -f "particle_list.txt" ]]; then
    echo "❌ particle_list.txt not found"
    exit 1
fi

TOTAL_PARTICLES=$(wc -l < particle_list.txt)
echo "🔢 Total particles in list: $TOTAL_PARTICLES"
echo ""

# Quick sampling - check first 10 particles to understand the structure
echo "🔍 Analyzing first 10 particles for structure..."
echo ""

completed_count=0
incomplete_count=0
error_count=0

for i in {1..10}; do
    if [[ $i -gt $TOTAL_PARTICLES ]]; then
        break
    fi
    
    # Get particle entry
    particle_entry=$(sed -n "${i}p" particle_list.txt)
    echo "Entry $i: $particle_entry"
    
    # Parse entry: PID,H5_FILE,OBJECT_COUNT,SIZE_CATEGORY (current format)
    IFS=',' read -r pid h5_file object_count size_category <<< "$particle_entry"
    
    # Extract halo info
    filename=$(basename "$h5_file")
    halo_id=$(echo "$filename" | sed 's/.*Halo\([0-9]\+\).*/\1/')
    
    # Determine data source
    data_source="unknown"
    if [[ "$filename" == *"eden_scaled"* ]]; then
        data_source="eden"
    elif [[ "$filename" == *"symphonyHR_scaled"* ]]; then
        data_source="symphony-hr" 
    elif [[ "$filename" == *"symphony_scaled"* ]]; then
        data_source="symphony"
    fi
    
    # Handle fallback
    if [[ "$filename" == "all_in_one.h5" ]] || [[ "$halo_id" == "$filename" ]]; then
        halo_id="000"
        if [[ "$data_source" == "unknown" ]]; then
            data_source="symphony"
        fi
    fi
    
    # Check paths
    h5_parent_dir=$(dirname "$h5_file")
    output_base_dir="$h5_parent_dir/tfp_output"
    model_dir="$output_base_dir/trained_flows/${data_source}/halo${halo_id}"
    samples_dir="$output_base_dir/samples/${data_source}/halo${halo_id}"
    
    model_file="$model_dir/model_pid${pid}.npz"
    samples_file_npz="$samples_dir/model_pid${pid}_samples.npz"
    samples_file_h5="$samples_dir/model_pid${pid}_samples.h5"
    
    echo "  📁 Expected model: $model_file"
    echo "  📁 Expected samples: $samples_file_npz or $samples_file_h5"
    
    # Check completion
    if [[ -f "$model_file" ]] && [[ -f "$samples_file_npz" || -f "$samples_file_h5" ]]; then
        echo "  ✅ COMPLETED"
        ((completed_count++))
    else
        echo "  ❌ INCOMPLETE"
        if [[ ! -f "$model_file" ]]; then
            echo "    Missing: model file"
        fi
        if [[ ! -f "$samples_file_npz" && ! -f "$samples_file_h5" ]]; then
            echo "    Missing: samples file"
        fi
        ((incomplete_count++))
    fi
    echo ""
done

echo "📊 SAMPLE RESULTS (first 10 particles):"
echo "✅ Completed: $completed_count"
echo "❌ Incomplete: $incomplete_count" 
echo ""

# Quick count of actual output files to estimate progress
echo "🔍 Counting existing output files..."

# Count model files
if command -v find >/dev/null 2>&1; then
    model_count=$(find /oak/stanford/orgs/kipac/users/caganze -name "model_pid*.npz" 2>/dev/null | wc -l || echo "0")
    sample_count=$(find /oak/stanford/orgs/kipac/users/caganze -name "*_samples.*" 2>/dev/null | wc -l || echo "0")
    
    echo "📁 Found model files: $model_count"
    echo "📁 Found sample files: $sample_count"
    echo ""
    
    # Rough estimate
    if [[ $model_count -gt 0 ]]; then
        estimated_completion=$((model_count * 100 / TOTAL_PARTICLES))
        echo "📈 Estimated completion: ~${estimated_completion}% ($model_count/$TOTAL_PARTICLES)"
    fi
else
    echo "⚠️  find command not available for counting"
fi

echo ""
echo "💡 NEXT STEPS:"
echo "============="
echo "If you want to run only incomplete particles:"
echo "1. Fix the filtering script issue (check particle_list.txt format)"
echo "2. Or manually count completed particles in your output directories"
echo "3. Or just submit - the jobs will skip completed particles automatically"
echo ""
echo "To submit despite filtering issues:"
echo "  ./submit_cpu_chunked.sh --chunk-size 500 --concurrent 5"


