#!/bin/bash

#=============================================================================
# SHERLOCK PATH CHECKER
# 
# This script helps you determine the correct BASE_DIR for your Sherlock setup
# Run this ON Sherlock to find your actual paths
#=============================================================================

echo "🔍 SHERLOCK PATH DISCOVERY"
echo "=========================="
echo "Username: $(whoami)"
echo "Hostname: $(hostname)"
echo ""

echo "📁 Home Directory Info:"
echo "HOME: $HOME"
echo "PWD: $(pwd)"
echo ""

echo "🔍 Checking common KIPAC paths..."

# Check different possible base directories
POSSIBLE_BASES=(
    "/oak/stanford/orgs/kipac/users/caganze"
    "/oak/stanford/orgs/kipac/users/$(whoami)"
    "$HOME"
    "/scratch/users/$(whoami)"
    "/scratch/groups/kipac/users/$(whoami)"
)

echo "Checking these potential base directories:"
for base in "${POSSIBLE_BASES[@]}"; do
    echo "  Testing: $base"
    if [[ -d "$base" ]]; then
        echo "    ✅ EXISTS"
        echo "    Contents:"
        ls -la "$base" 2>/dev/null | head -5 | while read line; do
            echo "      $line"
        done
        
        # Check for flows-tensorflow directory
        if [[ -d "$base/flows-tensorflow" ]]; then
            echo "    ✅ flows-tensorflow directory found!"
        else
            echo "    ⚠️  flows-tensorflow directory not found"
        fi
        
        # Check for halo files
        halo_count=$(find "$base" -name '*Halo*_*orig*.h5' -type f 2>/dev/null | wc -l)
        if [[ $halo_count -gt 0 ]]; then
            echo "    ✅ Found $halo_count halo files"
            echo "    First few halo files:"
            find "$base" -name '*Halo*_*orig*.h5' -type f 2>/dev/null | head -3 | while read file; do
                echo "      $(basename $file)"
            done
        else
            echo "    ⚠️  No halo files found"
        fi
        echo ""
    else
        echo "    ❌ DOES NOT EXIST"
    fi
    echo ""
done

echo "🔍 Searching for halo files in common locations..."
HALO_SEARCH_PATHS=(
    "/oak/stanford/orgs/kipac"
    "/scratch/groups/kipac"
    "$HOME"
)

for search_path in "${HALO_SEARCH_PATHS[@]}"; do
    if [[ -d "$search_path" ]]; then
        echo "Searching in: $search_path"
        halo_files=$(find "$search_path" -name '*Halo*_*orig*.h5' -type f 2>/dev/null | head -5)
        if [[ -n "$halo_files" ]]; then
            echo "  ✅ Found halo files:"
            echo "$halo_files" | while read file; do
                echo "    $file"
            done
        else
            echo "  ❌ No halo files found"
        fi
        echo ""
    fi
done

echo "📋 RECOMMENDATIONS:"
echo "=================="

# Find the best base directory
best_base=""
for base in "${POSSIBLE_BASES[@]}"; do
    if [[ -d "$base" ]]; then
        # Score each directory
        score=0
        [[ -d "$base/flows-tensorflow" ]] && score=$((score + 10))
        
        halo_count=$(find "$base" -name '*Halo*_*orig*.h5' -type f 2>/dev/null | wc -l)
        score=$((score + halo_count))
        
        if [[ $score -gt 0 ]]; then
            echo "✅ RECOMMENDED BASE_DIR: $base (score: $score)"
            best_base="$base"
            break
        fi
    fi
done

if [[ -n "$best_base" ]]; then
    echo ""
    echo "🔧 TO UPDATE THE SCRIPT:"
    echo "Replace this line in brute_force_gpu_job.sh:"
    echo '  BASE_DIR="/oak/stanford/orgs/kipac/users/caganze"'
    echo "With:"
    echo "  BASE_DIR=\"$best_base\""
    echo ""
    echo "Expected directory structure:"
    echo "  $best_base/flows-tensorflow/           # Your scripts"
    echo "  $best_base/tfp_flows_output/           # Output directory" 
    echo "  $best_base/**/Halo*_*orig*.h5          # Halo data files"
else
    echo "❌ Could not find a suitable base directory"
    echo "💡 You may need to:"
    echo "   1. Create the flows-tensorflow directory"
    echo "   2. Ensure halo files are accessible"
    echo "   3. Check your KIPAC group membership"
fi

echo ""
echo "🏁 Path discovery complete!"
