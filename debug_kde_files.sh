#!/bin/bash

# 🔍 Debug KDE File Recognition
# Check what KDE files exist vs what the filter is looking for

set -e

echo "🔍 KDE FILE RECOGNITION DEBUG"
echo "============================="
echo ""

# Parameters
OUTPUT_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/kde_output"
PARTICLE_LIST_FILE="${1:-particle_list.txt}"

if [[ ! -f "$PARTICLE_LIST_FILE" ]]; then
    echo "❌ Particle list not found: $PARTICLE_LIST_FILE"
    echo "Usage: $0 [particle_list_file]"
    exit 1
fi

echo "📁 KDE Output Directory: $OUTPUT_DIR"
echo "📋 Checking first 10 particles from: $PARTICLE_LIST_FILE"
echo ""

# Check if output directory exists
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "❌ Output directory doesn't exist: $OUTPUT_DIR"
    exit 1
fi

echo "🔍 CHECKING FILE PATTERNS"
echo "========================="

# Sample first 10 particles
head -10 "$PARTICLE_LIST_FILE" | while IFS=',' read -r PID HALO_ID SUITE OBJECT_COUNT SIZE_CATEGORY; do
    # Skip empty lines
    [[ -z "$PID" ]] && continue
    
    echo ""
    echo "🎯 Particle: PID $PID, Halo $HALO_ID, Suite $SUITE"
    
    # Expected directory
    SAMPLES_DIR="$OUTPUT_DIR/kde_samples/${SUITE}/halo${HALO_ID#Halo}"
    echo "   📁 Expected dir: $SAMPLES_DIR"
    
    # Expected filename (what filter looks for)
    EXPECTED_FILE="$SAMPLES_DIR/kde_samples_${HALO_ID}_pid${PID}.h5"
    echo "   📄 Expected file: kde_samples_${HALO_ID}_pid${PID}.h5"
    
    if [[ -d "$SAMPLES_DIR" ]]; then
        echo "   ✅ Directory exists"
        
        # List all .h5 files in directory
        H5_FILES=$(find "$SAMPLES_DIR" -name "*.h5" 2>/dev/null || echo "")
        if [[ -n "$H5_FILES" ]]; then
            echo "   📄 Found .h5 files:"
            echo "$H5_FILES" | sed 's/^/      /'
        else
            echo "   ❌ No .h5 files found"
        fi
        
        # Check specific expected file
        if [[ -f "$EXPECTED_FILE" ]]; then
            echo "   ✅ Expected file EXISTS"
        else
            echo "   ❌ Expected file MISSING"
        fi
    else
        echo "   ❌ Directory doesn't exist"
    fi
done

echo ""
echo "🔍 DIRECTORY STRUCTURE OVERVIEW"
echo "==============================="

if [[ -d "$OUTPUT_DIR/kde_samples" ]]; then
    echo "📊 KDE Samples Directory Structure:"
    find "$OUTPUT_DIR/kde_samples" -type f -name "*.h5" | head -20 | while read file; do
        echo "   $(basename "$file")"
    done | sort | uniq -c | sort -nr
else
    echo "❌ No kde_samples directory found"
fi

echo ""
echo "💡 RECOMMENDATIONS"
echo "=================="
echo "1. Check if filenames match the expected pattern: kde_samples_HaloXXX_pidYYY.h5"
echo "2. Verify the train_kde_conditional.py output format"
echo "3. Run the filter with --verbose to see detailed mismatches"
echo ""
echo "🔧 Test filter with verbose output:"
echo "   ./filter_completed_kde.sh --verbose --dry-run"


