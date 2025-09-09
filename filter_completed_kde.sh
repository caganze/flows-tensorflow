#!/bin/bash

# 🔍 Filter Completed KDE Particles
# Scans output directory and creates list of incomplete particles

set -e

echo "🔍 KDE COMPLETION FILTER"
echo "========================"
echo "📁 Checking: /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/kde_output/"
echo ""

# Default parameters
INPUT_FILE="particle_list.txt"
OUTPUT_FILE="particle_list_kde_incomplete.txt"
OUTPUT_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/kde_output"
DRY_RUN=false
VERBOSE=false

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --input FILE        Input particle list (default: particle_list.txt)"
    echo "  --output FILE       Output incomplete list (default: particle_list_kde_incomplete.txt)"
    echo "  --output-dir DIR    Base output directory to check"
    echo "  --verbose           Show detailed progress"
    echo "  --dry-run           Show results without creating output file"
    echo "  --help              Show this help"
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
        --output-dir)
            OUTPUT_DIR="$2"
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

echo "📋 Input: $INPUT_FILE"
echo "📝 Output: $OUTPUT_FILE"
echo "📁 Checking: $OUTPUT_DIR"
echo ""

TOTAL_PARTICLES=$(wc -l < "$INPUT_FILE")
COMPLETED_COUNT=0
INCOMPLETE_COUNT=0

# Temporary file for incomplete particles
TEMP_INCOMPLETE=$(mktemp)

echo "🔍 Scanning $TOTAL_PARTICLES particles..."
echo ""

while IFS=',' read -r PID HALO_ID SUITE OBJECT_COUNT SIZE_CATEGORY; do
    # Skip empty lines
    [[ -z "$PID" ]] && continue
    
    # Expected paths for this particle
    SAMPLES_DIR="$OUTPUT_DIR/kde_samples/${SUITE}/halo${HALO_ID#Halo}"
    
    # KDE samples file (only one file needed for KDE)
    SAMPLES_FILE="$SAMPLES_DIR/kde_samples_${HALO_ID}_pid${PID}.h5"
    
    # Check if completed (KDE samples exist)
    if [[ -f "$SAMPLES_FILE" ]]; then
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
        if [[ "$VERBOSE" == "true" ]]; then
            echo "✅ Completed: $HALO_ID PID $PID"
        fi
    else
        INCOMPLETE_COUNT=$((INCOMPLETE_COUNT + 1))
        echo "$PID,$HALO_ID,$SUITE,$OBJECT_COUNT,$SIZE_CATEGORY" >> "$TEMP_INCOMPLETE"
        if [[ "$VERBOSE" == "true" ]]; then
            echo "❌ Incomplete: $HALO_ID PID $PID"
            echo "   Missing: $SAMPLES_FILE"
        fi
    fi
    
    # Progress indicator (every 1000 particles)
    if [[ $((($COMPLETED_COUNT + $INCOMPLETE_COUNT) % 1000)) -eq 0 ]]; then
        echo "📊 Progress: $((COMPLETED_COUNT + INCOMPLETE_COUNT))/$TOTAL_PARTICLES"
    fi
    
done < "$INPUT_FILE"

echo ""
echo "📊 KDE FILTERING RESULTS"
echo "========================"
echo "✅ Completed: $COMPLETED_COUNT"
echo "❌ Incomplete: $INCOMPLETE_COUNT"
echo "📊 Total: $TOTAL_PARTICLES"

COMPLETION_RATE=$(( COMPLETED_COUNT * 100 / TOTAL_PARTICLES ))
echo "📈 Completion Rate: $COMPLETION_RATE%"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "🧪 DRY RUN - would create: $OUTPUT_FILE"
    echo "📊 Would contain $INCOMPLETE_COUNT incomplete particles"
else
    # Move temp file to final location
    if [[ $INCOMPLETE_COUNT -gt 0 ]]; then
        mv "$TEMP_INCOMPLETE" "$OUTPUT_FILE"
        echo ""
        echo "✅ Created: $OUTPUT_FILE ($INCOMPLETE_COUNT particles)"
    else
        rm -f "$TEMP_INCOMPLETE"
        # Remove old incomplete file if it exists
        [[ -f "$OUTPUT_FILE" ]] && rm "$OUTPUT_FILE"
        echo ""
        echo "🎉 All KDE particles completed! No incomplete list needed."
    fi
fi

# Cleanup
[[ -f "$TEMP_INCOMPLETE" ]] && rm -f "$TEMP_INCOMPLETE"

echo ""
if [[ $INCOMPLETE_COUNT -gt 0 ]]; then
    echo "💡 Next step: ./submit_cpu_kde_smart.sh"
else
    echo "🎉 KDE processing is complete!"
fi
