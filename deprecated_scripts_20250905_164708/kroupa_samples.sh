#!/bin/bash

# 🌟 Kroupa IMF Sampling Script
# Generate realistic stellar populations from trained flows

set -e

echo "🌟 KROUPA IMF SAMPLING"
echo "===================="
echo "🎯 Generate realistic stellar populations using Kroupa IMF"
echo ""

# Default parameters
BASE_DIR=""
PARTICLE_PID=""
OUTPUT_BASE=""
DRY_RUN=false
VERBOSE=false
FAILED_LOG="kroupa_failed_pids.txt"

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "REQUIRED:"
    echo "  --base-dir DIR      Base directory containing trained_flows subdirectory"
    echo ""
    echo "OPTIONS:"
    echo "  --particle-pid N    Process specific particle ID only"
    echo "  --output-base DIR   Output base directory (default: same as base-dir)"
    echo "  --failed-log FILE   File to log failed PIDs (default: kroupa_failed_pids.txt)"
    echo "  --dry-run           Show what would be processed without doing it"
    echo "  --verbose           Verbose output"
    echo "  --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 --base-dir /path/to/tfp_output"
    echo "  $0 --base-dir /path/to/tfp_output --particle-pid 123"
    echo "  $0 --base-dir /path/to/tfp_output --dry-run"
    echo ""
    echo "OUTPUT STRUCTURE:"
    echo "  \$BASE_DIR/kroupa-samples/"
    echo "    ├── eden/halo123/"
    echo "    │   ├── kroupa_pid1_samples.npz"
    echo "    │   └── kroupa_pid1_samples.h5"
    echo "    └── logs/"
    echo "        └── kroupa_sampling_pid1.log"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --particle-pid)
            PARTICLE_PID="$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --failed-log)
            FAILED_LOG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
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

# Validate inputs
if [[ -z "$BASE_DIR" ]]; then
    echo "❌ --base-dir is required"
    show_usage
    exit 1
fi

if [[ ! -d "$BASE_DIR" ]]; then
    echo "❌ Base directory does not exist: $BASE_DIR"
    exit 1
fi

# Check for trained_flows subdirectory
TRAINED_FLOWS_DIR="$BASE_DIR/trained_flows"
if [[ ! -d "$TRAINED_FLOWS_DIR" ]]; then
    echo "❌ trained_flows directory not found in: $BASE_DIR"
    echo "💡 Expected structure: $BASE_DIR/trained_flows/..."
    exit 1
fi

# Set output base if not provided
if [[ -z "$OUTPUT_BASE" ]]; then
    OUTPUT_BASE="$BASE_DIR"
fi

echo "🔍 CONFIGURATION"
echo "==============="
echo "📁 Base directory: $BASE_DIR"
echo "🏗️  Trained flows: $TRAINED_FLOWS_DIR"
echo "📤 Output base: $OUTPUT_BASE"
if [[ -n "$PARTICLE_PID" ]]; then
    echo "🎯 Target PID: $PARTICLE_PID"
else
    echo "🎯 Processing: All found models"
fi
echo ""

# Check Python environment
echo "🐍 Checking Python environment..."

# Check if we're in the right environment
if ! python -c "import tensorflow, tensorflow_probability" 2>/dev/null; then
    echo "❌ TensorFlow/TFP not available"
    echo "💡 Make sure you're in the right conda environment:"
    echo "   conda activate bosque"
    exit 1
fi

# Check for required modules
required_modules=("tfp_flows_gpu_solution" "kroupa_imf" "optimized_io")
for module in "${required_modules[@]}"; do
    if ! python -c "import $module" 2>/dev/null; then
        echo "❌ Required module not found: $module"
        echo "💡 Make sure you're in the flows-tensorflow directory"
        exit 1
    fi
done

echo "✅ Python environment ready"
echo ""

# Find available models
echo "🔍 Scanning for trained models..."
MODEL_COUNT=$(find "$TRAINED_FLOWS_DIR" -name "model_pid*.npz" | wc -l)

if [[ $MODEL_COUNT -eq 0 ]]; then
    echo "❌ No trained models found in $TRAINED_FLOWS_DIR"
    exit 1
fi

echo "✅ Found $MODEL_COUNT trained model(s)"

# Show some examples
echo "📋 Sample models found:"
find "$TRAINED_FLOWS_DIR" -name "model_pid*.npz" | head -5 | while read -r model_file; do
    # Extract info from path
    relative_path=$(echo "$model_file" | sed "s|$BASE_DIR/||")
    pid=$(basename "$model_file" | sed 's/model_pid\([0-9]*\)\.npz/\1/')
    echo "   PID $pid: $relative_path"
done

if [[ $MODEL_COUNT -gt 5 ]]; then
    echo "   ... and $((MODEL_COUNT - 5)) more"
fi
echo ""

# Prepare Python arguments
PYTHON_ARGS="--base-dir $BASE_DIR"

if [[ -n "$PARTICLE_PID" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --particle-pid $PARTICLE_PID"
fi

if [[ -n "$OUTPUT_BASE" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --output-base $OUTPUT_BASE"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --dry-run"
fi

if [[ "$VERBOSE" == "true" ]]; then
    PYTHON_ARGS="$PYTHON_ARGS --verbose"
fi

PYTHON_ARGS="$PYTHON_ARGS --failed-log $FAILED_LOG"

# Show what will be executed
if [[ "$DRY_RUN" == "true" ]]; then
    echo "🧪 DRY RUN MODE"
    echo "==============="
    echo "Command that would be executed:"
    echo "python kroupa_samples.py $PYTHON_ARGS"
    echo ""
fi

# Execute the Python script
echo "🚀 Starting Kroupa IMF sampling..."
echo "=================================="

if python kroupa_samples.py $PYTHON_ARGS; then
    echo ""
    echo "✅ KROUPA SAMPLING COMPLETED SUCCESSFULLY"
    echo "========================================"
    
    if [[ "$DRY_RUN" != "true" ]]; then
        echo "📁 Output directory: $OUTPUT_BASE/kroupa-samples/"
        echo ""
        
        # Show sample output structure
        if [[ -d "$OUTPUT_BASE/kroupa-samples" ]]; then
            echo "📋 Output structure:"
            ls -la "$OUTPUT_BASE/kroupa-samples/" | head -10
            echo ""
        fi
        
        # Check for failed PIDs
        if [[ -f "$FAILED_LOG" && -s "$FAILED_LOG" ]]; then
            FAILED_COUNT=$(wc -l < "$FAILED_LOG")
            echo "⚠️  Failed PIDs: $FAILED_COUNT (see $FAILED_LOG)"
            echo "📋 Failed PIDs:"
            head -10 "$FAILED_LOG" | while read -r pid; do
                echo "   PID $pid"
            done
            if [[ $FAILED_COUNT -gt 10 ]]; then
                echo "   ... and $((FAILED_COUNT - 10)) more"
            fi
        else
            echo "🎉 All PIDs processed successfully!"
        fi
    fi
    
else
    echo ""
    echo "❌ KROUPA SAMPLING FAILED"
    echo "========================"
    echo "Check the error messages above for details"
    exit 1
fi

