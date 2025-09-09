#!/bin/bash

# 🧪 Test Kroupa Sampling Script
# Test the Kroupa IMF sampling functionality

set -e

echo "🧪 KROUPA SAMPLING TEST SUITE"
echo "============================="
echo "🎯 Testing Kroupa IMF sampling functionality"
echo ""

# Configuration
TEST_BASE_DIR="/oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/tfp_output"
TEST_OUTPUT_DIR="/tmp/kroupa_test_output_$$"
TEST_PID=""
CLEANUP=true
VERBOSE=false

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --base-dir DIR      Base directory for testing (default: auto-detect)"
    echo "  --test-pid N        Test specific PID only"
    echo "  --output-dir DIR    Test output directory (default: temp)"
    echo "  --no-cleanup        Don't cleanup test files"
    echo "  --verbose           Verbose output"
    echo "  --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                               # Basic test"
    echo "  $0 --test-pid 123                # Test specific PID"
    echo "  $0 --verbose --no-cleanup        # Detailed test, keep files"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir)
            TEST_BASE_DIR="$2"
            shift 2
            ;;
        --test-pid)
            TEST_PID="$2"
            shift 2
            ;;
        --output-dir)
            TEST_OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-cleanup)
            CLEANUP=false
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

# Cleanup function
cleanup() {
    if [[ "$CLEANUP" == "true" && -d "$TEST_OUTPUT_DIR" ]]; then
        echo "🧹 Cleaning up test files..."
        rm -rf "$TEST_OUTPUT_DIR"
        echo "✅ Cleanup complete"
    fi
}

# Setup cleanup trap
trap cleanup EXIT

echo "🔧 TEST CONFIGURATION"
echo "===================="
echo "📁 Test base dir: $TEST_BASE_DIR"
echo "📤 Test output dir: $TEST_OUTPUT_DIR"
if [[ -n "$TEST_PID" ]]; then
    echo "🎯 Test PID: $TEST_PID"
else
    echo "🎯 Test mode: Auto-select first available PID"
fi
echo "🧹 Cleanup: $CLEANUP"
echo ""

# Test 1: Check environment
echo "🧪 TEST 1: Environment Check"
echo "=========================="

echo "🐍 Checking Python environment..."
if ! python -c "import tensorflow, tensorflow_probability" 2>/dev/null; then
    echo "❌ TensorFlow/TFP not available"
    echo "💡 Make sure you're in the right conda environment"
    exit 1
fi
echo "✅ TensorFlow/TFP available"

echo "📦 Checking required modules..."
required_modules=("tfp_flows_gpu_solution" "kroupa_imf" "optimized_io")
for module in "${required_modules[@]}"; do
    if ! python -c "import $module" 2>/dev/null; then
        echo "❌ Required module not found: $module"
        exit 1
    fi
    echo "✅ $module available"
done

echo "📁 Checking required scripts..."
required_scripts=("kroupa_samples.py" "kroupa_samples.sh")
for script in "${required_scripts[@]}"; do
    if [[ ! -f "$script" ]]; then
        echo "❌ Required script not found: $script"
        exit 1
    fi
    echo "✅ $script found"
done

echo ""

# Test 2: Find test data
echo "🧪 TEST 2: Data Discovery"
echo "======================="

echo "🔍 Looking for trained models in: $TEST_BASE_DIR"

if [[ ! -d "$TEST_BASE_DIR" ]]; then
    echo "⚠️  Test base directory not found, trying alternative locations..."
    
    # Try alternative locations
    ALT_DIRS=(
        "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/tfp_output"
        "/oak/stanford/orgs/kipac/users/caganze/tfp_output"
        "./test_output"
    )
    
    for alt_dir in "${ALT_DIRS[@]}"; do
        if [[ -d "$alt_dir/trained_flows" ]]; then
            TEST_BASE_DIR="$alt_dir"
            echo "✅ Found alternative: $TEST_BASE_DIR"
            break
        fi
    done
    
    if [[ ! -d "$TEST_BASE_DIR/trained_flows" ]]; then
        echo "❌ No trained_flows directory found"
        echo "💡 Available directories:"
        ls -la /oak/stanford/orgs/kipac/users/caganze/ | grep -E "(milkyway|symphony|tfp)" || echo "   None found"
        exit 1
    fi
fi

# Find available models
echo "🔍 Scanning for models..."
AVAILABLE_MODELS=($(find "$TEST_BASE_DIR/trained_flows" -name "model_pid*.npz" | head -10))

if [[ ${#AVAILABLE_MODELS[@]} -eq 0 ]]; then
    echo "❌ No trained models found"
    exit 1
fi

echo "✅ Found ${#AVAILABLE_MODELS[@]} model(s)"

# Select test PID
if [[ -z "$TEST_PID" ]]; then
    # Auto-select first available PID
    FIRST_MODEL="${AVAILABLE_MODELS[0]}"
    TEST_PID=$(basename "$FIRST_MODEL" | sed 's/model_pid\([0-9]*\)\.npz/\1/')
    echo "🎯 Auto-selected test PID: $TEST_PID"
else
    echo "🎯 Using specified test PID: $TEST_PID"
fi

echo "📋 Available models (first 5):"
for i in {0..4}; do
    if [[ $i -lt ${#AVAILABLE_MODELS[@]} ]]; then
        model="${AVAILABLE_MODELS[$i]}"
        pid=$(basename "$model" | sed 's/model_pid\([0-9]*\)\.npz/\1/')
        relative_path=$(echo "$model" | sed "s|$TEST_BASE_DIR/||")
        echo "   PID $pid: $relative_path"
    fi
done

echo ""

# Test 3: Dry run
echo "🧪 TEST 3: Dry Run Test"
echo "===================="

echo "🎲 Testing dry run functionality..."
mkdir -p "$TEST_OUTPUT_DIR"

DRY_RUN_ARGS="--base-dir $TEST_BASE_DIR --output-base $TEST_OUTPUT_DIR --particle-pid $TEST_PID --dry-run"

if [[ "$VERBOSE" == "true" ]]; then
    DRY_RUN_ARGS="$DRY_RUN_ARGS --verbose"
fi

echo "🚀 Command: python kroupa_samples.py $DRY_RUN_ARGS"

if python kroupa_samples.py $DRY_RUN_ARGS; then
    echo "✅ Dry run successful"
else
    echo "❌ Dry run failed"
    exit 1
fi

echo ""

# Test 4: Actual sampling (small test)
echo "🧪 TEST 4: Actual Sampling Test"
echo "============================="

echo "🌟 Testing actual Kroupa sampling for PID $TEST_PID..."

SAMPLING_ARGS="--base-dir $TEST_BASE_DIR --output-base $TEST_OUTPUT_DIR --particle-pid $TEST_PID"

if [[ "$VERBOSE" == "true" ]]; then
    SAMPLING_ARGS="$SAMPLING_ARGS --verbose"
fi

echo "🚀 Command: python kroupa_samples.py $SAMPLING_ARGS"

if python kroupa_samples.py $SAMPLING_ARGS; then
    echo "✅ Sampling successful"
else
    echo "❌ Sampling failed"
    exit 1
fi

echo ""

# Test 5: Verify output
echo "🧪 TEST 5: Output Verification"
echo "============================"

echo "🔍 Checking output files..."

KROUPA_DIR="$TEST_OUTPUT_DIR/kroupa-samples"
if [[ ! -d "$KROUPA_DIR" ]]; then
    echo "❌ kroupa-samples directory not created"
    exit 1
fi
echo "✅ kroupa-samples directory exists"

# Find output files
SAMPLE_FILES=($(find "$KROUPA_DIR" -name "kroupa_pid${TEST_PID}_samples.*"))

if [[ ${#SAMPLE_FILES[@]} -eq 0 ]]; then
    echo "❌ No sample files found for PID $TEST_PID"
    echo "📁 Contents of $KROUPA_DIR:"
    find "$KROUPA_DIR" -type f | head -10
    exit 1
fi

echo "✅ Found ${#SAMPLE_FILES[@]} sample file(s) for PID $TEST_PID:"
for file in "${SAMPLE_FILES[@]}"; do
    file_size=$(du -h "$file" | cut -f1)
    echo "   $(basename "$file") ($file_size)"
done

# Check log files
LOG_DIR="$KROUPA_DIR/logs"
if [[ -d "$LOG_DIR" ]]; then
    LOG_FILE="$LOG_DIR/kroupa_sampling_pid${TEST_PID}.log"
    if [[ -f "$LOG_FILE" ]]; then
        echo "✅ Log file created: $(basename "$LOG_FILE")"
        if [[ "$VERBOSE" == "true" ]]; then
            echo "📋 Last 10 lines of log:"
            tail -10 "$LOG_FILE" | sed 's/^/   /'
        fi
    fi
fi

echo ""

# Test 6: Content validation
echo "🧪 TEST 6: Content Validation"
echo "==========================="

echo "🔍 Validating sample file content..."

# Try to load and validate one of the sample files
python -c "
import numpy as np
import sys

try:
    # Find a .npz file
    import glob
    npz_files = glob.glob('$KROUPA_DIR/**/kroupa_pid${TEST_PID}_samples.npz', recursive=True)
    
    if not npz_files:
        print('❌ No .npz sample files found')
        sys.exit(1)
    
    # Load the file
    data = np.load(npz_files[0], allow_pickle=True)
    
    print('✅ Successfully loaded sample file')
    print(f'   Keys: {list(data.keys())}')
    
    # Check for required fields
    if 'samples' in data:
        samples = data['samples']
        print(f'   Samples shape: {samples.shape}')
        print(f'   Samples dtype: {samples.dtype}')
        
        if len(samples) > 0:
            print('✅ Non-empty samples array')
        else:
            print('❌ Empty samples array')
            sys.exit(1)
    
    if 'masses' in data:
        masses = data['masses']
        print(f'   Masses shape: {masses.shape}')
        print(f'   Total mass: {np.sum(masses):.2e}')
        print('✅ Masses data present')
    
    if 'metadata' in data:
        metadata = data['metadata'].item() if data['metadata'].ndim == 0 else data['metadata']
        print('✅ Metadata present')
        if 'n_samples' in metadata:
            print(f'   Number of samples: {metadata[\"n_samples\"]:,}')
        if 'actual_total_mass' in metadata:
            print(f'   Total stellar mass: {metadata[\"actual_total_mass\"]:.2e} M☉')
    
    print('✅ Content validation successful')
    
except Exception as e:
    print(f'❌ Content validation failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo ""

# Test 7: Shell script test
echo "🧪 TEST 7: Shell Script Test"
echo "=========================="

echo "🐚 Testing kroupa_samples.sh wrapper..."

SHELL_ARGS="--base-dir $TEST_BASE_DIR --output-base $TEST_OUTPUT_DIR --particle-pid $TEST_PID --dry-run"

if ./kroupa_samples.sh $SHELL_ARGS; then
    echo "✅ Shell script test successful"
else
    echo "❌ Shell script test failed"
    exit 1
fi

echo ""

# Final summary
echo "🎉 ALL TESTS PASSED!"
echo "==================="
echo "✅ Environment check: PASS"
echo "✅ Data discovery: PASS"
echo "✅ Dry run: PASS"
echo "✅ Actual sampling: PASS"
echo "✅ Output verification: PASS"
echo "✅ Content validation: PASS"
echo "✅ Shell script: PASS"
echo ""

if [[ "$CLEANUP" == "false" ]]; then
    echo "📁 Test files preserved in: $TEST_OUTPUT_DIR"
    echo "🔍 You can examine the output files manually"
else
    echo "🧹 Test files will be cleaned up automatically"
fi

echo ""
echo "🚀 Kroupa sampling system is ready for production use!"
echo "💡 Try: ./kroupa_samples.sh --base-dir /path/to/tfp_output"

