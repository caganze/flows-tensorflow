#!/bin/bash

# 🎯 END-TO-END SINGLE PARTICLE TEST
# Tests the complete pipeline: symlib data loading -> training -> model/sample output
# This will tell us definitively if the smart scripts will work

set -e

echo "🎯 END-TO-END SINGLE PARTICLE TEST"
echo "=================================="
echo "Testing complete pipeline from symlib data to final outputs"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Job: ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo ""

# Exit if not on compute node
if [[ -z "$SLURM_JOB_ID" ]]; then
    echo "❌ This test must run on a SLURM compute node"
    echo "Run with: srun --partition=owners --time=30:00 --mem=16GB --cpus-per-task=4 --pty bash"
    exit 1
fi

# Activate environment
echo "🐍 Activating bosque environment..."
source ~/.bashrc
conda activate bosque

# Test parameters - small particle for quick test
TEST_HALO="Halo939"
TEST_PID=1
TEST_SUITE="eden"
OUTPUT_BASE="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output"
MODEL_DIR="$OUTPUT_BASE/trained_flows/$TEST_SUITE/halo939"
SAMPLES_DIR="$OUTPUT_BASE/samples/$TEST_SUITE/halo939"

echo "📋 Test parameters:"
echo "   Halo: $TEST_HALO"
echo "   PID: $TEST_PID"
echo "   Suite: $TEST_SUITE"
echo "   Output base: $OUTPUT_BASE"
echo ""

# Create output directories
echo "📁 Creating output directories..."
mkdir -p "$MODEL_DIR" "$SAMPLES_DIR"

# Clean any existing test outputs
echo "🧹 Cleaning any existing test outputs..."
rm -f "$MODEL_DIR/model_pid${TEST_PID}.npz"
rm -f "$MODEL_DIR/model_pid${TEST_PID}_preprocessing.npz"
rm -f "$MODEL_DIR/model_pid${TEST_PID}_results.json"
rm -f "$SAMPLES_DIR/model_pid${TEST_PID}_samples.npz"
echo ""

# Step 1: Test symlib data loading
echo "🔍 STEP 1: Testing symlib data loading"
echo "======================================"

# Change to project directory for correct imports
cd /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow

SYMLIB_TEST=$(python3 -c "
import sys
sys.path.append('/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow')
from symlib_utils import load_particle_data
import time

start_time = time.time()
try:
    data, metadata = load_particle_data('$TEST_HALO', $TEST_PID, '$TEST_SUITE')
    load_time = time.time() - start_time
    
    print(f'SUCCESS: Loaded {data.shape[0]} particles in {load_time:.1f}s')
    print(f'Data shape: {data.shape}')
    print(f'Data type: {data.dtype}')
    print(f'Total mass: {metadata[\"stellar_mass\"]:.2e} M☉')
    
    # Basic validation - now expect 7 columns (pos + vel + mass)
    if data.shape[0] > 0 and data.shape[1] == 7:
        print('✅ Data format valid (7 columns: pos + vel + mass)')
        exit(0)
    else:
        print('❌ Invalid data format')
        exit(1)
        
except Exception as e:
    print(f'❌ FAILED: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1)

echo "$SYMLIB_TEST"
if [[ $? -ne 0 ]]; then
    echo "❌ STEP 1 FAILED: Symlib data loading failed"
    exit 1
fi
echo "✅ STEP 1 PASSED: Symlib data loading works"
echo ""

# Step 2: Test training script with minimal parameters
echo "🧠 STEP 2: Testing training with minimal parameters"
echo "=================================================="

# Use very small parameters for quick test
echo "🎛️ Using minimal training parameters for speed:"
echo "   Epochs: 5"
echo "   Batch size: 64"
echo "   Layers: 2"
echo "   Hidden units: 32"
echo "   Learning rate: 1e-3"
echo ""

echo "🚀 Starting training..."
TRAINING_START=$(date +%s)

# Change to project directory to ensure correct paths
cd /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow

TRAINING_OUTPUT=$(timeout 600 python3 train_tfp_flows.py \
    --halo_id "$TEST_HALO" \
    --particle_pid "$TEST_PID" \
    --suite "$TEST_SUITE" \
    --output_dir "$MODEL_DIR" \
    --epochs 5 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --n_layers 2 \
    --hidden_units 32 \
    --generate-samples \
    2>&1)

TRAINING_EXIT=$?
TRAINING_END=$(date +%s)
TRAINING_TIME=$((TRAINING_END - TRAINING_START))

echo "Training completed in ${TRAINING_TIME}s with exit code: $TRAINING_EXIT"
echo ""

if [[ $TRAINING_EXIT -ne 0 ]]; then
    echo "❌ STEP 2 FAILED: Training failed"
    echo "Training output:"
    echo "$TRAINING_OUTPUT"
    exit 1
fi

echo "✅ STEP 2 PASSED: Training completed successfully"
echo ""

# Step 3: Verify output files were created
echo "📂 STEP 3: Verifying output files were created"
echo "=============================================="

EXPECTED_FILES=(
    "$MODEL_DIR/model_pid${TEST_PID}.npz"
    "$SAMPLES_DIR/model_pid${TEST_PID}_samples.npz"
)

OPTIONAL_FILES=(
    "$MODEL_DIR/model_pid${TEST_PID}_preprocessing.npz"
    "$MODEL_DIR/model_pid${TEST_PID}_results.json"
)

echo "🔍 Checking required files:"
MISSING_REQUIRED=0
for file in "${EXPECTED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
        echo "   ✅ $file (${size} bytes)"
    else
        echo "   ❌ MISSING: $file"
        MISSING_REQUIRED=1
    fi
done

echo ""
echo "🔍 Checking optional files:"
for file in "${OPTIONAL_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
        echo "   ✅ $file (${size} bytes)"
    else
        echo "   ⚠️ Optional file not found: $file"
    fi
done

if [[ $MISSING_REQUIRED -eq 1 ]]; then
    echo ""
    echo "❌ STEP 3 FAILED: Required output files missing"
    exit 1
fi

echo ""
echo "✅ STEP 3 PASSED: All required output files created"
echo ""

# Step 4: Validate output file contents
echo "🔬 STEP 4: Validating output file contents"
echo "=========================================="

# Test model file
MODEL_TEST=$(python3 -c "
import numpy as np
try:
    model_data = np.load('$MODEL_DIR/model_pid${TEST_PID}.npz')
    print(f'Model file keys: {list(model_data.keys())}')
    print('✅ Model file loads successfully')
    
    # Test samples file
    samples_data = np.load('$SAMPLES_DIR/model_pid${TEST_PID}_samples.npz')
    print(f'Samples file keys: {list(samples_data.keys())}')
    
    # Check if samples have reasonable shape
    if 'samples_6d' in samples_data:
        samples = samples_data['samples_6d']
        print(f'Samples shape: {samples.shape}')
        if len(samples.shape) == 2 and samples.shape[1] == 6:
            print('✅ Samples have correct 6D format')
        else:
            print('❌ Samples have wrong format')
            exit(1)
    else:
        print('❌ No samples_6d found in samples file')
        exit(1)
        
    print('✅ Output files contain valid data')
    exit(0)
    
except Exception as e:
    print(f'❌ Error validating files: {e}')
    exit(1)
" 2>&1)

echo "$MODEL_TEST"
if [[ $? -ne 0 ]]; then
    echo "❌ STEP 4 FAILED: Output file validation failed"
    exit 1
fi

echo "✅ STEP 4 PASSED: Output files contain valid data"
echo ""

# Step 5: Test that filter script would recognize completion
echo "🔍 STEP 5: Testing filter script recognition"
echo "==========================================="

# Create a test particle list with our test particle
cat > test_single_particle.txt << EOF
${TEST_PID},${TEST_HALO},${TEST_SUITE},1000,Small
EOF

FILTER_TEST=$(./filter_completed_particles.sh --input test_single_particle.txt --output test_filtered.txt --dry-run 2>&1)

echo "$FILTER_TEST"

if echo "$FILTER_TEST" | grep -q "✅ Already completed: 1"; then
    echo "✅ STEP 5 PASSED: Filter script correctly recognizes completed particle"
else
    echo "❌ STEP 5 FAILED: Filter script doesn't recognize completed particle"
    echo "This means smart scripts might reprocess already completed particles"
    exit 1
fi

# Cleanup test files
rm -f test_single_particle.txt test_filtered.txt

echo ""

# Final summary
echo "🎉 END-TO-END TEST SUCCESSFUL!"
echo "==============================="
echo "✅ Symlib data loading: Working"
echo "✅ Training script: Produces valid outputs"
echo "✅ Model file: Created and valid"
echo "✅ Samples file: Created and valid (6D format)"
echo "✅ Filter script: Recognizes completion"
echo ""
echo "🚀 PIPELINE VALIDATION COMPLETE"
echo "=============================="
echo "The complete pipeline works end-to-end:"
echo "1. Symlib data loads correctly"
echo "2. Training produces model and sample files"
echo "3. Filter script recognizes completed work"
echo ""
echo "✅ SMART SCRIPTS SHOULD NOW WORK IN PRODUCTION!"
echo ""
echo "🎯 Recommended next steps:"
echo "1. Generate fresh particle list: ./generate_all_priority_halos.sh"
echo "2. Submit jobs: ./submit_gpu_smart.sh --chunk-size 50"
echo ""
echo "Training time: ${TRAINING_TIME}s for 5 epochs"
echo "Completed: $(date)"

