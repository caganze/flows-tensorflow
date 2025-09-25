#!/bin/bash

# 🧪 Test KDE NaN Fix
# Quick test for a single particle to verify fixes

set -e

echo "🧪 TESTING KDE NaN FIXES"
echo "========================"
echo ""

# Test parameters
TEST_HALO="Halo939"
TEST_PID="1"
TEST_SUITE="eden"

echo "🔧 Test Configuration:"
echo "   Halo: $TEST_HALO"
echo "   PID: $TEST_PID"
echo "   Suite: $TEST_SUITE"
echo ""

# Create test particle list
echo "$TEST_PID,$TEST_HALO,$TEST_SUITE,369193,Large" > test_kde_single.txt

echo "📋 Testing with single large particle (369k objects)"
echo ""

# Set environment variables to match job script
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2

echo "🧠 Running KDE with improved parameters..."
python train_kde_conditional.py \
    --halo_id "$TEST_HALO" \
    --parent_id "$TEST_PID" \
    --suite "$TEST_SUITE" \
    --output_dir "./test_kde_output" \
    --n_neighbors 96 \
    --sample_fraction 0.5 \
    --mass_range 0.08 100 \
    --seed 42

echo ""
echo "📊 Checking output quality..."

if [[ -f "./test_kde_output/kde_samples_${TEST_HALO}_pid${TEST_PID}.h5" ]]; then
    echo "✅ Output file created successfully"
    
    python -c "
import h5py
import numpy as np

with h5py.File('./test_kde_output/kde_samples_${TEST_HALO}_pid${TEST_PID}.h5', 'r') as f:
    pos = f['positions'][:]
    vel = f['velocities'][:]
    
    print(f'📊 Sample Analysis:')
    print(f'   Total samples: {len(pos)}')
    
    # Check for NaN/inf
    pos_finite = np.isfinite(pos).all(axis=1)
    vel_finite = np.isfinite(vel).all(axis=1)
    finite_samples = np.sum(pos_finite & vel_finite)
    
    if finite_samples == len(pos):
        print(f'   ✅ All {finite_samples} samples are finite')
        print(f'   Position range: [{np.min(pos):.2f}, {np.max(pos):.2f}]')
        print(f'   Velocity range: [{np.min(vel):.2f}, {np.max(vel):.2f}]')
        print(f'   Position std: {np.std(pos):.3f}')
        print(f'   Velocity std: {np.std(vel):.3f}')
        print(f'   🎯 SUCCESS: No NaN/inf values found!')
    else:
        print(f'   ❌ Warning: {len(pos)-finite_samples}/{len(pos)} samples contain NaN/inf')
        exit(1)
"
    
    PYTHON_EXIT=$?
    if [[ $PYTHON_EXIT -eq 0 ]]; then
        echo ""
        echo "✅ KDE NaN FIXES SUCCESSFUL!"
        echo "=========================="
        echo "🎯 The fixes have resolved the NaN/inf issue"
        echo "🚀 KDE jobs should now produce valid samples"
    else
        echo ""
        echo "❌ NaN issues still present"
        echo "🔧 May need further debugging"
    fi
else
    echo "❌ Output file not created"
    exit 1
fi

# Cleanup
rm -f test_kde_single.txt
rm -rf test_kde_output

echo ""
echo "💡 If successful, cancel current KDE jobs and resubmit with fixes:"
echo "   scancel -u \$USER --name=kde_chunk_*"
echo "   ./submit_cpu_kde_smart.sh"


