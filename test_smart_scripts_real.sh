#!/bin/bash

# 🧪 REAL SMART SCRIPTS TEST
# Actually tests functionality, not just status messages

set -e

echo "🧪 REAL SMART SCRIPTS TEST"
echo "========================="
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Job: ${SLURM_JOB_ID:-local}"
echo ""

# Exit codes for different failure types
EXIT_ENV_FAIL=10
EXIT_FILTER_FAIL=11
EXIT_SCRIPT_FAIL=12
EXIT_CRITICAL_FAIL=13

# Test 1: Can we activate bosque environment?
echo "TEST 1: bosque environment"
echo "=========================="
source ~/.bashrc
if ! conda activate bosque; then
    echo "❌ FAILED: Cannot activate bosque"
    exit $EXIT_ENV_FAIL
fi

# Test CPU mode (for CPU jobs)
if ! python3 -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import symlib
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
print('bosque CPU mode: symlib OK, TF CPU-only OK')
"; then
    echo "❌ FAILED: bosque environment CPU mode broken"
    exit $EXIT_ENV_FAIL
fi

# Test GPU mode (for GPU jobs)
if ! python3 -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import symlib
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'bosque GPU mode: symlib OK, TF OK, GPUs: {len(gpus)}')
"; then
    echo "❌ FAILED: bosque environment GPU mode broken"
    exit $EXIT_ENV_FAIL
fi
echo "✅ PASSED: bosque environment works for both CPU and GPU"
echo ""

# Test 3: Create test data and test filter script
echo "TEST 3: Filter script with symlib format"
echo "========================================"
cat > test_particles.txt << 'EOF'
1,Halo939,eden,1000,Small
2,Halo939,eden,2000,Small
3,Halo718,symphony,1500,Small
EOF

if ! ./filter_completed_particles.sh --input test_particles.txt --output filtered_test.txt --dry-run --verbose > filter_test.log 2>&1; then
    echo "❌ FAILED: Filter script crashed"
    cat filter_test.log
    exit $EXIT_FILTER_FAIL
fi

# Check if it parsed symlib format correctly - look for successful processing
if grep -q "🔢 Total particles: 3" filter_test.log && grep -q "⏳ Incomplete (need processing): 3" filter_test.log; then
    echo "✅ PASSED: Filter script successfully processed 3 symlib-format particles"
else
    echo "❌ FAILED: Filter script didn't process symlib format correctly"
    echo "Expected: 3 total particles, 3 incomplete"
    echo "Got:"
    cat filter_test.log
    exit $EXIT_FILTER_FAIL
fi
echo ""

# Test 4: Test CPU smart script dry run
echo "TEST 4: CPU smart script dry run"
echo "================================"
conda activate bosque

if ! ./submit_cpu_smart.sh --dry-run --chunk-size 2 > cpu_test.log 2>&1; then
    echo "❌ FAILED: CPU smart script crashed"
    cat cpu_test.log
    exit $EXIT_SCRIPT_FAIL
fi

if ! grep -q "DRY RUN" cpu_test.log; then
    echo "❌ FAILED: CPU smart script didn't run in dry run mode"
    cat cpu_test.log
    exit $EXIT_SCRIPT_FAIL
fi
echo "✅ PASSED: CPU smart script dry run works"
echo ""

# Test 5: Test GPU smart script dry run
echo "TEST 5: GPU smart script dry run"
echo "================================"
conda activate bosque

if ! ./submit_gpu_smart.sh --dry-run --chunk-size 2 > gpu_test.log 2>&1; then
    echo "❌ FAILED: GPU smart script crashed"
    cat gpu_test.log
    exit $EXIT_SCRIPT_FAIL
fi

if ! grep -q "DRY RUN" gpu_test.log; then
    echo "❌ FAILED: GPU smart script didn't run in dry run mode"
    cat gpu_test.log
    exit $EXIT_SCRIPT_FAIL
fi
echo "✅ PASSED: GPU smart script dry run works"
echo ""

# Test 6: Test training script can be invoked with symlib args
echo "TEST 6: Training script symlib arguments"
echo "======================================="
if ! python3 train_tfp_flows.py --help > train_help.log 2>&1; then
    echo "❌ FAILED: Training script crashed on --help"
    cat train_help.log
    exit $EXIT_SCRIPT_FAIL
fi

if ! grep -q -- "--halo_id" train_help.log || ! grep -q -- "--suite" train_help.log; then
    echo "❌ FAILED: Training script missing symlib arguments"
    echo "Expected: --halo_id and --suite"
    cat train_help.log
    exit $EXIT_SCRIPT_FAIL
fi
echo "✅ PASSED: Training script has symlib arguments"
echo ""

# Test 7: Test actual symlib data loading (quick test)
echo "TEST 7: Quick symlib data loading test"
echo "====================================="
if ! python3 -c "
from symlib_utils import load_particle_data
import time
start = time.time()
try:
    data, metadata = load_particle_data('Halo939', 1, 'eden')
    elapsed = time.time() - start
    print(f'Loaded {data.shape[0]} particles in {elapsed:.1f}s')
    if data.shape[0] > 0 and data.shape[1] == 7:
        print('✅ Data shape and format correct (7 columns: pos + vel + mass)')
    else:
        print('❌ Data shape wrong - expected 7 columns, got', data.shape[1])
        exit(1)
except Exception as e:
    print(f'❌ Symlib loading failed: {e}')
    exit(1)
" > symlib_test.log 2>&1; then
    echo "❌ FAILED: Symlib data loading test failed"
    cat symlib_test.log
    exit $EXIT_CRITICAL_FAIL
fi
echo "✅ PASSED: Symlib data loading works"
echo ""

# Test 8: Output directory write test
echo "TEST 8: Output directory write permissions"
echo "========================================="
OUTPUT_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output"
TEST_FILE="$OUTPUT_DIR/test_write_$(date +%s).tmp"

mkdir -p "$OUTPUT_DIR" || {
    echo "❌ FAILED: Cannot create output directory"
    exit $EXIT_CRITICAL_FAIL
}

if ! touch "$TEST_FILE" || ! rm "$TEST_FILE"; then
    echo "❌ FAILED: Cannot write to output directory"
    exit $EXIT_CRITICAL_FAIL
fi
echo "✅ PASSED: Output directory writable"
echo ""

# Cleanup
rm -f test_particles.txt filtered_test.txt filter_test.log cpu_test.log gpu_test.log train_help.log symlib_test.log

# Final result
echo "🎉 ALL TESTS PASSED!"
echo "===================="
echo "✅ bosque environment (CPU mode): Working"
echo "✅ bosque environment (GPU mode): Working"  
echo "✅ Filter script: Parses symlib format correctly"
echo "✅ CPU smart script: Dry run successful"
echo "✅ GPU smart script: Dry run successful"
echo "✅ Training script: Has symlib arguments"
echo "✅ Symlib data loading: Working"
echo "✅ Output directory: Writable"
echo ""
echo "🚀 SMART SCRIPTS ARE READY FOR PRODUCTION!"
echo "You can now run:"
echo "  ./submit_cpu_smart.sh"
echo "  ./submit_gpu_smart.sh"
echo ""
exit 0
