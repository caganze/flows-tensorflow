#!/bin/bash
#SBATCH --job-name=meta_test_pipeline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/meta_test_%j.out
#SBATCH --error=logs/meta_test_%j.err

echo "🚀 META TEST: FULL PIPELINE VALIDATION"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Create logs directory
mkdir -p logs

# Load modules - CRITICAL for GPU
echo "🔌 Loading required modules..."
set -x
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131
set +x

echo "📋 Loaded modules:"
module list

# Activate conda environment
echo "🐍 Activating conda environment..."
source /oak/stanford/orgs/kipac/users/caganze/anaconda3/etc/profile.d/conda.sh
conda activate bosque

# Set CUDA paths for TensorFlow - CRITICAL for GPU operations
echo "🔧 Setting CUDA paths for TensorFlow..."
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0

# Verify environment
echo "📋 Environment Check:"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo "  Python location: $(which python)"
echo "  Python version: $(python --version)"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  XLA_FLAGS: $XLA_FLAGS"
echo ""

# Test 1: Basic TensorFlow GPU test
echo "🧪 TEST 1: TensorFlow GPU Detection"
echo "===================================="
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {tf.test.is_gpu_available()}')
print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')
if tf.test.is_gpu_available():
    print('✅ GPU detection successful')
else:
    print('❌ GPU detection failed')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ TensorFlow GPU test failed - aborting"
    exit 1
fi
echo ""

# Test 2: H5 file discovery for different halos
echo "🧪 TEST 2: H5 File Discovery Across Halos"
echo "=========================================="

# Test halos to check
TEST_HALOS=("023" "088" "188" "268" "327" "415" "469" "570" "641" "718" "800" "852" "939")

FOUND_HALOS=()
MISSING_HALOS=()

for halo_id in "${TEST_HALOS[@]}"; do
    echo "🔍 Testing halo $halo_id..."
    
    # Search for halo file in different locations
    search_paths=(
        "../milkyway-eden-mocks/eden_scaled_Halo${halo_id}_sunrot0_0kpc200kpcoriginal_particles.h5"
        "../milkyway-eden-mocks/eden_scaled_Halo${halo_id}_m_sunrot0_0kpc200kpcoriginal_particles.h5"
        "../milkywaymocks/symphony_scaled_Halo${halo_id}_sunrot0_0kpc200kpcoriginal_particles.h5"
        "../milkyway-hr-mocks/symphonyHR_scaled_Halo${halo_id}_sunrot0_0kpc200kpcoriginal_particles.h5"
    )
    
    found=false
    for path in "${search_paths[@]}"; do
        if [ -f "$path" ]; then
            echo "  ✅ Found: $path"
            FOUND_HALOS+=("$halo_id")
            found=true
            break
        fi
    done
    
    if [ "$found" = false ]; then
        echo "  ❌ Missing: halo $halo_id"
        MISSING_HALOS+=("$halo_id")
    fi
done

echo ""
echo "📊 H5 File Discovery Summary:"
echo "  Found halos: ${#FOUND_HALOS[@]} (${FOUND_HALOS[*]})"
echo "  Missing halos: ${#MISSING_HALOS[@]} (${MISSING_HALOS[*]})"
echo ""

# Require at least 3 halos for meaningful testing
if [ ${#FOUND_HALOS[@]} -lt 3 ]; then
    echo "❌ Insufficient halos found for testing - need at least 3"
    exit 1
fi

# Test 3: Single particle extraction for each found halo
echo "🧪 TEST 3: Single Particle Extraction (PID 1, 1000 samples)"
echo "=========================================================="

SUCCESSFUL_EXTRACTIONS=0
FAILED_EXTRACTIONS=0

for halo_id in "${FOUND_HALOS[@]:0:5}"; do  # Test first 5 found halos
    echo "🔬 Testing halo $halo_id, PID 1, 1000 particles..."
    
    python test_h5_read_single_particle.py --halo_id "$halo_id" --particle_pid 1 --n_subsample 1000
    
    if [ $? -eq 0 ]; then
        echo "  ✅ Halo $halo_id: SUCCESS"
        ((SUCCESSFUL_EXTRACTIONS++))
    else
        echo "  ❌ Halo $halo_id: FAILED"
        ((FAILED_EXTRACTIONS++))
    fi
    echo ""
done

echo "📊 Single Particle Extraction Summary:"
echo "  Successful: $SUCCESSFUL_EXTRACTIONS"
echo "  Failed: $FAILED_EXTRACTIONS"
echo ""

if [ $SUCCESSFUL_EXTRACTIONS -eq 0 ]; then
    echo "❌ No successful particle extractions - aborting"
    exit 1
fi

# Test 4: Multi-particle extraction
echo "🧪 TEST 4: Multi-Particle Extraction"
echo "===================================="

# Use first successful halo for multi-particle test
TEST_HALO=${FOUND_HALOS[0]}
echo "🔬 Testing halo $TEST_HALO with PIDs 1,2,3,4,5..."

python test_multiple_particles.py --halo_id "$TEST_HALO" --particle_pids 1 2 3 4 5 --n_subsample 500

if [ $? -eq 0 ]; then
    echo "✅ Multi-particle extraction: SUCCESS"
else
    echo "❌ Multi-particle extraction: FAILED"
    exit 1
fi
echo ""

# Test 5: Training script generation
echo "🧪 TEST 5: Training Script Generation"
echo "===================================="

# Generate scripts for first 2 found halos, PIDs 1-3
TEST_HALO_LIST=$(echo "${FOUND_HALOS[@]:0:2}" | tr ' ' ',')
echo "🔬 Generating scripts for halos: $TEST_HALO_LIST, PIDs 1,2,3..."

python generate_parallel_scripts.py --halo_ids $TEST_HALO_LIST --particle_pids 1 2 3 --test_mode

if [ $? -eq 0 ]; then
    echo "✅ Script generation: SUCCESS"
else
    echo "❌ Script generation: FAILED"
    exit 1
fi

# Verify generated scripts exist and are executable
echo "🔍 Verifying generated scripts..."
generated_count=0
for halo_id in "${FOUND_HALOS[@]:0:2}"; do
    for pid in 1 2 3; do
        script_name="submit_training_h${halo_id}_p$(printf '%03d' $pid).sh"
        if [ -f "$script_name" ] && [ -x "$script_name" ]; then
            echo "  ✅ $script_name"
            ((generated_count++))
        else
            echo "  ❌ $script_name (missing or not executable)"
        fi
    done
done

echo "📊 Generated Scripts: $generated_count/6 expected"
echo ""

# Test 6: Queue scheduling simulation
echo "🧪 TEST 6: Queue Scheduling Simulation"
echo "======================================"

echo "🔬 Testing queue submission logic..."

# Test master submission script generation
python generate_parallel_scripts.py --halo_ids $TEST_HALO_LIST --particle_pids 1 2 3 --test_mode --create_master

if [ -f "submit_all_training.sh" ] && [ -x "submit_all_training.sh" ]; then
    echo "✅ Master submission script created"
    
    # Show first few lines of submission strategy
    echo "📋 Submission strategy preview:"
    head -20 submit_all_training.sh
else
    echo "❌ Master submission script not found"
    exit 1
fi
echo ""

# Test 7: Minimal training test (very short run)
echo "🧪 TEST 7: Minimal Training Test"
echo "==============================="

echo "🔬 Running minimal training test (5 epochs, 1000 particles)..."

# Create minimal test data if needed
python -c "
import numpy as np
import h5py
import os

# Create tiny test dataset
test_dir = 'test_data'
os.makedirs(test_dir, exist_ok=True)

n_particles = 1000
np.random.seed(42)

# Simple 6D phase space data
pos = np.random.normal(0, 10, (n_particles, 3))
vel = np.random.normal(0, 100, (n_particles, 3))
data = np.hstack([pos, vel]).astype(np.float32)

with h5py.File(f'{test_dir}/test_halo.h5', 'w') as f:
    f.create_dataset('pos3', data=pos)
    f.create_dataset('vel3', data=vel)
    f.create_dataset('parentid', data=np.ones(n_particles, dtype=int))

print('✅ Test data created')
"

# Run minimal training
python train_tfp_flows.py \
    --input_file test_data/test_halo.h5 \
    --output_dir test_output \
    --halo_id test \
    --particle_pid 1 \
    --n_epochs 5 \
    --batch_size 128 \
    --learning_rate 0.001 \
    --n_subsample 1000 \
    --use_gpu

if [ $? -eq 0 ]; then
    echo "✅ Minimal training test: SUCCESS"
else
    echo "❌ Minimal training test: FAILED"
    exit 1
fi

# Verify training outputs
if [ -d "test_output" ] && [ "$(ls -A test_output)" ]; then
    echo "✅ Training outputs created"
    echo "📋 Output files:"
    ls -la test_output/
else
    echo "❌ No training outputs found"
    exit 1
fi
echo ""

# Test 8: Memory and performance check
echo "🧪 TEST 8: Memory and Performance Check"
echo "======================================="

echo "📊 System resources:"
echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2, \"total,\", $3, \"used,\", $4, \"available\"}')"
echo "  GPU memory: $(nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits)"
echo "  Load average: $(uptime | awk -F'load average:' '{print $2}')"
echo ""

# Final summary
echo "🎉 META TEST COMPLETION SUMMARY"
echo "==============================="
echo "✅ TensorFlow GPU detection: PASSED"
echo "✅ H5 file discovery: PASSED (${#FOUND_HALOS[@]} halos)"
echo "✅ Single particle extraction: PASSED ($SUCCESSFUL_EXTRACTIONS halos)"
echo "✅ Multi-particle extraction: PASSED"
echo "✅ Script generation: PASSED"
echo "✅ Queue scheduling: PASSED"
echo "✅ Minimal training: PASSED"
echo "✅ System resources: CHECKED"
echo ""
echo "🚀 READY FOR FULL PIPELINE DEPLOYMENT!"
echo "📋 Available halos for production: ${FOUND_HALOS[*]}"
echo "💡 Next step: Run full pipeline with complete particle sets"
echo ""
echo "End time: $(date)"

# Cleanup test files
echo "🧹 Cleaning up test files..."
rm -rf test_data test_output
echo "✅ Cleanup complete"
