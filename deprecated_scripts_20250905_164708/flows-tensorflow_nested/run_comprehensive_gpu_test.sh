#!/bin/bash
#SBATCH --job-name=comprehensive_test
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=comprehensive_test_%j.out
#SBATCH --error=comprehensive_test_%j.err

# Comprehensive GPU test for the complete TensorFlow Probability flows pipeline
# Tests: Kroupa IMF + Optimized I/O + GPU training + Multiple particles
# NO JAX DEPENDENCIES

set -e  # Exit on any error

echo "🚀 COMPREHENSIVE GPU TEST ON SHERLOCK"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo

# Load required modules (compatible with our environment)
echo "📦 Loading modules..."
module purge
module load math
module load devel
module load nvhpc/24.7
module load cuda/12.2.0
module load cudnn/8.9.0.131

# Set CUDA environment for TensorFlow
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0
export CUDA_VISIBLE_DEVICES=0

# Show GPU info
echo "🔍 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo

# Activate conda environment
echo "🐍 Activating conda environment..."
source ~/.bashrc
conda activate bosque || {
    echo "❌ Failed to activate bosque environment"
    echo "Available environments:"
    conda env list
    exit 1
}

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo

# Verify no JAX is installed
echo "🔍 Checking for JAX conflicts..."
python -c "
import sys
forbidden = ['jax', 'jaxlib']
for mod in forbidden:
    if mod in sys.modules:
        print(f'❌ {mod} is imported!')
        sys.exit(1)
try:
    import jax
    print('❌ JAX is installed and importable!')
    sys.exit(1)
except ImportError:
    print('✅ JAX not available (good)')

print('✅ No JAX conflicts detected')
"

# Verify TensorFlow GPU
echo "🔍 Verifying TensorFlow GPU..."
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs detected: {len(gpus)}')
for i, gpu in enumerate(gpus):
    print(f'  GPU {i}: {gpu}')
if len(gpus) == 0:
    print('❌ No GPUs detected!')
    import sys; sys.exit(1)
print('✅ TensorFlow GPU ready')
"

# Set default parameters
# Smart H5 file discovery for parallel jobs
find_h5_file() {
    # If H5_FILE is set and exists, use it
    if [[ -n "$H5_FILE" && -f "$H5_FILE" ]]; then
        echo "$H5_FILE"
        return 0
    fi
    
    # Search paths for H5 files (most likely locations)
    local search_paths=(
        "/oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/"
        "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/"
        "/oak/stanford/orgs/kipac/users/caganze/milkyway-hr-mocks/"
        "/oak/stanford/orgs/kipac/users/caganze/milkywaymocks/"
        "/oak/stanford/orgs/kipac/users/caganze/"
        "../milkyway-eden-mocks/"
        "../symphony_mocks/"
        "../milkyway-hr-mocks/"
        "../milkywaymocks/"
    )
    
    # Send search messages to stderr to avoid interfering with return value
    echo "🔍 Searching for H5 files in common locations..." >&2
    for path in "${search_paths[@]}"; do
        if [[ -d "$path" ]]; then
            echo "  Checking: $path" >&2
            # Find any .h5 file in this directory
            h5_file=$(find "$path" -name "*.h5" -type f 2>/dev/null | head -1)
            if [[ -n "$h5_file" ]]; then
                echo "  ✅ Found: $h5_file" >&2
                echo "$h5_file"  # Only this goes to stdout for variable capture
                return 0
            fi
        fi
    done
    
    # If nothing found, return empty
    echo ""
    return 1
}

# Configuration with smart defaults
H5_FILE=$(find_h5_file)
if [[ -z "$H5_FILE" ]]; then
    H5_FILE="${H5_FILE:-/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/all_in_one.h5}"
fi
PARTICLE_PIDS="${PARTICLE_PIDS:-1 2 200}"
N_SAMPLES="${N_SAMPLES:-50000}"
EPOCHS="${EPOCHS:-5}"

echo "📋 Test parameters:"
echo "  H5 file: $H5_FILE"
echo "  Particle PIDs: $PARTICLE_PIDS"
echo "  Samples per particle: $N_SAMPLES"
echo "  Training epochs: $EPOCHS"
echo

# Check if H5 file exists
if [[ ! -f "$H5_FILE" ]]; then
    echo "❌ H5 file not found: $H5_FILE"
    echo "Please set H5_FILE environment variable or place file at default location"
    exit 1
fi

echo "✅ H5 file found: $(ls -lh $H5_FILE)"
echo

# Run the comprehensive test
echo "🧪 Starting comprehensive test..."
echo "Command: python comprehensive_gpu_test.py --h5_file $H5_FILE --particle_pids $PARTICLE_PIDS --n_samples $N_SAMPLES --epochs $EPOCHS"
echo

python comprehensive_gpu_test.py \
    --h5_file "$H5_FILE" \
    --particle_pids $PARTICLE_PIDS \
    --n_samples $N_SAMPLES \
    --epochs $EPOCHS

TEST_EXIT_CODE=$?

echo
echo "📊 Test completed with exit code: $TEST_EXIT_CODE"

if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    echo "🎉 COMPREHENSIVE TEST PASSED!"
    echo "✅ Pipeline is ready for full deployment"
    
    # Show output summary
    echo
    echo "📁 Generated outputs:"
    find test_output -type f -name "*.h5" -o -name "*.npz" -o -name "*.json" 2>/dev/null | head -10
    
else
    echo "❌ COMPREHENSIVE TEST FAILED!"
    echo "⚠️  Please check the errors above and fix before deployment"
fi

echo
echo "🏁 Test job completed at $(date)"
echo "Total runtime: $SECONDS seconds"

exit $TEST_EXIT_CODE
