#!/bin/bash

#=============================================================================
# DEPLOYMENT SCRIPT FOR SHERLOCK
# 
# This script helps you deploy the brute force GPU job system to Sherlock
# and ensures all paths and permissions are set up correctly.
#
# Usage: ./deploy_to_sherlock.sh
#=============================================================================

set -e

echo "🚀 DEPLOYING BRUTE FORCE SYSTEM TO SHERLOCK"
echo "============================================="

# Check if we're on Sherlock
if [[ $(hostname) != *"sherlock"* ]]; then
    echo "❌ This script should be run ON Sherlock after uploading the files"
    echo ""
    echo "📝 To deploy to Sherlock:"
    echo "1. From your local flows-tensorflow directory, sync all files:"
    echo "   rsync -av * caganze@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/"
    echo ""
    echo "2. SSH to Sherlock and run this script:"
    echo "   ssh caganze@login.sherlock.stanford.edu"
    echo "   cd /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow"
    echo "   ./deploy_to_sherlock.sh"
    exit 1
fi

echo "✅ Running on Sherlock: $(hostname)"
echo ""

# Define paths
BASE_DIR="/oak/stanford/orgs/kipac/users/caganze"
FLOWS_DIR="$BASE_DIR/flows-tensorflow"
OUTPUT_DIR="$BASE_DIR/tfp_flows_output"

echo "🔍 Checking directory structure..."

# Check if flows directory exists
if [[ ! -d "$FLOWS_DIR" ]]; then
    echo "❌ Flows directory not found: $FLOWS_DIR"
    echo "💡 Current location: $(pwd)"
    echo "💡 Creating flows directory..."
    mkdir -p "$FLOWS_DIR"
    echo "✅ Created: $FLOWS_DIR"
fi

# Navigate to flows directory
cd "$FLOWS_DIR"
echo "✅ Working directory: $(pwd)"

# Check required files
REQUIRED_FILES=("brute_force_gpu_job.sh" "monitor_brute_force.sh" "train_tfp_flows.py")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✅ Found: $file"
    else
        echo "❌ Missing: $file"
        MISSING_FILES+=("$file")
    fi
done

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    echo ""
    echo "❌ Missing required files. Please copy them to $FLOWS_DIR:"
    for file in "${MISSING_FILES[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

echo ""
echo "🔧 Setting up environment and permissions..."

# Make scripts executable
chmod +x brute_force_gpu_job.sh
chmod +x monitor_brute_force.sh
echo "✅ Scripts are executable"

# Create necessary directories
mkdir -p logs
mkdir -p success_logs
mkdir -p failed_jobs
mkdir -p "$OUTPUT_DIR/trained_flows"
mkdir -p "$OUTPUT_DIR/samples"
mkdir -p "$OUTPUT_DIR/metrics"

echo "✅ Created necessary directories"

# Check environment setup
echo ""
echo "🔍 Checking environment setup..."

# Check if conda environment exists
if command -v conda >/dev/null 2>&1; then
    echo "✅ Conda available"
    if conda env list | grep -q "bosque"; then
        echo "✅ Bosque environment found"
    else
        echo "⚠️  Bosque environment not found - you may need to create it"
    fi
else
    echo "⚠️  Conda not available - check your ~/.bashrc"
fi

# Check CUDA setup
if module list 2>&1 | grep -q "cuda"; then
    echo "✅ CUDA modules loaded"
else
    echo "ℹ️  CUDA modules not currently loaded (this is normal)"
fi

# Test basic Python imports
echo ""
echo "🧪 Testing Python environment..."
if conda activate bosque 2>/dev/null && python -c "
import tensorflow as tf
import numpy as np
import h5py
print(f'✅ TensorFlow: {tf.__version__}')
print(f'✅ NumPy: {np.__version__}')
print(f'✅ h5py: {h5py.__version__}')
" 2>/dev/null; then
    echo "✅ Python environment working"
else
    echo "⚠️  Python environment test failed - check conda setup"
fi

echo ""
echo "🔍 Checking for halo files..."

# Check for halo files
HALO_FILES=$(find "$BASE_DIR" -name '*Halo*_*orig*.h5' -type f 2>/dev/null | wc -l)
if [[ $HALO_FILES -gt 0 ]]; then
    echo "✅ Found $HALO_FILES halo files"
    echo "First few files:"
    find "$BASE_DIR" -name '*Halo*_*orig*.h5' -type f 2>/dev/null | head -3 | while read file; do
        echo "   $(basename $file)"
    done
else
    echo "⚠️  No halo files found - check data directories"
    echo "   Searched pattern: $BASE_DIR/*Halo*_*orig*.h5"
fi

echo ""
echo "🎯 Deployment summary:"
echo "   Base directory: $BASE_DIR"
echo "   Flows directory: $FLOWS_DIR"
echo "   Output directory: $OUTPUT_DIR"
echo "   Halo files: $HALO_FILES found"
echo ""

# Create a quick test script
cat > test_single_job.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name="test_brute_force"
#SBATCH --partition=gpu
#SBATCH --time=30:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

echo "🧪 Testing brute force environment..."
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load modules
module --force purge
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0

# Activate environment
source ~/.bashrc
conda activate bosque

# Test basic functionality
python -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
print(f'GPU available: {tf.test.is_gpu_available()}')
print(f'GPU devices: {len(tf.config.list_physical_devices(\"GPU\"))}')
"

echo "✅ Test completed successfully!"
EOF

chmod +x test_single_job.sh
echo "✅ Created test job script: test_single_job.sh"

echo ""
echo "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
echo "============================================="
echo ""
echo "🚀 Next steps:"
echo "1. Test the environment:"
echo "   sbatch test_single_job.sh"
echo ""
echo "2. Monitor the test:"
echo "   tail -f logs/test_*.out"
echo ""
echo "3. If test passes, submit the full job:"
echo "   sbatch brute_force_gpu_job.sh"
echo ""
echo "4. Monitor progress:"
echo "   ./monitor_brute_force.sh"
echo ""
echo "📚 For detailed instructions, see: BRUTE_FORCE_USAGE_GUIDE.md"
echo ""
echo "🌟 Ready to process all halos and PIDs!"
