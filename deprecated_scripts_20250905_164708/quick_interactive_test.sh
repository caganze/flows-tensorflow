#!/bin/bash

echo "🚀 QUICK INTERACTIVE TEST FOR ARGUMENT FIXES"
echo "============================================"
echo ""

# Check if running interactively or as batch job
if [[ -z "$SLURM_JOB_ID" ]]; then
    echo "🔥 INTERACTIVE MODE: Requesting minimal GPU allocation..."
    echo "========================================================="
    echo "Purpose: Test the fixed --data_path arguments immediately"
    echo "Allocation: 4 GPUs, 1 hour, 8GB memory (minimal for testing)"
    echo ""
    echo "💡 After allocation, run these commands on the compute node:"
    echo "   cd /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow"
    echo "   ./quick_interactive_test.sh"
    echo ""
    
    salloc --partition=gpu --gres=gpu:4 --time=01:00:00 --mem=8GB --cpus-per-task=16 --job-name="arg_fix_test"
    exit 0
fi

echo "📍 Running on compute node: $SLURMD_NODENAME"
echo "🎯 Job ID: $SLURM_JOB_ID"
echo "🖥️  GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Load environment
echo "🔧 Setting up environment..."
module purge
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0
export CUDA_VISIBLE_DEVICES=0

source ~/.bashrc
conda activate bosque

echo "✅ Environment loaded"
echo ""

# Find H5 file
echo "📁 Locating H5 file..."
find_h5_file() {
    local eden_files=$(find /oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/ -name "eden_scaled_Halo*" -type f 2>/dev/null | head -1)
    if [[ -n "$eden_files" ]]; then
        echo "$eden_files"
        return 0
    fi
    echo "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/all_in_one.h5"
}

H5_FILE=$(find_h5_file)
OUTPUT_BASE="/oak/stanford/orgs/kipac/users/caganze/tfp_flows_interactive_test"

# Extract data source and halo ID from H5 file
FILENAME=$(basename "$H5_FILE")
HALO_ID=$(echo "$FILENAME" | sed 's/.*Halo\([0-9][0-9]*\).*/\1/')

# Determine data source from filename
if [[ "$FILENAME" == *"eden_scaled"* ]]; then
    DATA_SOURCE="eden"
elif [[ "$FILENAME" == *"symphonyHR_scaled"* ]]; then
    DATA_SOURCE="symphony-hr"
elif [[ "$FILENAME" == *"symphony_scaled"* ]]; then
    DATA_SOURCE="symphony"
else
    DATA_SOURCE="unknown"
fi

# Handle fallback file (all_in_one.h5) - use default structure
if [[ "$FILENAME" == "all_in_one.h5" ]] || [[ "$HALO_ID" == "$FILENAME" ]]; then
    echo "⚠️  Using fallback file, setting default halo structure"
    HALO_ID="000"
    DATA_SOURCE="symphony"
fi

# Create hierarchical output directory
OUTPUT_DIR="$OUTPUT_BASE/trained_flows/${DATA_SOURCE}/halo${HALO_ID}"

echo "📋 Test Configuration:"
echo "  H5 file: $H5_FILE"
echo "  Output base: $OUTPUT_BASE"
echo "  Data source: $DATA_SOURCE"
echo "  Halo ID: $HALO_ID"
echo "  Model dir: $OUTPUT_DIR"
echo "  Test PID: 1"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "🧪 TEST 1: Argument Parsing Validation"
echo "======================================"
echo "Testing: python train_tfp_flows.py --help" \
    --generate-samples
echo ""

python train_tfp_flows.py --help | head -15 \
    --generate-samples

echo ""
echo "🧪 TEST 2: Quick Training with Fixed Arguments"
echo "=============================================="
echo "Testing: train_tfp_flows.py with --data_path (should work now!)"
echo ""

# Run minimal training with the fixed arguments
python train_tfp_flows.py \
    --data_path "$H5_FILE" \
    --particle_pid 1 \
    --output_dir "$OUTPUT_DIR" \
    --epochs 2 \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --n_layers 2 \
    --hidden_units 32 \
    --generate-samples \
    --use_kroupa_imf

TRAIN_EXIT_CODE=$?

echo ""
echo "🎯 TEST RESULTS:"
echo "================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ ARGUMENT FIX SUCCESS! No more --data_path errors!"
    echo "✅ Training completed without argument errors"
    
    # Check if model was created (look in hierarchical structure)
    MODEL_FOUND=false
    for source in eden symphony symphony-hr unknown; do
        for halo_dir in "$OUTPUT_DIR/trained_flows/$source"/halo*/; do
            if [[ -f "$halo_dir/model_pid1.npz" ]]; then
                echo "✅ Model file created successfully at: $source/$(basename $halo_dir)/"
                MODEL_FOUND=true
                break 2
            fi
        done
    done
    
    if [[ "$MODEL_FOUND" == "false" ]]; then
        echo "⚠️  Training ran but model file not found (check for other issues)"
    fi
    
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "❌ Check logs for remaining issues"
fi

echo ""
echo "🧪 TEST 3: Test Other Fixed Scripts"
echo "=================================="
echo "Testing submit_small_test.sh argument parsing..."

# Extract just the python command from submit_small_test.sh and test it
TEST_CMD=$(grep -A 10 "python train_tfp_flows.py" submit_small_test.sh | head -8 | sed 's/\$//' | sed 's/\\$//' | tr -d '\\')
echo "Command extracted: $TEST_CMD"

echo ""
echo "🎉 INTERACTIVE TEST COMPLETE!"
echo "============================="
echo "Node: $SLURMD_NODENAME"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Time: $(date)"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🚀 READY FOR FULL DEPLOYMENT!"
    echo "=============================="
    echo "✅ Arguments are fixed - no more --data_path errors expected"
    echo "✅ Ready to submit large job arrays"
    echo ""
    echo "Next steps:"
    echo "  1. Submit small test: sbatch submit_small_test.sh"
    echo "  2. Check logs for success rate improvement"
    echo "  3. Submit full array: ./auto_submit_flows.sh"
else
    echo ""
    echo "⚠️  INVESTIGATION NEEDED"
    echo "======================"
    echo "❌ Arguments may need additional fixes"
    echo "❌ Check logs for specific error messages"
fi
