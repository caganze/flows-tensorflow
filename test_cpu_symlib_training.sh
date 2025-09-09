#!/bin/bash
# Test script for CPU compute node - trains actual particles using symlib format

echo "🧪 CPU SYMLIB TRAINING TEST"
echo "============================"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-test}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-64}"
echo "Time: $(date)"
echo

# Environment setup - using working configuration from GitHub
module purge
module load math devel python/3.9.0

# CPU-specific TensorFlow optimizations
export TF_CPP_MIN_LOG_LEVEL=1
export CUDA_VISIBLE_DEVICES=""  # Force CPU-only
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}

source ~/.bashrc
conda activate bosque

echo "✅ Environment ready"
echo

# Check particle list
if [[ ! -f "particle_list.txt" ]]; then
    echo "❌ particle_list.txt not found"
    exit 1
fi

echo "📋 Particle list found: $(wc -l < particle_list.txt) particles"
echo "First 3 lines:"
head -3 particle_list.txt
echo

# Find test particles: 1 Large, 1 Medium, 1 Small
echo "🔍 Finding test particles..."
LARGE_PARTICLE=$(grep ",Large$" particle_list.txt | head -1)
MEDIUM_PARTICLE=$(grep ",Medium" particle_list.txt | head -1)
SMALL_PARTICLE=$(grep ",Small$" particle_list.txt | head -1)

if [[ -z "$LARGE_PARTICLE" ]]; then
    echo "❌ No Large particles found"
    exit 1
fi
if [[ -z "$MEDIUM_PARTICLE" ]]; then
    echo "❌ No Medium particles found"
    exit 1
fi
if [[ -z "$SMALL_PARTICLE" ]]; then
    echo "❌ No Small particles found"
    exit 1
fi

echo "✅ Test particles selected:"
echo "   Large: $LARGE_PARTICLE"
echo "   Medium: $MEDIUM_PARTICLE"
echo "   Small: $SMALL_PARTICLE"
echo

# Create output directories
OUTPUT_BASE="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output"
mkdir -p "$OUTPUT_BASE/trained_flows" "$OUTPUT_BASE/samples"
mkdir -p "test_logs"

# Test function
test_particle() {
    local particle_line="$1"
    local size_type="$2"
    
    echo "🧠 Testing $size_type particle: $particle_line"
    
    # Parse: PID,HALO_ID,SUITE,OBJECT_COUNT,SIZE_CATEGORY
    IFS=',' read -r PID HALO_ID SUITE OBJECT_COUNT SIZE_CATEGORY <<< "$particle_line"
    
    echo "   PID: $PID, Halo: $HALO_ID, Suite: $SUITE, Count: $OBJECT_COUNT"
    
    # Create output directories
    MODEL_DIR="$OUTPUT_BASE/trained_flows/${SUITE}/halo${HALO_ID#Halo}"
    SAMPLES_DIR="$OUTPUT_BASE/samples/${SUITE}/halo${HALO_ID#Halo}"
    mkdir -p "$MODEL_DIR" "$SAMPLES_DIR"
    
    echo "   Output: $MODEL_DIR"
    echo "   Samples: $SAMPLES_DIR"
    
    # Check if already completed
    if [[ -f "$MODEL_DIR/model_pid${PID}.npz" ]] && [[ -f "$SAMPLES_DIR/model_pid${PID}_samples.npz" ]]; then
        echo "   ✅ Already completed: $size_type particle"
        return 0
    fi
    
    # Adaptive hyperparameters based on particle size
    if [[ "$OBJECT_COUNT" -gt 100000 ]]; then
        EPOCHS=10
        BATCH_SIZE=1024
        N_LAYERS=3
        HIDDEN_UNITS=64
        LEARNING_RATE=0.001
    elif [[ "$OBJECT_COUNT" -gt 10000 ]]; then
        EPOCHS=15
        BATCH_SIZE=512
        N_LAYERS=4
        HIDDEN_UNITS=128
        LEARNING_RATE=0.002
    else
        EPOCHS=20
        BATCH_SIZE=256
        N_LAYERS=5
        HIDDEN_UNITS=256
        LEARNING_RATE=0.003
    fi
    
    echo "   Hyperparameters: epochs=$EPOCHS, batch=$BATCH_SIZE, layers=$N_LAYERS, units=$HIDDEN_UNITS, lr=$LEARNING_RATE"
    
    # Train the particle using symlib format
    echo "   🚀 Starting training..."
    python train_tfp_flows.py \
        --halo_id "$HALO_ID" \
        --particle_pid "$PID" \
        --suite "$SUITE" \
        --output_dir "$MODEL_DIR" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --n_layers $N_LAYERS \
        --hidden_units $HIDDEN_UNITS \
        --learning_rate $LEARNING_RATE \
        --generate-samples \
        > "test_logs/cpu_${size_type}_pid${PID}.log" 2>&1
    
    TRAIN_EXIT=$?
    
    if [[ $TRAIN_EXIT -eq 0 ]]; then
        # Check success
        if [[ -f "$MODEL_DIR/model_pid${PID}.npz" ]] && [[ -f "$SAMPLES_DIR/model_pid${PID}_samples.npz" ]]; then
            echo "   ✅ SUCCESS: $size_type particle trained and sampled"
            echo "   📁 Model: $MODEL_DIR/model_pid${PID}.npz"
            echo "   📁 Samples: $SAMPLES_DIR/model_pid${PID}_samples.npz"
            return 0
        else
            echo "   ❌ FAILED: Files not created"
            return 1
        fi
    else
        echo "   ❌ FAILED: Training exited with code $TRAIN_EXIT"
        echo "   📋 ERROR LOG:"
        echo "   ============="
        if [[ -f "test_logs/cpu_${size_type}_pid${PID}.log" ]]; then
            tail -20 "test_logs/cpu_${size_type}_pid${PID}.log" | sed 's/^/   /'
        else
            echo "   No log file found"
        fi
        echo "   ============="
        return 1
    fi
}

# Run tests
echo "🚀 Starting CPU training tests..."
echo "================================="

SUCCESS_COUNT=0
TOTAL_COUNT=0

# Test Large particle
TOTAL_COUNT=$((TOTAL_COUNT + 1))
if test_particle "$LARGE_PARTICLE" "Large"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
fi
echo

# Test Medium particle
TOTAL_COUNT=$((TOTAL_COUNT + 1))
if test_particle "$MEDIUM_PARTICLE" "Medium"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
fi
echo

# Test Small particle
TOTAL_COUNT=$((TOTAL_COUNT + 1))
if test_particle "$SMALL_PARTICLE" "Small"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
fi
echo

# Results
echo "🎉 CPU SYMLIB TRAINING TEST COMPLETE!"
echo "====================================="
echo "✅ Successful: $SUCCESS_COUNT/$TOTAL_COUNT"
echo "📁 Output directory: $OUTPUT_BASE"
echo "📋 Logs: test_logs/"

if [[ $SUCCESS_COUNT -eq $TOTAL_COUNT ]]; then
    echo "🚀 ALL TESTS PASSED! Ready for production!"
    exit 0
else
    echo "⚠️ Some tests failed. Check logs for details."
    exit 1
fi
