#!/bin/bash
#SBATCH --job-name="continuous_batch"
#SBATCH --partition=owners
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/continuous_batch_%j.out
#SBATCH --error=logs/continuous_batch_%j.err

set -e

echo "🚀 CONTINUOUS FLOW BATCH JOB - Job ID ${SLURM_JOB_ID:-test}"
echo "🔧 Mode: Batch Continuous Flow Processing (For Loop)"
echo "Started: $(date)"
echo "Node: ${SLURM_NODELIST:-local}"

# Get line range arguments
START_LINE=$1
END_LINE=$2

if [[ -z "$START_LINE" || -z "$END_LINE" ]]; then
    echo "❌ Usage: $0 <start_line> <end_line>"
    exit 1
fi

echo "📋 Processing lines $START_LINE to $END_LINE from particle_list_continuous_incomplete.txt"

# Environment setup
module --force purge
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0
export CUDA_VISIBLE_DEVICES=0

source ~/.bashrc
conda activate bosque

# Create directories
mkdir -p logs success_logs failed_jobs

# Check if incomplete particle list exists
PARTICLE_LIST_FILE="particle_list_continuous_incomplete.txt"
if [[ ! -f "$PARTICLE_LIST_FILE" ]]; then
    echo "❌ Incomplete particle list not found: $PARTICLE_LIST_FILE"
    exit 1
fi

TOTAL_LINES=$(wc -l < "$PARTICLE_LIST_FILE")
echo "📊 Total incomplete particles: $TOTAL_LINES"

if [[ $END_LINE -gt $TOTAL_LINES ]]; then
    END_LINE=$TOTAL_LINES
    echo "🔧 Adjusted end line to $END_LINE (file limit)"
fi

ACTUAL_PARTICLES=$((END_LINE - START_LINE + 1))
echo "🎯 Will process $ACTUAL_PARTICLES particles"
echo ""

# Process particles in for loop
PROCESSED=0
SUCCEEDED=0
FAILED=0

for (( line_num=START_LINE; line_num<=END_LINE; line_num++ )); do
    # Get the specific line
    PARTICLE_LINE=$(sed -n "${line_num}p" "$PARTICLE_LIST_FILE")
    
    if [[ -z "$PARTICLE_LINE" ]]; then
        echo "⚠️  Empty line $line_num, skipping"
        continue
    fi
    
    # Parse particle list line: PID,HALO_ID,SUITE,OBJECT_COUNT,SIZE_CATEGORY
    IFS=',' read -r SELECTED_PID HALO_ID SUITE OBJECT_COUNT SIZE_CATEGORY <<< "$PARTICLE_LINE"
    
    echo "🎯 Processing [$((++PROCESSED))/$ACTUAL_PARTICLES]: PID $SELECTED_PID from $HALO_ID (suite: $SUITE, objects: $OBJECT_COUNT)"
    
    # Output directories
    OUTPUT_BASE_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output_conditional"
    MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/${SUITE}/halo${HALO_ID#Halo}"
    SAMPLES_DIR="$OUTPUT_BASE_DIR/samples/${SUITE}/halo${HALO_ID#Halo}"
    
    mkdir -p "$MODEL_DIR" "$SAMPLES_DIR"
    
    # Check if already completed (safety check)
    if [[ -f "$MODEL_DIR/model_pid${SELECTED_PID}.npz" ]] && [[ -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.npz" || -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.h5" ]]; then
        echo "  ✅ Already completed: $HALO_ID PID $SELECTED_PID"
        echo "$(date) SUCCESS halo:$HALO_ID pid:$SELECTED_PID already_completed_batch" >> success_logs/continuous_flow_success.log
        SUCCEEDED=$((SUCCEEDED + 1))
        continue
    fi
    
    # Determine parameters based on particle size
    if [[ $OBJECT_COUNT -gt 100000 ]]; then
        EPOCHS=60
        BATCH_SIZE=512
        N_LAYERS=6
        HIDDEN_UNITS=512
        LEARNING_RATE=1e-4
        echo "  🧠 Large particle: epochs=60, layers=6, units=512, lr=1e-4"
    elif [[ $OBJECT_COUNT -gt 50000 ]]; then
        EPOCHS=50
        BATCH_SIZE=512
        N_LAYERS=5
        HIDDEN_UNITS=384
        LEARNING_RATE=2e-4
        echo "  🧠 Medium-large: epochs=50, layers=5, units=384, lr=2e-4"
    elif [[ $OBJECT_COUNT -lt 5000 ]]; then
        EPOCHS=40
        BATCH_SIZE=256
        N_LAYERS=4
        HIDDEN_UNITS=256
        LEARNING_RATE=3e-4
        echo "  🧠 Small particle: epochs=40, layers=4, units=256, lr=3e-4"
    else
        EPOCHS=45
        BATCH_SIZE=512
        N_LAYERS=4
        HIDDEN_UNITS=320
        LEARNING_RATE=2e-4
        echo "  🧠 Medium particle: epochs=45, layers=4, units=320, lr=2e-4"
    fi
    
    # Run continuous flow training
    echo "  🚀 Training conditional continuous flow..."
    
    TRAIN_EXIT=0
    python train_tfp_flows_conditional.py \
        --halo_id "$HALO_ID" \
        --particle_pid "$SELECTED_PID" \
        --suite "$SUITE" \
        --output_dir "$MODEL_DIR" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --n_layers $N_LAYERS \
        --hidden_units $HIDDEN_UNITS || TRAIN_EXIT=$?
    
    if [[ $TRAIN_EXIT -eq 0 ]]; then
        # Check success
        if [[ -f "$MODEL_DIR/model_pid${SELECTED_PID}.npz" ]] && [[ -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.npz" || -f "$SAMPLES_DIR/model_pid${SELECTED_PID}_samples.h5" ]]; then
            echo "  ✅ SUCCESS: Continuous Flow $HALO_ID PID $SELECTED_PID"
            echo "$(date) SUCCESS halo:$HALO_ID pid:$SELECTED_PID training_completed_batch" >> success_logs/continuous_flow_success.log
            SUCCEEDED=$((SUCCEEDED + 1))
        else
            echo "  ❌ FAILED: Missing output files"
            echo "$(date) FAILED halo:$HALO_ID pid:$SELECTED_PID missing_outputs_batch" >> failed_jobs/continuous_flow_failures.log
            FAILED=$((FAILED + 1))
        fi
    else
        echo "  ❌ FAILED: Training exit code $TRAIN_EXIT"
        echo "$(date) FAILED halo:$HALO_ID pid:$SELECTED_PID training_failed_${TRAIN_EXIT}_batch" >> failed_jobs/continuous_flow_failures.log
        FAILED=$((FAILED + 1))
    fi
    
    echo ""
done

echo "📊 BATCH PROCESSING SUMMARY"
echo "==========================="
echo "🎯 Processed: $PROCESSED particles"
echo "✅ Succeeded: $SUCCEEDED"
echo "❌ Failed: $FAILED"
echo "📈 Success rate: $(( SUCCEEDED * 100 / PROCESSED ))%"
echo ""
echo "🏁 Batch completed: $(date)"

