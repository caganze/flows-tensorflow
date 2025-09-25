#!/bin/bash

# Quick parameter sweep for different particle counts
# Tests a subset of parameters for faster execution

set -e  # Exit on any error

# Configuration
HALO_ID="Halo268"
SUITE="eden"
OUTPUT_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/quick_sweep_results"
LOG_DIR="${OUTPUT_DIR}/logs"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Particle count configurations (from large to small)
declare -a PARTICLE_COUNTS=(10000 5000 1000 500 100)

# Reduced parameter combinations for quick testing
declare -a N_LAYERS=(4 6)
declare -a HIDDEN_UNITS=(32 64)
declare -a LEARNING_RATES=(1e-4)
declare -a BATCH_SIZES=(512)
declare -a EPOCHS=(50)

# Test PIDs
declare -a TEST_PIDS=(20 21)

echo "🚀 Quick parameter sweep starting..."
echo "📊 Testing particle counts: ${PARTICLE_COUNTS[*]}"
echo "🎯 PIDs: ${TEST_PIDS[*]}"
echo ""

# Function to run training with specific parameters
run_training() {
    local pid=$1
    local particle_count=$2
    local n_layers=$3
    local hidden_units=$4
    local learning_rate=$5
    local batch_size=$6
    local epochs=$7
    
    local config_name="pid${pid}_${particle_count}p_${n_layers}l_${hidden_units}h_lr${learning_rate}_bs${batch_size}_ep${epochs}"
    local log_file="${LOG_DIR}/${config_name}.log"
    
    echo "🚀 Training ${config_name}..."
    
    # Run training
    python3 train_tfp_flows_conditional.py \
        --halo_id "${HALO_ID}" \
        --particle_pid "${pid}" \
        --suite "${SUITE}" \
        --n_layers "${n_layers}" \
        --hidden_units "${hidden_units}" \
        --learning_rate "${learning_rate}" \
        --batch_size "${batch_size}" \
        --epochs "${epochs}" \
        --validation_freq 10 \
        --clip_outliers 5.0 \
        --use_gmm_base \
        --gmm_components 5 \
        --output_dir "${OUTPUT_DIR}" \
        > "${log_file}" 2>&1
    
    # Check if training was successful
    if grep -q "✅ Training completed successfully" "${log_file}"; then
        echo "   ✅ SUCCESS: ${config_name}"
        
        # Extract final metrics
        local final_train_loss=$(grep "final_training_loss:" "${log_file}" | tail -1 | awk '{print $2}')
        local final_val_loss=$(grep "final_validation_loss:" "${log_file}" | tail -1 | awk '{print $2}')
        
        echo "   📊 Final train loss: ${final_train_loss}"
        echo "   📊 Final val loss: ${final_val_loss}"
    else
        echo "   ❌ FAILED: ${config_name}"
    fi
    echo ""
}

# Main execution
total_experiments=0
successful_experiments=0

for pid in "${TEST_PIDS[@]}"; do
    echo "🔍 Testing PID ${pid}..."
    
    for particle_count in "${PARTICLE_COUNTS[@]}"; do
        echo "   📊 Testing with ${particle_count} particles..."
        
        # Run parameter sweep for this particle count
        for n_layers in "${N_LAYERS[@]}"; do
            for hidden_units in "${HIDDEN_UNITS[@]}"; do
                for learning_rate in "${LEARNING_RATES[@]}"; do
                    for batch_size in "${BATCH_SIZES[@]}"; do
                        # Skip if batch size is larger than particle count
                        if [ "${batch_size}" -gt "${particle_count}" ]; then
                            continue
                        fi
                        
                        for epochs in "${EPOCHS[@]}"; do
                            total_experiments=$((total_experiments + 1))
                            
                            # Run training
                            run_training "${pid}" "${particle_count}" "${n_layers}" "${hidden_units}" "${learning_rate}" "${batch_size}" "${epochs}"
                            
                            # Check if successful
                            config_name="pid${pid}_${particle_count}p_${n_layers}l_${hidden_units}h_lr${learning_rate}_bs${batch_size}_ep${epochs}"
                            log_file="${LOG_DIR}/${config_name}.log"
                            
                            if [ -f "${log_file}" ] && grep -q "✅ Training completed successfully" "${log_file}"; then
                                successful_experiments=$((successful_experiments + 1))
                            fi
                        done
                    done
                done
            done
        done
    done
done

echo "🎉 Quick parameter sweep completed!"
echo "📊 Total experiments: ${total_experiments}"
echo "✅ Successful: ${successful_experiments}"
echo "❌ Failed: $((total_experiments - successful_experiments))"
echo "📁 Results saved in: ${OUTPUT_DIR}"
