#!/bin/bash

# Parameter sweep for different particle counts
# Sweeps from 100k+ particles down to 100 particles with sampling

set -e  # Exit on any error

# Configuration
HALO_ID="Halo268"
SUITE="eden"
OUTPUT_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/parameter_sweep_results"
LOG_DIR="${OUTPUT_DIR}/logs"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Particle count configurations (from large to small)
declare -a PARTICLE_COUNTS=(100000 50000 25000 10000 5000 2500 1000 500 250 100)

# Parameter combinations to test
declare -a N_LAYERS=(4 6 8)
declare -a HIDDEN_UNITS=(32 64 128)
declare -a LEARNING_RATES=(1e-3 1e-4 1e-5)
declare -a BATCH_SIZES=(256 512 1024)
declare -a EPOCHS=(100 200)

# Function to get particle count for a PID
get_particle_count() {
    local pid=$1
    python3 -c "
import sys
sys.path.append('.')
from symlib_utils import load_symlib_data
try:
    data, masses = load_symlib_data('${HALO_ID}', ${pid}, '${SUITE}')
    print(len(data))
except Exception as e:
    print(0)
"
}

# Function to sample data to target size
sample_data_to_size() {
    local pid=$1
    local target_size=$2
    local output_file=$3
    
    python3 -c "
import numpy as np
import sys
sys.path.append('.')
from symlib_utils import load_symlib_data

try:
    data, masses = load_symlib_data('${HALO_ID}', ${pid}, '${SUITE}')
    if len(data) >= ${target_size}:
        # Sample with replacement to get exactly target_size
        indices = np.random.choice(len(data), size=${target_size}, replace=True)
        sampled_data = data[indices]
        sampled_masses = masses[indices]
        
        # Save sampled data
        np.savez_compressed('${output_file}', 
                          data=sampled_data, 
                          masses=sampled_masses,
                          original_size=len(data),
                          target_size=${target_size})
        print('SUCCESS')
    else:
        print('INSUFFICIENT_DATA')
except Exception as e:
    print(f'ERROR: {e}')
"
}

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
    local result_file="${OUTPUT_DIR}/${config_name}_results.json"
    
    echo "🚀 Training ${config_name}..."
    echo "   PID: ${pid}, Particles: ${particle_count}, Layers: ${n_layers}, Hidden: ${hidden_units}"
    echo "   LR: ${learning_rate}, Batch: ${batch_size}, Epochs: ${epochs}"
    
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
        local runtime=$(grep "Total runtime:" "${log_file}" | tail -1 | awk '{print $3}')
        
        # Save results summary
        cat > "${result_file}" << EOF
{
    "config": "${config_name}",
    "pid": ${pid},
    "particle_count": ${particle_count},
    "n_layers": ${n_layers},
    "hidden_units": ${hidden_units},
    "learning_rate": ${learning_rate},
    "batch_size": ${batch_size},
    "epochs": ${epochs},
    "final_train_loss": ${final_train_loss},
    "final_val_loss": ${final_val_loss},
    "runtime_seconds": ${runtime},
    "status": "SUCCESS"
}
EOF
    else
        echo "   ❌ FAILED: ${config_name}"
        
        # Save failure results
        cat > "${result_file}" << EOF
{
    "config": "${config_name}",
    "pid": ${pid},
    "particle_count": ${particle_count},
    "n_layers": ${n_layers},
    "hidden_units": ${hidden_units},
    "learning_rate": ${learning_rate},
    "batch_size": ${batch_size},
    "epochs": ${epochs},
    "status": "FAILED",
    "error": "Training failed - check log file"
}
EOF
    fi
}

# Main execution
echo "🔬 Starting parameter sweep for particle sizes"
echo "📊 Testing particle counts: ${PARTICLE_COUNTS[*]}"
echo "📁 Output directory: ${OUTPUT_DIR}"
echo "📝 Log directory: ${LOG_DIR}"
echo ""

# Test PIDs (you can modify this list)
declare -a TEST_PIDS=(20 21 22 23 24)

# Initialize results summary
SUMMARY_FILE="${OUTPUT_DIR}/parameter_sweep_summary.json"
echo "[" > "${SUMMARY_FILE}"

total_experiments=0
successful_experiments=0

for pid in "${TEST_PIDS[@]}"; do
    echo "🔍 Testing PID ${pid}..."
    
    # Get original particle count
    original_count=$(get_particle_count "${pid}")
    echo "   Original particle count: ${original_count}"
    
    if [ "${original_count}" -lt 100 ]; then
        echo "   ⚠️  Skipping PID ${pid}: insufficient particles (${original_count} < 100)"
        continue
    fi
    
    for particle_count in "${PARTICLE_COUNTS[@]}"; do
        # Skip if target count is larger than available
        if [ "${particle_count}" -gt "${original_count}" ]; then
            echo "   ⚠️  Skipping ${particle_count} particles: larger than available (${original_count})"
            continue
        fi
        
        echo "   📊 Testing with ${particle_count} particles..."
        
        # Sample data to target size
        sample_file="${OUTPUT_DIR}/sampled_data_pid${pid}_${particle_count}.npz"
        sample_result=$(sample_data_to_size "${pid}" "${particle_count}" "${sample_file}")
        
        if [ "${sample_result}" != "SUCCESS" ]; then
            echo "   ❌ Failed to sample data: ${sample_result}"
            continue
        fi
        
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
                            result_file="${OUTPUT_DIR}/${config_name}_results.json"
                            
                            if [ -f "${result_file}" ] && grep -q '"status": "SUCCESS"' "${result_file}"; then
                                successful_experiments=$((successful_experiments + 1))
                                
                                # Add to summary
                                if [ "${total_experiments}" -gt 1 ]; then
                                    echo "," >> "${SUMMARY_FILE}"
                                fi
                                cat "${result_file}" >> "${SUMMARY_FILE}"
                            fi
                            
                            echo ""
                        done
                    done
                done
            done
        done
    done
done

# Close summary JSON
echo "]" >> "${SUMMARY_FILE}"

# Generate final report
REPORT_FILE="${OUTPUT_DIR}/parameter_sweep_report.txt"
cat > "${REPORT_FILE}" << EOF
Parameter Sweep Report
=====================
Date: $(date)
Halo: ${HALO_ID}
Suite: ${SUITE}

Total Experiments: ${total_experiments}
Successful: ${successful_experiments}
Failed: $((total_experiments - successful_experiments))
Success Rate: $(echo "scale=2; ${successful_experiments} * 100 / ${total_experiments}" | bc -l)%

Particle Counts Tested: ${PARTICLE_COUNTS[*]}
PIDs Tested: ${TEST_PIDS[*]}

Parameter Ranges:
- Layers: ${N_LAYERS[*]}
- Hidden Units: ${HIDDEN_UNITS[*]}
- Learning Rates: ${LEARNING_RATES[*]}
- Batch Sizes: ${BATCH_SIZES[*]}
- Epochs: ${EPOCHS[*]}

Results saved in: ${OUTPUT_DIR}
Logs saved in: ${LOG_DIR}
Summary JSON: ${SUMMARY_FILE}
EOF

echo ""
echo "🎉 Parameter sweep completed!"
echo "📊 Total experiments: ${total_experiments}"
echo "✅ Successful: ${successful_experiments}"
echo "❌ Failed: $((total_experiments - successful_experiments))"
echo "📁 Results saved in: ${OUTPUT_DIR}"
echo "📝 Report: ${REPORT_FILE}"
echo "📋 Summary: ${SUMMARY_FILE}"
