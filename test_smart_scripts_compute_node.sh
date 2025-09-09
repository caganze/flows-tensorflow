#!/bin/bash

# 🧪 SMART SCRIPTS COMPUTE NODE TEST
# Comprehensive test for both CPU and GPU smart submission scripts
# Tests bosque environment on compute node

set -e

echo "🧪 SMART SCRIPTS COMPUTE NODE TEST"
echo "=================================="
echo "Testing bosque environment for both CPU and GPU"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Started: $(date)"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
CRITICAL_FAILURES=()

# Helper function for test results
test_result() {
    local test_name="$1"
    local result="$2"
    local details="$3"
    local critical="$4"
    
    if [[ "$result" == "PASS" ]]; then
        echo -e "${GREEN}✅ PASS${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAIL${NC}: $test_name"
        ((TESTS_FAILED++))
        if [[ "$critical" == "true" ]]; then
            CRITICAL_FAILURES+=("$test_name")
        fi
    fi
    
    if [[ -n "$details" ]]; then
        echo "   $details"
    fi
    echo ""
}

# Create test particle list with symlib format
create_test_particle_list() {
    echo "📝 Creating test particle list..."
    cat > test_particle_list.txt << 'EOF'
1,Halo939,eden,1000,Small
2,Halo939,eden,2000,Small
3,Halo718,symphony,1500,Small
EOF
    echo "✅ Test particle list created with 3 test particles"
    echo ""
}

# Test 1: Basic environment setup
test_basic_environment() {
    echo -e "${BLUE}1️⃣ Testing basic compute node environment${NC}"
    echo "============================================="
    
    # Check if we're on a compute node
    if [[ -n "$SLURM_JOB_ID" ]]; then
        test_result "SLURM job environment" "PASS" "Job ID: $SLURM_JOB_ID, Node: ${SLURM_NODELIST:-unknown}"
    else
        test_result "SLURM job environment" "FAIL" "Not running in SLURM job" "true"
        return 1
    fi
    
    # Check essential files exist
    local missing_files=()
    local required_files=(
        "submit_gpu_smart.sh"
        "submit_cpu_smart.sh"
        "filter_completed_particles.sh"
        "brute_force_gpu_job.sh"
        "brute_force_cpu_parallel.sh"
        "train_tfp_flows.py"
        "symlib_utils.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -eq 0 ]]; then
        test_result "Required scripts present" "PASS" "All ${#required_files[@]} required files found"
    else
        test_result "Required scripts present" "FAIL" "Missing: ${missing_files[*]}" "true"
    fi
}

# Test 2: CPU Environment (bosque)
test_cpu_environment() {
    echo -e "${BLUE}2️⃣ Testing CPU environment (bosque)${NC}"
    echo "=========================================="
    
    # Test conda activation
    local cpu_env_test=$(timeout 60 bash -c '
        source ~/.bashrc
        conda activate bosque 2>/dev/null
        if [[ $? -eq 0 ]]; then
            echo "CPU_ENV_SUCCESS"
            # Test symlib import
            python3 -c "
import os
os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"\"
os.environ[\"TF_CPP_MIN_LOG_LEVEL\"] = \"2\"
try:
    import symlib
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], \"GPU\")
    print(\"SYMLIB_IMPORT_SUCCESS\")
except Exception as e:
    print(f\"SYMLIB_IMPORT_FAILED: {e}\")
            " 2>&1
        else
            echo "CPU_ENV_FAILED"
        fi
    ' 2>&1)
    
    if [[ "$cpu_env_test" == *"CPU_ENV_SUCCESS"* ]]; then
        if [[ "$cpu_env_test" == *"SYMLIB_IMPORT_SUCCESS"* ]]; then
            test_result "CPU environment (bosque)" "PASS" "Environment activated and symlib imports successfully"
        else
            test_result "CPU environment (bosque)" "FAIL" "Environment activated but symlib import failed: $cpu_env_test"
        fi
    else
        test_result "CPU environment (bosque)" "FAIL" "Could not activate bosque environment: $cpu_env_test"
    fi
}

# Test 3: GPU Environment (bosque)
test_gpu_environment() {
    echo -e "${BLUE}3️⃣ Testing GPU environment (bosque)${NC}"
    echo "===================================="
    
    # Test conda activation
    local gpu_env_test=$(timeout 60 bash -c '
        source ~/.bashrc
        conda activate bosque 2>/dev/null
        if [[ $? -eq 0 ]]; then
            echo "GPU_ENV_SUCCESS"
            # Test symlib import
            python3 -c "
import os
os.environ[\"TF_CPP_MIN_LOG_LEVEL\"] = \"2\"
try:
    import symlib
    import numpy as np
    import tensorflow as tf
    print(f\"TF_VERSION: {tf.__version__}\")
    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices(\"GPU\")
    print(f\"GPUS_AVAILABLE: {len(gpus)}\")
    print(\"SYMLIB_IMPORT_SUCCESS\")
except Exception as e:
    print(f\"SYMLIB_IMPORT_FAILED: {e}\")
            " 2>&1
        else
            echo "GPU_ENV_FAILED"
        fi
    ' 2>&1)
    
    if [[ "$gpu_env_test" == *"GPU_ENV_SUCCESS"* ]]; then
        if [[ "$gpu_env_test" == *"SYMLIB_IMPORT_SUCCESS"* ]]; then
            local gpu_count=$(echo "$gpu_env_test" | grep "GPUS_AVAILABLE:" | cut -d: -f2 | tr -d ' ')
            test_result "GPU environment (bosque)" "PASS" "Environment activated, symlib imports, GPUs: ${gpu_count:-unknown}"
        else
            test_result "GPU environment (bosque)" "FAIL" "Environment activated but symlib import failed: $gpu_env_test"
        fi
    else
        test_result "GPU environment (bosque)" "FAIL" "Could not activate bosque environment: $gpu_env_test"
    fi
}

# Test 4: Filter script with symlib format
test_filter_script() {
    echo -e "${BLUE}4️⃣ Testing filter script with symlib format${NC}"
    echo "=============================================="
    
    # Test filter script with our test particle list
    local filter_test=$(timeout 30 ./filter_completed_particles.sh --input test_particle_list.txt --output test_filtered.txt --dry-run --verbose 2>&1)
    local filter_exit=$?
    
    if [[ $filter_exit -eq 0 ]] && [[ "$filter_test" == *"📊 Parsed: pid=1, halo_id=Halo939, suite=eden"* ]]; then
        test_result "Filter script symlib parsing" "PASS" "Successfully parsed symlib format"
        
        # Test actual filtering (not dry run)
        ./filter_completed_particles.sh --input test_particle_list.txt --output test_filtered.txt >/dev/null 2>&1
        if [[ -f "test_filtered.txt" ]]; then
            local filtered_count=$(wc -l < test_filtered.txt)
            test_result "Filter script output creation" "PASS" "Created filtered list with $filtered_count particles"
        else
            test_result "Filter script output creation" "FAIL" "No filtered output file created"
        fi
    else
        test_result "Filter script symlib parsing" "FAIL" "Failed to parse symlib format correctly: $filter_test" "true"
    fi
}

# Test 5: CPU Smart Script (Dry Run)
test_cpu_smart_script() {
    echo -e "${BLUE}5️⃣ Testing CPU smart script (dry run)${NC}"
    echo "====================================="
    
    # Set up environment and test
    export PARTICLE_LIST_FILE="test_particle_list.txt"
    
    local cpu_test=$(timeout 120 bash -c '
        source ~/.bashrc
        conda activate bosque 2>/dev/null
        
        # Test CPU smart script dry run
        ./submit_cpu_smart.sh --dry-run --chunk-size 2 --concurrent 1 --verbose 2>&1
    ' 2>&1)
    local cpu_exit=$?
    
    if [[ $cpu_exit -eq 0 ]] && [[ "$cpu_test" == *"DRY RUN"* ]]; then
        if [[ "$cpu_test" == *"incomplete particles"* ]]; then
            test_result "CPU smart script (dry run)" "PASS" "Successfully executed dry run and found particles to process"
        else
            test_result "CPU smart script (dry run)" "PASS" "Dry run successful but no incomplete particles (expected if all completed)"
        fi
    else
        test_result "CPU smart script (dry run)" "FAIL" "CPU smart script dry run failed: $cpu_test" "true"
    fi
}

# Test 6: GPU Smart Script (Dry Run)
test_gpu_smart_script() {
    echo -e "${BLUE}6️⃣ Testing GPU smart script (dry run)${NC}"
    echo "====================================="
    
    # Set up environment and test
    export PARTICLE_LIST_FILE="test_particle_list.txt"
    
    local gpu_test=$(timeout 120 bash -c '
        source ~/.bashrc
        conda activate bosque 2>/dev/null
        
        # Test GPU smart script dry run
        ./submit_gpu_smart.sh --dry-run --chunk-size 2 --concurrent 1 --verbose 2>&1
    ' 2>&1)
    local gpu_exit=$?
    
    if [[ $gpu_exit -eq 0 ]] && [[ "$gpu_test" == *"DRY RUN"* ]]; then
        if [[ "$gpu_test" == *"incomplete particles"* ]]; then
            test_result "GPU smart script (dry run)" "PASS" "Successfully executed dry run and found particles to process"
        else
            test_result "GPU smart script (dry run)" "PASS" "Dry run successful but no incomplete particles (expected if all completed)"
        fi
    else
        test_result "GPU smart script (dry run)" "FAIL" "GPU smart script dry run failed: $gpu_test" "true"
    fi
}

# Test 7: Training Script Help (Quick validation)
test_training_script() {
    echo -e "${BLUE}7️⃣ Testing training script basic functionality${NC}"
    echo "=============================================="
    
    local train_test=$(timeout 60 bash -c '
        source ~/.bashrc
        conda activate bosque 2>/dev/null
        
        # Test training script help
        python3 train_tfp_flows.py --help 2>&1 | head -10
    ' 2>&1)
    local train_exit=$?
    
    if [[ $train_exit -eq 0 ]] && [[ "$train_test" == *"--halo_id"* ]] && [[ "$train_test" == *"--suite"* ]]; then
        test_result "Training script symlib arguments" "PASS" "Training script has correct symlib arguments (--halo_id, --suite)"
    else
        test_result "Training script symlib arguments" "FAIL" "Training script missing symlib arguments or failed to run: $train_test"
    fi
}

# Test 8: Output directory permissions
test_output_permissions() {
    echo -e "${BLUE}8️⃣ Testing output directory permissions${NC}"
    echo "======================================"
    
    local output_base="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output"
    
    if [[ -d "$output_base" ]]; then
        # Test write permissions
        local test_file="$output_base/compute_node_test_$(date +%s).tmp"
        if touch "$test_file" 2>/dev/null && rm "$test_file" 2>/dev/null; then
            test_result "Output directory permissions" "PASS" "Can write to $output_base"
        else
            test_result "Output directory permissions" "FAIL" "Cannot write to $output_base" "true"
        fi
    else
        # Try to create directory
        if mkdir -p "$output_base" 2>/dev/null; then
            test_result "Output directory creation" "PASS" "Created output directory: $output_base"
        else
            test_result "Output directory creation" "FAIL" "Cannot create output directory: $output_base" "true"
        fi
    fi
}

# Cleanup function
cleanup_test_files() {
    echo "🧹 Cleaning up test files..."
    rm -f test_particle_list.txt test_filtered.txt 2>/dev/null || true
    echo "✅ Cleanup completed"
    echo ""
}

# Main execution
main() {
    create_test_particle_list
    
    test_basic_environment
    test_cpu_environment  
    test_gpu_environment
    test_filter_script
    test_cpu_smart_script
    test_gpu_smart_script
    test_training_script
    test_output_permissions
    
    cleanup_test_files
    
    # Final summary
    echo -e "${BLUE}📊 FINAL TEST SUMMARY${NC}"
    echo "===================="
    echo -e "${GREEN}✅ Tests passed: $TESTS_PASSED${NC}"
    echo -e "${RED}❌ Tests failed: $TESTS_FAILED${NC}"
    
    total_tests=$((TESTS_PASSED + TESTS_FAILED))
    if [[ $total_tests -gt 0 ]]; then
        success_rate=$((TESTS_PASSED * 100 / total_tests))
        echo "📈 Success rate: ${success_rate}%"
    fi
    
    if [[ ${#CRITICAL_FAILURES[@]} -gt 0 ]]; then
        echo ""
        echo -e "${RED}🚨 CRITICAL FAILURES:${NC}"
        for failure in "${CRITICAL_FAILURES[@]}"; do
            echo "   ❌ $failure"
        done
    fi
    
    echo ""
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
        echo "✅ Both CPU and GPU smart scripts are ready for production use"
        echo ""
        echo -e "${BLUE}🚀 READY FOR PRODUCTION DEPLOYMENT${NC}"
        echo "You can now safely run:"
        echo "  ./submit_cpu_smart.sh --chunk-size 50 --concurrent 4"
        echo "  ./submit_gpu_smart.sh --chunk-size 100 --concurrent 2"
        exit 0
    else
        echo -e "${RED}💥 TESTS FAILED!${NC}"
        echo "❌ Fix the failed tests before deploying smart scripts to production"
        
        if [[ ${#CRITICAL_FAILURES[@]} -gt 0 ]]; then
            echo "🚨 Critical failures must be addressed immediately"
            exit 2
        else
            echo "⚠️ Non-critical failures - may still be usable with caution"
            exit 1
        fi
    fi
}

# Run main function
main "$@"

