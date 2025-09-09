#!/bin/bash

# 🧪 Test Continuous Flow Training
# Quick test to verify continuous flow setup works

set -e

echo "🧪 CONTINUOUS FLOW TEST"
echo "======================="
echo "🎯 Testing continuous flow training setup"
echo ""

# Test parameters
TEST_HALO_ID="Halo939"
TEST_PID="123"
TEST_SUITE="eden"

echo "🔧 Test Configuration:"
echo "   Halo ID: $TEST_HALO_ID"
echo "   Parent ID: $TEST_PID"
echo "   Suite: $TEST_SUITE"
echo ""

# Check required files
echo "📁 Checking required files..."
required_files=(
    "train_tfp_flows_conditional.py"
    "continuous_flow_gpu_job.sh"
    "submit_gpu_continuous_smart.sh"
    "submit_gpu_continuous_chunked.sh"
    "filter_completed_continuous.sh"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    else
        echo "   ✅ $file"
    fi
done

if [[ ${#missing_files[@]} -gt 0 ]]; then
    echo ""
    echo "❌ Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

echo ""
echo "🔍 Testing filter script..."
if ! ./filter_completed_continuous.sh --dry-run; then
    echo "❌ Filter script test failed"
    exit 1
fi

echo ""
echo "🚀 Testing chunked submission (dry run)..."
# Create a minimal test particle list
echo "$TEST_PID,$TEST_HALO_ID,$TEST_SUITE,10000,medium" > test_particle_list.txt

if ! ./submit_gpu_continuous_chunked.sh --particle-list test_particle_list.txt --dry-run; then
    echo "❌ Chunked submission test failed"
    exit 1
fi

echo ""
echo "🧠 Testing smart submission (dry run)..."
if ! ./submit_gpu_continuous_smart.sh --particle-list test_particle_list.txt --dry-run; then
    echo "❌ Smart submission test failed"
    exit 1
fi

# Cleanup
rm -f test_particle_list.txt

echo ""
echo "✅ CONTINUOUS FLOW TEST PASSED"
echo "=============================="
echo "🎉 All continuous flow scripts are working correctly!"
echo ""
echo "📋 Usage Examples:"
echo "   # Dry run to preview:"
echo "   ./submit_gpu_continuous_smart.sh --dry-run"
echo ""
echo "   # Actual submission:"
echo "   ./submit_gpu_continuous_smart.sh"
echo ""
echo "   # Custom chunk size:"
echo "   ./submit_gpu_continuous_smart.sh --chunk-size 100"
echo ""
echo "📁 Output will go to:"
echo "   /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output_conditional/"
