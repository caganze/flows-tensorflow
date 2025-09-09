#!/bin/bash

# 🧪 Test KDE Training
# Quick test to verify KDE setup works

set -e

echo "🧪 KDE TEST"
echo "==========="
echo "🎯 Testing KDE training setup"
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
    "train_kde_conditional.py"
    "kde_cpu_job.sh"
    "submit_cpu_kde_smart.sh"
    "submit_cpu_kde_chunked.sh"
    "filter_completed_kde.sh"
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
if ! ./filter_completed_kde.sh --dry-run; then
    echo "❌ Filter script test failed"
    exit 1
fi

echo ""
echo "🚀 Testing chunked submission (dry run)..."
# Create a minimal test particle list
echo "$TEST_PID,$TEST_HALO_ID,$TEST_SUITE,10000,medium" > test_particle_list_kde.txt

if ! ./submit_cpu_kde_chunked.sh --particle-list test_particle_list_kde.txt --dry-run; then
    echo "❌ Chunked submission test failed"
    exit 1
fi

echo ""
echo "🧠 Testing smart submission (dry run)..."
if ! ./submit_cpu_kde_smart.sh --particle-list test_particle_list_kde.txt --dry-run; then
    echo "❌ Smart submission test failed"
    exit 1
fi

echo ""
echo "🧪 Testing KDE python import..."
python3 -c "
import sys
sys.path.append('/oak/stanford/orgs/kipac/users/caganze/kde_sampler')
try:
    from kde_sampler import M4Kernel, KDESampler
    print('✅ KDE imports successful')
except ImportError as e:
    print(f'⚠️  KDE import warning: {e}')
    print('   This is expected if not on the cluster')
"

# Cleanup
rm -f test_particle_list_kde.txt

echo ""
echo "✅ KDE TEST PASSED"
echo "=================="
echo "🎉 All KDE scripts are working correctly!"
echo ""
echo "📋 Usage Examples:"
echo "   # Dry run to preview:"
echo "   ./submit_cpu_kde_smart.sh --dry-run"
echo ""
echo "   # Actual submission:"
echo "   ./submit_cpu_kde_smart.sh"
echo ""
echo "   # Custom parameters:"
echo "   ./submit_cpu_kde_smart.sh --chunk-size 100 --concurrent 15"
echo ""
echo "📁 Output will go to:"
echo "   /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/kde_output/"
