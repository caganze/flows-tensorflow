#!/bin/bash

# 🚀 Deploy and Test on Sherlock
# Upload local changes and run verification on Sherlock

set -e

echo "🚀 DEPLOY AND TEST ON SHERLOCK"
echo "==============================="
echo "📤 Uploading local changes and running verification"
echo ""

# Configuration
SHERLOCK_HOST="sherlock.stanford.edu"
SHERLOCK_USER="${USER}"  # Use current username
SHERLOCK_PATH="/oak/stanford/orgs/kipac/users/${USER}/flows-tensorflow"
LOCAL_PATH="."

# Files to upload
ESSENTIAL_FILES=(
    "train_tfp_flows.py"
    "kroupa_imf.py"
    "brute_force_gpu_job.sh"
    "brute_force_cpu_parallel.sh"
    "submit_tfp_array.sh"
    "submit_cpu_smart.sh"
    "submit_cpu_chunked.sh"
    "train_single_gpu.sh"
    "meta_test_full_pipeline.sh"
    "validate_deployment.sh"
    "filter_completed_particles.sh"
    "generate_particle_list.sh"
    "kroupa_samples.py"
    "kroupa_samples.sh"
    "final_verification.py"
)

# Test files to upload
TEST_FILES=(
    "comprehensive_consistency_check.py"
    "test_kroupa_samples.sh"
    "run_comprehensive_gpu_test.sh"
)

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --upload-only       Upload files without testing"
    echo "  --test-only         Skip upload, just run tests"
    echo "  --quick-test        Run quick verification only"
    echo "  --full-test         Run comprehensive tests"
    echo "  --user USERNAME     Sherlock username (default: $USER)"
    echo "  --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                          # Upload and run quick test"
    echo "  $0 --full-test             # Upload and run comprehensive tests"
    echo "  $0 --test-only --quick-test # Just run quick test on Sherlock"
    echo ""
}

# Parse arguments
UPLOAD=true
TEST=true
FULL_TEST=false
QUICK_TEST=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --upload-only)
            TEST=false
            shift
            ;;
        --test-only)
            UPLOAD=false
            shift
            ;;
        --quick-test)
            QUICK_TEST=true
            FULL_TEST=false
            shift
            ;;
        --full-test)
            FULL_TEST=true
            QUICK_TEST=false
            shift
            ;;
        --user)
            SHERLOCK_USER="$2"
            SHERLOCK_PATH="/oak/stanford/orgs/kipac/users/$2/flows-tensorflow"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

echo "📋 Configuration:"
echo "   Sherlock host: $SHERLOCK_HOST"
echo "   Sherlock user: $SHERLOCK_USER"
echo "   Sherlock path: $SHERLOCK_PATH"
echo "   Upload files: $UPLOAD"
echo "   Run tests: $TEST"
echo ""

# Upload files
if [[ "$UPLOAD" == "true" ]]; then
    echo "📤 UPLOADING FILES TO SHERLOCK"
    echo "==============================="
    
    # Check if we can connect
    echo "🔍 Testing connection..."
    if ! ssh -o ConnectTimeout=10 "$SHERLOCK_USER@$SHERLOCK_HOST" "echo 'Connection successful'" >/dev/null 2>&1; then
        echo "❌ Cannot connect to Sherlock"
        echo "💡 Make sure you have SSH access and VPN is connected"
        exit 1
    fi
    echo "✅ Connection successful"
    
    # Create remote directory if needed
    echo "📁 Ensuring remote directory exists..."
    ssh "$SHERLOCK_USER@$SHERLOCK_HOST" "mkdir -p $SHERLOCK_PATH"
    
    # Upload essential files
    echo "📤 Uploading essential files..."
    for file in "${ESSENTIAL_FILES[@]}"; do
        if [[ -f "$file" ]]; then
            echo "   📄 $file"
            scp "$file" "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/"
        else
            echo "   ⚠️  $file (not found locally)"
        fi
    done
    
    # Upload test files
    echo "📤 Uploading test files..."
    for file in "${TEST_FILES[@]}"; do
        if [[ -f "$file" ]]; then
            echo "   📄 $file"
            scp "$file" "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/"
        else
            echo "   ⚠️  $file (not found locally)"
        fi
    done
    
    # Make scripts executable
    echo "🔧 Making scripts executable..."
    ssh "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && chmod +x *.sh *.py"
    
    echo "✅ Upload complete"
    echo ""
fi

# Run tests
if [[ "$TEST" == "true" ]]; then
    echo "🧪 RUNNING TESTS ON SHERLOCK"
    echo "============================="
    
    if [[ "$QUICK_TEST" == "true" ]]; then
        echo "⚡ Running quick verification..."
        
        # Create a quick test script
        cat > /tmp/quick_sherlock_test.sh << 'EOF'
#!/bin/bash
cd /oak/stanford/orgs/kipac/users/$USER/flows-tensorflow

echo "🧪 QUICK SHERLOCK VERIFICATION"
echo "==============================="

# Load modules
module purge
module load math devel python/3.9.0
source ~/.bashrc
conda activate bosque

echo "✅ Environment loaded"

# Test argument parser
echo ""
echo "🔍 Testing train_tfp_flows.py argument parser..."
if python train_tfp_flows.py --help | grep -q "generate-samples"; then
    echo "✅ --generate-samples argument found"
else
    echo "❌ --generate-samples argument missing"
fi

if python train_tfp_flows.py --help | grep -q "use_kroupa_imf"; then
    echo "✅ --use_kroupa_imf argument found"
else
    echo "❌ --use_kroupa_imf argument missing"
fi

# Test Kroupa IMF imports
echo ""
echo "🌟 Testing Kroupa IMF imports..."
if python -c "from kroupa_imf import sample_with_kroupa_imf, get_stellar_mass_from_h5; print('✅ Kroupa IMF imports work')"; then
    echo "✅ Kroupa IMF functions accessible"
else
    echo "❌ Kroupa IMF import failed"
fi

echo ""
echo "🎯 Quick test complete!"
EOF

        # Upload and run quick test
        scp /tmp/quick_sherlock_test.sh "$SHERLOCK_USER@$SHERLOCK_HOST:/tmp/"
        ssh "$SHERLOCK_USER@$SHERLOCK_HOST" "bash /tmp/quick_sherlock_test.sh"
        
    elif [[ "$FULL_TEST" == "true" ]]; then
        echo "🔬 Running comprehensive verification..."
        
        # Run the full verification script
        ssh "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && module purge && module load math devel python/3.9.0 && source ~/.bashrc && conda activate bosque && python final_verification.py"
    fi
    
    echo ""
    echo "✅ Testing complete"
fi

echo ""
echo "🎉 DEPLOYMENT SUMMARY"
echo "====================="
if [[ "$UPLOAD" == "true" ]]; then
    echo "✅ Files uploaded to Sherlock"
fi
if [[ "$TEST" == "true" ]]; then
    echo "✅ Tests executed on Sherlock"
fi
echo ""
echo "🔗 To access Sherlock manually:"
echo "   ssh $SHERLOCK_USER@$SHERLOCK_HOST"
echo "   cd $SHERLOCK_PATH"
echo ""
echo "🚀 Ready for production testing on Sherlock!"

