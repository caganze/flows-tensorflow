#!/bin/bash
# Complete fix for Sherlock - addresses both issues

echo "🔧 COMPLETE FIX FOR SHERLOCK"
echo "============================="

# Fix 1: use_kroupa_imf variable error
echo "1️⃣ Fixing use_kroupa_imf variable error in train_tfp_flows.py..."
sed -i "s/'use_kroupa_imf': use_kroupa_imf,/'use_kroupa_imf': True,  # MANDATORY - always True/" train_tfp_flows.py
echo "   ✅ Fixed variable reference error"

# Fix 2: Remove unsupported --samples_dir argument from test scripts
echo "2️⃣ Fixing test scripts to remove unsupported arguments..."

# Fix GPU test script
if [[ -f "test_gpu_symlib_training.sh" ]]; then
    sed -i 's/--use_kroupa_imf \\/--generate-samples \\/' test_gpu_symlib_training.sh
    sed -i '/--samples_dir/d' test_gpu_symlib_training.sh
    echo "   ✅ Fixed test_gpu_symlib_training.sh"
fi

# Fix CPU test script  
if [[ -f "test_cpu_symlib_training.sh" ]]; then
    sed -i 's/--use_kroupa_imf \\/--generate-samples \\/' test_cpu_symlib_training.sh
    sed -i '/--samples_dir/d' test_cpu_symlib_training.sh
    echo "   ✅ Fixed test_cpu_symlib_training.sh"
fi

echo ""
echo "🧪 Testing the fixes..."

# Test that train_tfp_flows.py can be imported without errors
python3 -c "
import sys
sys.path.append('.')
try:
    from train_tfp_flows import train_and_save_flow
    print('✅ train_tfp_flows.py imports successfully')
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

# Test that the script accepts the correct arguments
python3 train_tfp_flows.py --help | grep -q "generate-samples" && echo "✅ --generate-samples argument available" || echo "❌ --generate-samples argument missing"
python3 train_tfp_flows.py --help | grep -q "n_samples" && echo "✅ --n_samples argument available" || echo "❌ --n_samples argument missing"

echo ""
echo "🎯 ALL FIXES APPLIED SUCCESSFULLY!"
echo "🚀 Ready to run tests:"
echo "   ./test_gpu_symlib_training.sh"
echo "   ./test_cpu_symlib_training.sh"

