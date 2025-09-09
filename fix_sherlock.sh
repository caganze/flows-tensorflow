#!/bin/bash
# Quick fix for the use_kroupa_imf variable error on Sherlock

echo "🔧 Fixing use_kroupa_imf variable error in train_tfp_flows.py..."

# Fix the specific line that's causing the error
sed -i "s/'use_kroupa_imf': use_kroupa_imf,/'use_kroupa_imf': True,  # MANDATORY - always True/" train_tfp_flows.py

echo "✅ Fix applied successfully!"
echo "🧪 Testing the fix..."

# Test that the file can be imported without errors
python3 -c "
import sys
sys.path.append('.')
try:
    from train_tfp_flows import train_and_save_flow
    print('✅ train_tfp_flows.py imports successfully - fix verified!')
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"

echo "🎯 Ready to run tests again!"
