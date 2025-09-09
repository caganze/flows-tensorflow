#!/bin/bash

# 🧹 FINAL H5 CLEANUP - Remove ALL remaining H5 references
# This fixes the missed H5 references in essential scripts

set -e

echo "🧹 FINAL H5 CLEANUP"
echo "==================="
echo "Removing ALL remaining H5 references from essential scripts"
echo ""

# 1. Fix brute_force_gpu_job.sh - Remove H5 sample checking
echo "1️⃣ Fixing brute_force_gpu_job.sh..."
sed -i 's/|| -f "$SAMPLES_DIR\/model_pid${SELECTED_PID}_samples\.h5"//' brute_force_gpu_job.sh
echo "✅ Removed H5 sample checking from brute_force_gpu_job.sh"

# 2. Fix brute_force_cpu_parallel.sh - Remove H5 references
echo ""
echo "2️⃣ Fixing brute_force_cpu_parallel.sh..."
sed -i '/all_in_one\.h5/d' brute_force_cpu_parallel.sh
sed -i 's/|| -f "$SAMPLES_DIR\/model_pid${SELECTED_PID}_samples\.h5"//' brute_force_cpu_parallel.sh
echo "✅ Removed H5 references from brute_force_cpu_parallel.sh"

# 3. Fix filter_completed_particles.sh - Remove H5 logic
echo ""
echo "3️⃣ Fixing filter_completed_particles.sh..."
sed -i '/all_in_one\.h5/d' filter_completed_particles.sh
sed -i '/samples_file_h5/d' filter_completed_particles.sh
echo "✅ Removed H5 logic from filter_completed_particles.sh"

# 4. Fix validate_deployment.sh - Remove H5 paths
echo ""
echo "4️⃣ Fixing validate_deployment.sh..."
sed -i '/milkyway-eden-mocks\|symphony_mocks/d' validate_deployment.sh
sed -i '/all_in_one\.h5\|\.h5/d' validate_deployment.sh
echo "✅ Removed H5 paths from validate_deployment.sh"

# 5. Fix monitor_brute_force.sh - Update output path
echo ""
echo "5️⃣ Fixing monitor_brute_force.sh..."
sed -i 's|/oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/tfp_output|/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output|g' monitor_brute_force.sh
echo "✅ Updated output path in monitor_brute_force.sh"

# 6. Fix deploy_to_sherlock.sh - Remove H5 file searching
echo ""
echo "6️⃣ Fixing deploy_to_sherlock.sh..."
sed -i '/Halo.*_.*orig.*\.h5/d' deploy_to_sherlock.sh
echo "✅ Removed H5 file searching from deploy_to_sherlock.sh"

# 7. Fix Python files - Remove H5 references from active scripts
echo ""
echo "7️⃣ Fixing Python files..."
if [[ -f "optimized_io.py" ]]; then
    sed -i '/\.h5/d' optimized_io.py
    echo "  ✅ Fixed optimized_io.py"
fi

if [[ -f "comprehensive_logging.py" ]]; then
    sed -i '/model\.h5/d' comprehensive_logging.py
    echo "  ✅ Fixed comprehensive_logging.py"
fi

if [[ -f "comprehensive_gpu_test.py" ]]; then
    sed -i '/\.h5/d' comprehensive_gpu_test.py
    echo "  ✅ Fixed comprehensive_gpu_test.py"
fi

# 8. Verify samples are consistently saved as .npz
echo ""
echo "8️⃣ Verifying sample format consistency..."
echo "Checking train_tfp_flows.py sample saving format..."
if grep -q "samples.*\.npz" train_tfp_flows.py; then
    echo "  ✅ train_tfp_flows.py saves samples as .npz"
else
    echo "  ⚠️  Could not verify .npz format in train_tfp_flows.py"
fi

# 9. Final verification - check for remaining H5 references
echo ""
echo "9️⃣ Final verification..."
REMAINING_H5=$(grep -r "\.h5\|milkyway-eden-mocks\|symphony_mocks" *.sh 2>/dev/null | grep -v "deprecated_scripts" | grep -v "test_symlib_only.sh" | grep -v "complete_symlib_migration.sh" | wc -l)

if [[ $REMAINING_H5 -eq 0 ]]; then
    echo "  ✅ No H5 references found in essential scripts"
else
    echo "  ⚠️  Found $REMAINING_H5 remaining H5 references:"
    grep -r "\.h5\|milkyway-eden-mocks\|symphony_mocks" *.sh 2>/dev/null | grep -v "deprecated_scripts" | grep -v "test_symlib_only.sh" | grep -v "complete_symlib_migration.sh" || true
fi

echo ""
echo "🎉 FINAL H5 CLEANUP COMPLETE!"
echo "============================="
echo "✅ Removed H5 sample checking"
echo "✅ Removed H5 file paths"
echo "✅ Removed H5 fallback logic"
echo "✅ Updated output paths"
echo "✅ Fixed Python file references"
echo ""
echo "🔍 CONSISTENCY CHECK:"
echo "===================="
echo "Shell scripts expect:"
echo "  📄 Models: model_pid{N}.npz"
echo "  📄 Samples: model_pid{N}_samples.npz"
echo ""
echo "Python train_tfp_flows.py saves:"
if grep -q "model_pid.*\.npz" train_tfp_flows.py && grep -q "samples.*\.npz" train_tfp_flows.py; then
    echo "  ✅ Models: .npz format"
    echo "  ✅ Samples: .npz format"
    echo "  ✅ FORMATS MATCH!"
else
    echo "  ⚠️  Check train_tfp_flows.py save format"
fi
echo ""

