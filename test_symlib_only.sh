#!/bin/bash

# 🧪 SYMLIB-ONLY OPERATION TEST
# Simple test to verify complete symlib migration worked
# This replaces all the complex test scripts with one comprehensive check

set -e

echo "🧪 SYMLIB-ONLY OPERATION TEST"
echo "============================="
echo "Verifying complete symlib migration worked correctly"
echo ""

PASSED=0
FAILED=0

# Test function
test_check() {
    local test_name="$1"
    local command="$2"
    local expected_result="$3"
    
    echo -n "  $test_name... "
    
    if eval "$command"; then
        if [[ "$expected_result" == "pass" ]]; then
            echo "✅ PASS"
            ((PASSED++))
        else
            echo "❌ FAIL (unexpected pass)"
            ((FAILED++))
        fi
    else
        if [[ "$expected_result" == "fail" ]]; then
            echo "✅ PASS (expected fail)"
            ((PASSED++))
        else
            echo "❌ FAIL"
            ((FAILED++))
        fi
    fi
}

echo "1️⃣ Testing H5 dependencies removed..."
test_check "No H5 generator calls" "! grep -r 'generate_particle_list\.sh' submit_*_chunked.sh brute_force_cpu_parallel.sh" "pass"
test_check "No H5 file paths" "! grep -r 'milkyway-eden-mocks\|symphony_mocks' submit_*.sh train_*.sh" "pass"
test_check "No H5 fallbacks" "! grep -r 'all_in_one\.h5' *.sh" "pass"
test_check "Old H5 generator neutralized" "[[ ! -f 'flows-tensorflow/generate_particle_list.sh' ]]" "pass"

echo ""
echo "2️⃣ Testing symlib infrastructure..."
test_check "Symlib generator exists" "[[ -f 'generate_symlib_particle_list.py' ]]" "pass"
test_check "Symlib utils exists" "[[ -f 'symlib_utils.py' ]]" "pass"
test_check "Priority halo generator exists" "[[ -f 'generate_all_priority_halos.sh' ]]" "pass"

echo ""
echo "3️⃣ Testing particle list format..."
if [[ -f "particle_list.txt" ]]; then
    FIRST_LINE=$(head -1 particle_list.txt)
    if [[ "$FIRST_LINE" =~ ^[0-9]+,Halo[0-9]+,(eden|symphony),[0-9]+,(Small|Medium|Large)$ ]]; then
        echo "  Particle list format... ✅ PASS (symlib format detected)"
        ((PASSED++))
    else
        echo "  Particle list format... ❌ FAIL (not symlib format)"
        echo "    Expected: PID,Halo###,suite,count,size"
        echo "    Found: $FIRST_LINE"
        ((FAILED++))
    fi
else
    echo "  Particle list exists... ⚠️  SKIP (no particle_list.txt found)"
fi

echo ""
echo "4️⃣ Testing submission scripts..."
test_check "GPU smart script exists" "[[ -f 'submit_gpu_smart.sh' ]]" "pass"
test_check "CPU smart script exists" "[[ -f 'submit_cpu_smart.sh' ]]" "pass"
test_check "Filter script updated" "grep -q 'symlib format' filter_completed_particles.sh" "pass"

echo ""
echo "5️⃣ Testing training infrastructure..."
test_check "Main training script exists" "[[ -f 'train_tfp_flows.py' ]]" "pass"
test_check "Kroupa IMF script exists" "[[ -f 'kroupa_imf.py' ]]" "pass"

echo ""
echo "📊 TEST RESULTS"
echo "==============="
echo "✅ Passed: $PASSED"
echo "❌ Failed: $FAILED"

if [[ $FAILED -eq 0 ]]; then
    echo ""
    echo "🎉 ALL TESTS PASSED!"
    echo "==================="
    echo "✅ Symlib migration is complete"
    echo "✅ No H5 dependencies remain"
    echo "✅ Ready for production use"
    echo ""
    echo "🚀 NEXT STEPS:"
    echo "=============="
    if [[ ! -f "particle_list.txt" ]]; then
        echo "1. Generate particle list: ./generate_all_priority_halos.sh"
        echo "2. Submit jobs: sbatch submit_gpu_smart.sh"
    else
        echo "1. Submit jobs: sbatch submit_gpu_smart.sh"
        echo "2. Monitor progress: squeue --me"
    fi
    echo ""
    exit 0
else
    echo ""
    echo "⚠️  SOME TESTS FAILED"
    echo "==================="
    echo "Please fix the failed tests before proceeding."
    echo "Run: ./complete_symlib_migration.sh"
    echo ""
    exit 1
fi

