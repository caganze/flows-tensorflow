#!/bin/bash

# 🧪 SYMLIB DATA ACCESS TEST
# Comprehensive test script to verify symlib environment and data access
# Must pass before any SLURM job submissions
# Based on symlib paper: https://arxiv.org/pdf/2005.05342

set -e

echo "🧪 SYMLIB DATA ACCESS TEST"
echo "=========================="
echo "📋 Testing symlib environment and data access before SLURM submissions"
echo "📄 Reference: https://arxiv.org/pdf/2005.05342 (symlib: A Python package for analyzing cosmological simulations)"
echo "Started: $(date)"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function for test results
test_result() {
    local test_name="$1"
    local result="$2"
    local details="$3"
    
    if [[ "$result" == "PASS" ]]; then
        echo -e "${GREEN}✅ PASS${NC}: $test_name"
        if [[ -n "$details" ]]; then
            echo "   $details"
        fi
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAIL${NC}: $test_name"
        if [[ -n "$details" ]]; then
            echo "   $details"
        fi
        ((TESTS_FAILED++))
    fi
    echo ""
}

# Test 1: Python environment and symlib import
echo -e "${BLUE}1️⃣ Testing Python environment and symlib import${NC}"
echo "================================================"

# Activate conda environment (if available)
if command -v conda &> /dev/null; then
    echo "🐍 Activating conda environment..."
    source ~/.bashrc
    conda activate bosque 2>/dev/null || echo "⚠️ Warning: Could not activate bosque environment"
fi

# Test basic Python and symlib import
python_test=$(timeout 60 python3 -c "
import sys
print(f'Python version: {sys.version}')

try:
    import symlib
    print(f'Symlib version: {symlib.__version__ if hasattr(symlib, \"__version__\") else \"Available\"}')
    print('✅ Symlib import successful')
    
    # Test basic symlib functionality
    import numpy as np
    import astropy.units as u
    from astropy.cosmology import FlatLambdaCDM
    
    # Test cosmology setup (required for symlib)
    cosmo = FlatLambdaCDM(H0=70, Om0=0.286)
    print(f'✅ Cosmology setup: H0={cosmo.H0}, Om0={cosmo.Om0}')
    
    # Test symlib constants
    print(f'✅ Symlib constants available: DWARF_GALAXY_HALO_MODEL = {symlib.DWARF_GALAXY_HALO_MODEL_NO_UM}')
    
    exit(0)
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Symlib test error: {e}')
    exit(2)
" 2>&1)

if [[ $? -eq 0 ]]; then
    test_result "Python and symlib import" "PASS" "$python_test"
else
    test_result "Python and symlib import" "FAIL" "$python_test"
fi

# Test 2: Symlib simulation directory access
echo -e "${BLUE}2️⃣ Testing symlib simulation directory access${NC}"
echo "=============================================="

# Test priority halos and suites from our configuration
priority_halos=("939" "718" "270" "925")
test_suites=("eden" "symphony")

directory_test_passed=true
directory_details=""

for suite in "${test_suites[@]}"; do
    for halo in "${priority_halos[@]}"; do
        halo_id="Halo${halo}"
        echo "🔍 Testing access to $suite/$halo_id..."
        
        # Test directory access using symlib
        dir_test=$(timeout 30 python3 -c "
import symlib
import os

halo_id = '$halo_id'
suite = '$suite'

try:
    # Test symlib.get_host_directory function
    if suite.lower() == 'eden':
        sim_dir = symlib.get_host_directory('/oak/stanford/orgs/kipac/users/phil1/simulations/ZoomIns/', 'EDEN_MilkyWay_8K', halo_id)
    elif suite.lower() == 'symphony':
        sim_dir = symlib.get_host_directory('/oak/stanford/orgs/kipac/users/phil1/simulations/ZoomIns/', 'SymphonyMilkyWay', halo_id)
    else:
        raise ValueError(f'Unknown suite: {suite}')
    
    print(f'✅ Directory found: {sim_dir}')
    
    # Check if directory exists and is accessible
    if os.path.exists(sim_dir):
        print(f'✅ Directory accessible: {sim_dir}')
        
        # Test symlib.simulation_parameters
        params = symlib.simulation_parameters(sim_dir)
        print(f'✅ Simulation parameters loaded: mp={params[\"mp\"]:.2e}, eps={params[\"eps\"]:.3f}')
        
        # Test if we can initialize Particles object
        part = symlib.Particles(sim_dir)
        print(f'✅ Particles object initialized for {halo_id}')
        
    else:
        print(f'❌ Directory not accessible: {sim_dir}')
        exit(1)
        
    exit(0)
except Exception as e:
    print(f'❌ Error accessing {suite}/{halo_id}: {e}')
    exit(1)
" 2>&1)

        if [[ $? -eq 0 ]]; then
            directory_details="$directory_details\n   ✅ $suite/$halo_id: Accessible"
        else
            directory_details="$directory_details\n   ❌ $suite/$halo_id: $dir_test"
            directory_test_passed=false
        fi
    done
done

if [[ "$directory_test_passed" == "true" ]]; then
    test_result "Symlib simulation directory access" "PASS" "$directory_details"
else
    test_result "Symlib simulation directory access" "FAIL" "$directory_details"
fi

# Test 3: Particle data loading test
echo -e "${BLUE}3️⃣ Testing particle data loading${NC}"
echo "================================="

# Test our symlib_utils.py integration
particle_loading_test=$(timeout 120 python3 -c "
import sys
sys.path.append('.')
from symlib_utils import load_particle_data, validate_symlib_environment

# Validate environment first
if not validate_symlib_environment():
    print('❌ Symlib environment validation failed')
    exit(1)

# Test loading a small particle from priority halo
try:
    halo_id = 'Halo939'
    particle_pid = 1
    suite = 'eden'
    
    print(f'🔍 Testing particle data loading: {halo_id} PID {particle_pid} from {suite}')
    
    data, metadata = load_particle_data(halo_id, particle_pid, suite)
    
    print(f'✅ Data shape: {data.shape}')
    print(f'✅ Data type: {data.dtype}')
    print(f'✅ Metadata keys: {list(metadata.keys())}')
    print(f'✅ Particle count: {metadata[\"n_particles\"]}')
    print(f'✅ Total stellar mass: {metadata[\"stellar_mass\"]:.2e} M☉')
    
    # Validate data ranges
    if data.shape[1] == 7:  # Should be 7D (position + velocity + mass)
        pos_range = metadata['pos_range']
        vel_range = metadata['vel_range']
        print(f'✅ Position range: [{pos_range[0]:.2f}, {pos_range[1]:.2f}] kpc')
        print(f'✅ Velocity range: [{vel_range[0]:.2f}, {vel_range[1]:.2f}] km/s')
    
    # Test if data is reasonable
    if metadata['n_particles'] > 0 and data.shape[0] == metadata['n_particles']:
        print('✅ Data consistency check passed')
        exit(0)
    else:
        print('❌ Data consistency check failed')
        exit(1)
        
except Exception as e:
    print(f'❌ Particle loading error: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1)

if [[ $? -eq 0 ]]; then
    test_result "Particle data loading" "PASS" "$particle_loading_test"
else
    test_result "Particle data loading" "FAIL" "$particle_loading_test"
fi

# Test 4: TensorFlow and TFP compatibility
echo -e "${BLUE}4️⃣ Testing TensorFlow and TFP compatibility${NC}"
echo "==========================================="

tf_test=$(timeout 180 python3 -c "
import os
# Configure TensorFlow for CPU-only (login node safe)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    
    # Force CPU configuration
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    print(f'✅ TensorFlow version: {tf.__version__}')
    print(f'✅ TensorFlow Probability version: {tfp.__version__}')
    
    # Test basic TFP functionality
    tfd = tfp.distributions
    tfb = tfp.bijectors
    
    # Test simple flow creation (like our training script)
    base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros(6), scale_diag=tf.ones(6))
    bijector = tfb.RealNVP(num_masked=3, shift_and_log_scale_fn=tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(6)
    ]))
    flow = tfd.TransformedDistribution(distribution=base_dist, bijector=bijector)
    
    # Test sampling
    samples = flow.sample(100)
    print(f'✅ Flow sampling test: {samples.shape}')
    
    # Test log probability
    log_prob = flow.log_prob(samples[:10])
    print(f'✅ Log probability test: {log_prob.shape}')
    
    print('✅ TensorFlow/TFP basic functionality verified')
    exit(0)
    
except Exception as e:
    print(f'❌ TensorFlow/TFP error: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
" 2>&1)

if [[ $? -eq 0 ]]; then
    test_result "TensorFlow and TFP compatibility" "PASS" "$tf_test"
else
    test_result "TensorFlow and TFP compatibility" "FAIL" "$tf_test"
fi

# Test 5: Output directory access and permissions
echo -e "${BLUE}5️⃣ Testing output directory access and permissions${NC}"
echo "=================================================="

output_test_passed=true
output_details=""

# Test main output directory
output_base="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output"

if [[ -d "$output_base" ]]; then
    output_details="$output_details\n   ✅ Base directory exists: $output_base"
    
    # Test write permissions
    test_file="$output_base/test_write_permissions_$(date +%s).tmp"
    if touch "$test_file" 2>/dev/null && rm "$test_file" 2>/dev/null; then
        output_details="$output_details\n   ✅ Write permissions verified"
        
        # Test subdirectory creation
        for suite in "eden" "symphony"; do
            for halo in "halo939" "halo718"; do
                test_dir="$output_base/trained_flows/$suite/$halo"
                if mkdir -p "$test_dir" 2>/dev/null; then
                    output_details="$output_details\n   ✅ Can create: $test_dir"
                else
                    output_details="$output_details\n   ❌ Cannot create: $test_dir"
                    output_test_passed=false
                fi
                
                test_dir="$output_base/samples/$suite/$halo"
                if mkdir -p "$test_dir" 2>/dev/null; then
                    output_details="$output_details\n   ✅ Can create: $test_dir"
                else
                    output_details="$output_details\n   ❌ Cannot create: $test_dir"
                    output_test_passed=false
                fi
            done
        done
    else
        output_details="$output_details\n   ❌ No write permissions"
        output_test_passed=false
    fi
else
    # Try to create the base directory
    if mkdir -p "$output_base" 2>/dev/null; then
        output_details="$output_details\n   ✅ Created base directory: $output_base"
    else
        output_details="$output_details\n   ❌ Cannot create base directory: $output_base"
        output_test_passed=false
    fi
fi

if [[ "$output_test_passed" == "true" ]]; then
    test_result "Output directory access and permissions" "PASS" "$output_details"
else
    test_result "Output directory access and permissions" "FAIL" "$output_details"
fi

# Test 6: Essential script availability
echo -e "${BLUE}6️⃣ Testing essential script availability${NC}"
echo "========================================"

essential_scripts=(
    "submit_gpu_smart.sh"
    "submit_cpu_smart.sh"  
    "filter_completed_particles.sh"
    "brute_force_gpu_job.sh"
    "brute_force_cpu_parallel.sh"
    "train_tfp_flows.py"
    "symlib_utils.py"
    "optimized_io.py"
    "kroupa_imf.py"
    "generate_all_priority_halos.sh"
)

script_test_passed=true
script_details=""

for script in "${essential_scripts[@]}"; do
    if [[ -f "$script" && -x "$script" ]]; then
        script_details="$script_details\n   ✅ $script: Available and executable"
    elif [[ -f "$script" ]]; then
        script_details="$script_details\n   ⚠️ $script: Available but not executable"
        chmod +x "$script" 2>/dev/null || true
    else
        script_details="$script_details\n   ❌ $script: Missing"
        script_test_passed=false
    fi
done

if [[ "$script_test_passed" == "true" ]]; then
    test_result "Essential script availability" "PASS" "$script_details"
else
    test_result "Essential script availability" "FAIL" "$script_details"
fi

# Test 7: Particle list generation capability
echo -e "${BLUE}7️⃣ Testing particle list generation capability${NC}"
echo "=============================================="

if [[ -f "generate_all_priority_halos.sh" ]]; then
    # Test particle list generation (dry run style)
    list_gen_test=$(timeout 60 ./generate_all_priority_halos.sh 2>&1 | head -20)
    
    if [[ $? -eq 0 && -f "particle_list.txt" ]]; then
        particle_count=$(wc -l < particle_list.txt 2>/dev/null || echo "0")
        if [[ $particle_count -gt 0 ]]; then
            # Validate format of first few lines
            format_test=$(head -3 particle_list.txt | python3 -c "
import sys
valid = True
for i, line in enumerate(sys.stdin):
    line = line.strip()
    if line:
        parts = line.split(',')
        if len(parts) != 5:
            print(f'❌ Line {i+1}: Wrong number of fields ({len(parts)}/5): {line}')
            valid = False
        else:
            print(f'✅ Line {i+1}: Valid format: PID={parts[0]}, HALO={parts[1]}, SUITE={parts[2]}')

if valid:
    exit(0)
else:
    exit(1)
" 2>&1)
            
            if [[ $? -eq 0 ]]; then
                test_result "Particle list generation" "PASS" "Generated $particle_count particles\n$format_test"
            else
                test_result "Particle list generation" "FAIL" "Format validation failed:\n$format_test"
            fi
        else
            test_result "Particle list generation" "FAIL" "Empty particle list generated"
        fi
    else
        test_result "Particle list generation" "FAIL" "Script failed or no output:\n$list_gen_test"
    fi
else
    test_result "Particle list generation" "FAIL" "generate_all_priority_halos.sh not found"
fi

# Final summary
echo ""
echo -e "${BLUE}📊 FINAL TEST SUMMARY${NC}"
echo "===================="
echo -e "${GREEN}✅ Tests passed: $TESTS_PASSED${NC}"
echo -e "${RED}❌ Tests failed: $TESTS_FAILED${NC}"

total_tests=$((TESTS_PASSED + TESTS_FAILED))
if [[ $total_tests -gt 0 ]]; then
    success_rate=$((TESTS_PASSED * 100 / total_tests))
    echo "📈 Success rate: ${success_rate}%"
fi

echo ""
if [[ $TESTS_FAILED -eq 0 ]]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
    echo "✅ Symlib environment is ready for SLURM job submissions"
    echo ""
    echo -e "${BLUE}🚀 READY FOR DEPLOYMENT${NC}"
    echo "You can now safely run:"
    echo "  ./submit_gpu_smart.sh"
    echo "  ./submit_cpu_smart.sh"
    echo ""
    echo "📄 Reference: https://arxiv.org/pdf/2005.05342"
    echo "Finished: $(date)"
    exit 0
else
    echo -e "${RED}💥 TESTS FAILED!${NC}"
    echo "❌ Do NOT submit SLURM jobs until all tests pass"
    echo "🔧 Fix the failed tests before proceeding"
    echo ""
    echo "📄 Reference: https://arxiv.org/pdf/2005.05342"
    echo "Finished: $(date)"
    exit 1
fi


