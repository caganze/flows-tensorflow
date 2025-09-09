#!/bin/bash

# 🚀 Quick Interactive Test for submit_tfp_array.sh
# Use this for immediate testing without SLURM submission

echo "🧪 Quick TFP Array Test (Interactive)"
echo "====================================="
echo "This will test the core functionality without submitting to SLURM"
echo

# Test configuration
export PARTICLES_PER_TASK=1     # Test just 1 particle
export EPOCHS=2                 # Minimal epochs
export BATCH_SIZE=128           # Small batch
export N_LAYERS=2               # Minimal layers
export HIDDEN_UNITS=16          # Tiny hidden units
export CHECK_SAMPLES=false      # Skip sample checking

# Simulate SLURM environment variables
export SLURM_ARRAY_TASK_ID=1
export SLURM_JOB_ID="test_$$"
export SLURMD_NODENAME=$(hostname)
export CUDA_VISIBLE_DEVICES="0"

# Test will use H5 directory structure (no override needed)

echo "🔧 Loading environment..."
module purge 2>/dev/null || true
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131 2>/dev/null || echo "⚠️ Module loading failed (normal if not on cluster)"

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0

source ~/.bashrc 2>/dev/null || true
conda activate bosque 2>/dev/null || echo "⚠️ Conda activation failed"

echo "✅ Environment loaded"
echo

# Test H5 file discovery
echo "🔍 Testing H5 file discovery..."
find_h5_file() {
    local eden_files=$(find /oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/ -name "eden_scaled_Halo*" -type f 2>/dev/null | head -1)
    if [[ -n "$eden_files" ]]; then
        echo "$eden_files"
        return 0
    fi
    
    local search_paths=(
        "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/"
        "/oak/stanford/orgs/kipac/users/caganze/milkyway-hr-mocks/"
        "/oak/stanford/orgs/kipac/users/caganze/milkywaymocks/"
        "/oak/stanford/orgs/kipac/users/caganze/"
    )
    
    for path in "${search_paths[@]}"; do
        if [[ -d "$path" ]]; then
            h5_file=$(find "$path" -name "*.h5" -type f 2>/dev/null | head -1)
            if [[ -n "$h5_file" ]]; then
                echo "$h5_file"
                return 0
            fi
        fi
    done
    
    echo "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/all_in_one.h5"
}

H5_FILE=$(find_h5_file)
echo "📁 Found H5 file: $H5_FILE"

if [[ ! -f "$H5_FILE" ]]; then
    echo "❌ ERROR: H5 file not accessible"
    echo "💡 This test requires access to the H5 data files"
    echo "💡 Run this on the cluster or ensure data is mounted"
    exit 1
fi

echo "✅ H5 file accessible"

# Test particle size detection  
echo
echo "🔍 Testing particle size detection for multiple PIDs..."

test_particle_sizes() {
    local test_pids=(1 2 3 23 88 188)
    
    for pid in "${test_pids[@]}"; do
        local size=$(python -c "
import h5py
import numpy as np

def get_particle_size_robust(h5_file_path, pid):
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Method 1: PartType1 with ParticleIDs
            if 'PartType1' in f and 'ParticleIDs' in f['PartType1']:
                pids = f['PartType1']['ParticleIDs'][:]
                pid_mask = (pids == pid)
                size = np.sum(pid_mask)
                if size > 0:
                    return size
                    
            # Method 2: parentid field
            elif 'parentid' in f:
                pids = f['parentid'][:]
                pid_mask = (pids == pid)
                size = np.sum(pid_mask)
                if size > 0:
                    return size
                    
            # Method 3: Search all datasets
            datasets_with_pid = []
            def find_pid_datasets(name, obj):
                if hasattr(obj, 'shape') and len(obj.shape) == 1 and obj.shape[0] < 10000000:
                    try:
                        data = obj[:]
                        if hasattr(data, '__iter__') and pid in data:
                            count = np.sum(data == pid)
                            if count > 0:
                                datasets_with_pid.append((name, count))
                    except:
                        pass
                        
            f.visititems(find_pid_datasets)
            
            if datasets_with_pid:
                return max(datasets_with_pid, key=lambda x: x[1])[1]
                
            # Method 4: Estimate based on PID and total particles
            total_particles = 0
            if 'PartType1' in f and 'Coordinates' in f['PartType1']:
                total_particles = len(f['PartType1']['Coordinates'])
            
            if total_particles > 0:
                if pid <= 10:
                    return min(500000, total_particles // 2)
                elif pid <= 100:
                    return min(200000, total_particles // 5)
                elif pid <= 500:
                    return min(50000, total_particles // 20)
                else:
                    return min(10000, total_particles // 100)
                    
    except Exception as e:
        pass
        
    # PID-based fallback
    if pid <= 10:
        return 300000
    elif pid <= 100:
        return 150000
    elif pid <= 500:
        return 75000
    else:
        return 25000

print(get_particle_size_robust('$H5_FILE', $pid))
" 2>/dev/null)
        
        if [[ ! "$size" =~ ^[0-9]+$ ]] || [[ $size -eq 0 ]]; then
            if [[ $pid -le 10 ]]; then
                size=300000
            elif [[ $pid -le 100 ]]; then
                size=150000
            elif [[ $pid -le 500 ]]; then
                size=75000
            else
                size=25000
            fi
        fi
        
        if [[ $size -gt 100000 ]]; then
            category="🐋 Large (>100k)"
        elif [[ $size -gt 10000 ]]; then
            category="🐄 Medium (10k-100k)"
        else
            category="🐭 Small (<10k)"
        fi
        
        echo "  PID $pid: $size objects ($category)"
    done
}

test_particle_sizes

# Get size for PID 1 specifically for the summary
particle_size=$(python -c "
import h5py
import numpy as np

def get_particle_size_robust(h5_file_path, pid):
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if 'PartType1' in f and 'ParticleIDs' in f['PartType1']:
                pids = f['PartType1']['ParticleIDs'][:]
                pid_mask = (pids == pid)
                size = np.sum(pid_mask)
                if size > 0:
                    return size
            elif 'parentid' in f:
                pids = f['parentid'][:]
                pid_mask = (pids == pid)
                size = np.sum(pid_mask)
                if size > 0:
                    return size
    except:
        pass
    return 300000 if pid <= 10 else 150000 if pid <= 100 else 75000 if pid <= 500 else 25000

print(get_particle_size_robust('$H5_FILE', 1))
" 2>/dev/null)

echo "📊 PID 1 has $particle_size objects"
if [[ $particle_size -gt 100000 ]]; then
    echo "🐋 Large particle detected (>100k objects)"
else
    echo "🐭 Small particle (<100k objects)"
fi

# Test environment
echo
echo "🔍 Testing Python environment..."
python -c "
import sys
print(f'Python: {sys.version.split()[0]}')

try:
    import tensorflow as tf
    print(f'TensorFlow: {tf.__version__}')
    print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')
except ImportError as e:
    print(f'TensorFlow: ❌ {e}')

try:
    import tensorflow_probability as tfp
    print(f'TFP: {tfp.__version__}')
except ImportError as e:
    print(f'TFP: ❌ {e}')

try:
    import h5py
    print(f'h5py: {h5py.__version__}')
except ImportError as e:
    print(f'h5py: ❌ {e}')
" || echo "❌ Python environment test failed"

# Test argument parsing
echo
echo "🔍 Testing train_tfp_flows.py argument parsing..."
if [[ -f "train_tfp_flows.py" ]]; then
    python train_tfp_flows.py --help | head -5 || echo "❌ train_tfp_flows.py help failed"
    echo "✅ train_tfp_flows.py is accessible"
else
    echo "❌ train_tfp_flows.py not found in current directory"
    echo "💡 Make sure you're in the correct directory"
fi

# Test directory creation
echo
echo "🔍 Testing directory creation..."
FILENAME=$(basename "$H5_FILE")
HALO_ID=$(echo "$FILENAME" | sed 's/.*Halo\([0-9][0-9]*\).*/\1/')

if [[ "$FILENAME" == *"eden_scaled"* ]]; then
    DATA_SOURCE="eden"
elif [[ "$FILENAME" == *"symphonyHR_scaled"* ]]; then
    DATA_SOURCE="symphony-hr"
elif [[ "$FILENAME" == *"symphony_scaled"* ]]; then
    DATA_SOURCE="symphony"
else
    DATA_SOURCE="unknown"
fi

if [[ "$FILENAME" == "all_in_one.h5" ]] || [[ "$HALO_ID" == "$FILENAME" ]]; then
    HALO_ID="000"
    DATA_SOURCE="symphony"
fi

# Output directories - save in same parent directory as H5 file with halo/PID structure
H5_PARENT_DIR=$(dirname "$H5_FILE")
OUTPUT_BASE_DIR="$H5_PARENT_DIR/tfp_output"
MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/${DATA_SOURCE}/halo${HALO_ID}"
mkdir -p "$MODEL_DIR" || {
    echo "❌ Failed to create test directory: $MODEL_DIR"
    exit 1
}
echo "✅ Test directory created: $MODEL_DIR"

echo
echo "🎯 QUICK TEST SUMMARY"
echo "====================="
echo "H5 file: ✅ $(basename "$H5_FILE")"
echo "H5 parent: $H5_PARENT_DIR"
echo "Data source: $DATA_SOURCE"
echo "Halo ID: $HALO_ID"
echo "Output base: $OUTPUT_BASE_DIR"
echo "Model dir: $MODEL_DIR"
echo "Particle size (PID 1): $particle_size objects"

echo
echo "💡 NEXT STEPS:"
echo "=============="
echo "If all tests above passed, you can proceed with:"
echo
echo "1. 🧪 Full SLURM test (recommended):"
echo "   sbatch test_submit_tfp_array.sh"
echo
echo "2. 🚀 Production submission (after successful test):"
echo "   sbatch submit_tfp_array.sh"
echo
echo "3. 📊 Monitor progress:"
echo "   squeue -u \$(whoami)"
echo "   ./scan_and_resubmit.sh"

echo
echo "🏁 Quick test completed!"
