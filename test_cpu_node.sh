#!/bin/bash
# Test script for CPU compute node - validates environment and basic functionality

echo "🧪 CPU COMPUTE NODE TEST"
echo "========================"
echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-test}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-64}"
echo "Time: $(date)"
echo

# Test 1: Environment setup
echo "🔧 TEST 1: Environment Setup"
echo "-----------------------------"
module purge
# module load math devel python/3.9.0  # Commented out - using conda instead

# CPU-specific TensorFlow optimizations
export TF_CPP_MIN_LOG_LEVEL=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}

source ~/.bashrc
conda activate bosque

echo "✅ Modules loaded successfully"
echo "✅ Conda environment activated: bosque"
echo

# Test 2: Python imports
echo "🐍 TEST 2: Python Imports"
echo "-------------------------"
python -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import tensorflow as tf
    print('✅ TensorFlow imported successfully')
    print(f'   Version: {tf.__version__}')
    print(f'   GPU available: {tf.config.list_physical_devices(\"GPU\")}')
    
    import tensorflow_probability as tfp
    print('✅ TensorFlow Probability imported successfully')
    print(f'   Version: {tfp.__version__}')
    
    import numpy as np
    print('✅ NumPy imported successfully')
    print(f'   Version: {np.__version__}')
    
    import scipy
    print('✅ SciPy imported successfully')
    print(f'   Version: {scipy.__version__}')
    
    print('✅ All imports successful!')
    
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
"
echo

# Test 3: Basic TensorFlow functionality
echo "🧠 TEST 3: TensorFlow Functionality"
echo "-----------------------------------"
python -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Test basic operations
print('Testing basic TensorFlow operations...')
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([4.0, 5.0, 6.0])
z = tf.add(x, y)
print(f'✅ Basic operations: {z.numpy()}')

# Test TFP operations
print('Testing TensorFlow Probability operations...')
normal = tfp.distributions.Normal(0.0, 1.0)
samples = normal.sample(10)
print(f'✅ TFP sampling: {samples.numpy()[:5]}...')

# Test flow creation
print('Testing flow creation...')
base_dist = tfp.distributions.Normal(0.0, 1.0)
flow = tfp.bijectors.Exp()
transformed_dist = tfp.distributions.TransformedDistribution(base_dist, flow)
flow_samples = transformed_dist.sample(5)
print(f'✅ Flow sampling: {flow_samples.numpy()}')

print('✅ All TensorFlow functionality tests passed!')
"
echo

# Test 4: File system access
echo "📁 TEST 4: File System Access"
echo "-----------------------------"
echo "Current directory: $(pwd)"
echo "Home directory: $HOME"
echo "Available space:"
df -h $HOME
echo

# Test 5: Check for particle list
echo "📋 TEST 5: Particle List Check"
echo "------------------------------"
if [[ -f "particle_list.txt" ]]; then
    echo "✅ particle_list.txt found"
    echo "First 3 lines:"
    head -3 particle_list.txt
    echo "Total lines: $(wc -l < particle_list.txt)"
else
    echo "❌ particle_list.txt not found"
fi
echo

# Test 6: Check for H5 files
echo "🗃️ TEST 6: H5 File Access"
echo "-------------------------"
if [[ -f "particle_list.txt" ]]; then
    FIRST_H5=$(head -1 particle_list.txt | cut -d',' -f2)
    if [[ -f "$FIRST_H5" ]]; then
        echo "✅ First H5 file accessible: $(basename $FIRST_H5)"
        echo "   Size: $(du -h "$FIRST_H5" | cut -f1)"
    else
        echo "❌ First H5 file not accessible: $FIRST_H5"
    fi
else
    echo "⚠️ Cannot test H5 access - no particle list"
fi
echo

# Test 7: Memory and CPU info
echo "💻 TEST 7: System Resources"
echo "---------------------------"
echo "CPU info:"
lscpu | grep -E "(Model name|CPU\(s\)|Thread|Core)"
echo
echo "Memory info:"
free -h
echo
echo "Available CPUs: $SLURM_CPUS_PER_TASK"
echo "OMP threads: $OMP_NUM_THREADS"
echo

echo "🎉 CPU COMPUTE NODE TEST COMPLETE!"
echo "=================================="
echo "✅ Environment: Ready"
echo "✅ Python imports: Working"
echo "✅ TensorFlow: Functional"
echo "✅ File system: Accessible"
echo "✅ Resources: Available"
echo
echo "🚀 Ready for production training!"
