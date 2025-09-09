#!/bin/bash
# Reinstall TensorFlow using the working configuration from GitHub repository

echo "🔧 REINSTALLING TENSORFLOW"
echo "=========================="
echo "Using working configuration from GitHub repository"
echo "Time: $(date)"
echo

# Environment setup - using working configuration from GitHub
module purge
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131

# Set CUDA environment for TensorFlow (from your working GitHub repo)
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0
export CUDA_VISIBLE_DEVICES=0

source ~/.bashrc
conda activate bosque

echo "✅ Environment loaded"
echo "🔧 Reinstalling TensorFlow and dependencies..."

# Uninstall existing TensorFlow packages
echo "1. Uninstalling existing packages..."
pip uninstall -y tensorflow tensorflow-probability tensorflow-gpu

# Reinstall TensorFlow with specific versions (from your working setup)
echo "2. Installing TensorFlow 2.15.0..."
pip install tensorflow==2.15.0

echo "3. Installing TensorFlow Probability 0.23.0..."
pip install tensorflow-probability==0.23.0

echo "4. Installing other dependencies..."
pip install h5py numpy scipy matplotlib tqdm

echo "5. Verifying installation..."
python -c "
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import h5py

print('✅ TensorFlow:', tf.__version__)
print('✅ TensorFlow Probability:', tfp.__version__)
print('✅ NumPy:', np.__version__)
print('✅ h5py:', h5py.__version__)

# Test GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'✅ GPUs detected: {len(gpus)}')
for i, gpu in enumerate(gpus):
    print(f'  GPU {i}: {gpu}')

# Test basic operations
x = tf.constant([1.0, 2.0, 3.0])
y = tf.constant([4.0, 5.0, 6.0])
z = tf.add(x, y)
print(f'✅ Basic operations: {z.numpy()}')

# Test TFP operations
normal = tfp.distributions.Normal(0.0, 1.0)
samples = normal.sample(5)
print(f'✅ TFP sampling: {samples.numpy()}')

print('🎉 TensorFlow reinstallation successful!')
"

echo "✅ TensorFlow reinstallation complete!"
echo "🚀 Ready to test symlib training"
