#!/bin/bash
# Fix CUDA paths for TensorFlow JIT compilation
# Run this before your Python scripts on Sherlock

echo "🔧 Setting CUDA paths for TensorFlow..."

# Set the CUDA data directory for XLA
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0

# Verify the path exists
if [ -d "/share/software/user/open/cuda/12.2.0" ]; then
    echo "✅ CUDA 12.2.0 path found: /share/software/user/open/cuda/12.2.0"
else
    echo "⚠️ CUDA path not found, trying alternative..."
    export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME
    echo "✅ Using CUDA_HOME: $CUDA_HOME"
fi

# Also set NVCC path if needed
if [ -f "/share/software/user/open/cuda/12.2.0/bin/nvcc" ]; then
    export CUDA_NVCC_EXECUTABLE=/share/software/user/open/cuda/12.2.0/bin/nvcc
    echo "✅ NVCC executable: $CUDA_NVCC_EXECUTABLE"
fi

echo "✅ CUDA paths configured for TensorFlow"
echo "💡 XLA_FLAGS: $XLA_FLAGS"
echo ""
