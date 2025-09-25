#!/usr/bin/env python3
"""
Simple symlib test that uses the working environment setup from GitHub
"""

import os
import sys

# Set CUDA paths for TensorFlow before importing (from your working GitHub repo)
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging

# Configure for CPU-only to avoid GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print("🧪 Simple Symlib Test")
print("====================")

try:
    print("1. Testing basic imports...")
    import numpy as np
    print("   ✅ NumPy imported")
    
    import tensorflow as tf
    print("   ✅ TensorFlow imported")
    
    import tensorflow_probability as tfp
    print("   ✅ TensorFlow Probability imported")
    
    print("2. Testing symlib import...")
    import symlib
    print("   ✅ Symlib imported successfully!")
    
    print("3. Testing symlib functionality...")
    # Test basic symlib functionality
    print(f"   Symlib version: {symlib.__version__}")
    
    print("4. Testing particle data loading...")
    # Test loading particle data for Halo939
    try:
        from symlib_utils import load_particle_data
        print("   ✅ symlib_utils imported")
        
        # Test loading a small particle
        data, metadata = load_particle_data("Halo939", 27, "eden")
        print(f"   ✅ Loaded particle data: {data.shape[0]:,} particles")
        print(f"   Stellar mass: {metadata.get('stellar_mass', 'N/A')}")
        
    except Exception as e:
        print(f"   ❌ symlib_utils failed: {e}")
        sys.exit(1)
    
    print("5. Testing TensorFlow operations...")
    # Test basic TF operations
    x = tf.constant([1.0, 2.0, 3.0])
    y = tf.constant([4.0, 5.0, 6.0])
    z = tf.add(x, y)
    print(f"   ✅ TensorFlow operations: {z.numpy()}")
    
    print("6. Testing TFP operations...")
    # Test TFP operations
    normal = tfp.distributions.Normal(0.0, 1.0)
    samples = normal.sample(5)
    print(f"   ✅ TFP sampling: {samples.numpy()}")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Environment is ready for symlib training")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("🔧 This is the GLIBC/CXXABI compatibility issue")
    print("💡 Need to use compatible environment setup")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

