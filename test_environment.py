#!/usr/bin/env python3
"""Quick environment test for TensorFlow Probability"""

import sys
import os

def test_basic_imports():
    """Test basic Python imports"""
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        
        import tensorflow_probability as tfp
        print(f"✅ TensorFlow Probability {tfp.__version__}")
        
        import h5py
        print(f"✅ h5py {h5py.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability"""
    try:
        import tensorflow as tf
        
        print(f"GPU available: {tf.test.is_gpu_available()}")
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"GPU devices: {gpu_devices}")
            return True
        else:
            print("No GPU devices found")
            return False
            
    except Exception as e:
        print(f"GPU test error: {e}")
        return False

def test_tfp_flows():
    """Test basic TFP flow functionality"""
    try:
        import tensorflow as tf
        import tensorflow_probability as tfp
        
        tfd = tfp.distributions
        tfb = tfp.bijectors
        
        # Create simple flow
        base_dist = tfd.Normal(0., 1.)
        bijector = tfb.Shift(1.0)
        flow = tfd.TransformedDistribution(distribution=base_dist, bijector=bijector)
        
        # Test sampling and log_prob
        samples = flow.sample(10)
        log_probs = flow.log_prob(samples)
        
        print(f"✅ TFP flow test passed")
        print(f"   Samples shape: {samples.shape}")
        print(f"   Log probs shape: {log_probs.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ TFP flow test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Environment Test")
    print("=" * 30)
    
    print("\n📦 Testing imports...")
    imports_ok = test_basic_imports()
    
    print("\n🖥️ Testing GPU...")
    gpu_ok = test_gpu_availability()
    
    print("\n🔄 Testing TFP flows...")
    flows_ok = test_tfp_flows()
    
    print("\n📋 Summary:")
    print(f"Imports: {'✅' if imports_ok else '❌'}")
    print(f"GPU: {'✅' if gpu_ok else '❌'}")
    print(f"Flows: {'✅' if flows_ok else '❌'}")
    
    if imports_ok and flows_ok:
        print("\n🎉 Environment ready for TFP flows!")
        if gpu_ok:
            print("💫 GPU acceleration available")
        else:
            print("⚠️ Using CPU only")
    else:
        print("\n❌ Environment setup incomplete")
        sys.exit(1)
