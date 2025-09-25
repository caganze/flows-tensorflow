#!/usr/bin/env python3
"""
Debug Kroupa IMF NaN sampling issue
"""

import numpy as np
import tensorflow as tf
import sys
from pathlib import Path

# Import our modules
from tfp_flows_gpu_solution import TFPNormalizingFlow
from kroupa_imf import sample_with_kroupa_imf

def test_flow_sampling():
    """Test if flow sampling produces NaN values"""
    
    print("🔍 DEBUGGING FLOW SAMPLING")
    print("=" * 50)
    
    # Create a simple test flow
    print("1️⃣ Creating test flow...")
    flow = TFPNormalizingFlow(input_dim=6, n_layers=2, hidden_units=32)
    print("✅ Test flow created")
    
    # Test basic sampling
    print("\n2️⃣ Testing basic flow sampling...")
    try:
        samples = flow.sample(100, seed=42)
        print(f"✅ Basic sampling successful: shape {samples.shape}")
        
        # Check for NaN/Inf
        has_nan = tf.reduce_any(tf.math.is_nan(samples))
        has_inf = tf.reduce_any(tf.math.is_inf(samples))
        
        print(f"   Has NaN: {has_nan.numpy()}")
        print(f"   Has Inf: {has_inf.numpy()}")
        print(f"   Sample range: [{tf.reduce_min(samples).numpy():.3f}, {tf.reduce_max(samples).numpy():.3f}]")
        
        if has_nan.numpy() or has_inf.numpy():
            print("❌ UNTRAINED FLOW PRODUCES NaN/Inf!")
            return False
        else:
            print("✅ Untrained flow produces valid samples")
            
    except Exception as e:
        print(f"❌ Basic sampling failed: {e}")
        return False
    
    # Test with preprocessing stats
    print("\n3️⃣ Testing with preprocessing stats...")
    try:
        # Create fake preprocessing stats
        preprocessing_stats = {
            'mean': tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32),
            'std': tf.constant([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=tf.float32)
        }
        
        # Test Kroupa sampling
        samples, masses = sample_with_kroupa_imf(
            flow=flow,
            n_target_mass=1e8,  # 100 million solar masses
            preprocessing_stats=preprocessing_stats,
            seed=42,
            max_samples=1000  # Small test
        )
        
        print(f"✅ Kroupa sampling successful: {len(samples)} samples")
        print(f"   Sample shape: {samples.shape}")
        print(f"   Masses shape: {masses.shape}")
        
        # Check for NaN
        samples_has_nan = tf.reduce_any(tf.math.is_nan(samples))
        masses_has_nan = np.any(np.isnan(masses))
        
        print(f"   Samples have NaN: {samples_has_nan.numpy()}")
        print(f"   Masses have NaN: {masses_has_nan}")
        
        if samples_has_nan.numpy() or masses_has_nan:
            print("❌ KROUPA SAMPLING PRODUCES NaN!")
            return False
        else:
            print("✅ Kroupa sampling produces valid results")
            return True
            
    except Exception as e:
        print(f"❌ Kroupa sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_flow_sampling()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - Flow sampling works correctly")
        print("   The issue may be with specific trained models or preprocessing stats")
    else:
        print("\n❌ TESTS FAILED - Flow sampling has issues")
        print("   This indicates a fundamental problem with the flow implementation")
    
    sys.exit(0 if success else 1)

