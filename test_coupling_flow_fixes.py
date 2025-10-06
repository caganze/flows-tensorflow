#!/usr/bin/env python3
"""
Test script to verify the coupling flow fixes work correctly.
This tests the shape guards and helper methods.
"""

import numpy as np
import tensorflow as tf
from train_coupling_flows_conditional import ConditionalCouplingFlow

def test_coupling_flow_fixes():
    """Test the comprehensive fixes for shape handling"""
    print("🧪 Testing ConditionalCouplingFlow fixes...")
    
    # Create a flow instance (use same args as training)
    flow = ConditionalCouplingFlow(
        input_dim=6, 
        n_mass_bins=2,  # Use 2 bins to match our filtered data
        n_layers=2, 
        hidden_units=(32, 32),  # Should be a sequence
        embedding_dim=4
    )
    
    print(f"✅ Flow created successfully")
    
    # Test 1: batched input
    print("\n1️⃣ Testing batched input...")
    x = np.random.randn(5, 6).astype(np.float32)
    c = np.random.randint(0, 2, size=(5, 1)).astype(np.int32)
    try:
        log_prob_shape = flow.log_prob(x, c).shape
        print(f"✅ Batched log_prob shape: {log_prob_shape} (expected: (5,))")
        assert log_prob_shape == (5,), f"Expected (5,), got {log_prob_shape}"
    except Exception as e:
        print(f"❌ Batched test failed: {e}")
        return False
    
    # Test 2: single-sample input (rank-1 x, scalar condition)
    print("\n2️⃣ Testing single-sample input...")
    x1 = x[0]         # shape (6,)
    c1 = int(c[0,0])  # scalar
    try:
        log_prob_shape = flow.log_prob(x1, c1).shape
        print(f"✅ Single-sample log_prob shape: {log_prob_shape} (expected: (1,))")
        assert log_prob_shape == (1,), f"Expected (1,), got {log_prob_shape}"
    except Exception as e:
        print(f"❌ Single-sample test failed: {e}")
        return False
    
    # Test 3: sample() sanity check
    print("\n3️⃣ Testing sample() method...")
    try:
        # pass conditions shaped (n_samples,1)
        conditions = np.random.randint(0, 2, size=(10,1)).astype(np.int32)
        samples = flow.sample(10, conditions=conditions)
        print(f"✅ Samples shape: {samples.shape} (expected: (10,6))")
        assert samples.shape == (10, 6), f"Expected (10, 6), got {samples.shape}"
    except Exception as e:
        print(f"❌ Sample test failed: {e}")
        return False
    
    # Test 4: Test batch size mismatch detection
    print("\n4️⃣ Testing batch size mismatch detection...")
    try:
        x_mismatch = np.random.randn(3, 6).astype(np.float32)
        c_mismatch = np.random.randint(0, 2, size=(5, 1)).astype(np.int32)
        flow.log_prob(x_mismatch, c_mismatch)
        print("❌ Batch size mismatch should have failed!")
        return False
    except Exception as e:
        if "Batch size mismatch" in str(e):
            print("✅ Batch size mismatch correctly detected")
        else:
            print(f"❌ Unexpected error: {e}")
            return False
    
    print("\n🎉 All tests passed! The coupling flow fixes are working correctly.")
    return True

if __name__ == "__main__":
    # Set up TensorFlow to avoid warnings
    tf.get_logger().setLevel('ERROR')
    
    success = test_coupling_flow_fixes()
    if success:
        print("\n✅ Test script completed successfully!")
    else:
        print("\n❌ Test script failed!")
        exit(1)








