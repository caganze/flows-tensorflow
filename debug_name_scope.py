#!/usr/bin/env python3
"""
Debug script to isolate the name scope issue
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

def debug_name_scope_issue():
    """Debug the name scope issue step by step"""
    print("🔍 Debugging name scope issue...")
    
    # Test 1: Simple RealNVP without custom shift_and_log_scale_fn
    print("\n1️⃣ Testing simple RealNVP...")
    try:
        # Create a simple shift and log scale function
        def simple_shift_and_log_scale_fn(x, conditional_input=None):
            return tf.zeros_like(x), tf.zeros_like(x)
        
        simple_bijector = tfb.RealNVP(
            num_masked=3,
            shift_and_log_scale_fn=simple_shift_and_log_scale_fn,
            name="simple_real_nvp"
        )
        print("✅ Simple RealNVP created successfully")
        
        # Test it
        x_test = tf.random.normal((5, 6))
        y = simple_bijector.forward(x_test)
        print(f"✅ Simple RealNVP forward pass successful, shape: {y.shape}")
        
    except Exception as e:
        print(f"❌ Simple RealNVP failed: {e}")
        return False
    
    # Test 2: Test with a Keras layer as shift_and_log_scale_fn
    print("\n2️⃣ Testing RealNVP with Keras layer...")
    try:
        class SimpleNet(tf.keras.layers.Layer):
            def __init__(self, name=None):
                super().__init__(name=name)
                self.dense = tf.keras.layers.Dense(3 * 2)  # 3 for shift + 3 for log_scale
            
            def call(self, x, conditional_input=None):
                output = self.dense(x)
                shift, log_scale = tf.split(output, num_or_size_splits=2, axis=-1)
                return shift, log_scale
        
        simple_net = SimpleNet(name="simple_net")
        bijector_with_net = tfb.RealNVP(
            num_masked=3,
            shift_and_log_scale_fn=simple_net,
            name="real_nvp_with_net"
        )
        print("✅ RealNVP with Keras layer created successfully")
        
        # Test it
        x_test = tf.random.normal((5, 6))
        y = bijector_with_net.forward(x_test)
        print(f"✅ RealNVP with Keras layer forward pass successful, shape: {y.shape}")
        
    except Exception as e:
        print(f"❌ RealNVP with Keras layer failed: {e}")
        return False
    
    # Test 3: Test the actual ConditionalCouplingNet
    print("\n3️⃣ Testing ConditionalCouplingNet...")
    try:
        from train_coupling_flows_conditional import ConditionalCouplingNet
        
        coupling_net = ConditionalCouplingNet(
            input_dim=3,
            hidden_units=(32, 32),
            n_mass_bins=2,
            embedding_dim=4,
            name="test_coupling_net"
        )
        print("✅ ConditionalCouplingNet created successfully")
        
        # Test it
        x_test = tf.random.normal((5, 3))
        c_test = tf.constant([[0], [1], [0], [1], [0]], dtype=tf.int32)
        shift, log_scale = coupling_net(x_test, c_test)
        print(f"✅ ConditionalCouplingNet call successful, shapes: {shift.shape}, {log_scale.shape}")
        
    except Exception as e:
        print(f"❌ ConditionalCouplingNet failed: {e}")
        return False
    
    # Test 4: Test RealNVP with ConditionalCouplingNet
    print("\n4️⃣ Testing RealNVP with ConditionalCouplingNet...")
    try:
        coupling_net = ConditionalCouplingNet(
            input_dim=3,
            hidden_units=(32, 32),
            n_mass_bins=2,
            embedding_dim=4,
            name="test_coupling_net_2"
        )
        
        bijector_with_coupling = tfb.RealNVP(
            num_masked=3,
            shift_and_log_scale_fn=coupling_net,
            name="real_nvp_with_coupling"
        )
        print("✅ RealNVP with ConditionalCouplingNet created successfully")
        
        # Test it
        x_test = tf.random.normal((5, 6))
        c_test = tf.constant([[0], [1], [0], [1], [0]], dtype=tf.int32)
        
        # We need to create a function that passes the condition
        def shift_and_log_scale_fn(x, conditional_input=None):
            return coupling_net(x, c_test)
        
        bijector_with_coupling = tfb.RealNVP(
            num_masked=3,
            shift_and_log_scale_fn=shift_and_log_scale_fn,
            name="real_nvp_with_coupling_final"
        )
        
        y = bijector_with_coupling.forward(x_test)
        print(f"✅ RealNVP with ConditionalCouplingNet forward pass successful, shape: {y.shape}")
        
    except Exception as e:
        print(f"❌ RealNVP with ConditionalCouplingNet failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The issue might be elsewhere.")
    return True

if __name__ == "__main__":
    # Set up TensorFlow to avoid warnings
    tf.get_logger().setLevel('ERROR')
    
    success = debug_name_scope_issue()
    if success:
        print("\n✅ Debug script completed successfully!")
    else:
        print("\n❌ Debug script failed!")
        exit(1)
