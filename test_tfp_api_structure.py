#!/usr/bin/env python3
"""
Test script to verify TFP API structure and catch FlowJAX legacy issues
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors

def test_tfp_structure():
    """Test the exact API structure of our TFP flow"""
    print("🔍 Testing TFP API Structure...")
    print(f"TF version: {tf.__version__}")
    print(f"TFP version: {tfp.__version__}")
    print()
    
    # Create a simple flow like ours
    input_dim = 6
    
    # 1. Create AutoregressiveNetwork (our component)
    print("1️⃣ Testing AutoregressiveNetwork...")
    try:
        made = tfb.AutoregressiveNetwork(
            params=2,
            hidden_units=[64, 64],
            event_shape=[input_dim],
            dtype=tf.float32,
            name='test_autoregressive'
        )
        print("✅ AutoregressiveNetwork created successfully")
        
        # Check if it has trainable_variables
        if hasattr(made, 'trainable_variables'):
            print(f"✅ AutoregressiveNetwork.trainable_variables exists: {len(made.trainable_variables)} vars")
        else:
            print("❌ AutoregressiveNetwork.trainable_variables does NOT exist")
            
    except Exception as e:
        print(f"❌ AutoregressiveNetwork failed: {e}")
        return False
    
    # 2. Create MaskedAutoregressiveFlow
    print("\n2️⃣ Testing MaskedAutoregressiveFlow...")
    try:
        maf = tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=made,
            name='test_maf'
        )
        print("✅ MaskedAutoregressiveFlow created successfully")
        
        # Check if it has trainable_variables
        if hasattr(maf, 'trainable_variables'):
            print(f"✅ MaskedAutoregressiveFlow.trainable_variables exists: {len(maf.trainable_variables)} vars")
        else:
            print("❌ MaskedAutoregressiveFlow.trainable_variables does NOT exist")
            
    except Exception as e:
        print(f"❌ MaskedAutoregressiveFlow failed: {e}")
        return False
    
    # 3. Create Chain
    print("\n3️⃣ Testing Chain...")
    try:
        chain = tfb.Chain([maf], name='test_chain')
        print("✅ Chain created successfully")
        
        # Check various attributes
        if hasattr(chain, 'trainable_variables'):
            print(f"✅ Chain.trainable_variables exists: {len(chain.trainable_variables)} vars")
        else:
            print("❌ Chain.trainable_variables does NOT exist")
            
        if hasattr(chain, 'bijectors'):
            print(f"✅ Chain.bijectors exists: {len(chain.bijectors)} bijectors")
        else:
            print("❌ Chain.bijectors does NOT exist")
            
    except Exception as e:
        print(f"❌ Chain failed: {e}")
        return False
    
    # 4. Create TransformedDistribution
    print("\n4️⃣ Testing TransformedDistribution...")
    try:
        base_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(input_dim, dtype=tf.float32),
            scale_diag=tf.ones(input_dim, dtype=tf.float32)
        )
        
        flow = tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=chain,
            name='test_flow'
        )
        print("✅ TransformedDistribution created successfully")
        
        # Check various attributes
        if hasattr(flow, 'trainable_variables'):
            print(f"✅ TransformedDistribution.trainable_variables exists: {len(flow.trainable_variables)} vars")
        else:
            print("❌ TransformedDistribution.trainable_variables does NOT exist")
            
        if hasattr(flow, 'bijector'):
            print("✅ TransformedDistribution.bijector exists")
            if hasattr(flow.bijector, 'trainable_variables'):
                print(f"✅ TransformedDistribution.bijector.trainable_variables exists: {len(flow.bijector.trainable_variables)} vars")
            else:
                print("❌ TransformedDistribution.bijector.trainable_variables does NOT exist")
        else:
            print("❌ TransformedDistribution.bijector does NOT exist")
            
    except Exception as e:
        print(f"❌ TransformedDistribution failed: {e}")
        return False
    
    # 5. Test actual operations
    print("\n5️⃣ Testing Operations...")
    try:
        # Test sampling
        samples = flow.sample(10)
        print(f"✅ Sampling works: shape {samples.shape}")
        
        # Test log_prob
        log_probs = flow.log_prob(samples)
        print(f"✅ log_prob works: shape {log_probs.shape}")
        
        # Test gradient computation
        with tf.GradientTape() as tape:
            test_log_probs = flow.log_prob(samples)
            loss = -tf.reduce_mean(test_log_probs)
        
        # Try different ways to get trainable variables
        print("\n🔍 Testing trainable variable access methods:")
        
        methods = [
            ("flow.trainable_variables", lambda: flow.trainable_variables),
            ("flow.bijector.trainable_variables", lambda: flow.bijector.trainable_variables),
            ("made.trainable_variables", lambda: made.trainable_variables),
            ("chain.trainable_variables", lambda: chain.trainable_variables),
        ]
        
        working_method = None
        for name, method in methods:
            try:
                vars = method()
                print(f"✅ {name}: {len(vars)} variables")
                if working_method is None:
                    working_method = (name, vars)
            except AttributeError as e:
                print(f"❌ {name}: AttributeError - {e}")
            except Exception as e:
                print(f"❌ {name}: Other error - {e}")
        
        if working_method:
            print(f"\n🎯 RECOMMENDED METHOD: {working_method[0]}")
            print(f"   Variables: {len(working_method[1])}")
            
            # Test gradient computation
            try:
                gradients = tape.gradient(loss, working_method[1])
                if gradients and any(g is not None for g in gradients):
                    print("✅ Gradient computation works!")
                else:
                    print("❌ Gradients are None - check variable tracking")
            except Exception as e:
                print(f"❌ Gradient computation failed: {e}")
        else:
            print("❌ No working method found for trainable variables!")
            
    except Exception as e:
        print(f"❌ Operations failed: {e}")
        return False
    
    print("\n🏆 API Structure Test Complete!")
    return True

if __name__ == "__main__":
    test_tfp_structure()
