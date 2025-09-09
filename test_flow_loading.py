#!/usr/bin/env python3
"""
Test script to verify flow loading and Kroupa sampling works
"""

import sys
sys.path.append('.')
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from train_tfp_flows import load_flow_from_model, generate_samples_separately

def test_flow_loading():
    """Test loading a flow model and generating samples"""
    
    model_path = '/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output/trained_flows/eden/halo939/model_pid3.npz'
    
    try:
        print("🧪 Testing flow loading...")
        
        # Load the flow
        flow = load_flow_from_model(model_path)
        print(f"✅ Flow loaded successfully: {type(flow)}")
        
        # Test basic sampling
        print("🧪 Testing basic flow sampling...")
        test_samples = flow.sample(10)
        print(f"✅ Basic sampling works! Shape: {test_samples.shape}")
        print(f"Sample values: {test_samples.numpy()[:3]}")
        
        # Test Kroupa sampling
        print("🧪 Testing Kroupa sampling...")
        samples_path = generate_samples_separately(
            model_path=model_path,
            n_samples=1000  # Small test
        )
        print(f"✅ Kroupa sampling successful! Samples saved to: {samples_path}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_flow_loading()
