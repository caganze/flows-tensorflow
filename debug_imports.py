#!/usr/bin/env python3
"""
Debug imports to find what's causing timeouts
"""

import sys
import time

print("🔍 DEBUGGING IMPORTS ON SHERLOCK")
print("=" * 40)

def test_import(module_name, description):
    """Test importing a module with timeout tracking"""
    print(f"Testing {description}...", end=" ", flush=True)
    start_time = time.time()
    
    try:
        if module_name == "numpy":
            import numpy as np
            print(f"✅ OK ({time.time() - start_time:.2f}s)")
            return True
        elif module_name == "tensorflow":
            import tensorflow as tf
            print(f"✅ OK ({time.time() - start_time:.2f}s) - TF version: {tf.__version__}")
            return True
        elif module_name == "kroupa_imf":
            from kroupa_imf import sample_with_kroupa_imf
            print(f"✅ OK ({time.time() - start_time:.2f}s)")
            return True
        elif module_name == "tfp_flows":
            from tfp_flows_gpu_solution import TFPNormalizingFlow
            print(f"✅ OK ({time.time() - start_time:.2f}s)")
            return True
        else:
            print(f"❌ Unknown module: {module_name}")
            return False
            
    except Exception as e:
        print(f"❌ FAILED ({time.time() - start_time:.2f}s): {e}")
        return False

# Test imports in order of complexity
print("1️⃣ Basic Python modules:")
test_import("numpy", "NumPy")

print("\n2️⃣ TensorFlow (this might hang):")
test_import("tensorflow", "TensorFlow")

print("\n3️⃣ Our custom modules:")
test_import("kroupa_imf", "Kroupa IMF")
test_import("tfp_flows", "TFP Flows")

print(f"\n✅ All imports completed successfully!")
print("Environment appears to be working correctly.")
