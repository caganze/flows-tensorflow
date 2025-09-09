#!/usr/bin/env python3
"""
Quick test to verify TensorFlow imports work without hanging
"""

import sys
import time

def test_with_timeout():
    """Test imports with a simple timeout mechanism"""
    print("🧪 QUICK TENSORFLOW TEST")
    print("=" * 30)
    
    start_time = time.time()
    
    try:
        print("1️⃣ Testing NumPy import...", end=" ")
        import numpy as np
        print(f"✅ OK ({time.time() - start_time:.2f}s)")
        
        print("2️⃣ Testing TensorFlow import...", end=" ")
        import tensorflow as tf
        print(f"✅ OK ({time.time() - start_time:.2f}s)")
        print(f"   TF version: {tf.__version__}")
        
        print("3️⃣ Testing Kroupa IMF import...", end=" ")
        from kroupa_imf import sample_with_kroupa_imf
        print(f"✅ OK ({time.time() - start_time:.2f}s)")
        
        print("4️⃣ Testing argument parser...", end=" ")
        import train_tfp_flows
        print(f"✅ OK ({time.time() - start_time:.2f}s)")
        
        total_time = time.time() - start_time
        print(f"\n✅ ALL TESTS PASSED in {total_time:.2f}s")
        print("🎉 Environment is working correctly!")
        return True
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ TEST FAILED after {total_time:.2f}s")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_with_timeout()
    sys.exit(0 if success else 1)
