#!/usr/bin/env python3
"""
Test script for sampling from trained coupling flow models
"""

import subprocess
import sys
import os

def test_sampling():
    """Test sampling from trained coupling flow models"""
    
    print("🧪 Testing coupling flow sampling...")
    
    # Test basic sampling
    print("\n📊 Test 1: Basic Sampling")
    cmd = [
        "python", "sample_coupling_flow.py",
        "--base_dir", "coupling_output/eden/halo718/pid1",
        "--halo_id", "Halo718",
        "--particle_pid", "1",
        "--n_samples", "1000",  # Small number for testing
        "--seed", "42"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Basic sampling test completed!")
            print("Output:")
            print(result.stdout)
            
            # Check if output file was created
            output_file = "coupling_output/eden/halo718/pid1/samples_Halo718_1.npz"
            if os.path.exists(output_file):
                print(f"✅ Output file created: {output_file}")
            else:
                print(f"⚠️ Output file not found: {output_file}")
                
        else:
            print("❌ Basic sampling test failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out (2 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False
    
    # Test Kroupa sampling
    print("\n🌟 Test 2: Kroupa IMF Sampling")
    cmd_kroupa = [
        "python", "sample_coupling_flow.py",
        "--base_dir", "coupling_output/eden/halo718/pid1",
        "--halo_id", "Halo718",
        "--particle_pid", "1",
        "--n_samples", "1000",
        "--use_kroupa",
        "--seed", "123"
    ]
    
    print(f"Running: {' '.join(cmd_kroupa)}")
    
    try:
        result = subprocess.run(cmd_kroupa, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Kroupa sampling test completed!")
            print("Output:")
            print(result.stdout)
        else:
            print("❌ Kroupa sampling test failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Kroupa test timed out (2 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running Kroupa test: {e}")
        return False
    
    print("\n🎉 All sampling tests completed successfully!")
    print("You can now use the sampling script to generate samples from your trained models.")
    
    return True

if __name__ == "__main__":
    success = test_sampling()
    sys.exit(0 if success else 1)







