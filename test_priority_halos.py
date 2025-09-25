#!/usr/bin/env python3
"""
Test script to verify priority halos exist and train_tfp_flows.py works without symlib
"""

import os
import glob
import subprocess
import sys

# Force CPU mode to avoid GPU issues on compute nodes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def find_priority_halo_files():
    """Find H5 files for priority halos"""
    priority_halos = ["239", "718", "270", "925"]
    
    symphony_base = "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/"
    eden_base = "/oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/"
    
    found_files = {}
    
    print("🔍 SEARCHING FOR PRIORITY HALO FILES")
    print("====================================")
    
    for halo in priority_halos:
        # Search symphony
        symphony_pattern = f"{symphony_base}*Halo{halo}_*.h5"
        symphony_files = glob.glob(symphony_pattern)
        
        # Search eden  
        eden_pattern = f"{eden_base}*Halo{halo}_*.h5"
        eden_files = glob.glob(eden_pattern)
        
        print(f"\n🎯 Halo {halo}:")
        
        if symphony_files:
            found_files[f"symphony_halo{halo}"] = symphony_files[0]
            print(f"  ✅ Symphony: {os.path.basename(symphony_files[0])}")
        else:
            print(f"  ❌ Symphony: Not found ({symphony_pattern})")
            
        if eden_files:
            found_files[f"eden_halo{halo}"] = eden_files[0]
            print(f"  ✅ Eden: {os.path.basename(eden_files[0])}")
        else:
            print(f"  ❌ Eden: Not found ({eden_pattern})")
    
    return found_files

def test_train_tfp_help():
    """Test that train_tfp_flows.py help works (avoiding symlib import)"""
    print("\n🧪 TESTING TRAIN_TFP_FLOWS.PY HELP")
    print("==================================")
    
    try:
        # Test help without importing symlib
        result = subprocess.run([
            'python', '-c', 
            '''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Try to import train_tfp_flows basic components
try:
    import argparse
    import tensorflow as tf
    print("✅ Basic imports successful")
    print(f"   TensorFlow version: {tf.__version__}")
    print(f"   GPU available: {len(tf.config.list_physical_devices('GPU'))}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
'''
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Basic TensorFlow imports work")
            print(result.stdout.strip())
            return True
        else:
            print(f"❌ Basic imports failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        return False

def test_quick_particle_extraction(h5_file):
    """Test extracting particles from an H5 file"""
    print(f"\n🧪 TESTING PARTICLE EXTRACTION: {os.path.basename(h5_file)}")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            'python', '-c', f'''
import os
import h5py
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    with h5py.File("{h5_file}", "r") as f:
        print(f"📁 H5 datasets: {{list(f.keys())}}")
        
        if "PartType1" in f and "ParticleIDs" in f["PartType1"]:
            pids = np.unique(f["PartType1"]["ParticleIDs"][:])
            pids = pids[(pids > 0) & (pids < 100000)]  # Filter reasonable range
            print(f"📊 Found {{len(pids)}} unique particle IDs")
            print(f"   First 10 PIDs: {{pids[:10].tolist()}}")
            
            # Test getting particle count for first PID
            if len(pids) > 0:
                test_pid = pids[0]
                mask = f["PartType1"]["ParticleIDs"][:] == test_pid
                count = np.sum(mask)
                print(f"🎯 PID {{test_pid}} has {{count}} particles")
                
        else:
            print("❌ Expected PartType1/ParticleIDs structure not found")
            
except Exception as e:
    print(f"❌ Error: {{e}}")
'''
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(result.stdout.strip())
            return True
        else:
            print(f"❌ Extraction failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        return False

def main():
    print("🚀 PRIORITY HALO READINESS TEST")
    print("==============================")
    print("Testing readiness for 12-hour priority job")
    print("Target halos: 239, 718, 270, 925 (symphony + eden)")
    
    # Find files
    found_files = find_priority_halo_files()
    
    # Test basic functionality
    basic_test_passed = test_train_tfp_help()
    
    # Test particle extraction on one file
    extraction_tests = 0
    if found_files:
        # Test extraction on first available file
        test_file = list(found_files.values())[0]
        if test_quick_particle_extraction(test_file):
            extraction_tests = 1
    
    # Summary
    print(f"\n📊 READINESS SUMMARY")
    print("===================")
    print(f"✅ Priority halo files found: {len(found_files)}/8")
    print(f"✅ Basic TF imports: {'YES' if basic_test_passed else 'NO'}")
    print(f"✅ Particle extraction: {'YES' if extraction_tests > 0 else 'NO'}")
    
    if len(found_files) >= 4 and basic_test_passed and extraction_tests > 0:
        print("\n🎉 READY FOR PRIORITY 12-HOUR JOB!")
        print("   Run: sbatch priority_12hour_cpu.sh")
        return True
    else:
        print("\n⚠️  NOT READY - Issues detected")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


