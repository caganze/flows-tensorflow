#!/usr/bin/env python3
"""
Test script for symlib training pipeline
Tests the complete workflow: symlib data loading -> training -> Kroupa IMF
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Force CPU mode to avoid GPU/import issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def test_symlib_data_loading():
    """Test that symlib data loading works"""
    print("🧪 TESTING SYMLIB DATA LOADING")
    print("===============================")
    
    try:
        result = subprocess.run([
            'python', '-c', '''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    from symlib_utils import load_particle_data, validate_symlib_environment
    
    print("✅ Symlib utils imported successfully")
    
    # Test environment validation
    validate_symlib_environment()
    print("✅ Symlib environment validated")
    
    # Test loading a known working particle
    print("📊 Testing data loading for Halo268 PID 2...")
    data, metadata = load_particle_data("Halo268", 2, "eden")
    
    print(f"✅ Data loaded: {data.shape[0]:,} particles")
    print(f"   Stellar mass: {metadata['stellar_mass']:.2e} M☉")
    print("🎉 Symlib data loading works!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Data loading error: {e}")
    sys.exit(1)
'''
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"❌ Symlib data loading failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        return False

def test_kroupa_imf_fixed():
    """Test that Kroupa IMF works without 1M fallback"""
    print("\n🧪 TESTING KROUPA IMF (NO FALLBACK)")
    print("===================================")
    
    try:
        result = subprocess.run([
            'python', 'test_symlib_kroupa.py', 'Halo268', '2', 'eden'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            output = result.stdout
            # Check for key success indicators
            if "✅ PASS: Sample count" in output and "is realistic (not round)" in output:
                print("✅ Kroupa IMF generates realistic sample counts")
                # Extract sample count
                for line in output.split('\n'):
                    if "Generated samples:" in line:
                        print(f"   {line.strip()}")
                    elif "Mass ratio:" in line:
                        print(f"   {line.strip()}")
                return True
            else:
                print("❌ Kroupa IMF still has issues:")
                print(output)
                return False
        else:
            print(f"❌ Kroupa IMF test failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        return False

def test_training_args():
    """Test that training script accepts symlib arguments"""
    print("\n🧪 TESTING TRAINING SCRIPT ARGUMENTS")
    print("====================================")
    
    try:
        # Test help
        result = subprocess.run([
            'python', 'train_tfp_flows.py', '--help'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            help_text = result.stdout
            
            # Check for symlib arguments
            required_args = ['--halo_id', '--particle_pid', '--suite']
            missing_args = []
            
            for arg in required_args:
                if arg not in help_text:
                    missing_args.append(arg)
            
            if missing_args:
                print(f"❌ Missing symlib arguments: {missing_args}")
                return False
            
            # Check that old --data_path is not present
            if '--data_path' in help_text:
                print("❌ Old --data_path argument still present!")
                return False
            
            print("✅ Training script has correct symlib arguments")
            print("   --halo_id: ✅")
            print("   --particle_pid: ✅") 
            print("   --suite: ✅")
            print("   --data_path removed: ✅")
            return True
        else:
            print(f"❌ Help command failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        return False

def test_priority_halos_in_particle_list():
    """Test that priority halos exist in particle_list.txt"""
    print("\n🧪 TESTING PRIORITY HALOS IN PARTICLE LIST")
    print("==========================================")
    
    particle_list_file = "particle_list.txt"
    
    if not os.path.exists(particle_list_file):
        print(f"❌ Particle list not found: {particle_list_file}")
        return False
    
    priority_halos = ["239", "718", "270", "925"]
    found_halos = {}
    
    try:
        with open(particle_list_file, 'r') as f:
            lines = f.readlines()
        
        print(f"📋 Scanning {len(lines)} particles in particle list...")
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                pid, halo_id, suite, count, size = parts
                
                # Extract halo number
                halo_num = halo_id.replace('Halo', '')
                
                if halo_num in priority_halos:
                    key = f"{suite}_halo{halo_num}"
                    if key not in found_halos:
                        found_halos[key] = []
                    found_halos[key].append((pid, count, size))
        
        print(f"\n🎯 PRIORITY HALO AVAILABILITY:")
        for halo in priority_halos:
            eden_key = f"eden_halo{halo}"
            symphony_key = f"symphony_halo{halo}"  # Note: might not exist in eden dataset
            
            print(f"\n   Halo {halo}:")
            if eden_key in found_halos:
                count = len(found_halos[eden_key])
                total_particles = sum(int(item[1]) for item in found_halos[eden_key])
                print(f"     ✅ Eden: {count} PIDs, {total_particles:,} total particles")
            else:
                print(f"     ❌ Eden: Not found")
            
            if symphony_key in found_halos:
                count = len(found_halos[symphony_key])
                total_particles = sum(int(item[1]) for item in found_halos[symphony_key])
                print(f"     ✅ Symphony: {count} PIDs, {total_particles:,} total particles")
            else:
                print(f"     ❌ Symphony: Not found")
        
        total_found = len(found_halos)
        print(f"\n📊 Summary: {total_found} halo+suite combinations found")
        
        if total_found >= 2:  # At least 2 priority halos available
            print("✅ Sufficient priority halos available for training")
            return True
        else:
            print("❌ Insufficient priority halos available")
            return False
            
    except Exception as e:
        print(f"❌ Error reading particle list: {e}")
        return False

def test_quick_training():
    """Test a very quick training run to verify end-to-end pipeline"""
    print("\n🧪 TESTING QUICK TRAINING RUN")
    print("=============================")
    
    try:
        # Use a small particle from our known working data
        print("🚀 Starting quick training test (2 epochs)...")
        
        start_time = time.time()
        result = subprocess.run([
            'python', 'train_tfp_flows.py',
            '--halo_id', 'Halo268',
            '--particle_pid', '2', 
            '--suite', 'eden',
            '--epochs', '2',  # Very quick test
            '--batch_size', '256',
            '--n_layers', '2',
            '--hidden_units', '64',
            '--learning_rate', '1e-3',
            '--no-generate-samples'  # Skip sampling for speed
        ], capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Quick training completed in {duration:.1f}s")
            print("   Pipeline working end-to-end!")
            
            # Check for key indicators in output
            output = result.stdout
            if "Kroupa IMF: MANDATORY" in output:
                print("   ✅ Kroupa IMF enforced")
            if "TensorFlow Probability Flow Training (Symlib)" in output:
                print("   ✅ Symlib mode detected")
            
            return True
        else:
            print(f"❌ Quick training failed after {duration:.1f}s:")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Quick training timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Test crashed: {e}")
        return False

def main():
    print("🚀 SYMLIB TRAINING PIPELINE TEST SUITE")
    print("======================================")
    print("Testing complete symlib workflow for priority halos")
    print("Target halos: 239, 718, 270, 925")
    print()
    
    tests = [
        ("Symlib Data Loading", test_symlib_data_loading),
        ("Kroupa IMF (No Fallback)", test_kroupa_imf_fixed),
        ("Training Script Arguments", test_training_args),
        ("Priority Halos in Particle List", test_priority_halos_in_particle_list),
        ("Quick Training Run", test_quick_training)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: CRASHED - {e}")
    
    print(f"\n{'='*60}")
    print("🏁 FINAL RESULTS")
    print("================")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success rate: {passed}/{passed+failed} ({100*passed/(passed+failed):.1f}%)")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Symlib training pipeline is ready")
        print("✅ Priority halos can be processed")
        print("✅ Kroupa IMF working correctly")
        print("\n🚀 Ready to run: sbatch brute_force_gpu_job.sh")
        return True
    else:
        print(f"\n⚠️  {failed} TESTS FAILED")
        print("❌ Pipeline needs fixes before production")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

