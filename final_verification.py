#!/usr/bin/env python3
"""
Final verification script - double-check everything is actually working
"""

import os
import re
import subprocess

def test_argument_parser():
    """Test that train_tfp_flows.py actually accepts the Kroupa arguments"""
    print("🧪 TESTING ARGUMENT PARSER")
    print("=" * 30)
    
    try:
        # Use a longer timeout since TensorFlow initialization takes ~4 minutes on Sherlock
        print("⏳ Testing argument parser (this may take 5+ minutes due to TensorFlow initialization)...")
        result = subprocess.run(['python', 'train_tfp_flows.py', '--help'], 
                              capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            help_text = result.stdout
            
            # Check for required arguments
            has_generate_samples = '--generate-samples' in help_text
            has_use_kroupa = '--use_kroupa_imf' in help_text
            has_n_samples = '--n_samples' in help_text
            
            print(f"✅ train_tfp_flows.py --help works")
            print(f"   --generate-samples: {'✅' if has_generate_samples else '❌'}")
            print(f"   --use_kroupa_imf: {'✅' if has_use_kroupa else '❌'}")
            print(f"   --n_samples: {'✅' if has_n_samples else '❌'}")
            
            if not (has_generate_samples and has_use_kroupa and has_n_samples):
                print("❌ CRITICAL: Missing required arguments in parser!")
                return False
            else:
                print("✅ All required arguments present")
                return True
        else:
            print(f"❌ train_tfp_flows.py --help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing argument parser: {e}")
        return False

def verify_essential_scripts():
    """Verify each essential script has the right configuration"""
    print(f"\n🔍 VERIFYING ESSENTIAL SCRIPTS")
    print("=" * 35)
    
    essential_scripts = {
        'brute_force_gpu_job.sh': {
            'expected_partition': 'owners',
            'should_have_kroupa': True,
            'kroupa_patterns': ['--generate-samples']
        },
        'brute_force_cpu_parallel.sh': {
            'expected_partition': 'kipac',
            'should_have_kroupa': True,
            'kroupa_patterns': ['--generate-samples']
        },
        'submit_tfp_array.sh': {
            'expected_partition': 'owners',
            'should_have_kroupa': True,
            'kroupa_patterns': ['--use_kroupa_imf']
        },
        'train_single_gpu.sh': {
            'expected_partition': 'owners',
            'should_have_kroupa': True,
            'kroupa_patterns': ['--generate-samples']
        }
    }
    
    all_good = True
    
    for script, config in essential_scripts.items():
        print(f"\n📄 {script}:")
        
        if not os.path.exists(script):
            print(f"   ❌ File not found!")
            all_good = False
            continue
            
        try:
            with open(script, 'r') as f:
                content = f.read()
        except:
            print(f"   ❌ Cannot read file!")
            all_good = False
            continue
        
        # Check partition
        sbatch_partition = re.search(r'#SBATCH\s+--partition[=\s]+(\w+)', content)
        if sbatch_partition:
            partition = sbatch_partition.group(1)
            if partition == config['expected_partition']:
                print(f"   ✅ Partition: {partition}")
            else:
                print(f"   ❌ Partition: {partition} (expected: {config['expected_partition']})")
                all_good = False
        else:
            print(f"   ❌ No SBATCH partition found!")
            all_good = False
        
        # Check Kroupa IMF
        if config['should_have_kroupa']:
            has_kroupa = any(pattern in content for pattern in config['kroupa_patterns'])
            if has_kroupa:
                found_patterns = [p for p in config['kroupa_patterns'] if p in content]
                print(f"   ✅ Kroupa IMF: {', '.join(found_patterns)}")
            else:
                print(f"   ❌ Missing Kroupa IMF arguments: {config['kroupa_patterns']}")
                all_good = False
        
        # Check if it calls train_tfp_flows.py
        calls_training = 'python train_tfp_flows.py' in content
        if calls_training:
            print(f"   ✅ Calls train_tfp_flows.py")
        else:
            print(f"   ⚠️  Does not directly call train_tfp_flows.py")
    
    return all_good

def test_kroupa_imf_function():
    """Test that the Kroupa IMF functions are actually accessible"""
    print(f"\n🌟 TESTING KROUPA IMF FUNCTIONS")
    print("=" * 35)
    
    try:
        # Test import
        print("⏳ Testing Kroupa IMF imports (this may take 5+ minutes due to TensorFlow initialization)...")
        result = subprocess.run(['python', '-c', 
                               'from kroupa_imf import sample_with_kroupa_imf, get_stellar_mass_from_h5; print("✅ Kroupa IMF imports work")'], 
                              capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print(result.stdout.strip())
            return True
        else:
            print(f"❌ Kroupa IMF import failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Kroupa IMF: {e}")
        return False

def check_file_permissions():
    """Check that essential scripts are executable"""
    print(f"\n🔧 CHECKING FILE PERMISSIONS")
    print("=" * 30)
    
    essential_scripts = [
        'brute_force_gpu_job.sh',
        'brute_force_cpu_parallel.sh', 
        'submit_tfp_array.sh',
        'submit_cpu_smart.sh',
        'train_single_gpu.sh'
    ]
    
    all_executable = True
    
    for script in essential_scripts:
        if os.path.exists(script):
            is_executable = os.access(script, os.X_OK)
            print(f"   {script}: {'✅ executable' if is_executable else '❌ not executable'}")
            if not is_executable:
                all_executable = False
        else:
            print(f"   {script}: ❌ not found")
            all_executable = False
    
    return all_executable

def main():
    print("🔍 FINAL VERIFICATION")
    print("=" * 25)
    print("Double-checking everything is actually working...\n")
    
    tests = [
        ("Argument Parser", test_argument_parser),
        ("Essential Scripts", verify_essential_scripts),
        ("Kroupa IMF Functions", test_kroupa_imf_function),
        ("File Permissions", check_file_permissions)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print(f"\n📊 FINAL RESULTS")
    print("=" * 20)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n🎯 OVERALL STATUS:")
    if all_passed:
        print("✅ ALL TESTS PASSED - System is ready!")
        print("🚀 Your scripts should work correctly for Kroupa IMF sampling")
    else:
        print("❌ SOME TESTS FAILED - Issues remain!")
        print("⚠️  Your scripts may not work as expected")
    
    return all_passed

if __name__ == '__main__':
    main()

