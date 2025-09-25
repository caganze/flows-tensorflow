#!/usr/bin/env python3
"""
Test script to verify symlib argument parsing in train_tfp_flows.py
"""

import subprocess
import sys

def test_help():
    """Test that --help works with new symlib arguments"""
    print("🧪 Testing argument parser help...")
    
    result = subprocess.run([
        'python', 'train_tfp_flows.py', '--help'
    ], capture_output=True, text=True, timeout=60)
    
    if result.returncode != 0:
        print(f"❌ Help failed: {result.stderr}")
        return False
    
    # Check for new symlib arguments
    help_text = result.stdout
    required_args = ['--halo_id', '--particle_pid', '--suite']
    
    missing_args = []
    for arg in required_args:
        if arg not in help_text:
            missing_args.append(arg)
    
    if missing_args:
        print(f"❌ Missing required arguments in help: {missing_args}")
        return False
    
    # Check that old --data_path is not present
    if '--data_path' in help_text:
        print("❌ Old --data_path argument still present!")
        return False
    
    print("✅ Argument parser help looks correct")
    return True

def test_basic_validation():
    """Test basic argument validation"""
    print("🧪 Testing basic argument validation...")
    
    # Test missing required arguments
    result = subprocess.run([
        'python', 'train_tfp_flows.py'
    ], capture_output=True, text=True, timeout=30)
    
    if result.returncode == 0:
        print("❌ Should have failed with missing required arguments")
        return False
    
    if 'required' not in result.stderr.lower():
        print(f"❌ Expected 'required' in error message, got: {result.stderr}")
        return False
    
    print("✅ Properly rejects missing required arguments")
    return True

def main():
    print("🚀 TESTING SYMLIB ARGUMENT PARSING")
    print("==================================")
    
    tests = [
        ("Help Command", test_help),
        ("Argument Validation", test_basic_validation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 {test_name}...")
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            failed += 1
    
    print(f"\n📊 RESULTS:")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED - Symlib arguments working!")
        return True
    else:
        print("❌ SOME TESTS FAILED - Check implementation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


