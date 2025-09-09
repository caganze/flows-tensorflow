#!/usr/bin/env python3
"""
Fast verification script - checks syntax and structure without full TensorFlow initialization
"""

import os
import re
import ast
import sys
from pathlib import Path

def check_python_syntax(file_path: str) -> bool:
    """Check if Python file has valid syntax"""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse to check syntax
        ast.parse(source)
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return False

def check_argument_parser_syntax():
    """Check train_tfp_flows.py syntax and argument parser structure"""
    print("🧪 FAST ARGUMENT PARSER CHECK")
    print("=" * 35)
    
    file_path = "train_tfp_flows.py"
    
    # Check syntax first
    if not check_python_syntax(file_path):
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for required argument patterns
        checks = {
            "--generate-samples": "--generate-samples" in content,
            "--use_kroupa_imf": "--use_kroupa_imf" in content,
            "--n_samples": "--n_samples" in content,
            "Kroupa validation": "Kroupa IMF is MANDATORY" in content,
            "No fallbacks": "use_kroupa_imf: bool = True" not in content  # Should be removed
        }
        
        print(f"✅ Syntax: Valid")
        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            print(f"{status} {check_name}: {'Present' if result else 'Missing'}")
        
        all_passed = all(checks.values())
        if all_passed:
            print("✅ All argument parser checks passed")
        else:
            print("❌ Some argument parser checks failed")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error checking argument parser: {e}")
        return False

def check_kroupa_imf_syntax():
    """Check kroupa_imf.py syntax and structure"""
    print(f"\n🌟 FAST KROUPA IMF CHECK")
    print("=" * 30)
    
    file_path = "kroupa_imf.py"
    
    # Check syntax
    if not check_python_syntax(file_path):
        return False
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for required functions and improvements
        checks = {
            "sample_with_kroupa_imf": "def sample_with_kroupa_imf" in content,
            "get_stellar_mass_from_h5": "def get_stellar_mass_from_h5" in content,
            "CPU configuration": "CUDA_VISIBLE_DEVICES" in content,
            "Threading limits": "set_inter_op_parallelism_threads" in content,
        }
        
        print(f"✅ Syntax: Valid")
        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            print(f"{status} {check_name}: {'Present' if result else 'Missing'}")
        
        all_passed = all(checks.values())
        if all_passed:
            print("✅ All Kroupa IMF checks passed")
        else:
            print("❌ Some Kroupa IMF checks failed")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error checking Kroupa IMF: {e}")
        return False

def verify_script_configurations():
    """Verify essential scripts have correct configurations"""
    print(f"\n🔍 SCRIPT CONFIGURATION CHECK")
    print("=" * 35)
    
    scripts = [
        ("brute_force_gpu_job.sh", "owners", "--generate-samples"),
        ("brute_force_cpu_parallel.sh", "kipac", "--generate-samples"),
        ("submit_tfp_array.sh", "owners", "--use_kroupa_imf"),
        ("train_single_gpu.sh", "owners", "--generate-samples"),
    ]
    
    all_passed = True
    
    for script_name, expected_partition, expected_kroupa in scripts:
        if not os.path.exists(script_name):
            print(f"❌ {script_name}: Not found")
            all_passed = False
            continue
        
        try:
            with open(script_name, 'r') as f:
                content = f.read()
            
            has_partition = f"--partition={expected_partition}" in content
            has_kroupa = expected_kroupa in content
            calls_train = "train_tfp_flows.py" in content
            
            status = "✅" if (has_partition and has_kroupa and calls_train) else "❌"
            print(f"{status} {script_name}: Partition({expected_partition}), Kroupa({expected_kroupa}), Calls train_tfp_flows")
            
            if not (has_partition and has_kroupa and calls_train):
                all_passed = False
                
        except Exception as e:
            print(f"❌ {script_name}: Error reading file - {e}")
            all_passed = False
    
    return all_passed

def check_file_permissions():
    """Check that essential scripts are executable"""
    print(f"\n🔧 FILE PERMISSIONS CHECK")
    print("=" * 30)
    
    scripts = [
        "brute_force_gpu_job.sh",
        "brute_force_cpu_parallel.sh", 
        "submit_tfp_array.sh",
        "submit_cpu_smart.sh",
        "train_single_gpu.sh"
    ]
    
    all_executable = True
    
    for script in scripts:
        if os.path.exists(script):
            is_executable = os.access(script, os.X_OK)
            status = "✅" if is_executable else "❌"
            print(f"   {script}: {status} {'executable' if is_executable else 'not executable'}")
            if not is_executable:
                all_executable = False
        else:
            print(f"   {script}: ❌ not found")
            all_executable = False
    
    return all_executable

def main():
    """Run fast verification checks"""
    print("🚀 FAST VERIFICATION (No TensorFlow Initialization)")
    print("=" * 55)
    print("This checks syntax and structure without importing TensorFlow")
    print()
    
    results = {
        "Argument Parser": check_argument_parser_syntax(),
        "Kroupa IMF": check_kroupa_imf_syntax(), 
        "Script Configs": verify_script_configurations(),
        "File Permissions": check_file_permissions()
    }
    
    print(f"\n📊 FAST VERIFICATION RESULTS")
    print("=" * 35)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    print(f"\n🎯 OVERALL STATUS:")
    if all_passed:
        print("✅ ALL FAST CHECKS PASSED!")
        print("💡 Run final_verification.py for full TensorFlow testing (takes 5+ minutes)")
    else:
        print("❌ SOME FAST CHECKS FAILED!")
        print("🔧 Fix the issues above before running full verification")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
