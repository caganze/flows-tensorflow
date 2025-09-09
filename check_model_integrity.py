#!/usr/bin/env python3
"""
Check integrity of saved flow models
"""

import numpy as np
import sys
from pathlib import Path
import glob

def check_model_file(model_path: str):
    """Check if a model file is corrupted"""
    try:
        data = np.load(model_path)
        keys = list(data.keys())
        n_variables = len([k for k in keys if k.startswith('var_')])
        
        # Check for expected structure
        has_config = 'config' in keys
        has_architecture = 'architecture' in keys
        
        print(f"📁 {model_path}")
        print(f"   Keys: {len(keys)} total")
        print(f"   Variables: {n_variables}")
        print(f"   Has config: {has_config}")
        print(f"   Has architecture: {has_architecture}")
        
        if n_variables == 0:
            print("   ❌ CORRUPTED: No variables found")
            return False
        elif n_variables < 10:
            print("   ⚠️ SUSPICIOUS: Very few variables")
            return False
        else:
            print("   ✅ HEALTHY: Adequate variables")
            return True
            
    except Exception as e:
        print(f"📁 {model_path}")
        print(f"   ❌ ERROR: {e}")
        return False

def scan_models(base_dir: str, max_check: int = 20):
    """Scan for model files and check their integrity"""
    
    print("🔍 SCANNING MODEL INTEGRITY")
    print("=" * 50)
    print(f"Base directory: {base_dir}")
    print(f"Max files to check: {max_check}")
    print("")
    
    # Find all model files
    pattern = f"{base_dir}/**/model_pid*.npz"
    model_files = glob.glob(pattern, recursive=True)
    
    print(f"📋 Found {len(model_files)} model files")
    print("")
    
    if len(model_files) == 0:
        print("❌ No model files found!")
        return
    
    # Check a sample of files
    files_to_check = model_files[:max_check]
    
    healthy = 0
    corrupted = 0
    suspicious = 0
    
    for model_file in files_to_check:
        result = check_model_file(model_file)
        if result:
            if "HEALTHY" in str(result):
                healthy += 1
            else:
                suspicious += 1
        else:
            corrupted += 1
        print("")
    
    print("📊 SUMMARY")
    print("=" * 20)
    print(f"✅ Healthy models: {healthy}")
    print(f"⚠️ Suspicious models: {suspicious}")
    print(f"❌ Corrupted models: {corrupted}")
    print(f"📊 Total checked: {len(files_to_check)}")
    
    corruption_rate = corrupted / len(files_to_check) * 100
    print(f"💥 Corruption rate: {corruption_rate:.1f}%")
    
    if corruption_rate > 50:
        print("\n🚨 HIGH CORRUPTION DETECTED!")
        print("   Most models are corrupted - there's a systematic issue")
    elif corruption_rate > 10:
        print("\n⚠️ MODERATE CORRUPTION DETECTED")
        print("   Some models are corrupted - investigate saving process")
    else:
        print("\n✅ LOW CORRUPTION RATE")
        print("   Most models appear healthy")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_model_integrity.py <base_directory> [max_check]")
        print("Example: python check_model_integrity.py /path/to/tfp_output 50")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    max_check = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    scan_models(base_dir, max_check)
