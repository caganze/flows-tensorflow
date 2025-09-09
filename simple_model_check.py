#!/usr/bin/env python3
"""
Simple model integrity check without TensorFlow dependencies
"""

import numpy as np
import sys
from pathlib import Path
import glob

def detailed_file_check(model_path: str):
    """Detailed check of a model file without TensorFlow"""
    try:
        print(f"📁 Checking: {model_path}")
        
        # Get file stats
        path_obj = Path(model_path)
        file_size = path_obj.stat().st_size
        print(f"   File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        # Try to load with numpy
        try:
            data = np.load(model_path, allow_pickle=True)
            keys = list(data.keys())
            print(f"   Total keys: {len(keys)}")
            
            # Categorize keys
            var_keys = [k for k in keys if k.startswith('var_')]
            config_keys = [k for k in keys if 'config' in k.lower()]
            other_keys = [k for k in keys if not k.startswith('var_') and 'config' not in k.lower()]
            
            print(f"   Variable keys (var_*): {len(var_keys)}")
            print(f"   Config keys: {len(config_keys)}")
            print(f"   Other keys: {len(other_keys)}")
            
            # Show first few keys of each type
            if var_keys:
                print(f"   Sample var keys: {var_keys[:3]}")
            if config_keys:
                print(f"   Config keys: {config_keys}")
            if other_keys:
                print(f"   Other keys: {other_keys[:5]}")
            
            # Check if keys have actual data
            non_empty_keys = 0
            for key in keys[:10]:  # Check first 10 keys
                try:
                    value = data[key]
                    if hasattr(value, 'shape'):
                        if value.size > 0:
                            non_empty_keys += 1
                    elif value is not None:
                        non_empty_keys += 1
                except:
                    pass
            
            print(f"   Non-empty keys (first 10): {non_empty_keys}/10")
            
            # Status determination
            if len(var_keys) == 0:
                if file_size > 1000:  # File has size but no variables
                    print("   🤔 SUSPICIOUS: Large file but no variables found")
                    print("      This might be a key naming issue, not corruption")
                else:
                    print("   ❌ CORRUPTED: No variables and small size")
                return False
            elif len(var_keys) < 10:
                print("   ⚠️ SUSPICIOUS: Very few variables")
                return False
            else:
                print("   ✅ HEALTHY: Adequate variables found")
                return True
                
        except Exception as load_error:
            print(f"   ❌ LOAD ERROR: {load_error}")
            return False
            
    except Exception as e:
        print(f"   ❌ FILE ERROR: {e}")
        return False

def check_recent_models(base_dir: str, max_check: int = 10):
    """Check the most recent model files"""
    
    print("🔍 SIMPLE MODEL INTEGRITY CHECK")
    print("=" * 50)
    print(f"Base directory: {base_dir}")
    print(f"Max files to check: {max_check}")
    print("(No TensorFlow required)")
    print("")
    
    # Find all model files
    pattern = f"{base_dir}/**/model_pid*.npz"
    model_files = glob.glob(pattern, recursive=True)
    
    print(f"📋 Found {len(model_files)} model files")
    
    if len(model_files) == 0:
        print("❌ No model files found!")
        return
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    
    print(f"🕒 Checking {max_check} most recent files:")
    print("")
    
    # Check files
    healthy = 0
    corrupted = 0
    suspicious = 0
    
    for i, model_file in enumerate(model_files[:max_check]):
        print(f"[{i+1}/{max_check}]")
        result = detailed_file_check(model_file)
        if result:
            healthy += 1
        else:
            if "SUSPICIOUS" in str(result):
                suspicious += 1
            else:
                corrupted += 1
        print("")
    
    print("📊 SUMMARY")
    print("=" * 20)
    print(f"✅ Healthy models: {healthy}")
    print(f"⚠️ Suspicious models: {suspicious}")
    print(f"❌ Corrupted models: {corrupted}")
    print(f"📊 Total checked: {min(max_check, len(model_files))}")
    
    total_checked = min(max_check, len(model_files))
    if total_checked > 0:
        corruption_rate = corrupted / total_checked * 100
        print(f"💥 Corruption rate: {corruption_rate:.1f}%")
        
        if corruption_rate > 50:
            print("\n🚨 HIGH CORRUPTION!")
        elif corruption_rate > 10:
            print("\n⚠️ MODERATE CORRUPTION")
        else:
            print("\n✅ LOW CORRUPTION")
    
    print(f"\n🔍 DIAGNOSIS:")
    if healthy > 0:
        print("   Some models appear healthy - issue may be intermittent")
    elif suspicious > 0:
        print("   Models have data but unexpected structure - check key naming")
    else:
        print("   All models appear corrupted - systematic issue likely")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_model_check.py <base_directory> [max_check]")
        print("Example: python simple_model_check.py /oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/tfp_output/trained_flows 10")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    max_check = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    check_recent_models(base_dir, max_check)
