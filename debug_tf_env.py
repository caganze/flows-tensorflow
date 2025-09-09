#!/usr/bin/env python3
"""
Debug TensorFlow environment issues on Sherlock
"""

import sys
import os
from pathlib import Path

def debug_python_environment():
    """Debug the current Python environment"""
    
    print("🔍 PYTHON ENVIRONMENT DEBUG")
    print("=" * 50)
    
    # Basic Python info
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path[:3]}...")  # First 3 entries
    print("")
    
    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    conda_prefix = os.environ.get('CONDA_PREFIX', 'None')
    print(f"Conda environment: {conda_env}")
    print(f"Conda prefix: {conda_prefix}")
    print("")
    
    # Check for TensorFlow in different ways
    print("🔍 TENSORFLOW DETECTION")
    print("-" * 30)
    
    # Method 1: Direct import
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow imported successfully: {tf.__version__}")
        print(f"   TensorFlow location: {tf.__file__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
    except Exception as e:
        print(f"❌ TensorFlow import error: {e}")
    
    # Method 2: Check if module exists in path
    try:
        import importlib.util
        spec = importlib.util.find_spec("tensorflow")
        if spec is not None:
            print(f"✅ TensorFlow spec found: {spec.origin}")
        else:
            print("❌ TensorFlow spec not found")
    except Exception as e:
        print(f"❌ Error finding TensorFlow spec: {e}")
    
    # Method 3: Check site-packages
    site_packages = [p for p in sys.path if 'site-packages' in p]
    print(f"\n📦 Site-packages directories ({len(site_packages)}):")
    for sp in site_packages[:3]:  # Show first 3
        print(f"   {sp}")
        tf_path = Path(sp) / "tensorflow"
        if tf_path.exists():
            print(f"   ✅ TensorFlow found in: {tf_path}")
        else:
            print(f"   ❌ No TensorFlow in: {sp}")
    
    # Check other ML libraries
    print(f"\n🔍 OTHER ML LIBRARIES")
    print("-" * 30)
    
    libraries = ['numpy', 'scipy', 'matplotlib', 'pandas', 'sklearn']
    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {lib}: {version}")
        except ImportError:
            print(f"❌ {lib}: not found")
    
    # Environment variables
    print(f"\n🌍 RELEVANT ENVIRONMENT VARIABLES")
    print("-" * 40)
    env_vars = ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        if len(value) > 100:
            value = value[:100] + "..."
        print(f"{var}: {value}")

if __name__ == "__main__":
    debug_python_environment()
