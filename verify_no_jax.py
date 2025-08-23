#!/usr/bin/env python3
"""
Verify that no JAX dependencies exist in the codebase
This prevents TensorFlow-JAX conflicts in the environment
"""

import os
import re
import sys
from pathlib import Path

def check_file_for_jax(filepath):
    """Check a single file for JAX imports or usage."""
    jax_patterns = [
        r'import\s+jax',
        r'from\s+jax',
        r'import\s+jaxlib',
        r'from\s+jaxlib',
        r'jax\.',
        r'jnp\.',
        r'@jax\.',
        r'@jit(?!\s*#)',  # JAX jit decorator (but not in comments)
    ]
    
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                for pattern in jax_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(f"Line {i}: {line.strip()}")
    except Exception as e:
        issues.append(f"Error reading file: {e}")
    
    return issues

def verify_no_jax_dependencies(directory="."):
    """Scan all Python files for JAX dependencies."""
    print("🔍 Scanning for JAX dependencies...")
    print(f"Directory: {os.path.abspath(directory)}")
    print()
    
    python_files = list(Path(directory).glob("**/*.py"))
    total_issues = 0
    
    for filepath in python_files:
        if '__pycache__' in str(filepath) or filepath.name == 'verify_no_jax.py':
            continue
            
        issues = check_file_for_jax(filepath)
        
        if issues:
            print(f"⚠️  {filepath}:")
            for issue in issues:
                print(f"    {issue}")
            print()
            total_issues += len(issues)
    
    print(f"📊 Scan Results:")
    print(f"  Files scanned: {len(python_files)}")
    print(f"  JAX references found: {total_issues}")
    
    if total_issues == 0:
        print("✅ No JAX dependencies found - codebase is clean!")
        return True
    else:
        print("❌ JAX dependencies found - please remove before deployment")
        return False

def check_runtime_jax():
    """Check if JAX is importable at runtime."""
    print("\n🔍 Checking runtime JAX availability...")
    
    try:
        __import__('jax')  # Import without creating jax variable
        print("❌ JAX is importable and will conflict with TensorFlow!")
        # Get version without keeping jax in scope
        jax_module = sys.modules.get('jax')
        if jax_module and hasattr(jax_module, '__version__'):
            print(f"   JAX version: {jax_module.__version__}")
        return False
    except ImportError:
        print("✅ JAX not importable (good for TensorFlow)")
        return True

def main():
    """Main verification function."""
    print("🧪 JAX DEPENDENCY VERIFICATION")
    print("=" * 40)
    
    # Check code for JAX dependencies
    code_clean = verify_no_jax_dependencies()
    
    # Check runtime availability
    runtime_clean = check_runtime_jax()
    
    print("\n" + "=" * 40)
    print("📋 VERIFICATION RESULTS")
    print("=" * 40)
    print(f"Code scan:      {'✅ CLEAN' if code_clean else '❌ ISSUES'}")
    print(f"Runtime check:  {'✅ CLEAN' if runtime_clean else '❌ ISSUES'}")
    
    overall_clean = code_clean and runtime_clean
    print(f"Overall:        {'✅ READY' if overall_clean else '❌ FIX NEEDED'}")
    
    if overall_clean:
        print("\n🚀 Codebase is JAX-free and ready for TensorFlow deployment!")
    else:
        print("\n⚠️  Please remove JAX dependencies before proceeding.")
        print("   Suggested fixes:")
        print("   - conda remove jax jaxlib")
        print("   - Remove JAX imports from code")
        print("   - Use TensorFlow alternatives")
    
    return overall_clean

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
