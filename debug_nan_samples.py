#!/usr/bin/env python3
"""
Debug NaN samples in flow generation
"""

import numpy as np
import tensorflow as tf
import sys
from pathlib import Path

# Import our modules
from tfp_flows_gpu_solution import load_trained_flow
from kroupa_imf import sample_with_kroupa_imf

def check_nan_samples(model_dir: str, particle_pid: int):
    """Debug where NaN values are coming from in flow sampling"""
    
    print("🔍 DEBUGGING NaN SAMPLES")
    print("=" * 50)
    print(f"Model directory: {model_dir}")
    print(f"Particle PID: {particle_pid}")
    print("")
    
    # Load the trained flow
    try:
        print("1️⃣ Loading trained flow...")
        model_path = Path(model_dir) / f"model_pid{particle_pid}.npz"
        print(f"   Model path: {model_path}")
        
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return
            
        flow = load_trained_flow(str(model_path))
        print("✅ Flow loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load flow: {e}")
        return
    
    # Load preprocessing stats
    try:
        print("\n2️⃣ Loading preprocessing stats...")
        preprocessing_path = Path(model_dir) / f"model_pid{particle_pid}_preprocessing.npz"
        preprocessing_data = np.load(preprocessing_path)
        preprocessing_stats = {
            'mean': preprocessing_data['mean'],
            'std': preprocessing_data['std']
        }
        print(f"✅ Preprocessing stats loaded")
        print(f"   Mean shape: {preprocessing_stats['mean'].shape}")
        print(f"   Std shape: {preprocessing_stats['std'].shape}")
        
        # Check for NaN/Inf in preprocessing stats
        mean_has_nan = np.any(np.isnan(preprocessing_stats['mean']))
        std_has_nan = np.any(np.isnan(preprocessing_stats['std']))
        mean_has_inf = np.any(np.isinf(preprocessing_stats['mean']))
        std_has_inf = np.any(np.isinf(preprocessing_stats['std']))
        
        print(f"   Mean has NaN: {mean_has_nan}")
        print(f"   Mean has Inf: {mean_has_inf}")
        print(f"   Std has NaN: {std_has_nan}")
        print(f"   Std has Inf: {std_has_inf}")
        
        if mean_has_nan or std_has_nan or mean_has_inf or std_has_inf:
            print("❌ FOUND ISSUE: Preprocessing stats contain NaN/Inf values!")
            print("   Mean:", preprocessing_stats['mean'])
            print("   Std:", preprocessing_stats['std'])
            return
            
    except Exception as e:
        print(f"❌ Failed to load preprocessing: {e}")
        return
    
    # Test basic flow sampling  
    try:
        print("\n3️⃣ Testing basic flow sampling...")
        n_test = 100
        samples_raw = flow.sample(n_test, seed=42)
        print(f"✅ Basic sampling successful: shape {samples_raw.shape}")
        
        # Check for NaN in raw samples
        raw_has_nan = tf.reduce_any(tf.math.is_nan(samples_raw))
        raw_has_inf = tf.reduce_any(tf.math.is_inf(samples_raw))
        
        print(f"   Raw samples have NaN: {raw_has_nan.numpy()}")
        print(f"   Raw samples have Inf: {raw_has_inf.numpy()}")
        
        if raw_has_nan.numpy() or raw_has_inf.numpy():
            print("❌ FOUND ISSUE: Flow produces NaN/Inf samples directly!")
            print("   This indicates the trained model is corrupted")
            return
            
    except Exception as e:
        print(f"❌ Failed basic sampling: {e}")
        return
    
    # Test unstandardization
    try:
        print("\n4️⃣ Testing unstandardization...")
        mean_tf = tf.constant(preprocessing_stats['mean'], dtype=tf.float32)
        std_tf = tf.constant(preprocessing_stats['std'], dtype=tf.float32)
        
        # Check TF constants
        mean_tf_has_nan = tf.reduce_any(tf.math.is_nan(mean_tf))
        std_tf_has_nan = tf.reduce_any(tf.math.is_nan(std_tf))
        std_tf_has_zero = tf.reduce_any(tf.equal(std_tf, 0.0))
        
        print(f"   TF mean has NaN: {mean_tf_has_nan.numpy()}")
        print(f"   TF std has NaN: {std_tf_has_nan.numpy()}")
        print(f"   TF std has zeros: {std_tf_has_zero.numpy()}")
        
        if std_tf_has_zero.numpy():
            print("❌ FOUND ISSUE: Standard deviation contains zeros!")
            print("   This will cause division by zero in standardization")
            zero_indices = tf.where(tf.equal(std_tf, 0.0))
            print(f"   Zero std indices: {zero_indices.numpy().flatten()}")
            return
        
        # Perform unstandardization
        samples_unstd = samples_raw * std_tf + mean_tf
        unstd_has_nan = tf.reduce_any(tf.math.is_nan(samples_unstd))
        unstd_has_inf = tf.reduce_any(tf.math.is_inf(samples_unstd))
        
        print(f"   Unstandardized samples have NaN: {unstd_has_nan.numpy()}")
        print(f"   Unstandardized samples have Inf: {unstd_has_inf.numpy()}")
        
        if unstd_has_nan.numpy() or unstd_has_inf.numpy():
            print("❌ FOUND ISSUE: Unstandardization creates NaN/Inf!")
            return
            
    except Exception as e:
        print(f"❌ Failed unstandardization test: {e}")
        return
    
    # Test Kroupa IMF sampling
    try:
        print("\n5️⃣ Testing full Kroupa IMF sampling...")
        stellar_mass = 1.5e8  # Test stellar mass
        
        samples_6d, masses = sample_with_kroupa_imf(
            flow=flow,
            n_target_mass=stellar_mass,
            preprocessing_stats=preprocessing_stats,
            seed=42
        )
        
        # Check final results
        samples_has_nan = tf.reduce_any(tf.math.is_nan(samples_6d))
        masses_has_nan = np.any(np.isnan(masses))
        
        print(f"✅ Kroupa sampling completed")
        print(f"   Final samples have NaN: {samples_has_nan.numpy()}")
        print(f"   Final masses have NaN: {masses_has_nan}")
        print(f"   Sample shape: {samples_6d.shape}")
        print(f"   Sample range: [{tf.reduce_min(samples_6d).numpy():.3f}, {tf.reduce_max(samples_6d).numpy():.3f}]")
        
        if samples_has_nan.numpy() or masses_has_nan:
            print("❌ FOUND ISSUE: Final Kroupa sampling produces NaN!")
        else:
            print("✅ No NaN issues found - sampling appears healthy")
            
    except Exception as e:
        print(f"❌ Failed Kroupa IMF sampling: {e}")
        return
    
    print("\n🏁 DIAGNOSIS COMPLETE")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python debug_nan_samples.py <model_dir> <particle_pid>")
        print("Example: python debug_nan_samples.py /path/to/model/dir 1")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    particle_pid = int(sys.argv[2])
    
    check_nan_samples(model_dir, particle_pid)
