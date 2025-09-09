#!/usr/bin/env python3
"""
Kroupa Initial Mass Function (IMF) implementation for TensorFlow
Translates JAX version to TensorFlow for realistic stellar mass sampling
"""

import os
import numpy as np

# Configure TensorFlow for CPU-only to prevent hanging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Hide GPUs to force CPU

import tensorflow as tf
from typing import Tuple

# Configure TensorFlow for minimal resource usage
# Use environment variables to avoid RuntimeError if TF already initialized
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# Try to configure GPU visibility (may fail if TF already initialized)
try:
    tf.config.set_visible_devices([], 'GPU')  # Disable GPU
except RuntimeError:
    # TensorFlow already initialized, skip configuration
    pass

def kroupa_masses(nsample: int, seed: int = 42) -> np.ndarray:
    """Generate Kroupa IMF masses in solar masses.
    
    Args:
        nsample: Number of mass samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Array of stellar masses in solar masses
    """
    np.random.seed(seed)
    
    def sample_from_powerlaw(alpha: float, xmin: float, xmax: float, nsample: int) -> np.ndarray:
        """Sample from power law distribution."""
        u = np.random.uniform(0, 1, nsample)
        if alpha == -1:
            return xmin * (xmax/xmin)**u
        else:
            return (u * (xmax**(alpha + 1) - xmin**(alpha + 1)) + xmin**(alpha + 1))**(1/(alpha + 1))
    
    # Kroupa IMF broken power law segments
    m0 = sample_from_powerlaw(-0.3, xmin=0.03, xmax=0.08, nsample=int(nsample))   # Brown dwarfs
    m1 = sample_from_powerlaw(-1.3, xmin=0.08, xmax=0.5, nsample=int(nsample))    # Low mass stars
    m2 = sample_from_powerlaw(-2.3, xmin=0.5, xmax=120, nsample=int(nsample))     # High mass stars
    
    # Combine all segments
    m = np.concatenate([m0, m1, m2]).flatten()
    
    # Apply mass range limits
    m_range = [0.08, 120]  # Solar masses
    mask = np.logical_and(m > m_range[0], m < m_range[1])
    
    # Randomly select nsample masses from valid range
    valid_masses = m[mask]
    if len(valid_masses) < nsample:
        # If not enough valid masses, repeat the process
        return kroupa_masses(nsample, seed=seed+1)
    
    masses = np.random.choice(valid_masses, int(nsample), replace=True)
    return masses

def estimate_ntot(M_target: float, ntest: int = 100000, seed: int = 42) -> Tuple[int, float]:
    """Fast approximate N needed to reach total mass M_target using Kroupa IMF.
    
    Args:
        M_target: Target total stellar mass in solar masses
        ntest: Number of test samples to estimate mean mass
        seed: Random seed
        
    Returns:
        Tuple of (estimated_n_particles, mean_mass_per_particle)
    """
    mtest = kroupa_masses(ntest, seed=seed)
    mean_mass = mtest.mean()
    ntot = int(np.round(M_target / mean_mass))
    return ntot, mean_mass

def sample_with_kroupa_imf(flow, n_target_mass: float, preprocessing_stats: dict, 
                          seed: int = 42) -> Tuple[tf.Tensor, np.ndarray]:
    """Sample from flow using Kroupa IMF to determine realistic particle count.
    
    Args:
        flow: Trained TensorFlow Probability flow
        n_target_mass: Total stellar mass to sample (solar masses)
        preprocessing_stats: Dict with 'mean' and 'std' for unstandardization
        seed: Random seed
        
    Returns:
        Tuple of (samples_6d, masses) where:
        - samples_6d: TensorFlow tensor of shape (n_particles, 6) with pos+vel
        - masses: NumPy array of stellar masses for each particle
    """
    # Estimate number of particles needed
    ntot, mean_mass = estimate_ntot(n_target_mass, seed=seed)
    
    # Add Poisson noise to particle count (realistic stellar formation stochasticity)
    np.random.seed(seed)
    n_poisson = int(np.random.poisson(ntot))
    n_poisson = max(n_poisson, 1)  # At least 1 particle
    
    print(f"📊 Kroupa IMF sampling:")
    print(f"   Target stellar mass: {n_target_mass:.2e} M☉")
    print(f"   Estimated particles: {ntot:,} (mean mass: {mean_mass:.3f} M☉)")
    print(f"   Poisson sample: {n_poisson:,} particles")
    
    # Validate flow model before sampling
    print(f"   🔍 Validating flow model...")
    try:
        # Test with a small sample first
        test_samples = flow.sample(10, seed=seed + 999)
        test_has_nan = tf.reduce_any(tf.math.is_nan(test_samples))
        test_has_inf = tf.reduce_any(tf.math.is_inf(test_samples))
        
        if test_has_nan.numpy() or test_has_inf.numpy():
            raise ValueError("❌ Flow model produces NaN/Inf even in test sampling - model is corrupted")
            
        print(f"   ✅ Flow model validation passed")
        
    except Exception as e:
        print(f"   ❌ Flow model validation failed: {e}")
        raise ValueError(f"❌ Flow model is not ready for sampling: {e}")
    
    # Sample from the trained flow using batching for large particle counts
    tf.random.set_seed(seed)
    
    print(f"   🔍 Sampling {n_poisson:,} particles from trained flow...")
    
    # Use batching for large particle counts to avoid memory/NaN issues
    batch_size = min(50000, n_poisson)  # Max 50K per batch
    n_batches = (n_poisson + batch_size - 1) // batch_size
    
    if n_batches > 1:
        print(f"   📦 Using {n_batches} batches of {batch_size:,} particles each")
    
    try:
        all_samples = []
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_poisson)
            batch_size_actual = batch_end - batch_start
            
            if n_batches > 1:
                print(f"   🔄 Batch {batch_idx + 1}/{n_batches}: {batch_size_actual:,} particles")
            
            batch_samples = flow.sample(batch_size_actual, seed=seed + batch_idx)
            
            # Check for NaN/Inf in this batch
            has_nan = tf.reduce_any(tf.math.is_nan(batch_samples))
            has_inf = tf.reduce_any(tf.math.is_inf(batch_samples))
            
            if has_nan.numpy():
                print(f"   ❌ CRITICAL: Batch {batch_idx + 1} produced NaN samples!")
                raise ValueError("❌ Flow sampling produced NaN values - model may be corrupted")
                
            if has_inf.numpy():
                print(f"   ❌ CRITICAL: Batch {batch_idx + 1} produced Inf samples!")
                raise ValueError("❌ Flow sampling produced Inf values - model may be corrupted")
            
            all_samples.append(batch_samples)
        
        # Concatenate all batches
        samples_standardized = tf.concat(all_samples, axis=0)
        print(f"   ✅ Batched sampling completed: shape {samples_standardized.shape}")
        
    except Exception as e:
        print(f"   ❌ Flow sampling failed: {e}")
        raise ValueError(f"❌ Flow sampling error: {e}")
    
    # Unstandardize the samples
    print(f"   🔄 Unstandardizing samples...")
    mean = tf.constant(preprocessing_stats['mean'], dtype=tf.float32)
    std = tf.constant(preprocessing_stats['std'], dtype=tf.float32)
    
    # Check preprocessing stats for NaN/Inf
    mean_has_nan = tf.reduce_any(tf.math.is_nan(mean))
    std_has_nan = tf.reduce_any(tf.math.is_nan(std))
    std_has_zero = tf.reduce_any(tf.equal(std, 0.0))
    
    if mean_has_nan.numpy() or std_has_nan.numpy():
        raise ValueError("❌ Preprocessing stats contain NaN values")
        
    if std_has_zero.numpy():
        print(f"   ⚠️ Warning: Some std values are zero, adding small epsilon")
        std = tf.where(tf.equal(std, 0.0), tf.ones_like(std) * 1e-8, std)
    
    samples_6d = samples_standardized * std + mean
    
    # Final validation of unstandardized samples
    final_has_nan = tf.reduce_any(tf.math.is_nan(samples_6d))
    if final_has_nan.numpy():
        print(f"   ❌ CRITICAL: Unstandardized samples contain NaN!")
        print(f"   Mean range: [{tf.reduce_min(mean).numpy():.3f}, {tf.reduce_max(mean).numpy():.3f}]")
        print(f"   Std range: [{tf.reduce_min(std).numpy():.3f}, {tf.reduce_max(std).numpy():.3f}]")
        raise ValueError("❌ Unstandardization produced NaN values")
    
    print(f"   ✅ Unstandardization successful: range [{tf.reduce_min(samples_6d).numpy():.3f}, {tf.reduce_max(samples_6d).numpy():.3f}]")
    
    # Generate corresponding stellar masses using Kroupa IMF
    masses = kroupa_masses(n_poisson, seed=seed + 1000)
    
    # Verify total mass is approximately correct
    actual_total_mass = np.sum(masses)
    mass_ratio = actual_total_mass / n_target_mass
    
    print(f"   Generated {n_poisson:,} particles")
    print(f"   Actual total mass: {actual_total_mass:.2e} M☉")
    print(f"   Mass ratio (actual/target): {mass_ratio:.3f}")
    
    if mass_ratio < 0.5 or mass_ratio > 2.0:
        print(f"   ⚠️ Mass ratio outside [0.5, 2.0] range - may need adjustment")
    
    return samples_6d, masses

def get_stellar_mass_from_h5(data_dict: dict, particle_pid: int) -> float:
    """Extract total stellar mass for a specific particle ID from H5 data.
    
    Args:
        data_dict: Dictionary containing H5 datasets including 'mass' and 'parentid'
        particle_pid: Parent ID to extract mass for
        
    Returns:
        Total stellar mass in solar masses
    """
    if 'mass' not in data_dict or 'parentid' not in data_dict:
        raise ValueError("H5 data must contain 'mass' and 'parentid' datasets")
    
    # Create mask for this particle
    mask = data_dict['parentid'] == particle_pid
    
    # Sum masses for this particle (handle NaN values)
    particle_masses = data_dict['mass'][mask]
    stellar_mass = np.nansum(particle_masses)
    
    n_particles = np.sum(mask)
    mean_mass = np.nanmean(particle_masses)
    
    print(f"📊 Particle {particle_pid} stellar mass analysis:")
    print(f"   Number of particles: {n_particles:,}")
    print(f"   Total stellar mass: {stellar_mass:.2e} M☉")
    print(f"   Mean particle mass: {mean_mass:.3f} M☉")
    
    return float(stellar_mass)

# Test function
def test_kroupa_imf():
    """Test the Kroupa IMF implementation."""
    print("🧪 Testing Kroupa IMF implementation...")
    
    # Test mass generation
    masses = kroupa_masses(10000, seed=42)
    print(f"Generated {len(masses)} masses")
    print(f"Mass range: [{masses.min():.3f}, {masses.max():.3f}] M☉")
    print(f"Mean mass: {masses.mean():.3f} M☉")
    print(f"Median mass: {np.median(masses):.3f} M☉")
    
    # Test particle count estimation
    target_mass = 1e6  # 1 million solar masses
    ntot, mean_mass = estimate_ntot(target_mass)
    print(f"\nFor target mass {target_mass:.0e} M☉:")
    print(f"Estimated particles: {ntot:,}")
    print(f"Mean mass: {mean_mass:.3f} M☉")
    
    print("✅ Kroupa IMF test completed")

if __name__ == "__main__":
    test_kroupa_imf()
