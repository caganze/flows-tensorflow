#!/usr/bin/env python3
"""
Lightweight Kroupa IMF implementation without TensorFlow dependencies
For testing and environments where TF might hang
"""

import numpy as np
from typing import Tuple

def kroupa_masses_simple(nsample: int, seed: int = 42) -> np.ndarray:
    """Generate Kroupa IMF masses in solar masses - NumPy only version.
    
    Args:
        nsample: Number of mass samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Array of stellar masses in solar masses
    """
    np.random.seed(seed)
    
    def sample_from_powerlaw(alpha: float, xmin: float, xmax: float, n: int) -> np.ndarray:
        """Sample from power law distribution."""
        u = np.random.uniform(0, 1, n)
        if alpha == -1:
            return xmin * (xmax/xmin)**u
        else:
            return (u * (xmax**(alpha + 1) - xmin**(alpha + 1)) + xmin**(alpha + 1))**(1/(alpha + 1))
    
    # Sample more masses than needed to account for filtering
    oversample_factor = 3
    n_total = int(nsample * oversample_factor)
    
    # Kroupa IMF broken power law segments
    # Segment proportions: ~30% brown dwarfs, ~60% low mass, ~10% high mass
    n_bd = int(n_total * 0.3)
    n_low = int(n_total * 0.6) 
    n_high = n_total - n_bd - n_low
    
    m0 = sample_from_powerlaw(-0.3, xmin=0.03, xmax=0.08, n=n_bd)      # Brown dwarfs
    m1 = sample_from_powerlaw(-1.3, xmin=0.08, xmax=0.5, n=n_low)     # Low mass stars
    m2 = sample_from_powerlaw(-2.3, xmin=0.5, xmax=120, n=n_high)     # High mass stars
    
    # Combine all segments
    m = np.concatenate([m0, m1, m2])
    
    # Apply mass range limits (main sequence stars only)
    m_range = [0.08, 120]  # Solar masses
    mask = np.logical_and(m >= m_range[0], m <= m_range[1])
    valid_masses = m[mask]
    
    # If we don't have enough valid masses, use what we have and repeat
    if len(valid_masses) < nsample:
        # Repeat to get enough samples
        repeats = int(np.ceil(nsample / len(valid_masses)))
        valid_masses = np.tile(valid_masses, repeats)
    
    # Randomly select the requested number
    selected_indices = np.random.choice(len(valid_masses), nsample, replace=False)
    masses = valid_masses[selected_indices]
    
    return masses

def test_kroupa_simple():
    """Test the simple Kroupa implementation"""
    print("🧪 Testing simple Kroupa IMF implementation...")
    
    # Test basic functionality
    masses = kroupa_masses_simple(1000, seed=42)
    
    print(f"✅ Generated {len(masses)} masses")
    print(f"   Range: {masses.min():.3f} - {masses.max():.3f} M☉")
    print(f"   Mean: {masses.mean():.3f} M☉")
    print(f"   Median: {np.median(masses):.3f} M☉")
    
    # Check mass distribution
    low_mass = np.sum(masses < 0.5) / len(masses) * 100
    high_mass = np.sum(masses > 1.0) / len(masses) * 100
    
    print(f"   {low_mass:.1f}% low mass (<0.5 M☉)")
    print(f"   {high_mass:.1f}% high mass (>1.0 M☉)")
    
    return True

if __name__ == "__main__":
    test_kroupa_simple()
