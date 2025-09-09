#!/usr/bin/env python3
"""
Test Kroupa IMF integration with symlib data
Quick test to ensure sample counts are realistic (not round numbers)
"""

import numpy as np
import sys
from symlib_utils import load_particle_data
from kroupa_imf import sample_with_kroupa_imf

def test_kroupa_with_symlib(halo_id: str, particle_pid: int, suite: str = 'eden'):
    """
    Test that Kroupa IMF produces realistic sample counts with symlib data
    """
    print(f"🧪 TESTING KROUPA IMF WITH SYMLIB")
    print(f"==================================")
    print(f"Halo: {halo_id}, PID: {particle_pid}, Suite: {suite}")
    print()
    
    try:
        # Load symlib data
        print("1️⃣ Loading symlib data...")
        data, metadata = load_particle_data(halo_id, particle_pid, suite)
        
        stellar_mass = metadata['stellar_mass']
        print(f"✅ Loaded {data.shape[0]:,} particles")
        print(f"✅ Total stellar mass: {stellar_mass:.2e} M☉")
        print()
        
        # Create dummy preprocessing stats (normally from training)
        print("2️⃣ Creating mock preprocessing stats...")
        mean_stats = np.mean(data, axis=0)
        std_stats = np.std(data, axis=0)
        
        preprocessing_stats = {
            'mean': mean_stats,
            'std': std_stats
        }
        print(f"✅ Mock preprocessing stats created")
        print()
        
        # Test Kroupa IMF sampling
        print("3️⃣ Testing Kroupa IMF sampling...")
        print(f"🌟 Sampling for stellar mass: {stellar_mass:.2e} M☉")
        
        # Create a dummy flow (we just need something that can sample)
        class MockFlow:
            def sample(self, n_samples, seed=42):
                # Return random 6D samples (this is just for testing Kroupa logic)
                np.random.seed(seed)
                return np.random.randn(n_samples, 6).astype(np.float32)
        
        mock_flow = MockFlow()
        
        # Test Kroupa sampling
        samples, masses = sample_with_kroupa_imf(
            flow=mock_flow,
            n_target_mass=stellar_mass,
            preprocessing_stats=preprocessing_stats,
            seed=42
        )
        
        n_samples = len(samples)
        total_sampled_mass = np.sum(masses)
        
        print(f"✅ Kroupa sampling successful!")
        print(f"📊 RESULTS:")
        print(f"   Generated samples: {n_samples:,}")
        print(f"   Target mass: {stellar_mass:.2e} M☉")
        print(f"   Sampled mass: {total_sampled_mass:.2e} M☉")
        print(f"   Mass ratio: {total_sampled_mass/stellar_mass:.3f}")
        print()
        
        # KEY TEST: Check if sample count is realistic (not round number)
        print("🔍 KROUPA IMF VALIDATION:")
        
        # Check if it's a round number (bad)
        round_numbers = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        is_round = n_samples in round_numbers
        
        if is_round:
            print(f"❌ FAIL: Sample count {n_samples:,} is a round number!")
            print(f"   This suggests fallback sampling, not Kroupa IMF")
            return False
        else:
            print(f"✅ PASS: Sample count {n_samples:,} is realistic (not round)")
        
        # Check mass range is realistic
        min_mass, max_mass = masses.min(), masses.max()
        print(f"✅ Mass range: {min_mass:.3f} - {max_mass:.3f} M☉")
        
        # Kroupa IMF should produce masses in range ~0.08-120 M☉
        if min_mass >= 0.05 and max_mass <= 150:
            print(f"✅ PASS: Mass range is realistic for Kroupa IMF")
        else:
            print(f"❌ FAIL: Mass range {min_mass:.3f}-{max_mass:.3f} is unrealistic")
            return False
        
        # Check mass distribution
        low_mass_fraction = np.sum(masses < 0.5) / len(masses)
        high_mass_fraction = np.sum(masses > 2.0) / len(masses)
        
        print(f"✅ Low mass (<0.5 M☉): {low_mass_fraction:.1%}")
        print(f"✅ High mass (>2.0 M☉): {high_mass_fraction:.1%}")
        
        # Kroupa IMF should be dominated by low-mass stars
        if low_mass_fraction > 0.5:  # Should be >50% low mass
            print(f"✅ PASS: Mass distribution matches Kroupa IMF expectation")
        else:
            print(f"⚠️  WARNING: Low mass fraction {low_mass_fraction:.1%} seems low for Kroupa IMF")
        
        print()
        print(f"🎉 KROUPA IMF + SYMLIB INTEGRATION: ✅ SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ KROUPA IMF TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_symlib_kroupa.py <halo_id> <particle_pid> [suite]")
        print("Example: python test_symlib_kroupa.py Halo268 2 eden")
        sys.exit(1)
    
    halo_id = sys.argv[1]
    particle_pid = int(sys.argv[2])
    suite = sys.argv[3] if len(sys.argv) > 3 else 'eden'
    
    success = test_kroupa_with_symlib(halo_id, particle_pid, suite)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

