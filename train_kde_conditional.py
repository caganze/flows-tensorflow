#!/usr/bin/env python3
"""
KDE-based Conditional Flow Training Script

This script uses Kernel Density Estimation (KDE) instead of reading particles directly 
from symlib. It's designed to work with the existing shell script infrastructure while
using a different density estimation approach.

Usage:
    python train_kde_flows_conditional.py --halo_id <halo_id> --parent_id <parent_id> [options]
"""

import sys
import os
import argparse
import numpy as np
import random
import h5py
from pathlib import Path

# Add KDE sampler to path
sys.path.append("/oak/stanford/orgs/kipac/users/caganze/kde_sampler")
sys.path.append("/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow")

try:
    from kde_sampler import M4Kernel, KDESampler
except ImportError:
    print("Warning: kde_sampler not found. Please ensure the path is correct.")
    # Fallback imports for development/testing
    pass

# Import existing utilities
try:
    from symlib_utils import get_halo_data
except ImportError:
    print("Warning: symlib_utils not found. Using placeholder function.")
    def get_halo_data(halo_name, suite='eden'):
        """Placeholder function for development"""
        return {
            'parentid': np.random.randint(0, 1000, 10000),
            'pos3': np.random.randn(10000, 3) * 100,
            'vel3': np.random.randn(10000, 3) * 50,
            'mass': np.random.lognormal(0, 1, 10000) * 1e8
        }

try:
    from kroupa_imf import kroupa_masses_fast
except ImportError:
    print("Warning: kroupa_imf not found. Using built-in implementation.")


class KDESamplerKroupa(KDESampler):
    """
    KDESampler that smooths a density field represented by a set of points
    of variable mass. This smooth density field can be sampled efficiently
    with Kroupa IMF mass distribution.
    """
    
    def __init__(self, kernel, k, x, m, method="split", gen=None, boxsize=None, mass_range=(0.1, 120)):
        """
        Initialize KDE sampler with Kroupa IMF capabilities.
        
        Parameters:
        -----------
        kernel : Kernel object
            The kernel to use for density estimation
        k : int
            Number of neighbors for KDE
        x : array-like
            Particle positions/phase-space coordinates
        m : array-like
            Particle masses
        method : str
            Sampling method ("split" or other)
        gen : numpy.random.Generator
            Random number generator
        boxsize : float
            Size of simulation box
        mass_range : tuple
            Mass range for IMF sampling (min_mass, max_mass)
        """
        if gen is None:
            gen = np.random.default_rng()
        
        super().__init__(kernel, k, x, m, method=method, gen=gen, boxsize=boxsize)
        
        # IMF sampling range
        self.mass_range = mass_range
        self.counts = None

    def _kroupa_masses_fast(self, nsample):
        """
        Generate Kroupa IMF masses using piecewise power law.
        
        Parameters:
        -----------
        nsample : int
            Number of mass samples to generate
            
        Returns:
        --------
        masses : array
            Array of stellar masses following Kroupa IMF
        """
        def sample_from_powerlaw(alpha, xmin, xmax, nsample):
            """Sample from power law distribution"""
            u = np.random.uniform(0, 1, nsample)
            if alpha == -1:
                return xmin * (xmax/xmin)**u
            else:
                return (u * (xmax**(alpha + 1) - xmin**(alpha + 1)) + xmin**(alpha + 1))**(1/(alpha + 1))
        
        m_range = self.mass_range
        
        # Kroupa IMF segments
        m0 = sample_from_powerlaw(-0.3, xmin=0.03, xmax=0.08, nsample=int(nsample))
        m1 = sample_from_powerlaw(-1.3, xmin=0.08, xmax=0.5, nsample=int(nsample))
        m2 = sample_from_powerlaw(-2.3, xmin=0.5, xmax=100, nsample=int(nsample))
        
        # Combine all mass segments
        m = np.concatenate([m0, m1, m2]).flatten()
        
        # Apply mass range filter
        mask = np.logical_and(m > m_range[0], m < m_range[1])
        masses = np.random.choice(m[mask], int(nsample))
        
        return masses
    
    def _estimate_ntot(self, M_target, ntest=1_000_000):
        """
        Fast approximate N needed to reach total mass M_target using Kroupa mass sampler.
        Returns Poisson draws to account for stochasticity.
        
        Parameters:
        -----------
        M_target : float
            Target total mass
        ntest : int
            Number of test samples for mean mass estimation
            
        Returns:
        --------
        ntot : int
            Estimated number of particles needed (Poisson draw)
        """
        # Step 1: measure mean mass with a large test draw
        mtest = self._kroupa_masses_fast(ntest)
        mean_mass = mtest.mean()
        
        # Step 2: estimate N ~ M_target / mean_mass
        ntot = int(np.round(M_target / mean_mass))
        
        return self.gen.poisson(ntot)
    
    def sample(self, frac=1.0):
        """
        Sample the KDE's underlying PDF with Kroupa IMF mass distribution.
        
        Parameters:
        -----------
        frac : float
            Fraction of samples from total number calculated based on IMF.
            Can be used for upsampling (frac > 1) or downsampling (frac < 1).
            
        Returns:
        --------
        out : array
            Sampled phase-space coordinates
        """
        n_bins = len(self.mass_bins) 
        
        # Compute total mass in each mass bin
        m_tot = [np.nansum(self.m[self.starts[i]: self.ends[i]]) for i in range(n_bins)]
        
        # Compute total counts in each mass bin using Kroupa IMF
        counts = [self._estimate_ntot(int(m * frac)) for m in m_tot]
        self.counts = counts
        
        # Sample phase-space in each mass bin
        ends = np.cumsum(counts)
        starts = ends - counts
        out = np.zeros((np.sum(counts), self.kernel.dim))
        
        for i in range(n_bins):
            # Start and end of each mass bin in the input array
            start_i, end_i = self.starts[i], self.ends[i]
            # Start and end of each mass bin in the output array
            start_o, end_o = starts[i], ends[i]

            idx = self.gen.integers(start_i, end_i, counts[i])
            xr = self.kernel.sample(self.gen, counts[i])

            for dim in range(self.kernel.dim):
                offset = xr[:, dim] * self.r[idx]
                out[start_o:end_o, dim] = self.x[idx, dim] + offset

        # Shuffle the output to remove any ordering bias
        self.gen.shuffle(out)
        return out


def load_halo_data(halo_id, parent_id, suite='eden'):
    """
    Load halo data and extract particles for specified parent ID.
    
    Parameters:
    -----------
    halo_id : str
        Halo identifier (e.g., 'Halo939')
    parent_id : int
        Parent ID to select particles
    suite : str
        Simulation suite name
        
    Returns:
    --------
    phase_space : array
        6D phase space coordinates (position + velocity)
    masses : array
        Particle masses
    """
    print(f"Loading halo data for {halo_id}, parent_id={parent_id}")
    
    # Load halo data
    nimbus = get_halo_data(halo_id, suite=suite)
    
    # Select particles with specified parent ID
    id_select = nimbus['parentid'] == parent_id
    
    # Combine position and velocity into 6D phase space
    nimbus_6d = np.hstack([nimbus['pos3'][id_select], nimbus['vel3'][id_select]])
    
    # Scale masses (divide by 100 as in original code)
    nimbus_masses = nimbus['mass'][id_select] / 100
    
    print(f"Selected {len(nimbus_masses)} particles")
    print(f"Mass range: {nimbus_masses.min():.2e} - {nimbus_masses.max():.2e}")
    
    return nimbus_6d, nimbus_masses


def generate_kde_samples(phase_space, masses, n_neighbors=64, sample_fraction=1.0, 
                        mass_range=(0.1, 120), seed=42):
    """
    Generate samples using KDE with Kroupa IMF.
    
    Parameters:
    -----------
    phase_space : array
        6D phase space coordinates
    masses : array
        Particle masses
    n_neighbors : int
        Number of neighbors for KDE
    sample_fraction : float
        Fraction of samples to generate
    mass_range : tuple
        Mass range for IMF sampling
    seed : int
        Random seed
        
    Returns:
    --------
    samples : array
        Generated samples from KDE
    """
    print(f"Generating KDE samples with {n_neighbors} neighbors, fraction={sample_fraction}")
    
    # Initialize random generator
    gen = np.random.default_rng(seed=seed)
    
    # Initialize M4 kernel for 6D phase space
    kernel = M4Kernel(dim=6)
    
    # Create KDE sampler with Kroupa IMF
    kroupa_sampler = KDESamplerKroupa(
        kernel, 
        n_neighbors, 
        phase_space, 
        masses, 
        method="split",
        gen=gen,
        mass_range=mass_range
    )
    
    # Generate samples
    kroupa_samples = kroupa_sampler.sample(frac=sample_fraction)
    
    print(f"Generated {len(kroupa_samples)} samples")
    
    return kroupa_samples


def save_samples(samples, output_path, halo_id, parent_id):
    """
    Save generated samples to HDF5 file in format compatible with training scripts.
    
    Parameters:
    -----------
    samples : array
        Generated samples
    output_path : str or Path
        Output file path
    halo_id : str
        Halo identifier
    parent_id : int
        Parent ID
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving samples to {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Save samples as positions and velocities
        f.create_dataset('positions', data=samples[:, :3])
        f.create_dataset('velocities', data=samples[:, 3:])
        
        # Save metadata
        f.attrs['halo_id'] = halo_id
        f.attrs['parent_id'] = parent_id
        f.attrs['n_samples'] = len(samples)
        f.attrs['method'] = 'kde_kroupa'
        f.attrs['generated_by'] = 'train_kde_flows_conditional.py'
    
    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='KDE-based Conditional Flow Training')
    
    parser.add_argument('--halo_id', type=str, required=True,
                       help='Halo identifier (e.g., Halo939)')
    parser.add_argument('--parent_id', type=int, required=True,
                       help='Parent ID to select particles')
    parser.add_argument('--suite', type=str, default='eden',
                       help='Simulation suite name')
    parser.add_argument('--n_neighbors', type=int, default=64,
                       help='Number of neighbors for KDE')
    parser.add_argument('--sample_fraction', type=float, default=1.0,
                       help='Fraction of samples to generate')
    parser.add_argument('--mass_range', type=float, nargs=2, default=[0.1, 120],
                       help='Mass range for IMF sampling')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./kde_samples',
                       help='Output directory for samples')
    
    args = parser.parse_args()
    
    try:
        # Load halo data
        phase_space, masses = load_halo_data(args.halo_id, args.parent_id, args.suite)
        
        # Generate KDE samples
        samples = generate_kde_samples(
            phase_space, 
            masses, 
            n_neighbors=args.n_neighbors,
            sample_fraction=args.sample_fraction,
            mass_range=tuple(args.mass_range),
            seed=args.seed
        )
        
        # Save samples
        output_filename = f"kde_samples_{args.halo_id}_pid{args.parent_id}.h5"
        output_path = Path(args.output_dir) / output_filename
        
        save_samples(samples, output_path, args.halo_id, args.parent_id)
        
        print(f"KDE sampling completed successfully!")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during KDE sampling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
