#!/usr/bin/env python3
"""Create example data for testing TFP flows"""

import numpy as np
import h5py
import argparse

def create_example_data(n_samples=50000, output_file="data/example_data.h5"):
    """Create synthetic 6D phase space data"""
    print(f"Creating {n_samples:,} synthetic particles...")
    
    np.random.seed(42)
    
    # Positions (kpc) - spherical distribution
    r = np.random.exponential(scale=8.0, size=n_samples)  # Exponential disk
    theta = np.random.uniform(0, np.pi, n_samples)
    phi = np.random.uniform(0, 2*np.pi, n_samples)
    
    pos_x = r * np.sin(theta) * np.cos(phi)
    pos_y = r * np.sin(theta) * np.sin(phi)
    pos_z = r * np.cos(theta) * 0.5  # Flattened disk
    
    # Velocities (km/s) - with realistic dispersions
    vel_dispersion = 30.0 + 40.0 * np.exp(-r/5.0)  # Higher dispersion in center
    
    vel_x = np.random.normal(0, vel_dispersion, n_samples)
    vel_y = np.random.normal(10.0, vel_dispersion, n_samples)  # Rotation
    vel_z = np.random.normal(0, vel_dispersion * 0.5, n_samples)  # Lower z-dispersion
    
    # Save to HDF5
    pos3 = np.column_stack([pos_x, pos_y, pos_z])
    vel3 = np.column_stack([vel_x, vel_y, vel_z])
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('pos3', data=pos3, compression='gzip')
        f.create_dataset('vel3', data=vel3, compression='gzip')
        
        # Add metadata
        f.attrs['n_samples'] = n_samples
        f.attrs['description'] = 'Synthetic galactic disk particles'
        f.attrs['position_units'] = 'kpc'
        f.attrs['velocity_units'] = 'km/s'
        f.attrs['created_by'] = 'create_example_data.py'
    
    print(f"✅ Created {output_file}")
    print(f"   Samples: {n_samples:,}")
    print(f"   Position range: [{pos3.min():.1f}, {pos3.max():.1f}] kpc")
    print(f"   Velocity range: [{vel3.min():.1f}, {vel3.max():.1f}] km/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--output", default="data/example_data.h5")
    args = parser.parse_args()
    
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    create_example_data(args.n_samples, args.output)
