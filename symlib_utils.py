#!/usr/bin/env python3
"""
Symlib utilities for TensorFlow Probability flow training
Helper functions for symlib integration
"""

import numpy as np
import symlib
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import h5py
from typing import Dict, Tuple, Any
import os

# Symlib simulation parameters
cosmo = FlatLambdaCDM(H0=70, Om0=0.286)

def get_halo_data(halo_id: str, suite: str = 'eden', gal_halo=symlib.DWARF_GALAXY_HALO_MODEL_NO_UM) -> Dict[str, Any]:
    """
    Get halo data from symlib simulation
    
    Args:
        halo_id: Halo ID (e.g., 'Halo268')
        suite: Simulation suite ('eden', 'mwest', 'symphony', 'symphony-hr')
        gal_halo: Galaxy halo model
        
    Returns:
        Dictionary with particle data organized by parentid
    """
    # Get simulation directory
    sim_dir = None
    
    if suite.lower() == 'mwest':
        sim_dir = symlib.get_host_directory("/oak/stanford/orgs/kipac/users/phil1/simulations/ZoomIns/", "MWest", halo_id)
    elif suite.lower() == 'eden':
        sim_dir = symlib.get_host_directory("/oak/stanford/orgs/kipac/users/phil1/simulations/ZoomIns/", "EDEN_MilkyWay_8K", halo_id)
    elif suite.lower() == 'symphony':
        sim_dir = symlib.get_host_directory("/oak/stanford/orgs/kipac/users/phil1/simulations/ZoomIns/", "SymphonyMilkyWay", halo_id)
    elif suite.lower() == 'symphony-hr':
        sim_dir = symlib.get_host_directory("/oak/stanford/orgs/kipac/users/phil1/simulations/ZoomIns/", "SymphonyMilkyWayHR", halo_id)
    else:
        raise ValueError(f"Unknown suite: {suite}")
    
    # Reading particle data
    snapshot = 235
    params = symlib.simulation_parameters(sim_dir)
    part = symlib.Particles(sim_dir)
    sim_p = part.read(snapshot, mode='stars')
    stars, gals, ranks = symlib.tag_stars(sim_dir, gal_halo)
    
    # Extract stellar properties
    Fe_H = np.concatenate([stars[idx]['Fe_H'] for idx in range(1, len(stars))])
    mp_star = np.concatenate([stars[idx]['mp'] for idx in range(1, len(stars))])
    scale_forms = np.concatenate([stars[idx]['a_form'] for idx in range(1, len(stars))])
    t_form = (cosmo.age(1/scale_forms-1)).to(u.yr)
    
    # Extract positions and velocities
    pos = np.concatenate([sim_p[idx]["x"] for idx in range(1, len(sim_p))], axis=0)
    vel = np.concatenate([sim_p[idx]["v"] for idx in range(1, len(sim_p))], axis=0)
    
    # Create owner array (parentid)
    n_halo = len(sim_p)
    n_part = [len(halo) for halo in sim_p]
    owner = np.concatenate([np.ones(n_part[i], dtype=int)*i for i in range(1, n_halo)])
    
    # Validation
    assert(len(Fe_H) == len(mp_star))
    assert(len(Fe_H) == len(pos))
    assert(len(pos) == len(vel))
    assert(len(owner) == len(vel))
    
    # Organize data dictionary
    nparts = len(pos)
    halo_data = {
        'pos3': pos,  # position in kpc
        'vel3': vel,  # velocity in km/s
        'mass': mp_star,  # mass in solar masses
        'age': np.log10((cosmo.age(0).to(u.yr)-t_form).value),  # log age in yr
        'feh': Fe_H,
        'a_form': scale_forms,
        'parentid': owner,
        'gals': gals,
        'n_total_particles': nparts,
        'halo_id': halo_id,
        'suite': suite
    }
    
    return halo_data

def load_particle_data(halo_id: str, particle_pid: int, suite: str = 'eden') -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load specific particle data for training
    Replaces the old load_astrophysical_data function
    
    Args:
        halo_id: Halo ID (e.g., 'Halo268')
        particle_pid: Particle ID to extract
        suite: Simulation suite
        
    Returns:
        Tuple of (data_array, metadata_dict)
    """
    print(f"📊 Loading symlib data for {halo_id} PID {particle_pid} from {suite}")
    
    # Get halo data
    halo_data = get_halo_data(halo_id, suite)
    
    # Extract specific particle
    mask = halo_data['parentid'] == particle_pid
    
    if not np.any(mask):
        raise ValueError(f"❌ No data found for particle PID {particle_pid} in {halo_id}")
    
    # Extract 7D data (3D position + 3D velocity + mass)
    pos = halo_data['pos3'][mask]  # Shape: (N, 3)
    vel = halo_data['vel3'][mask]  # Shape: (N, 3)
    mass = halo_data['mass'][mask]  # Shape: (N,)
    
    # Combine into 7D array: [x, y, z, vx, vy, vz, mass]
    data = np.column_stack([pos, vel, mass])  # Shape: (N, 7)
    
    print(f"✅ Extracted {data.shape[0]} particles for PID {particle_pid}")
    print(f"   Position range: [{pos.min():.2f}, {pos.max():.2f}] kpc")
    print(f"   Velocity range: [{vel.min():.2f}, {vel.max():.2f}] km/s")
    
    # Create metadata
    particle_masses = mass  # Use the mass data we already extracted
    total_stellar_mass = np.sum(particle_masses)
    
    metadata = {
        'halo_id': halo_id,
        'suite': suite,
        'particle_pid': particle_pid,
        'n_particles': data.shape[0],
        'stellar_mass': total_stellar_mass,
        'mean_stellar_mass': np.mean(particle_masses),
        'pos_range': [pos.min(), pos.max()],
        'vel_range': [vel.min(), vel.max()],
        'data_shape': data.shape,
        'feature_names': ['x', 'y', 'z', 'vx', 'vy', 'vz', 'mass'],
        'position_units': 'kpc',
        'velocity_units': 'km/s',
        'mass_units': 'solar_masses'
    }
    
    print(f"📋 Metadata: {data.shape[0]} particles, {total_stellar_mass:.2e} M☉ total mass")
    
    return data.astype(np.float32), metadata

def get_output_paths(halo_id: str, particle_pid: int, suite: str = 'eden', 
                    base_dir: str = '/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow') -> Dict[str, str]:
    """
    Get standardized output paths for symlib-based training
    
    Args:
        halo_id: Halo ID
        particle_pid: Particle ID
        suite: Simulation suite
        base_dir: Base output directory
        
    Returns:
        Dictionary with output paths
    """
    # Create directory structure: base_dir/tfp_output/{trained_flows,samples}/suite/halo_id/
    trained_flows_dir = f"{base_dir}/tfp_output/trained_flows/{suite}/{halo_id.lower()}"
    samples_dir = f"{base_dir}/tfp_output/samples/{suite}/{halo_id.lower()}"
    
    paths = {
        'trained_flows_dir': trained_flows_dir,
        'samples_dir': samples_dir,
        'model_file': f"{trained_flows_dir}/model_pid{particle_pid}.npz",
        'preprocessing_file': f"{trained_flows_dir}/model_pid{particle_pid}_preprocessing.npz",
        'results_file': f"{trained_flows_dir}/model_pid{particle_pid}_results.json",
        'samples_file': f"{samples_dir}/model_pid{particle_pid}_samples.npz"
    }
    
    return paths

def validate_symlib_environment():
    """
    Check if symlib and dependencies are available
    """
    try:
        import symlib
        import astropy
        print("✅ Symlib environment validated")
        return True
    except ImportError as e:
        print(f"❌ Symlib environment error: {e}")
        return False

if __name__ == "__main__":
    # Test the utilities
    print("🧪 Testing symlib utilities...")
    
    if validate_symlib_environment():
        print("✅ Environment check passed")
    else:
        print("❌ Environment check failed")
        exit(1)
    
    # Test data loading (if arguments provided)
    import sys
    if len(sys.argv) >= 3:
        halo_id = sys.argv[1]
        particle_pid = int(sys.argv[2])
        suite = sys.argv[3] if len(sys.argv) > 3 else 'eden'
        
        try:
            data, metadata = load_particle_data(halo_id, particle_pid, suite)
            print("✅ Data loading test passed")
        except Exception as e:
            print(f"❌ Data loading test failed: {e}")
    else:
        print("💡 To test data loading: python symlib_utils.py Halo268 2 eden")
