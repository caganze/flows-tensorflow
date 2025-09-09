#!/usr/bin/env python3
"""
Generate particle list from symlib simulation data
Replaces the old H5-based generate_particle_list.sh
"""

import numpy as np
import matplotlib.pyplot as plt
import symlib
import matplotlib as mpl
import numpy as np
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
import h5py
import glob
import astropy.units as u
import sys
import argparse
from pathlib import Path

# Symlib simulation parameters
cosmo = FlatLambdaCDM(H0=70, Om0=0.286)

def get_halo(hlid, suite='eden', gal_halo=symlib.DWARF_GALAXY_HALO_MODEL_NO_UM):
    """
    Get halo data from symlib simulation
    
    Args:
        hlid: Halo ID (e.g., 'Halo268')
        suite: Simulation suite ('eden', 'mwest', 'symphony', 'symphony-hr')
        gal_halo: Galaxy halo model
        
    Returns:
        Dictionary with particle data organized by parentid
    """
    # Get simulation directory
    catalog_dir = None
    sim_dir = None
    
    if suite.lower() == 'mwest':
        sim_dir = symlib.get_host_directory("/oak/stanford/orgs/kipac/users/phil1/simulations/ZoomIns/", "MWest", hlid)
    elif suite.lower() == 'eden':
        sim_dir = symlib.get_host_directory("/oak/stanford/orgs/kipac/users/phil1/simulations/ZoomIns/", "EDEN_MilkyWay_8K", hlid)
    elif suite.lower() == 'symphony':
        sim_dir = symlib.get_host_directory("/oak/stanford/orgs/kipac/users/phil1/simulations/ZoomIns/", "SymphonyMilkyWay", hlid)
    elif suite.lower() == 'symphony-hr':
        sim_dir = symlib.get_host_directory("/oak/stanford/orgs/kipac/users/phil1/simulations/ZoomIns/", "SymphonyMilkyWayHR", hlid)
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
    
    # Organize data
    nparts = len(pos)
    p = {}
    p['pos3'] = pos  # position in kpc
    p['vel3'] = vel  # velocity in km/s
    p['mass'] = mp_star  # mass in solar masses
    p['age'] = np.log10((cosmo.age(0).to(u.yr)-t_form).value)  # log age in yr
    p['feh'] = Fe_H
    p['a_form'] = scale_forms
    p['parentid'] = owner
    p['gals'] = gals
    
    return p

def generate_particle_list(halo_id, suite='eden', output_file='particle_list.txt'):
    """
    Generate particle list file from symlib data
    
    Args:
        halo_id: Halo ID (e.g., 'Halo268')
        suite: Simulation suite
        output_file: Output file path
    """
    print(f"🔍 GENERATING PARTICLE LIST")
    print(f"============================")
    print(f"Halo ID: {halo_id}")
    print(f"Suite: {suite}")
    print(f"Output: {output_file}")
    print()
    
    try:
        # Get halo data
        print("📊 Loading halo data from symlib...")
        res = get_halo(halo_id, suite)
        
        # Get unique particle IDs and count objects
        unique_pids = np.unique(res['parentid'])
        print(f"✅ Found {len(unique_pids)} unique particle IDs")
        print()
        
        # Create particle list entries
        particle_entries = []
        
        for pid in unique_pids:
            mask = np.array(res['parentid']) == pid
            n_particles = len(np.array(res['parentid'])[mask])
            
            # Classify size category
            if n_particles >= 100000:
                size_category = "Large"
            elif n_particles >= 50000:
                size_category = "Medium-Large"
            elif n_particles >= 5000:
                size_category = "Medium"
            else:
                size_category = "Small"
            
            # Format: PID,HALO_ID,SUITE,OBJECT_COUNT,SIZE_CATEGORY
            entry = f"{pid},{halo_id},{suite},{n_particles},{size_category}"
            particle_entries.append(entry)
            
            print(f"pid {pid} ########### Nparticles {n_particles} ({size_category})")
        
        # Write to file
        print(f"\n📝 Writing {len(particle_entries)} entries to {output_file}...")
        with open(output_file, 'w') as f:
            for entry in particle_entries:
                f.write(entry + '\n')
        
        print(f"✅ Particle list generated successfully!")
        print(f"📋 Summary:")
        print(f"   Total PIDs: {len(unique_pids)}")
        print(f"   Total particles: {len(res['parentid']):,}")
        print(f"   Size distribution:")
        
        # Count by size category
        size_counts = {}
        for entry in particle_entries:
            size_cat = entry.split(',')[4]
            size_counts[size_cat] = size_counts.get(size_cat, 0) + 1
        
        for size_cat, count in sorted(size_counts.items()):
            print(f"     {size_cat}: {count} PIDs")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating particle list: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Generate particle list from symlib simulation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_symlib_particle_list.py Halo268 --suite eden
  python generate_symlib_particle_list.py Halo023 --suite symphony --output custom_list.txt
        """
    )
    
    parser.add_argument("halo_id", help="Halo ID (e.g., Halo268)")
    parser.add_argument("--suite", default="eden", 
                       choices=["eden", "mwest", "symphony", "symphony-hr"],
                       help="Simulation suite (default: eden)")
    parser.add_argument("--output", default="particle_list.txt",
                       help="Output file path (default: particle_list.txt)")
    
    args = parser.parse_args()
    
    success = generate_particle_list(args.halo_id, args.suite, args.output)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
