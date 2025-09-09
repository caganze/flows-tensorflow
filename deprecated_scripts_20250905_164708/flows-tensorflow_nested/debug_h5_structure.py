#!/usr/bin/env python3

"""
Debug H5 file structure to understand the correct particle size detection
"""

import h5py
import numpy as np
import sys

def analyze_h5_structure(h5_file_path):
    print(f"🔍 Analyzing H5 file: {h5_file_path}")
    print("=" * 80)
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            print("📋 Top-level keys:")
            for key in f.keys():
                print(f"  - {key}: {type(f[key])}")
                if hasattr(f[key], 'shape'):
                    print(f"    Shape: {f[key].shape}")
                elif hasattr(f[key], 'keys'):
                    print(f"    Sub-keys: {list(f[key].keys())}")
            print()
            
            # Check PartType1 structure (common for simulation data)
            if 'PartType1' in f:
                print("🔍 PartType1 structure:")
                part1 = f['PartType1']
                for key in part1.keys():
                    print(f"  - {key}: shape={part1[key].shape}, dtype={part1[key].dtype}")
                print()
                
                # Check ParticleIDs specifically
                if 'ParticleIDs' in part1:
                    pids = part1['ParticleIDs'][:]
                    print(f"📊 ParticleIDs analysis:")
                    print(f"  Total entries: {len(pids)}")
                    print(f"  Unique PIDs: {len(np.unique(pids))}")
                    print(f"  PID range: {pids.min()} → {pids.max()}")
                    print(f"  First 20 PIDs: {np.unique(pids)[:20]}")
                    print()
                    
                    # Test specific PID counts
                    test_pids = [1, 2, 3, 23, 88, 188]
                    print(f"🧪 Object counts for test PIDs:")
                    for test_pid in test_pids:
                        if test_pid <= pids.max():
                            count = np.sum(pids == test_pid)
                            print(f"  PID {test_pid}: {count:,} objects")
                    print()
                
                # Check if there are other relevant arrays
                if 'Coordinates' in part1:
                    coords = part1['Coordinates']
                    print(f"📍 Coordinates: shape={coords.shape}")
                    
                if 'Velocities' in part1:
                    vels = part1['Velocities']
                    print(f"🚀 Velocities: shape={vels.shape}")
                    
                if 'Masses' in part1:
                    masses = part1['Masses']
                    print(f"⚖️  Masses: shape={masses.shape}")
                    
            # Check other possible structures
            elif 'particles' in f:
                print("🔍 particles structure:")
                particles = f['particles']
                for key in particles.keys():
                    print(f"  - {key}: shape={particles[key].shape}, dtype={particles[key].dtype}")
                    
            elif 'data' in f:
                print("🔍 data structure:")
                data = f['data']
                for key in data.keys():
                    print(f"  - {key}: shape={data[key].shape}, dtype={data[key].dtype}")
                    
            else:
                print("❓ Unknown H5 structure - checking all top-level datasets:")
                def print_dataset_info(name, obj):
                    if hasattr(obj, 'shape'):
                        print(f"  Dataset {name}: shape={obj.shape}, dtype={obj.dtype}")
                        
                f.visititems(print_dataset_info)
                
    except Exception as e:
        print(f"❌ Error analyzing H5 file: {e}")
        return False
        
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        h5_file = sys.argv[1]
    else:
        # Default to your eden file
        h5_file = "/oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/eden_scaled_Halo570_sunrot90_0kpc200kpcoriginal_particles.h5"
    
    analyze_h5_structure(h5_file)
