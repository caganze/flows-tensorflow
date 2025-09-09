#!/usr/bin/env python3
import h5py
import numpy as np

h5_file = '/oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/eden_scaled_Halo570_sunrot90_0kpc200kpcoriginal_particles.h5'

with h5py.File(h5_file, 'r') as f:
    print("=== H5 File Analysis ===")
    print(f"File: {h5_file}")
    print(f"Keys: {list(f.keys())}")
    
    if 'parentid' in f:
        pids = f['parentid'][:]
        unique_pids, counts = np.unique(pids, return_counts=True)
        
        print(f"\nTotal particles: {len(pids)}")
        print(f"Unique PIDs: {len(unique_pids)}")
        print(f"PID range: {unique_pids.min()} to {unique_pids.max()}")
        
        # Show largest particle groups
        sorted_idx = np.argsort(counts)[::-1]
        print(f"\nTop 10 most populous PIDs:")
        for i in range(min(10, len(unique_pids))):
            pid = unique_pids[sorted_idx[i]]
            count = counts[sorted_idx[i]]
            print(f"  PID {pid}: {count:,} particles")
            
        print(f"\nRecommended test PIDs (largest): {unique_pids[sorted_idx[:5]]}")
