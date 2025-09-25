#!/usr/bin/env python3
"""
Analyze particle size distributions across different PIDs in a halo
Helps understand the data distribution for better parameter selection
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from symlib_utils import load_particle_data, validate_symlib_environment

def analyze_particle_sizes(halo_id, suite='eden', max_pid=50):
    """Analyze particle size distribution across PIDs"""
    
    print(f"📊 Analyzing particle sizes for {halo_id} in {suite}")
    print("=" * 60)
    
    if not validate_symlib_environment():
        print("❌ Symlib environment not available")
        return
    
    results = []
    
    for pid in range(1, max_pid + 1):
        try:
            data, metadata = load_particle_data(halo_id, pid, suite)
            
            if len(data) > 0:
                n_particles = len(data)
                stellar_mass = metadata.get('stellar_mass', 0)
                
                # Calculate additional metrics
                if data.shape[1] > 6:  # If mass column exists
                    masses = data[:, 6]
                    mass_range = [np.min(masses), np.max(masses)]
                    mean_mass = np.mean(masses)
                    median_mass = np.median(masses)
                else:
                    mass_range = [0, 0]
                    mean_mass = 0
                    median_mass = 0
                
                # Phase space metrics
                positions = data[:, :3]
                velocities = data[:, 3:6]
                
                pos_std = np.std(positions, axis=0)
                vel_std = np.std(velocities, axis=0)
                
                results.append({
                    'pid': pid,
                    'n_particles': n_particles,
                    'stellar_mass': stellar_mass,
                    'mass_min': mass_range[0],
                    'mass_max': mass_range[1],
                    'mean_mass': mean_mass,
                    'median_mass': median_mass,
                    'pos_std_x': pos_std[0],
                    'pos_std_y': pos_std[1],
                    'pos_std_z': pos_std[2],
                    'vel_std_x': vel_std[0],
                    'vel_std_y': vel_std[1],
                    'vel_std_z': vel_std[2],
                    'pos_std_avg': np.mean(pos_std),
                    'vel_std_avg': np.mean(vel_std)
                })
                
                print(f"PID {pid:2d}: {n_particles:6d} particles, {stellar_mass:8.2e} M☉")
                
        except Exception as e:
            print(f"PID {pid:2d}: Error - {str(e)}")
            continue
    
    if not results:
        print("❌ No valid particle data found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Create size categories based on particle count
    df['size_category'] = pd.cut(
        df['n_particles'], 
        bins=[0, 500, 2000, 5000, 10000, 20000, float('inf')],
        labels=['tiny', 'very_small', 'small', 'medium', 'large', 'huge']
    )
    
    # Create mass categories based on stellar mass
    df['mass_category'] = pd.cut(
        df['stellar_mass'], 
        bins=[0, 1e4, 1e5, 1e6, 1e7, float('inf')],
        labels=['low_mass', 'medium_mass', 'high_mass', 'very_high_mass', 'extreme_mass']
    )
    
    # Print summary statistics
    print(f"\n📈 SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total PIDs analyzed: {len(df)}")
    print(f"Total particles: {df['n_particles'].sum():,}")
    print(f"Particle count range: {df['n_particles'].min():,} - {df['n_particles'].max():,}")
    print(f"Stellar mass range: {df['stellar_mass'].min():.2e} - {df['stellar_mass'].max():.2e} M☉")
    
    print(f"\n📊 SIZE DISTRIBUTION:")
    size_counts = df['size_category'].value_counts().sort_index()
    for category, count in size_counts.items():
        pids = df[df['size_category'] == category]['pid'].tolist()
        print(f"  {category:12s}: {count:2d} PIDs {pids}")
    
    print(f"\n📊 MASS DISTRIBUTION:")
    mass_counts = df['mass_category'].value_counts().sort_index()
    for category, count in mass_counts.items():
        pids = df[df['mass_category'] == category]['pid'].tolist()
        print(f"  {category:15s}: {count:2d} PIDs {pids}")
    
    # Recommendations for parameter selection
    print(f"\n🎯 PARAMETER RECOMMENDATIONS:")
    print("=" * 60)
    
    for category in ['tiny', 'very_small', 'small', 'medium', 'large', 'huge']:
        category_data = df[df['size_category'] == category]
        if len(category_data) > 0:
            n_particles_range = (category_data['n_particles'].min(), category_data['n_particles'].max())
            pids = category_data['pid'].tolist()
            
            # Recommend parameters based on size
            if category == 'tiny':
                rec = "Hidden: [16,16], Epochs: 50, LR: 1e-3, Batch: 256"
            elif category == 'very_small':
                rec = "Hidden: [32,32], Epochs: 100, LR: 1e-4, Batch: 512"
            elif category == 'small':
                rec = "Hidden: [32,32], Epochs: 100, LR: 1e-4, Batch: 512"
            elif category == 'medium':
                rec = "Hidden: [64,64], Epochs: 150, LR: 5e-5, Batch: 1024"
            elif category == 'large':
                rec = "Hidden: [64,64], Epochs: 200, LR: 5e-5, Batch: 1024"
            else:  # huge
                rec = "Hidden: [64,64,64], Epochs: 300, LR: 1e-5, Batch: 2048"
            
            print(f"{category:12s} ({n_particles_range[0]:,}-{n_particles_range[1]:,} particles): {rec}")
            print(f"             PIDs: {pids}")
    
    # Save results
    output_file = f"particle_size_analysis_{halo_id}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")
    
    return df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze particle size distributions")
    parser.add_argument("--halo_id", required=True, help="Halo ID (e.g., Halo268)")
    parser.add_argument("--suite", default="eden", help="Simulation suite name")
    parser.add_argument("--max_pid", type=int, default=50, help="Maximum PID to analyze")
    
    args = parser.parse_args()
    
    df = analyze_particle_sizes(args.halo_id, args.suite, args.max_pid)
    
    if df is not None:
        print(f"\n💡 Next steps:")
        print(f"1. Run quick_parameter_test.py with recommended parameters")
        print(f"2. Focus on PIDs with similar particle counts")
        print(f"3. Use the size categories to group similar training runs")

if __name__ == "__main__":
    main()
