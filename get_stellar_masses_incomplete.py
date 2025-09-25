#!/usr/bin/env python3
"""
Calculate stellar masses for incomplete KDE particles
Reads particle_list_kde_incomplete.txt and extracts stellar mass for each particle
"""

import sys
import os
import csv
import argparse
from typing import Dict, List, Tuple
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from symlib_utils import load_particle_data, validate_symlib_environment
    print("✅ Successfully imported symlib utilities")
except ImportError as e:
    print(f"❌ Failed to import symlib utilities: {e}")
    print("   Make sure you're running on Sherlock with symlib environment loaded")
    sys.exit(1)

def read_incomplete_particles(filename: str) -> List[Dict]:
    """
    Read the incomplete particle list and parse the data
    
    Args:
        filename: Path to the incomplete particle list file
        
    Returns:
        List of dictionaries with particle information
    """
    particles = []
    
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        print("   Make sure you've run ./filter_completed_kde.sh first")
        return particles
    
    print(f"📖 Reading incomplete particles from: {filename}")
    
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                # Parse CSV format: PID,HALO_ID,SUITE,OBJECT_COUNT,SIZE_CATEGORY
                parts = line.split(',')
                if len(parts) >= 3:
                    particle = {
                        'pid': int(parts[0]),
                        'halo_id': parts[1].strip(),
                        'suite': parts[2].strip(),
                        'object_count': int(parts[3]) if len(parts) > 3 and parts[3].strip() else 0,
                        'size_category': parts[4].strip() if len(parts) > 4 else 'unknown'
                    }
                    particles.append(particle)
                else:
                    print(f"⚠️ Skipping malformed line {line_num}: {line}")
                    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return []
    
    print(f"✅ Successfully read {len(particles)} incomplete particles")
    return particles

def get_stellar_mass_for_particle(particle: Dict) -> Tuple[float, bool]:
    """
    Get stellar mass for a single particle using symlib
    
    Args:
        particle: Dictionary with particle information
        
    Returns:
        Tuple of (stellar_mass, success_flag)
    """
    try:
        # Load particle data using symlib
        data, metadata = load_particle_data(particle['halo_id'], particle['pid'], particle['suite'])
        
        # Extract stellar mass from metadata
        stellar_mass = metadata.get('stellar_mass', 0.0)
        
        print(f"✅ {particle['halo_id']} PID {particle['pid']}: {stellar_mass:.2e} M☉")
        return stellar_mass, True
        
    except Exception as e:
        print(f"❌ Failed to get stellar mass for {particle['halo_id']} PID {particle['pid']}: {e}")
        return 0.0, False

def main():
    """Main function to process incomplete particles and extract stellar masses"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Calculate stellar masses for incomplete KDE particles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 get_stellar_masses_incomplete.py
  python3 get_stellar_masses_incomplete.py particle_list_flow_incomplete.txt
  python3 get_stellar_masses_incomplete.py my_particles.txt
        """
    )
    parser.add_argument(
        'filename',
        nargs='?',
        default='particle_list_kde_incomplete.txt',
        help='Path to incomplete particle list file (default: particle_list_kde_incomplete.txt)'
    )
    
    args = parser.parse_args()
    
    print("🌟 STELLAR MASS CALCULATOR FOR INCOMPLETE PARTICLES")
    print("=" * 60)
    print(f"📋 Input file: {args.filename}")
    print()
    
    # Validate symlib environment
    if not validate_symlib_environment():
        print("❌ Symlib environment not available")
        print("   Make sure you're running on Sherlock with the correct environment")
        sys.exit(1)
    
    print("✅ Symlib environment validated")
    print()
    
    # Read incomplete particles
    particles = read_incomplete_particles(args.filename)
    
    if not particles:
        print("❌ No incomplete particles found")
        sys.exit(1)
    
    print(f"🔍 Processing {len(particles)} incomplete particles...")
    print()
    
    # Process each particle and collect results
    results = []
    successful_count = 0
    failed_count = 0
    
    for i, particle in enumerate(particles):
        print(f"📊 Processing {i+1}/{len(particles)}: {particle['halo_id']} PID {particle['pid']}")
        
        stellar_mass, success = get_stellar_mass_for_particle(particle)
        
        result = {
            'pid': particle['pid'],
            'halo_id': particle['halo_id'],
            'suite': particle['suite'],
            'stellar_mass': stellar_mass,
            'success': success,
            'object_count': particle['object_count'],
            'size_category': particle['size_category']
        }
        results.append(result)
        
        if success:
            successful_count += 1
        else:
            failed_count += 1
        
        # Progress indicator every 50 particles
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(particles)} ({successful_count} successful, {failed_count} failed)")
        
        print()
    
    # Summary statistics
    print("📊 STELLAR MASS SUMMARY")
    print("=" * 40)
    print(f"✅ Successfully processed: {successful_count}")
    print(f"❌ Failed to process: {failed_count}")
    print(f"📊 Total particles: {len(particles)}")
    print()
    
    # Filter successful results and sort by stellar mass
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        stellar_masses = [r['stellar_mass'] for r in successful_results]
        
        print(f"📈 Stellar mass statistics (successful particles only):")
        print(f"   Minimum: {min(stellar_masses):.2e} M☉")
        print(f"   Maximum: {max(stellar_masses):.2e} M☉")
        print(f"   Mean: {np.mean(stellar_masses):.2e} M☉")
        print(f"   Median: {np.median(stellar_masses):.2e} M☉")
        print(f"   Total: {sum(stellar_masses):.2e} M☉")
        print()
        
        # Sort by stellar mass (descending)
        successful_results.sort(key=lambda x: x['stellar_mass'], reverse=True)
        
        print("🌟 INCOMPLETE PARTICLES RANKED BY STELLAR MASS")
        print("=" * 60)
        print(f"{'Rank':<6} {'Halo ID':<15} {'PID':<8} {'Suite':<10} {'Stellar Mass (M☉)':<18} {'Size':<10}")
        print("-" * 80)
        
        for i, result in enumerate(successful_results):
            print(f"{i+1:<6} {result['halo_id']:<15} {result['pid']:<8} {result['suite']:<10} {result['stellar_mass']:<18.2e} {result['size_category']:<10}")
        
        print()
        print("📝 Raw data (copy-paste friendly):")
        print("PID, Halo_ID, Suite, Stellar_Mass_MSun, Size_Category")
        for result in successful_results:
            print(f"{result['pid']}, {result['halo_id']}, {result['suite']}, {result['stellar_mass']:.2e}, {result['size_category']}")
    
    else:
        print("❌ No particles were successfully processed")

if __name__ == "__main__":
    main()
