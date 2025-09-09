#!/usr/bin/env python3
"""
Generate particle lists for priority halos: 239, 718, 270, 925
Uses symlib to scan available halos and create comprehensive particle lists
"""

import os
import sys
import numpy as np
from pathlib import Path

# Force CPU mode to avoid GPU issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def get_symlib_directory(suite, halo_id):
    """Get the symlib directory for a given suite and halo"""
    import symlib
    
    base_dir = "/oak/stanford/orgs/kipac/users/phil1/simulations/ZoomIns/"
    
    if suite.lower() == 'eden':
        return symlib.get_host_directory(base_dir, "EDEN_MilkyWay_8K", halo_id)
    elif suite.lower() == 'symphony':
        return symlib.get_host_directory(base_dir, "SymphonyMilkyWay", halo_id)
    elif suite.lower() == 'symphony-hr':
        return symlib.get_host_directory(base_dir, "SymphonyMilkyWayHR", halo_id)
    elif suite.lower() == 'mwest':
        return symlib.get_host_directory(base_dir, "MWest", halo_id)
    else:
        raise ValueError(f"Unknown suite: {suite}")

def check_halo_availability(halo_id, suite):
    """Check if a halo is available in a given suite"""
    try:
        sim_dir = get_symlib_directory(suite, halo_id)
        print(f"   {suite}: {sim_dir}")
        
        # Try to load the simulation
        import symlib
        sim = symlib.simulation.Simulation(suite)
        
        # Try to get halo data
        try:
            data = sim.get_halo(int(halo_id))
            n_particles = len(data['parentid']) if 'parentid' in data else 0
            unique_pids = np.unique(data['parentid']) if 'parentid' in data else []
            
            print(f"      ✅ Available: {n_particles:,} total particles, {len(unique_pids)} unique PIDs")
            return True, n_particles, len(unique_pids)
        except Exception as e:
            print(f"      ❌ Data access failed: {e}")
            return False, 0, 0
            
    except Exception as e:
        print(f"      ❌ Directory not found: {e}")
        return False, 0, 0

def generate_particle_list_for_halo(halo_id, suite, output_file=None):
    """Generate particle list for a specific halo and suite"""
    if output_file is None:
        output_file = f"particle_list_{suite}_halo{halo_id}.txt"
    
    print(f"\n🔧 GENERATING PARTICLE LIST: {suite} Halo{halo_id}")
    print("=" * 50)
    
    try:
        # Import symlib and load data
        import symlib
        from symlib_utils import load_particle_data
        
        # Try to load the full halo data
        sim = symlib.simulation.Simulation(suite)
        res = sim.get_halo(int(halo_id))
        
        print(f"✅ Loaded {suite} Halo{halo_id}")
        print(f"   Total particles: {len(res['parentid']):,}")
        
        # Get unique particle IDs
        unique_pids = np.unique(res['parentid'])
        print(f"   Unique PIDs: {len(unique_pids)}")
        
        # Create particle list entries
        particle_entries = []
        
        for pid in unique_pids:
            mask = np.array(res['parentid']) == pid
            n_particles = len(np.array(res['parentid'])[mask])
            
            # Classify particle size
            if n_particles > 100000:
                size_category = "Large"
            elif n_particles > 50000:
                size_category = "Medium-Large"
            elif n_particles > 5000:
                size_category = "Medium"
            else:
                size_category = "Small"
            
            # Format: PID,HALO_ID,SUITE,OBJECT_COUNT,SIZE_CATEGORY
            entry = f"{pid},Halo{halo_id},{suite},{n_particles},{size_category}"
            particle_entries.append(entry)
            
            print(f"   PID {pid}: {n_particles:,} particles ({size_category})")
        
        # Write to file
        print(f"\n📝 Writing {len(particle_entries)} entries to {output_file}...")
        with open(output_file, 'w') as f:
            for entry in particle_entries:
                f.write(entry + '\n')
        
        print(f"✅ Particle list saved: {output_file}")
        
        # Summary
        size_counts = {}
        for entry in particle_entries:
            size_cat = entry.split(',')[4]
            size_counts[size_cat] = size_counts.get(size_cat, 0) + 1
        
        print(f"📊 Size distribution:")
        for size_cat, count in sorted(size_counts.items()):
            print(f"   {size_cat}: {count} PIDs")
        
        return True, len(particle_entries)
        
    except Exception as e:
        print(f"❌ Failed to generate particle list: {e}")
        import traceback
        traceback.print_exc()
        return False, 0

def scan_priority_halos():
    """Scan all priority halos across different suites"""
    priority_halos = ["239", "718", "270", "925"]
    suites = ["eden", "symphony"]  # Start with main suites
    
    print("🔍 SCANNING PRIORITY HALOS")
    print("=" * 40)
    print(f"Priority halos: {', '.join(priority_halos)}")
    print(f"Suites: {', '.join(suites)}")
    print()
    
    available_halos = []
    
    for halo_id in priority_halos:
        print(f"\n🎯 Halo {halo_id}:")
        
        for suite in suites:
            is_available, n_particles, n_pids = check_halo_availability(halo_id, suite)
            
            if is_available:
                available_halos.append({
                    'halo_id': halo_id,
                    'suite': suite,
                    'n_particles': n_particles,
                    'n_pids': n_pids
                })
    
    print(f"\n📊 AVAILABLE PRIORITY HALOS:")
    print("=" * 40)
    
    if available_halos:
        for halo in available_halos:
            print(f"✅ {halo['suite']} Halo{halo['halo_id']}: {halo['n_pids']} PIDs, {halo['n_particles']:,} particles")
    else:
        print("❌ No priority halos found!")
    
    return available_halos

def generate_all_priority_lists():
    """Generate particle lists for all available priority halos"""
    print("🚀 PRIORITY HALO PARTICLE LIST GENERATOR")
    print("=" * 50)
    
    # First, scan to see what's available
    available_halos = scan_priority_halos()
    
    if not available_halos:
        print("\n❌ No priority halos available - check symlib setup")
        return False
    
    print(f"\n🔧 GENERATING PARTICLE LISTS")
    print("=" * 30)
    
    all_entries = []
    successful_halos = []
    
    for halo in available_halos:
        success, n_entries = generate_particle_list_for_halo(
            halo['halo_id'], 
            halo['suite']
        )
        
        if success:
            successful_halos.append(halo)
            # Read the generated file and add to combined list
            filename = f"particle_list_{halo['suite']}_halo{halo['halo_id']}.txt"
            try:
                with open(filename, 'r') as f:
                    entries = f.readlines()
                    all_entries.extend([e.strip() for e in entries if e.strip()])
            except Exception as e:
                print(f"⚠️  Could not read {filename}: {e}")
    
    # Create combined particle list
    if all_entries:
        combined_file = "particle_list_priority_halos.txt"
        print(f"\n📝 Creating combined list: {combined_file}")
        
        with open(combined_file, 'w') as f:
            for entry in all_entries:
                f.write(entry + '\n')
        
        print(f"✅ Combined particle list created: {len(all_entries)} total particles")
        print(f"   File: {combined_file}")
        
        # Update the main particle list
        main_file = "particle_list.txt"
        print(f"\n🔄 Updating main particle list: {main_file}")
        
        # Backup existing file
        if os.path.exists(main_file):
            backup_file = f"{main_file}.backup"
            os.rename(main_file, backup_file)
            print(f"   Backed up existing file to: {backup_file}")
        
        # Copy combined list to main file
        os.rename(combined_file, main_file)
        print(f"   Updated: {main_file}")
        
        return True
    else:
        print("\n❌ No particle lists generated successfully")
        return False

def main():
    print("🎯 PRIORITY HALO PARTICLE LIST GENERATOR")
    print("=======================================")
    print("Generating particle lists for halos: 239, 718, 270, 925")
    
    try:
        success = generate_all_priority_lists()
        
        if success:
            print("\n🎉 SUCCESS!")
            print("✅ Priority halo particle lists generated")
            print("✅ Main particle_list.txt updated")
            print("\n🚀 Next steps:")
            print("   1. Run: python test_priority_particles.py")
            print("   2. Run: sbatch brute_force_gpu_job.sh")
        else:
            print("\n❌ FAILED!")
            print("⚠️  Check symlib setup and halo availability")
        
        return success
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

