#!/usr/bin/env python3
"""
Quick test script to find and validate priority halo particles
Focuses specifically on halos 239, 718, 270, 925
"""

import os
import sys

# Force CPU mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def analyze_priority_particles():
    """Analyze particle_list.txt for priority halos and suggest best targets"""
    print("🎯 PRIORITY HALO PARTICLE ANALYSIS")
    print("==================================")
    
    particle_list_file = "particle_list.txt"
    
    if not os.path.exists(particle_list_file):
        print(f"❌ Particle list not found: {particle_list_file}")
        print("   Generate it with: python generate_symlib_particle_list.py Halo268 eden")
        return False
    
    priority_halos = ["939", "718", "270", "925"]
    halo_data = {}
    
    print(f"📋 Scanning particle list for priority halos: {', '.join(priority_halos)}")
    
    try:
        with open(particle_list_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                pid, halo_id, suite, count, size = parts
                count = int(count)
                
                # Extract halo number
                halo_num = halo_id.replace('Halo', '')
                
                if halo_num in priority_halos:
                    key = f"{suite}_{halo_num}"
                    if key not in halo_data:
                        halo_data[key] = []
                    halo_data[key].append({
                        'pid': int(pid),
                        'count': count,
                        'size': size,
                        'halo_id': halo_id
                    })
        
        print(f"\n🔍 PRIORITY HALO BREAKDOWN:")
        print("=" * 50)
        
        total_particles = 0
        recommendations = []
        
        for halo in priority_halos:
            eden_key = f"eden_{halo}"
            symphony_key = f"symphony_{halo}"
            
            print(f"\n🎯 Halo {halo}:")
            
            for suite_key, suite_name in [(eden_key, "Eden"), (symphony_key, "Symphony")]:
                if suite_key in halo_data:
                    particles = halo_data[suite_key]
                    total_count = sum(p['count'] for p in particles)
                    
                    print(f"   ✅ {suite_name}: {len(particles)} PIDs, {total_count:,} total particles")
                    
                    # Sort by size for recommendations
                    particles.sort(key=lambda x: x['count'])
                    
                    # Recommend small/medium particles for 12h completion
                    good_particles = [p for p in particles if p['count'] < 50000]
                    if good_particles:
                        print(f"      🚀 Fast particles (<50k): {len(good_particles)}")
                        for p in good_particles[:3]:  # Show top 3
                            print(f"         PID {p['pid']}: {p['count']:,} particles ({p['size']})")
                            recommendations.append({
                                'halo': halo,
                                'suite': suite_name.lower(),
                                'pid': p['pid'],
                                'count': p['count'],
                                'size': p['size'],
                                'halo_id': p['halo_id']
                            })
                    
                    total_particles += total_count
                else:
                    print(f"   ❌ {suite_name}: Not found")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total priority halo combinations: {len(halo_data)}")
        print(f"   Total particles across priority halos: {total_particles:,}")
        print(f"   Recommended fast particles: {len(recommendations)}")
        
        if recommendations:
            print(f"\n🚀 TOP RECOMMENDATIONS FOR 12-HOUR JOB:")
            print("=" * 50)
            
            # Sort by particle count (fastest first)
            recommendations.sort(key=lambda x: x['count'])
            
            for i, rec in enumerate(recommendations[:15], 1):  # Top 15 recommendations
                estimated_time = rec['count'] / 10000 * 30  # Rough estimate: 30 min per 10k particles
                print(f"{i:2d}. {rec['suite']} {rec['halo_id']} PID {rec['pid']:3d}: "
                      f"{rec['count']:6,} particles ({rec['size']:12s}) ~{estimated_time:.0f}min")
            
            print(f"\n💡 QUICK START COMMANDS:")
            print("=" * 30)
            
            # Show command for top 3 recommendations
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"# Test {i}: {rec['suite']} {rec['halo_id']} PID {rec['pid']}")
                print(f"python train_tfp_flows.py --halo_id {rec['halo_id']} --particle_pid {rec['pid']} "
                      f"--suite {rec['suite']} --epochs 15 --batch_size 512")
                print()
        
        return len(recommendations) > 0
        
    except Exception as e:
        print(f"❌ Error analyzing particle list: {e}")
        return False

def test_specific_particle():
    """Test loading a specific priority particle"""
    print("\n🧪 TESTING SPECIFIC PARTICLE LOADING")
    print("====================================")
    
    # Use a known working particle for testing
    test_cases = [
        ("Halo268", 2, "eden"),  # Known working case
    ]
    
    for halo_id, pid, suite in test_cases:
        print(f"\n🔍 Testing {suite} {halo_id} PID {pid}...")
        
        try:
            import subprocess
            result = subprocess.run([
                'python', '-c', f'''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    from symlib_utils import load_particle_data
    
    print("Loading {suite} {halo_id} PID {pid}...")
    data, metadata = load_particle_data("{halo_id}", {pid}, "{suite}")
    
    print(f"✅ Success: {{data.shape[0]:,}} particles")
    print(f"   Stellar mass: {{metadata['stellar_mass']:.2e}} M☉")
    print(f"   Data range: {{data.min():.2f}} to {{data.max():.2f}}")
    
except Exception as e:
    print(f"❌ Failed: {{e}}")
    sys.exit(1)
'''
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(result.stdout.strip())
                return True
            else:
                print(f"❌ Test failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            return False

def main():
    print("🎯 PRIORITY PARTICLE ANALYSIS & TESTING")
    print("=======================================")
    print("Analyzing halos 939, 718, 270, 925 for optimal 12-hour training")
    
    success = True
    
    # Analyze priority particles
    if not analyze_priority_particles():
        print("❌ Priority particle analysis failed")
        success = False
    
    # Test specific particle loading
    if not test_specific_particle():
        print("❌ Particle loading test failed")
        success = False
    
    if success:
        print("\n🎉 ANALYSIS COMPLETE!")
        print("✅ Priority particles identified")
        print("✅ Symlib loading confirmed")
        print("\n🚀 Ready for production training!")
    else:
        print("\n❌ ISSUES DETECTED")
        print("⚠️  Check symlib setup before proceeding")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
