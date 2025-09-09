#!/usr/bin/env python3
"""
Test script for multiple particles to verify H5 reading works across different parent IDs
"""

import argparse
import h5py
import numpy as np
import os
import sys

# Set CUDA paths for TensorFlow before importing
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0'

def read_h5_to_dict(filename):
    """Reads an HDF5 file and returns a dictionary with all datasets."""
    data = {}

    def recursively_load(h5obj, prefix=''):
        for key in h5obj:
            item = h5obj[key]
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(item, h5py.Dataset):
                data[path] = item[()]
            elif isinstance(item, h5py.Group):
                recursively_load(item, path)

    with h5py.File(filename, 'r') as f:
        recursively_load(f)
    return data

def test_multiple_particles(halo_id="023", particle_pids=[1, 2, 3, 4, 5], n_subsample=500):
    """Test reading H5 file and extracting multiple particle PIDs"""
    
    print(f"🧪 TESTING MULTIPLE PARTICLES FOR HALO {halo_id}")
    print("=" * 60)
    print(f"Target PIDs: {particle_pids}")
    print(f"Subsample per PID: {n_subsample}")
    print()
    
    # Search for halo file
    search_paths = [
        f"../milkyway-eden-mocks/eden_scaled_Halo{halo_id}_sunrot0_0kpc200kpcoriginal_particles.h5",
        f"../milkyway-eden-mocks/eden_scaled_Halo{halo_id}_m_sunrot0_0kpc200kpcoriginal_particles.h5",
        f"../milkywaymocks/symphony_scaled_Halo{halo_id}_sunrot0_0kpc200kpcoriginal_particles.h5",
        f"../milkyway-hr-mocks/symphonyHR_scaled_Halo{halo_id}_sunrot0_0kpc200kpcoriginal_particles.h5"
    ]
    
    halo_file = None
    for path in search_paths:
        if os.path.exists(path):
            halo_file = path
            break
    
    if halo_file is None:
        print("❌ No halo file found. Searched:")
        for path in search_paths:
            print(f"   {path}")
        return False
    
    print(f"✅ Found halo file: {halo_file}")
    
    try:
        # Load H5 file
        print(f"\n📋 Loading H5 file...")
        star_particles = read_h5_to_dict(halo_file)
        
        print(f"   Loaded datasets: {list(star_particles.keys())}")
        print(f"   Total particles: {len(star_particles['pos3'])}")
        
        if 'parentid' in star_particles:
            unique_pids = np.unique(star_particles['parentid'])
            print(f"   Available parent IDs: {len(unique_pids)} total")
            print(f"   PID range: {unique_pids.min()} to {unique_pids.max()}")
            
            # Test each requested PID
            successful_extractions = 0
            total_particles_extracted = 0
            all_data = []
            
            for pid in particle_pids:
                print(f"\n🔍 Processing parent ID {pid}:")
                
                # Extract particles for this PID
                bool_mask = star_particles['parentid'] == pid
                n_particles = np.sum(bool_mask)
                
                if n_particles > 0:
                    print(f"   ✅ Found {n_particles:,} particles")
                    
                    # Extract 6D phase space data
                    particle_data = np.hstack([
                        star_particles['pos3'][bool_mask], 
                        star_particles['vel3'][bool_mask]
                    ])
                    
                    print(f"   ✅ 6D data shape: {particle_data.shape}")
                    print(f"   ✅ Data type: {particle_data.dtype}")
                    
                    # Subsample if needed
                    if len(particle_data) > n_subsample:
                        indices = np.random.choice(len(particle_data), n_subsample, replace=False)
                        subsampled_data = particle_data[indices]
                        print(f"   ✅ Subsampled to {n_subsample} particles")
                    else:
                        subsampled_data = particle_data
                        print(f"   ✅ Using all {len(particle_data)} particles")
                    
                    # Basic statistics
                    pos_mean = np.mean(subsampled_data[:, :3], axis=0)
                    vel_mean = np.mean(subsampled_data[:, 3:], axis=0)
                    print(f"   📊 Position center: [{pos_mean[0]:.2f}, {pos_mean[1]:.2f}, {pos_mean[2]:.2f}] kpc")
                    print(f"   📊 Velocity center: [{vel_mean[0]:.2f}, {vel_mean[1]:.2f}, {vel_mean[2]:.2f}] km/s")
                    
                    successful_extractions += 1
                    total_particles_extracted += len(subsampled_data)
                    all_data.append(subsampled_data)
                    
                else:
                    print(f"   ❌ No particles found with parent ID {pid}")
                    if pid not in unique_pids:
                        print(f"   💡 PID {pid} not in dataset (available: {unique_pids[:10]}...)")
            
            # Summary
            print(f"\n📊 SUMMARY:")
            print(f"   Requested PIDs: {len(particle_pids)}")
            print(f"   Successful extractions: {successful_extractions}")
            print(f"   Total particles extracted: {total_particles_extracted:,}")
            print(f"   Average particles per PID: {total_particles_extracted/max(1, successful_extractions):.0f}")
            
            if successful_extractions > 0:
                # Test basic TensorFlow operations
                print(f"\n🧪 Testing TensorFlow with combined data:")
                try:
                    import tensorflow as tf
                    print(f"   ✅ TensorFlow {tf.__version__} imported")
                    
                    # Combine all extracted data
                    combined_data = np.vstack(all_data)
                    print(f"   ✅ Combined data shape: {combined_data.shape}")
                    
                    # Convert to TensorFlow tensor
                    tf_data = tf.constant(combined_data, dtype=tf.float32)
                    print(f"   ✅ TensorFlow tensor created: {tf_data.shape}")
                    
                    # Basic operations
                    mean_vals = tf.reduce_mean(tf_data, axis=0)
                    std_vals = tf.math.reduce_std(tf_data, axis=0)
                    
                    print(f"   ✅ Mean values computed: {mean_vals.shape}")
                    print(f"   ✅ Std values computed: {std_vals.shape}")
                    print(f"   ✅ TensorFlow operations successful!")
                    
                    # GPU check
                    gpu_available = tf.test.is_gpu_available()
                    gpu_devices = tf.config.list_physical_devices('GPU')
                    print(f"   📱 GPU available: {gpu_available}")
                    print(f"   📱 GPU devices: {len(gpu_devices)}")
                    
                    return True
                    
                except ImportError as e:
                    print(f"   ❌ TensorFlow import failed: {e}")
                    return False
                except Exception as e:
                    print(f"   ❌ TensorFlow test failed: {e}")
                    print(f"   💡 Error type: {type(e).__name__}")
                    return False
            else:
                print(f"\n❌ No particles successfully extracted")
                return False
                
        else:
            print(f"   ❌ No 'parentid' field found")
            return False
                
    except Exception as e:
        print(f"❌ Error reading H5 file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test H5 file reading for multiple particles")
    parser.add_argument("--halo_id", default="023", help="Halo ID (e.g., 023, 088, 852)")
    parser.add_argument("--particle_pids", nargs='+', type=int, default=[1, 2, 3, 4, 5], 
                       help="Parent IDs to extract (e.g., 1 2 3 4 5)")
    parser.add_argument("--n_subsample", type=int, default=500, 
                       help="Number of particles to subsample per PID")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting multi-particle H5 read test...")
    print(f"   Halo ID: {args.halo_id}")
    print(f"   Target PIDs: {args.particle_pids}")
    print(f"   Subsample per PID: {args.n_subsample}")
    print()
    
    success = test_multiple_particles(args.halo_id, args.particle_pids, args.n_subsample)
    
    if success:
        print(f"\n🎉 SUCCESS! Multi-particle extraction working:")
        print(f"   ✅ H5 file reading works")
        print(f"   ✅ Multiple particle extraction works") 
        print(f"   ✅ Subsampling works")
        print(f"   ✅ TensorFlow integration works")
        print(f"\n🚀 Ready for parallel training across PIDs!")
        print(f"💡 Next: Run generate_parallel_scripts.py to create training jobs")
    else:
        print(f"\n❌ Test failed. Check particle IDs and H5 file structure.")

if __name__ == "__main__":
    main()
