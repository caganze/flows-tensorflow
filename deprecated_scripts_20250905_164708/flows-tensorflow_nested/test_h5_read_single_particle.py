#!/usr/bin/env python3
"""
Tiny test script to verify:
1. Can read .h5 file from ../milkyway* folders
2. Can extract single particle (PID)
3. Can subsample to 1000 particles
4. Can run basic TensorFlow operation
"""

import argparse
import h5py
import numpy as np
import os
import sys

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

def extract_particle_data(data_dict: dict, particle_pid: int) -> np.ndarray:
    """
    Extract particle data for a specific PID from H5 data dictionary
    
    Args:
        data_dict: Dictionary containing H5 datasets
        particle_pid: Parent ID to extract
        
    Returns:
        Array of 6D particle data (pos3 + vel3) for the specified PID
    """
    # Check if we have the required datasets
    if 'parentid' not in data_dict:
        raise ValueError("Dataset 'parentid' not found in H5 data")
    
    if 'pos3' not in data_dict or 'vel3' not in data_dict:
        raise ValueError("Datasets 'pos3' and 'vel3' not found in H5 data")
    
    # Create boolean mask for this particle
    bool_mask = data_dict['parentid'] == particle_pid
    n_particles = np.sum(bool_mask)
    
    if n_particles == 0:
        print(f"⚠️  No particles found for PID {particle_pid}")
        return np.empty((0, 6))
    
    print(f"🎯 Found {n_particles} particles for PID {particle_pid}")
    
    # Extract particle data using the exact method from test_h5_read
    particle_data = np.hstack([data_dict['pos3'][bool_mask], data_dict['vel3'][bool_mask]])
    
    print(f"✅ Extracted particle data shape: {particle_data.shape}")
    return particle_data

def test_h5_read(halo_id="852", particle_pid=1, n_subsample=1000):
    """Test reading H5 file and extracting particle data using your method"""
    
    print(f"🧪 TESTING H5 READ FOR HALO {halo_id}, PARTICLE PID {particle_pid}")
    print("=" * 60)
    
    # Search for halo file in different locations - with actual naming patterns
    search_paths = [
        f"../milkyway-eden-mocks/eden_scaled_Halo{halo_id}_sunrot0_0kpc200kpcoriginal_particles.h5",
        f"../milkyway-eden-mocks/eden_scaled_Halo{halo_id}_m_sunrot0_0kpc200kpcoriginal_particles.h5",  # Some have _m suffix
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
        print("\n💡 Available files:")
        for base in ["../milkyway-eden-mocks", "../milkywaymocks"]:
            if os.path.exists(base):
                print(f"   {base}:")
                try:
                    for item in os.listdir(base):
                        if item.startswith("Halo"):
                            print(f"     {item}/")
                except:
                    print(f"     (cannot list directory)")
        return False
    
    print(f"✅ Found halo file: {halo_file}")
    
    try:
        # Use your method to read the H5 file
        print(f"\n📋 Loading H5 file using your method...")
        star_particles = read_h5_to_dict(halo_file)
        
        print(f"   File: {halo_file}")
        print(f"   Loaded datasets: {list(star_particles.keys())}")
        
        # Show dataset info
        if 'pos3' in star_particles and 'vel3' in star_particles:
            print(f"   Total particles in dataset: {len(star_particles['pos3'])}")
            print(f"   Position shape: {star_particles['pos3'].shape}")
            print(f"   Velocity shape: {star_particles['vel3'].shape}")
            print(f"   Position range: [{star_particles['pos3'].min():.2f}, {star_particles['pos3'].max():.2f}]")
            print(f"   Velocity range: [{star_particles['vel3'].min():.2f}, {star_particles['vel3'].max():.2f}]")
            
            if 'parentid' in star_particles:
                unique_pids = np.unique(star_particles['parentid'])
                print(f"   Available parent IDs: {unique_pids[:10]}... (showing first 10)")
                print(f"   Total unique parent IDs: {len(unique_pids)}")
            
            # Extract particles for specific parent ID using your exact method
            print(f"\n🔍 Extracting particles with parent ID {particle_pid} using your method:")
            
            if 'parentid' in star_particles:
                # Your exact method: bool1 = star_particles['parentid'] == 1
                bool_mask = star_particles['parentid'] == particle_pid
                n_particles = np.sum(bool_mask)
                
                if n_particles > 0:
                    print(f"   Found {n_particles} particles with parent ID {particle_pid}")
                    
                    # Your exact method: x1 = hstack([star_particles['pos3'][bool1], star_particles['vel3'][bool1]])
                    particle_data = np.hstack([star_particles['pos3'][bool_mask], star_particles['vel3'][bool_mask]])
                    print(f"   Combined 6D phase space shape: {particle_data.shape}")
                    print(f"   Data type: {particle_data.dtype}")
                    print(f"   Sample values (first 3 particles): {particle_data[:3]}")
                    
                    # Subsample to requested size
                    if len(particle_data) > n_subsample:
                        indices = np.random.choice(len(particle_data), n_subsample, replace=False)
                        subsampled_data = particle_data[indices]
                        print(f"   Subsampled to {n_subsample} particles: shape {subsampled_data.shape}")
                    else:
                        subsampled_data = particle_data
                        print(f"   Using all {len(particle_data)} particles (less than {n_subsample})")
                    
                    # Test basic TensorFlow operation
                    print(f"\n🧪 Testing TensorFlow on particle data:")
                    try:
                        import tensorflow as tf
                        print(f"   ✅ TensorFlow import successful: {tf.__version__}")
                        
                        # Check GPU availability
                        gpu_available = tf.test.is_gpu_available()
                        gpu_devices = tf.config.list_physical_devices('GPU')
                        print(f"   GPU available: {gpu_available}")
                        print(f"   GPU devices: {gpu_devices}")
                        
                        # Convert to TensorFlow tensor
                        tf_data = tf.constant(subsampled_data, dtype=tf.float32)
                        print(f"   ✅ TensorFlow tensor created: {tf_data.shape}")
                        
                        # Basic operations
                        mean_vals = tf.reduce_mean(tf_data, axis=0)
                        std_vals = tf.math.reduce_std(tf_data, axis=0)
                        
                        print(f"   ✅ Mean values: {mean_vals.numpy()}")
                        print(f"   ✅ Std values: {std_vals.numpy()}")
                        print(f"   ✅ TensorFlow operations successful!")
                        
                        return True
                        
                    except ImportError as e:
                        print(f"   ❌ TensorFlow import failed: {e}")
                        print(f"   💡 Make sure TensorFlow is installed: pip install tensorflow")
                        return False
                    except Exception as e:
                        print(f"   ❌ TensorFlow test failed: {e}")
                        print(f"   💡 Error type: {type(e).__name__}")
                        import traceback
                        print(f"   💡 Full traceback:")
                        traceback.print_exc()
                        return False
                
                else:
                    print(f"   ❌ No particles found with parent ID {particle_pid}")
                    print(f"   Available parent IDs: {unique_pids[:20]}")
                    return False
            else:
                print(f"   ❌ No 'parentid' field found")
                return False
        else:
            print(f"   ❌ Expected 'pos3' and 'vel3' datasets not found")
            print(f"   Available datasets: {list(star_particles.keys())}")
            return False
                
    except Exception as e:
        print(f"❌ Error reading H5 file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test H5 file reading for single particle")
    parser.add_argument("--halo_id", default="023", help="Halo ID (e.g., 023, 088, 852)")
    parser.add_argument("--particle_pid", type=int, default=1, help="Parent ID to extract")
    parser.add_argument("--n_subsample", type=int, default=1000, help="Number of particles to subsample")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting H5 read test...")
    print(f"   Halo ID: {args.halo_id}")
    print(f"   Particle PID: {args.particle_pid}")
    print(f"   Subsample size: {args.n_subsample}")
    print()
    
    success = test_h5_read(args.halo_id, args.particle_pid, args.n_subsample)
    
    if success:
        print(f"\n🎉 SUCCESS! Ready for full pipeline:")
        print(f"   ✅ H5 file reading works")
        print(f"   ✅ Particle extraction works") 
        print(f"   ✅ Subsampling works")
        print(f"   ✅ TensorFlow integration works")
        print(f"\n🚀 Next: Update generate_parallel_scripts.py and deploy!")
    else:
        print(f"\n❌ Test failed. Need to debug H5 file structure.")
        print(f"💡 Check the H5 file format and adjust extraction logic.")

if __name__ == "__main__":
    main()
