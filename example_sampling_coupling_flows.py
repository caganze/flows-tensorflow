#!/usr/bin/env python3
"""
Example script showing how to sample from trained coupling flow models
"""

import subprocess
import os

def example_sampling():
    """Show examples of how to sample from trained coupling flow models"""
    
    print("🎯 Coupling Flow Sampling Examples")
    print("=" * 60)
    
    # Example 1: Basic sampling with uniform mass bin distribution
    print("\n📊 Example 1: Basic Sampling (Uniform Mass Bins)")
    print("Command:")
    cmd1 = [
        "python", "sample_coupling_flow.py",
        "--base_dir", "coupling_output/eden/halo718/pid1",
        "--halo_id", "Halo718",
        "--particle_pid", "1",
        "--n_samples", "10000",
        "--seed", "42"
    ]
    print(" ".join(cmd1))
    print("Output: samples_Halo718_1.npz")
    
    # Example 2: Sampling with custom mass bin distribution
    print("\n🎯 Example 2: Custom Mass Bin Distribution")
    print("Command:")
    cmd2 = [
        "python", "sample_coupling_flow.py",
        "--base_dir", "coupling_output/eden/halo718/pid12",
        "--halo_id", "Halo718", 
        "--particle_pid", "12",
        "--n_samples", "50000",
        "--mass_bin_probs", "0.1,0.2,0.3,0.4",  # Custom distribution
        "--seed", "123"
    ]
    print(" ".join(cmd2))
    print("Output: samples_Halo718_12.npz")
    
    # Example 3: Sampling with Kroupa IMF distribution
    print("\n🌟 Example 3: Kroupa IMF Mass Distribution")
    print("Command:")
    cmd3 = [
        "python", "sample_coupling_flow.py",
        "--base_dir", "coupling_output/eden/halo718/pid1",
        "--halo_id", "Halo718",
        "--particle_pid", "1", 
        "--n_samples", "100000",
        "--use_kroupa",  # Use Kroupa IMF
        "--seed", "456"
    ]
    print(" ".join(cmd3))
    print("Output: samples_Halo718_1.npz (with Kroupa mass distribution)")
    
    # Example 4: Sampling with custom output file
    print("\n💾 Example 4: Custom Output File")
    print("Command:")
    cmd4 = [
        "python", "sample_coupling_flow.py",
        "--base_dir", "coupling_output/eden/halo718/pid12",
        "--halo_id", "Halo718",
        "--particle_pid", "12",
        "--n_samples", "25000",
        "--output_file", "my_custom_samples.npz",
        "--seed", "789"
    ]
    print(" ".join(cmd4))
    print("Output: my_custom_samples.npz")
    
    print("\n💡 Key Features:")
    print("  - Automatic model loading and weight restoration")
    print("  - Preprocessing statistics loading and unstandardization")
    print("  - Flexible mass bin distribution options")
    print("  - Kroupa IMF integration for realistic mass distributions")
    print("  - Comprehensive sample statistics and validation")
    
    print("\n📁 Output File Contents:")
    print("  - samples: Generated samples in physical units")
    print("  - conditions: Mass bin conditions used")
    print("  - mass_bin_probs: Mass bin probability distribution")
    print("  - n_samples: Number of samples generated")
    print("  - seed: Random seed used")
    
    print("\n🔧 Usage Tips:")
    print("  - Use --use_kroupa for astrophysically realistic mass distributions")
    print("  - Use --mass_bin_probs for custom mass bin weighting")
    print("  - Larger n_samples for better statistics")
    print("  - Set --seed for reproducible results")
    print("  - Check output statistics to validate sample quality")
    
    return True

if __name__ == "__main__":
    example_sampling()







