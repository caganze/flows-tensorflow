#!/usr/bin/env python3
"""
Example usage of KDE-based flow training

This script demonstrates how to use the new KDE sampling approach
as a drop-in replacement for symlib-based particle reading.
"""

import sys
import numpy as np
from pathlib import Path

# Import KDE functionality
from train_kde_flows_conditional import (
    load_halo_data, 
    generate_kde_samples, 
    save_samples,
    KDESamplerKroupa
)

def example_basic_kde_sampling():
    """Basic example of KDE sampling"""
    print("=" * 60)
    print("EXAMPLE 1: Basic KDE Sampling")
    print("=" * 60)
    
    # Parameters (modify these for your use case)
    halo_id = 'Halo939'
    parent_id = 123  # Replace with your parent ID
    suite = 'eden'
    
    try:
        # Load halo data
        print(f"Loading halo data: {halo_id}, parent_id={parent_id}")
        phase_space, masses = load_halo_data(halo_id, parent_id, suite)
        
        # Generate KDE samples
        print("Generating KDE samples...")
        samples = generate_kde_samples(
            phase_space, 
            masses, 
            n_neighbors=64,
            sample_fraction=1.0,
            mass_range=(0.1, 120),
            seed=42
        )
        
        # Save samples
        output_dir = Path('./example_output')
        output_file = output_dir / f"example_kde_samples_{halo_id}_pid{parent_id}.h5"
        save_samples(samples, output_file, halo_id, parent_id)
        
        print(f"Success! Saved {len(samples)} samples to {output_file}")
        
        # Print some statistics
        print("\nSample Statistics:")
        print(f"  Position range: {samples[:, :3].min():.2f} to {samples[:, :3].max():.2f}")
        print(f"  Velocity range: {samples[:, 3:].min():.2f} to {samples[:, 3:].max():.2f}")
        
    except Exception as e:
        print(f"Error in basic KDE sampling: {e}")


def example_wrapper_usage():
    """Example of using the wrapper script"""
    print("=" * 60)
    print("EXAMPLE 2: Wrapper Script Usage")
    print("=" * 60)
    
    print("To use the KDE approach with existing shell scripts, you can:")
    print()
    print("1. Direct usage:")
    print("   python kde_flow_wrapper.py --halo_id Halo939 --parent_id 123 --flow_type tfp")
    print()
    print("2. Integration with existing scripts:")
    print("   # Modify your shell script to call kde_flow_wrapper.py instead")
    print("   # The arguments remain the same!")
    print()
    print("3. Batch processing:")
    print("   # Your existing shell scripts should work with minimal changes")
    print("   # Just replace the training script name with kde_flow_wrapper.py")


def example_advanced_kde_options():
    """Example of advanced KDE options"""
    print("=" * 60)
    print("EXAMPLE 3: Advanced KDE Options")
    print("=" * 60)
    
    print("Advanced KDE sampling options:")
    print()
    print("1. Adjust number of neighbors (affects smoothing):")
    print("   --n_neighbors 32   # Less smoothing")
    print("   --n_neighbors 128  # More smoothing")
    print()
    print("2. Sample fraction control:")
    print("   --sample_fraction 0.5  # Downsample to 50%")
    print("   --sample_fraction 2.0   # Upsample to 200%")
    print()
    print("3. Mass range control:")
    print("   --mass_range 0.08 50   # Focus on specific mass range")
    print()
    print("4. Different flow types:")
    print("   --flow_type tfp   # TensorFlow Probability flows")
    print("   --flow_type cnf   # Continuous Normalizing Flows")


def example_shell_script_modification():
    """Example of how to modify existing shell scripts"""
    print("=" * 60)
    print("EXAMPLE 4: Shell Script Modification")
    print("=" * 60)
    
    print("To adapt your existing shell scripts for KDE:")
    print()
    print("BEFORE (using symlib directly):")
    print("```bash")
    print("python train_tfp_flows_conditional.py \\")
    print("    --halo_id $HALO_ID \\")
    print("    --parent_id $PARENT_ID \\")
    print("    --n_epochs 100 \\")
    print("    --batch_size 1024")
    print("```")
    print()
    print("AFTER (using KDE):")
    print("```bash")
    print("python kde_flow_wrapper.py \\")
    print("    --halo_id $HALO_ID \\")
    print("    --parent_id $PARENT_ID \\")
    print("    --flow_type tfp \\")
    print("    --n_epochs 100 \\")
    print("    --batch_size 1024 \\")
    print("    --n_neighbors 64 \\")
    print("    --sample_fraction 1.0")
    print("```")
    print()
    print("The wrapper handles:")
    print("- KDE sample generation")
    print("- Data format conversion")
    print("- Flow training")
    print("- Output organization")


def main():
    """Run all examples"""
    print("KDE Flow Training Examples")
    print("This demonstrates the new KDE-based approach")
    print()
    
    # Run examples
    example_wrapper_usage()
    print()
    
    example_advanced_kde_options()
    print()
    
    example_shell_script_modification()
    print()
    
    # Only run actual KDE sampling if user confirms
    response = input("Do you want to run the actual KDE sampling example? (y/N): ")
    if response.lower() == 'y':
        example_basic_kde_sampling()
    else:
        print("Skipping actual KDE sampling example.")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("New files created:")
    print("- train_kde_flows_conditional.py  # Core KDE implementation")
    print("- kde_flow_wrapper.py             # Shell script compatibility wrapper")
    print("- example_kde_usage.py            # This example file")
    print()
    print("Key benefits of KDE approach:")
    print("- Uses kernel density estimation instead of direct symlib access")
    print("- Kroupa IMF mass distribution")
    print("- Compatible with existing shell scripts")
    print("- Configurable smoothing and sampling parameters")
    print("- Same output format as existing training scripts")


if __name__ == "__main__":
    main()
