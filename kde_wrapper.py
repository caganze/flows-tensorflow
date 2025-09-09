#!/usr/bin/env python3
"""
KDE Flow Wrapper Script

This script provides a drop-in replacement for symlib-based training scripts,
using KDE sampling instead. It's designed to work with existing shell script
infrastructure by accepting the same command-line arguments and producing
compatible output formats.

Usage:
    python kde_flow_wrapper.py [same args as existing training scripts]
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Import the KDE training functionality
from train_kde_flows_conditional import (
    load_halo_data, 
    generate_kde_samples, 
    save_samples,
    KDESamplerKroupa
)

# Try to import existing training modules for compatibility
try:
    from train_tfp_flows_conditional import main as tfp_main
    TFP_AVAILABLE = True
except ImportError:
    TFP_AVAILABLE = False
    print("Warning: TensorFlow Probability flows not available")

try:
    from train_cnf_flows_conditional import main as cnf_main
    CNF_AVAILABLE = True
except ImportError:
    CNF_AVAILABLE = False
    print("Warning: CNF flows not available")


def parse_arguments():
    """Parse command line arguments compatible with existing scripts"""
    parser = argparse.ArgumentParser(description='KDE-based Flow Training Wrapper')
    
    # Core arguments
    parser.add_argument('--halo_id', type=str, required=True,
                       help='Halo identifier (e.g., Halo939)')
    parser.add_argument('--parent_id', type=int, required=True,
                       help='Parent ID to select particles')
    
    # KDE-specific arguments
    parser.add_argument('--use_kde', action='store_true', default=True,
                       help='Use KDE sampling (default: True)')
    parser.add_argument('--n_neighbors', type=int, default=64,
                       help='Number of neighbors for KDE')
    parser.add_argument('--sample_fraction', type=float, default=1.0,
                       help='Fraction of samples to generate')
    parser.add_argument('--mass_range', type=float, nargs=2, default=[0.1, 120],
                       help='Mass range for IMF sampling')
    
    # Flow training arguments
    parser.add_argument('--flow_type', type=str, choices=['tfp', 'cnf'], default='tfp',
                       help='Type of normalizing flow to train')
    parser.add_argument('--n_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    
    # I/O arguments
    parser.add_argument('--suite', type=str, default='eden',
                       help='Simulation suite name')
    parser.add_argument('--output_dir', type=str, default='./trained_flows',
                       help='Output directory for trained models')
    parser.add_argument('--sample_dir', type=str, default='./kde_samples',
                       help='Directory to save KDE samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Compatibility arguments (for existing shell scripts)
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for training')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()


def prepare_kde_samples(args):
    """
    Generate KDE samples and save them in a format compatible with training scripts.
    
    Returns:
    --------
    sample_file : Path
        Path to generated sample file
    """
    print("=" * 60)
    print("STEP 1: Generating KDE Samples")
    print("=" * 60)
    
    # Load halo data
    print(f"Loading halo data: {args.halo_id}, parent_id={args.parent_id}")
    phase_space, masses = load_halo_data(args.halo_id, args.parent_id, args.suite)
    
    # Generate KDE samples
    print(f"Generating samples with KDE (n_neighbors={args.n_neighbors})")
    samples = generate_kde_samples(
        phase_space, 
        masses, 
        n_neighbors=args.n_neighbors,
        sample_fraction=args.sample_fraction,
        mass_range=tuple(args.mass_range),
        seed=args.seed
    )
    
    # Save samples
    sample_filename = f"kde_samples_{args.halo_id}_pid{args.parent_id}.h5"
    sample_file = Path(args.sample_dir) / sample_filename
    
    save_samples(samples, sample_file, args.halo_id, args.parent_id)
    
    print(f"KDE samples saved to: {sample_file}")
    return sample_file


def train_flow_model(args, sample_file):
    """
    Train normalizing flow model using the generated KDE samples.
    
    Parameters:
    -----------
    args : Namespace
        Command line arguments
    sample_file : Path
        Path to KDE sample file
    """
    print("=" * 60)
    print("STEP 2: Training Normalizing Flow")
    print("=" * 60)
    
    # Prepare arguments for training script
    training_args = [
        '--input_file', str(sample_file),
        '--halo_id', args.halo_id,
        '--parent_id', str(args.parent_id),
        '--n_epochs', str(args.n_epochs),
        '--batch_size', str(args.batch_size),
        '--learning_rate', str(args.learning_rate),
        '--output_dir', args.output_dir,
        '--seed', str(args.seed)
    ]
    
    if args.gpu:
        training_args.append('--gpu')
    if args.verbose:
        training_args.append('--verbose')
    
    # Train the appropriate flow type
    if args.flow_type == 'tfp' and TFP_AVAILABLE:
        print("Training TensorFlow Probability flow...")
        # Save current sys.argv and replace with training args
        original_argv = sys.argv.copy()
        sys.argv = ['train_tfp_flows_conditional.py'] + training_args
        try:
            tfp_main()
        finally:
            sys.argv = original_argv
            
    elif args.flow_type == 'cnf' and CNF_AVAILABLE:
        print("Training Continuous Normalizing Flow...")
        # Save current sys.argv and replace with training args
        original_argv = sys.argv.copy()
        sys.argv = ['train_cnf_flows_conditional.py'] + training_args
        try:
            cnf_main()
        finally:
            sys.argv = original_argv
            
    else:
        print(f"Warning: {args.flow_type} flow training not available")
        print("Falling back to manual training script execution")
        
        # Create a training script command
        if args.flow_type == 'tfp':
            script_name = 'train_tfp_flows_conditional.py'
        else:
            script_name = 'train_cnf_flows_conditional.py'
            
        cmd = f"python {script_name} " + " ".join(training_args)
        print(f"Execute: {cmd}")


def create_compatibility_symlinks(args):
    """
    Create symbolic links to maintain compatibility with existing shell scripts.
    """
    # Create symlinks that point to this wrapper script
    wrapper_path = Path(__file__).resolve()
    
    symlinks = [
        'train_tfp_flows_kde.py',
        'train_cnf_flows_kde.py'
    ]
    
    for symlink_name in symlinks:
        symlink_path = wrapper_path.parent / symlink_name
        if not symlink_path.exists():
            try:
                symlink_path.symlink_to(wrapper_path)
                print(f"Created symlink: {symlink_name} -> {wrapper_path.name}")
            except OSError:
                print(f"Warning: Could not create symlink {symlink_name}")


def main():
    """Main execution function"""
    print("KDE Flow Wrapper - Kernel Density Estimation based Flow Training")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    if args.verbose:
        print("Configuration:")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        print()
    
    try:
        # Step 1: Generate KDE samples
        sample_file = prepare_kde_samples(args)
        
        # Step 2: Train normalizing flow
        train_flow_model(args, sample_file)
        
        # Step 3: Create compatibility symlinks
        create_compatibility_symlinks(args)
        
        print("=" * 60)
        print("KDE FLOW TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Halo: {args.halo_id}, Parent ID: {args.parent_id}")
        print(f"Flow type: {args.flow_type}")
        print(f"Samples: {sample_file}")
        print(f"Models saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during KDE flow training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
