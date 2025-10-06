#!/usr/bin/env python3
"""
Script to load and sample from trained coupling flow models
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
import pickle
import argparse
from typing import Optional, Tuple
import json

# Ensure repo root is importable when running from any directory
import sys
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import the coupling flow class (do not import bin_by_mass)
from train_coupling_flows_conditional import ConditionalCouplingFlow, estimate_ntot_from_target_mass, load_particle_data_with_mass
from kroupa_imf import kroupa_masses, estimate_ntot

tfd = tfp.distributions
tfb = tfp.bijectors


def find_required_files(base_dir: str, halo_id: str, particle_pid: int) -> Tuple[str, str, str, str]:
    """
    Find the required files for loading a trained model
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing the model files
    halo_id : str
        Halo identifier (e.g., 'Halo718')
    particle_pid : int
        Particle type identifier (e.g., 1, 12)
        
    Returns:
    --------
    config_path : str
        Path to the model configuration file
    weights_path : str
        Path to the model weights (without extension)
    preprocessing_path : str
        Path to the preprocessing statistics file
    mass_bin_path : str
        Path to the mass bin information file
    """
    # Normalize base directory path to avoid double slashes
    base_dir = os.path.normpath(base_dir)
    
    # Construct expected file paths
    config_path = os.path.join(base_dir, f"flow_config_{halo_id}_{particle_pid}.pkl")
    weights_path = os.path.join(base_dir, f"flow_weights_{halo_id}_{particle_pid}")
    preprocessing_path = os.path.join(base_dir, f"coupling_flow_pid{particle_pid}_preprocessing.npz")
    mass_bin_path = os.path.join(base_dir, f"mass_bin_info_{halo_id}_{particle_pid}.npz")
    
    # Check if files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(weights_path + ".index"):
        raise FileNotFoundError(f"Weights file not found: {weights_path}.index")
    if not os.path.exists(preprocessing_path):
        raise FileNotFoundError(f"Preprocessing file not found: {preprocessing_path}")
    if not os.path.exists(mass_bin_path):
        raise FileNotFoundError(f"Mass bin info file not found: {mass_bin_path}")
    
    return config_path, weights_path, preprocessing_path, mass_bin_path


def estimate_ntot_from_halo_data(halo_id: str, particle_pid: int, suite: str = "eden") -> int:
    """
    Estimate total number of particles (ntot) based on actual halo data and Kroupa IMF
    
    Parameters:
    -----------
    halo_id : str
        Halo identifier (e.g., 'Halo718')
    particle_pid : int
        Particle type identifier (e.g., 1, 12)
    suite : str
        Simulation suite name
        
    Returns:
    --------
    ntot : int
        Estimated total number of particles based on Kroupa IMF
    """
    print(f"📊 Estimating ntot for {halo_id}, PID {particle_pid} from {suite}...")
    
    try:
        # Load actual particle data to get total mass
        _, masses, metadata = load_particle_data_with_mass(halo_id, particle_pid, suite)
        total_mass = np.sum(masses)
        
        print(f"   Total stellar mass: {total_mass:.2e} M☉")
        
        # Estimate ntot using Kroupa IMF
        ntot, mean_mass = estimate_ntot_from_target_mass(total_mass)
        
        print(f"   Estimated ntot: {ntot:,} particles")
        print(f"   Mean mass per particle: {mean_mass:.3f} M☉")
        
        return ntot
        
    except Exception as e:
        print(f"⚠️ Could not load data for ntot estimation: {e}")
        print("Using fallback estimation...")
        
        # Fallback: rough estimates based on halo number
        try:
            halo_num = int(halo_id.replace("Halo", ""))
        except:
            halo_num = 718
        
        # Rough stellar mass estimates based on typical halo properties
        if halo_num < 100:
            M_target = np.random.uniform(1e6, 1e7)  # 1-10 million M☉
        elif halo_num < 500:
            M_target = np.random.uniform(1e7, 1e8)  # 10-100 million M☉
        else:
            M_target = np.random.uniform(1e8, 1e9)  # 100 million - 1 billion M☉
        
        ntot, _ = estimate_ntot_from_target_mass(M_target)
        print(f"   Fallback M_target: {M_target:.2e} M☉")
        print(f"   Fallback ntot: {ntot:,} particles")
        
        return ntot


def rebuild_and_restore_flow(config_path: str, weights_path: str) -> ConditionalCouplingFlow:
    """
    Rebuild the flow architecture and restore weights from saved files
    
    Parameters:
    -----------
    config_path : str
        Path to the model configuration file
    weights_path : str
        Path to the model weights (without extension)
        
    Returns:
    --------
    flow : ConditionalCouplingFlow
        Restored flow model
    """
    # Load configuration
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    print(f"📋 Model configuration:")
    print(f"   Input dim: {config['input_dim']}")
    print(f"   Layers: {config['n_layers']}")
    print(f"   Hidden units: {config['hidden_units']}")
    print(f"   Mass bins: {config['n_mass_bins']}")
    print(f"   Embedding dim: {config['embedding_dim']}")
    
    # Create flow with same architecture
    # Handle hidden_units - convert single int to tuple if needed
    hidden_units = config['hidden_units']
    if isinstance(hidden_units, int):
        hidden_units = (hidden_units, hidden_units)
    
    flow = ConditionalCouplingFlow(
        input_dim=config['input_dim'],
        n_layers=config['n_layers'],
        hidden_units=hidden_units,
        n_mass_bins=config['n_mass_bins'],
        embedding_dim=config['embedding_dim']
    )
    
    # Initialize variables
    flow.initialize_variables()
    
    # Restore weights using TensorFlow checkpoint
    checkpoint = tf.train.Checkpoint(flow=flow)
    checkpoint.restore(weights_path)
    
    print(f"✅ Model restored from {weights_path}")
    return flow


def load_preprocessing_stats(preprocessing_path: str) -> dict:
    """
    Load preprocessing statistics from NPZ file
    
    Parameters:
    -----------
    preprocessing_path : str
        Path to the preprocessing statistics file
        
    Returns:
    --------
    stats : dict
        Dictionary containing preprocessing statistics
    """
    stats = np.load(preprocessing_path)
    print(f"📊 Preprocessing statistics loaded:")
    print(f"   Standardize: {stats.get('standardize', False)}")
    print(f"   Clip outliers: {stats.get('clip_outliers', None)}")
    print(f"   Mass bins: {stats.get('n_mass_bins', 'Unknown')}")
    
    return dict(stats)


def load_mass_bin_info(mass_bin_path: str) -> dict:
    """
    Load mass bin information from NPZ file
    
    Parameters:
    -----------
    mass_bin_path : str
        Path to the mass bin information file
        
    Returns:
    --------
    mass_bin_info : dict
        Dictionary containing mass bin information
    """
    mass_bin_info = np.load(mass_bin_path)
    print(f"📊 Mass bin information loaded:")
    print(f"   Total mass: {mass_bin_info['total_mass']:.2e} M☉")
    print(f"   Number of mass bins: {mass_bin_info['n_mass_bins']}")
    print(f"   Mass per bin: {mass_bin_info['mass_bin_sums']}")
    
    return dict(mass_bin_info)


def unstandardize_samples(samples: np.ndarray, preprocessing_stats: dict) -> np.ndarray:
    """
    Convert standardized samples back to physical units
    
    Parameters:
    -----------
    samples : np.ndarray
        Standardized samples from the flow
    preprocessing_stats : dict
        Preprocessing statistics
        
    Returns:
    --------
    physical_samples : np.ndarray
        Samples in physical units
    """
    if preprocessing_stats.get('standardize', False):
        x_mean = preprocessing_stats['ps_mean']
        x_std = preprocessing_stats['ps_std']
        
        # Unstandardize: x_physical = x_std * x_standardized + x_mean
        physical_samples = samples * x_std + x_mean
        print(f"✅ Unstandardized samples using mean and std")
    else:
        physical_samples = samples
        print(f"ℹ️ No standardization applied")
    
    return physical_samples


def sample_from_flow(flow: ConditionalCouplingFlow, n_samples: int, 
                    mass_bin_probs: Optional[np.ndarray] = None,
                    seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample from the trained flow model
    
    Parameters:
    -----------
    flow : ConditionalCouplingFlow
        Trained flow model
    n_samples : int
        Number of samples to generate
    mass_bin_probs : np.ndarray, optional
        Probability distribution over mass bins. If None, uses uniform distribution.
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    samples : np.ndarray
        Generated samples
    conditions : np.ndarray
        Mass bin conditions used
    """
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
    
    n_mass_bins = flow.embedding_layer.input_dim
    
    if mass_bin_probs is None:
        # Uniform distribution over mass bins
        conditions = np.random.randint(0, n_mass_bins, size=(n_samples, 1))
        print(f"🎲 Using uniform distribution over {n_mass_bins} mass bins")
    else:
        # Use provided distribution
        if len(mass_bin_probs) != n_mass_bins:
            raise ValueError(f"mass_bin_probs length ({len(mass_bin_probs)}) must match number of mass bins ({n_mass_bins})")
        
        # Normalize probabilities
        mass_bin_probs = mass_bin_probs / mass_bin_probs.sum()
        
        # Sample mass bins according to distribution
        conditions = np.random.choice(n_mass_bins, size=n_samples, p=mass_bin_probs).reshape(-1, 1)
        print(f"🎯 Using custom mass bin distribution: {mass_bin_probs}")
    
    # Generate samples from the flow
    samples = flow.sample(n_samples, conditional_input=conditions)
    
    print(f"✅ Generated {n_samples} samples with shape {samples.shape}")
    print(f"   Condition distribution: {np.bincount(conditions.flatten())}")
    
    return samples.numpy(), conditions


def sample_from_flow_by_mass_bins(flow: ConditionalCouplingFlow, M_target: float,
                                 mass_bin_masses: Optional[np.ndarray] = None,
                                 seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample from the trained flow model using the correct approach:
    For each mass bin, estimate ntot using Kroupa IMF, then generate samples from that bin.
    
    Parameters:
    -----------
    flow : ConditionalCouplingFlow
        Trained flow model
    M_target : float
        Total target stellar mass in solar masses
    mass_bin_masses : np.ndarray, optional
        Mass budget for each bin. If None, distributes M_target uniformly across bins.
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    samples : np.ndarray
        Generated samples from all bins concatenated
    conditions : np.ndarray
        Mass bin conditions used
    masses : np.ndarray
        Individual Kroupa masses for each particle
    """
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
    
    n_mass_bins = flow.embedding_layer.input_dim
    
    # Distribute total mass across bins
    if mass_bin_masses is None:
        # Uniform distribution of mass across bins
        mass_bin_masses = np.full(n_mass_bins, M_target / n_mass_bins)
        print(f"🎯 Distributing {M_target:.2e} M☉ uniformly across {n_mass_bins} mass bins")
    else:
        if len(mass_bin_masses) != n_mass_bins:
            raise ValueError(f"mass_bin_masses length ({len(mass_bin_masses)}) must match number of mass bins ({n_mass_bins})")
        print(f"🎯 Using provided mass distribution across {n_mass_bins} mass bins")
    
    print(f"   Mass per bin: {mass_bin_masses}")
    
    all_samples = []
    all_conditions = []
    all_masses = []
    
    total_particles = 0
    
    # Process each mass bin
    for bin_idx in range(n_mass_bins):
        M_bin = mass_bin_masses[bin_idx]
        
        if M_bin <= 0:
            print(f"   Bin {bin_idx}: Skipped (M_bin = {M_bin:.2e} M☉)")
            continue
        
        # Estimate ntot for this specific bin using Kroupa IMF WITHOUT any cap
        # Draw a large Kroupa sample to estimate mean mass, then Poisson around M_bin / mean_mass
        ntest = 200000
        mtest = kroupa_masses(ntest, seed=(seed or 0) + bin_idx)
        mean_mass_bin = float(np.mean(mtest)) if len(mtest) > 0 else 1.0
        base_count = max(1, int(round(M_bin / mean_mass_bin)))
        np.random.seed((seed or 0) + bin_idx + 123)
        ntot_bin = int(np.random.poisson(base_count))
        ntot_bin = max(ntot_bin, 1)
        
        print(f"   Bin {bin_idx}: M_bin = {M_bin:.2e} M☉, ntot = {ntot_bin:,} particles, mean_mass = {mean_mass_bin:.3f} M☉")
        
        if ntot_bin <= 0:
            print(f"   Bin {bin_idx}: Skipped (ntot = {ntot_bin})")
            continue
        
        # Generate samples from this specific mass bin
        conditions_bin = np.full((ntot_bin, 1), bin_idx, dtype=np.int32)
        samples_bin = flow.sample(ntot_bin, conditional_input=conditions_bin)
        
        # Generate individual Kroupa masses for particles in this bin
        masses_bin = kroupa_masses(ntot_bin, seed=seed + bin_idx + 1000)
        
        # Store results
        all_samples.append(samples_bin.numpy())
        all_conditions.append(conditions_bin)
        all_masses.append(masses_bin)
        
        total_particles += ntot_bin
        
        print(f"   Bin {bin_idx}: Generated {ntot_bin:,} samples, total mass = {np.sum(masses_bin):.2e} M☉")
    
    # Concatenate all samples
    if len(all_samples) == 0:
        raise ValueError("No samples generated from any mass bin")
    
    samples = np.concatenate(all_samples, axis=0)
    conditions = np.concatenate(all_conditions, axis=0)
    masses = np.concatenate(all_masses, axis=0)
    
    print(f"✅ Generated {total_particles:,} total particles from {len(all_samples)} mass bins")
    print(f"   Total generated mass: {np.sum(masses):.2e} M☉")
    print(f"   Target mass: {M_target:.2e} M☉")
    print(f"   Mass ratio: {np.sum(masses) / M_target:.3f}")
    print(f"   Sample shape: {samples.shape}")
    print(f"   Condition distribution: {np.bincount(conditions.flatten())}")
    
    return samples, conditions, masses


def main():
    """Main function to load and sample from trained coupling flow models"""
    
    parser = argparse.ArgumentParser(description="Sample from trained coupling flow models")
    parser.add_argument("--base_dir", type=str, default=None,
                       help="Base directory containing model files. If not provided, will auto-construct from halo_id, suite, and pid")
    parser.add_argument("--halo_id", type=str, required=True,
                       help="Halo identifier (e.g., Halo718)")
    parser.add_argument("--particle_pid", type=int, required=True,
                       help="Particle type identifier (e.g., 1, 12)")
    parser.add_argument("--suite", type=str, default="eden",
                       help="Simulation suite (default: eden)")
    parser.add_argument("--n_samples", type=int, default=None,
                       help="Number of samples to generate. If not provided, will estimate from Kroupa IMF")
    parser.add_argument("--M_target", type=float, default=None,
                       help="Target stellar mass in solar masses for ntot estimation (required when using --use_kroupa)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file to save samples (default: auto-generated)")
    parser.add_argument("--mass_bin_probs", type=str, default=None,
                       help="Comma-separated probabilities for mass bins (e.g., '0.1,0.2,0.3,0.4')")
    parser.add_argument("--use_kroupa", action="store_true",
                       help="Use correct Kroupa IMF sampling: estimate ntot for each mass bin, then generate samples from each bin")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Ensure base_dir points to the correct PID folder
    expected_dir = f"coupling_output/{args.suite}/{args.halo_id.lower()}/pid{args.particle_pid}"
    if args.base_dir is None:
        args.base_dir = expected_dir
        print(f"🔧 Auto-constructed base directory: {args.base_dir}")
    else:
        # If provided base_dir does not match the requested PID, override it
        if os.path.normpath(args.base_dir) != os.path.normpath(expected_dir):
            print(f"⚠️ Provided base_dir ({args.base_dir}) does not match PID {args.particle_pid}. Using {expected_dir} instead.")
            args.base_dir = expected_dir
    
    print("🎯 Coupling Flow Sampling")
    print("=" * 50)
    print(f"Base directory: {args.base_dir}")
    print(f"Halo ID: {args.halo_id}")
    print(f"Particle PID: {args.particle_pid}")
    print(f"Suite: {args.suite}")
    print(f"Random seed: {args.seed}")
    print("=" * 50)
    
    # Validate arguments
    # Note: M_target is optional - if not provided, will use total_mass from training data
    
    try:
        # Determine number of samples
        if args.use_kroupa:
            # When using Kroupa, n_samples will be determined by the sampling function
            # based on mass bin sums and Kroupa IMF
            n_samples = None
            print(f"✅ Using Kroupa IMF - n_samples will be determined during sampling")
        elif args.n_samples is not None:
            n_samples = args.n_samples
            print(f"📊 Using specified n_samples: {n_samples:,}")
        else:
            print(f"📊 Estimating n_samples from Kroupa IMF...")
            if args.M_target is not None:
                print(f"   Using specified M_target: {args.M_target:.2e} M☉")
                n_samples, _ = estimate_ntot_from_target_mass(args.M_target)
            else:
                n_samples = estimate_ntot_from_halo_data(args.halo_id, args.particle_pid, args.suite)
            print(f"✅ Estimated n_samples: {n_samples:,}")
        
        # Find required files
        print("🔍 Finding model files...")
        config_path, weights_path, preprocessing_path, mass_bin_path = find_required_files(
            args.base_dir, args.halo_id, args.particle_pid
        )
        
        # Load and restore model
        print("🏗️ Loading and restoring model...")
        flow = rebuild_and_restore_flow(config_path, weights_path)
        
        # Load preprocessing statistics
        print("📊 Loading preprocessing statistics...")
        preprocessing_stats = load_preprocessing_stats(preprocessing_path)
        
        # Load mass bin information
        print("📊 Loading mass bin information...")
        mass_bin_info = load_mass_bin_info(mass_bin_path)
        
        # Determine sampling approach
        if args.use_kroupa:
            print("🌟 Using Kroupa IMF for correct mass bin sampling...")
            # Use the actual mass bin sums from training data
            mass_bin_sums = mass_bin_info['mass_bin_sums']
            total_mass = mass_bin_info['total_mass']
            print(f"   Using actual mass bin sums from training data")

            # Per-bin Kroupa counts (accurate): estimate ntot per bin and sample per bin
            samples, conditions, masses = sample_from_flow_by_mass_bins(
                flow, args.M_target or total_mass,
                mass_bin_masses=mass_bin_sums, seed=args.seed
            )
        else:
            # Use the old approach for backward compatibility
            mass_bin_probs = None
            if args.mass_bin_probs:
                print("🎯 Using custom mass bin distribution...")
                mass_bin_probs = np.array([float(x) for x in args.mass_bin_probs.split(',')])
            
            # Generate samples
            print("🎲 Generating samples...")
            samples, conditions = sample_from_flow(
                flow, n_samples, mass_bin_probs, args.seed
            )
            masses = None
        
        # Unstandardize samples
        print("🔄 Converting to physical units...")
        physical_samples = unstandardize_samples(samples, preprocessing_stats)
        
        # Save samples
        if args.output_file is None:
            # Use the same directory structure as training outputs
            output_file = os.path.join(args.base_dir, f"samples_{args.halo_id}_{args.particle_pid}.npz")
        else:
            output_file = args.output_file
        
        # Prepare save data
        save_data = {
            'samples': physical_samples,
            'conditions': conditions,
            'n_samples': len(physical_samples),
            'seed': args.seed,
            'halo_id': args.halo_id,
            'particle_pid': args.particle_pid,
            'suite': args.suite
        }
        
        # Add masses if available (from Kroupa sampling)
        if masses is not None:
            save_data['masses'] = masses
            save_data['total_mass'] = np.sum(masses)
            save_data['M_target'] = args.M_target or 1e6
        
        # Add mass_bin_probs if available (from old sampling)
        if 'mass_bin_probs' in locals() and mass_bin_probs is not None:
            save_data['mass_bin_probs'] = mass_bin_probs
        
        np.savez(output_file, **save_data)
        
        print(f"💾 Samples saved to: {output_file}")
        print(f"   Sample shape: {physical_samples.shape}")
        print(f"   Sample range: [{physical_samples.min():.3f}, {physical_samples.max():.3f}]")
        
        # Print some statistics
        print(f"\n📈 Sample Statistics:")
        print(f"   Mean: {physical_samples.mean(axis=0)}")
        print(f"   Std:  {physical_samples.std(axis=0)}")
        print(f"   Min:  {physical_samples.min(axis=0)}")
        print(f"   Max:  {physical_samples.max(axis=0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

