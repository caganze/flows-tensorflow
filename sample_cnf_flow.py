#!/usr/bin/env python3
"""
CNF sampling utility
Generate samples from trained Continuous Normalizing Flows (CNFs)
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

# Configure TensorFlow environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

from cnf_flows_solution import load_cnf_flow, CNFNormalizingFlow


def load_preprocessing_stats(preprocessing_path: str) -> Dict[str, tf.Tensor]:
    """Load preprocessing statistics"""
    try:
        preprocessing_data = np.load(preprocessing_path, allow_pickle=True)
        preprocessing_stats = {k: tf.constant(v) for k, v in preprocessing_data.items()}
        print(f"✅ Preprocessing statistics loaded from {preprocessing_path}")
        return preprocessing_stats
    except Exception as e:
        print(f"❌ Error loading preprocessing statistics: {e}")
        raise


def inverse_preprocess_samples(
    samples: tf.Tensor,
    preprocessing_stats: Dict[str, tf.Tensor]
) -> tf.Tensor:
    """Apply inverse preprocessing to samples"""
    
    processed_samples = samples
    
    # Check if standardization was applied
    if preprocessing_stats.get('standardize', True):
        # Inverse standardization
        mean = preprocessing_stats['mean']
        std = preprocessing_stats['std']
        processed_samples = processed_samples * std + mean
    
    return processed_samples


def sample_cnf_flow(
    model_path: str,
    n_samples: int = 10000,
    output_dir: Optional[str] = None,
    seed: int = 42
) -> tf.Tensor:
    """Generate samples from a CNF
    
    Args:
        model_path: Path to saved CNF model
        n_samples: Number of samples to generate
        output_dir: Directory to save samples (optional)
        seed: Random seed
        
    Returns:
        phase_space_samples: Generated samples in physical units
    """
    
    print(f"🎲 Generating {n_samples} CNF samples")
    
    # Load the CNF model
    flow = load_cnf_flow(model_path)
    
    # Load preprocessing statistics
    preprocessing_path = model_path.replace('.npz', '_preprocessing.npz')
    if not Path(preprocessing_path).exists():
        raise FileNotFoundError(f"Preprocessing file not found: {preprocessing_path}")
    
    preprocessing_stats = load_preprocessing_stats(preprocessing_path)
    
    # Generate samples from the CNF
    print("🔄 Sampling from CNF...")
    phase_space_samples_processed = flow.sample(
        n_samples=n_samples,
        seed=seed
    )
    
    # Inverse preprocess the samples
    phase_space_samples = inverse_preprocess_samples(
        phase_space_samples_processed, preprocessing_stats
    )
    
    print(f"✅ Generated {n_samples} CNF samples")
    print(f"   Phase space shape: {phase_space_samples.shape}")
    
    # Save samples if output directory is provided
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save samples
        samples_file = output_path / f"cnf_samples_{n_samples}.npz"
        
        np.savez_compressed(
            samples_file,
            phase_space_samples=phase_space_samples.numpy(),
            n_samples=n_samples,
            seed=seed,
            model_type='cnf'
        )
        
        print(f"💾 Samples saved to {samples_file}")
        
        # Save metadata
        metadata = {
            'n_samples': int(n_samples),
            'seed': int(seed),
            'model_path': str(model_path),
            'model_type': 'cnf',
            'sample_statistics': {
                'phase_space_mean': tf.reduce_mean(phase_space_samples, axis=0).numpy().tolist(),
                'phase_space_std': tf.math.reduce_std(phase_space_samples, axis=0).numpy().tolist(),
                'phase_space_min': tf.reduce_min(phase_space_samples, axis=0).numpy().tolist(),
                'phase_space_max': tf.reduce_max(phase_space_samples, axis=0).numpy().tolist()
            }
        }
        
        metadata_file = output_path / f"cnf_samples_{n_samples}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return phase_space_samples


def generate_kroupa_masses_for_cnf_samples(
    n_samples: int,
    stellar_mass_total: Optional[float] = None,
    seed: int = 42
) -> np.ndarray:
    """Generate Kroupa IMF masses to accompany CNF samples
    
    Args:
        n_samples: Number of mass samples to generate
        stellar_mass_total: Total stellar mass to normalize to (optional)
        seed: Random seed
        
    Returns:
        masses: Array of stellar masses in solar masses
    """
    from kroupa_imf import kroupa_masses
    
    print(f"🌟 Generating Kroupa IMF masses for {n_samples} CNF samples")
    
    # Generate Kroupa masses
    masses = kroupa_masses(n_samples, seed=seed)
    
    # Optionally normalize to target total mass
    if stellar_mass_total is not None:
        current_total = np.sum(masses)
        masses = masses * (stellar_mass_total / current_total)
        print(f"   Normalized to total stellar mass: {stellar_mass_total:.2e} M☉")
    
    print(f"✅ Generated Kroupa masses: [{np.min(masses):.3e}, {np.max(masses):.3e}] M☉")
    
    return masses


def sample_cnf_with_kroupa_masses(
    model_path: str,
    n_samples: int = 10000,
    stellar_mass_total: Optional[float] = None,
    output_dir: Optional[str] = None,
    seed: int = 42
) -> Tuple[tf.Tensor, np.ndarray]:
    """Generate CNF samples with accompanying Kroupa IMF masses
    
    Args:
        model_path: Path to saved CNF model
        n_samples: Number of samples to generate
        stellar_mass_total: Total stellar mass to normalize to (optional)
        output_dir: Directory to save samples (optional)
        seed: Random seed
        
    Returns:
        Tuple of (phase_space_samples, masses)
    """
    
    print(f"🎯 Generating {n_samples} CNF samples with Kroupa masses")
    
    # Generate CNF samples
    phase_space_samples = sample_cnf_flow(
        model_path=model_path,
        n_samples=n_samples,
        output_dir=None,  # Don't save yet
        seed=seed
    )
    
    # Generate Kroupa masses
    masses = generate_kroupa_masses_for_cnf_samples(
        n_samples=n_samples,
        stellar_mass_total=stellar_mass_total,
        seed=seed
    )
    
    # Save combined samples if output directory is provided
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save samples and masses together
        samples_file = output_path / f"cnf_samples_with_masses_{n_samples}.npz"
        
        np.savez_compressed(
            samples_file,
            phase_space_samples=phase_space_samples.numpy(),
            masses=masses,
            n_samples=n_samples,
            seed=seed,
            model_type='cnf',
            has_kroupa_masses=True,
            stellar_mass_total=stellar_mass_total if stellar_mass_total else np.sum(masses)
        )
        
        print(f"💾 CNF samples with masses saved to {samples_file}")
        
        # Save metadata
        metadata = {
            'n_samples': int(n_samples),
            'seed': int(seed),
            'model_path': str(model_path),
            'model_type': 'cnf',
            'has_kroupa_masses': True,
            'stellar_mass_total': float(stellar_mass_total) if stellar_mass_total else float(np.sum(masses)),
            'sample_statistics': {
                'phase_space_mean': tf.reduce_mean(phase_space_samples, axis=0).numpy().tolist(),
                'phase_space_std': tf.math.reduce_std(phase_space_samples, axis=0).numpy().tolist(),
                'mass_mean': float(np.mean(masses)),
                'mass_std': float(np.std(masses)),
                'mass_min': float(np.min(masses)),
                'mass_max': float(np.max(masses))
            }
        }
        
        metadata_file = output_path / f"cnf_samples_with_masses_{n_samples}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return phase_space_samples, masses


def analyze_cnf_samples(
    phase_space_samples: tf.Tensor,
    masses: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Analyze generated CNF samples
    
    Args:
        phase_space_samples: Generated phase space samples
        masses: Optional mass array for mass-dependent analysis
        
    Returns:
        analysis_results: Dictionary with analysis results
    """
    
    print("📈 Analyzing CNF samples...")
    
    # Basic statistics
    positions = phase_space_samples[:, :3]  # x, y, z
    velocities = phase_space_samples[:, 3:]  # vx, vy, vz
    
    # Calculate radii and speeds
    radii = tf.sqrt(tf.reduce_sum(tf.square(positions), axis=1))
    speeds = tf.sqrt(tf.reduce_sum(tf.square(velocities), axis=1))
    
    analysis = {
        'n_samples': int(len(phase_space_samples)),
        'position_statistics': {
            'mean_radius': float(tf.reduce_mean(radii)),
            'std_radius': float(tf.math.reduce_std(radii)),
            'min_radius': float(tf.reduce_min(radii)),
            'max_radius': float(tf.reduce_max(radii))
        },
        'velocity_statistics': {
            'mean_speed': float(tf.reduce_mean(speeds)),
            'std_speed': float(tf.math.reduce_std(speeds)),
            'min_speed': float(tf.reduce_min(speeds)),
            'max_speed': float(tf.reduce_max(speeds))
        },
        'phase_space_bounds': {
            'position_range': [
                [float(tf.reduce_min(positions[:, i])), float(tf.reduce_max(positions[:, i]))]
                for i in range(3)
            ],
            'velocity_range': [
                [float(tf.reduce_min(velocities[:, i])), float(tf.reduce_max(velocities[:, i]))]
                for i in range(3)
            ]
        }
    }
    
    # Mass-dependent analysis if masses are provided
    if masses is not None:
        analysis['mass_statistics'] = {
            'mean_mass': float(np.mean(masses)),
            'std_mass': float(np.std(masses)),
            'min_mass': float(np.min(masses)),
            'max_mass': float(np.max(masses)),
            'total_mass': float(np.sum(masses))
        }
        
        # Mass-radius correlation
        masses_tf = tf.constant(masses, dtype=tf.float32)
        mass_radius_corr = tfp.stats.correlation(masses_tf, radii, event_axis=None)
        mass_speed_corr = tfp.stats.correlation(masses_tf, speeds, event_axis=None)
        
        analysis['mass_correlations'] = {
            'mass_radius_correlation': float(mass_radius_corr),
            'mass_speed_correlation': float(mass_speed_corr)
        }
        
        # Binned analysis by mass
        mass_median = tf.reduce_median(masses_tf)
        low_mass_mask = masses_tf < mass_median
        high_mass_mask = masses_tf >= mass_median
        
        low_mass_radius = tf.reduce_mean(tf.boolean_mask(radii, low_mass_mask))
        high_mass_radius = tf.reduce_mean(tf.boolean_mask(radii, high_mass_mask))
        low_mass_speed = tf.reduce_mean(tf.boolean_mask(speeds, low_mass_mask))
        high_mass_speed = tf.reduce_mean(tf.boolean_mask(speeds, high_mass_mask))
        
        analysis['mass_binned_analysis'] = {
            'mass_median': float(mass_median),
            'low_mass_mean_radius': float(low_mass_radius),
            'high_mass_mean_radius': float(high_mass_radius),
            'low_mass_mean_speed': float(low_mass_speed),
            'high_mass_mean_speed': float(high_mass_speed)
        }
    
    print(f"✅ Analysis complete:")
    print(f"   Mean radius: {analysis['position_statistics']['mean_radius']:.2f} kpc")
    print(f"   Mean speed: {analysis['velocity_statistics']['mean_speed']:.2f} km/s")
    if masses is not None:
        print(f"   Mean mass: {analysis['mass_statistics']['mean_mass']:.3e} M☉")
        print(f"   Mass-radius correlation: {analysis['mass_correlations']['mass_radius_correlation']:.3f}")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Generate samples from CNF flows")
    
    # Required arguments
    parser.add_argument("--model_path", required=True, help="Path to saved CNF model")
    
    # Sampling arguments
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--stellar_mass_total", type=float, help="Total stellar mass to normalize Kroupa masses to")
    parser.add_argument("--with_masses", action="store_true", help="Generate Kroupa IMF masses alongside CNF samples")
    
    # Output arguments
    parser.add_argument("--output_dir", help="Directory to save generated samples")
    parser.add_argument("--analyze", action="store_true", help="Perform analysis of generated samples")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if not args.output_dir:
        model_dir = Path(args.model_path).parent
        args.output_dir = str(model_dir / "cnf_samples")
    
    print("🎯 CNF Sampling")
    print("=" * 30)
    print(f"Model: {args.model_path}")
    print(f"Samples: {args.n_samples}")
    print(f"With masses: {args.with_masses}")
    if args.stellar_mass_total:
        print(f"Target stellar mass: {args.stellar_mass_total:.3e} M☉")
    print(f"Output: {args.output_dir}")
    print()
    
    # Set random seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        if args.with_masses:
            # Generate CNF samples with Kroupa masses
            phase_space_samples, masses = sample_cnf_with_kroupa_masses(
                model_path=args.model_path,
                n_samples=args.n_samples,
                stellar_mass_total=args.stellar_mass_total,
                output_dir=args.output_dir,
                seed=args.seed
            )
            
            print(f"✅ CNF sampling with masses completed!")
            print(f"   Generated {len(phase_space_samples)} samples")
            print(f"   Phase space dimensions: {phase_space_samples.shape[1]}")
            print(f"   Mass range: [{np.min(masses):.3e}, {np.max(masses):.3e}] M☉")
            print(f"   Total stellar mass: {np.sum(masses):.3e} M☉")
            
            # Analysis
            if args.analyze:
                analysis = analyze_cnf_samples(phase_space_samples, masses)
                
                # Save analysis results
                analysis_file = Path(args.output_dir) / f"cnf_analysis_{args.n_samples}.json"
                with open(analysis_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
                print(f"📊 Analysis saved to {analysis_file}")
        
        else:
            # Generate CNF samples only
            phase_space_samples = sample_cnf_flow(
                model_path=args.model_path,
                n_samples=args.n_samples,
                output_dir=args.output_dir,
                seed=args.seed
            )
            
            print(f"✅ CNF sampling completed!")
            print(f"   Generated {len(phase_space_samples)} samples")
            print(f"   Phase space dimensions: {phase_space_samples.shape[1]}")
            
            # Analysis
            if args.analyze:
                analysis = analyze_cnf_samples(phase_space_samples)
                
                # Save analysis results
                analysis_file = Path(args.output_dir) / f"cnf_analysis_{args.n_samples}.json"
                with open(analysis_file, 'w') as f:
                    json.dump(analysis, f, indent=2)
                print(f"📊 Analysis saved to {analysis_file}")
        
    except Exception as e:
        print(f"❌ CNF sampling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
