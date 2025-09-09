#!/usr/bin/env python3
"""
Conditional flow sampling utility
Generate samples from trained conditional flows with specified mass conditions
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

from train_tfp_flows_conditional import ConditionalTFPNormalizingFlow


def load_conditional_flow(model_path: str) -> ConditionalTFPNormalizingFlow:
    """Load a conditional flow model from saved file"""
    try:
        # Load the saved data
        data = np.load(model_path, allow_pickle=True)
        
        # Reconstruct flow from configuration
        config = data['config'].item()
        flow = ConditionalTFPNormalizingFlow(
            input_dim=config['input_dim'],
            condition_dim=config['condition_dim'],
            n_layers=config['n_layers'],
            hidden_units=config['hidden_units'],
            activation=config['activation'],
            name=config['name']
        )
        
        # Force initialization by running a dummy forward pass
        dummy_input = tf.zeros((1, config['input_dim']), dtype=tf.float32)
        dummy_conditions = tf.zeros((1, config['condition_dim']), dtype=tf.float32)
        _ = flow.log_prob(dummy_input, dummy_conditions)  # This creates all the variables
        
        # Load the saved variables
        n_variables = int(data['n_variables'])
        variables = flow.trainable_variables
        
        if len(variables) != n_variables:
            raise ValueError(f"Model structure mismatch: expected {n_variables} variables, got {len(variables)}")
        
        # Assign the loaded values to variables
        for i, var in enumerate(variables):
            loaded_value = data[f'variable_{i}']
            var.assign(loaded_value)
        
        print(f"✅ Conditional flow loaded from {model_path}")
        print(f"   Loaded {len(variables)} variables")
        print(f"   Input dim: {config['input_dim']}, Condition dim: {config['condition_dim']}")
        return flow
        
    except Exception as e:
        print(f"❌ Error loading conditional flow model: {e}")
        raise


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
        ps_mean = preprocessing_stats['ps_mean']
        ps_std = preprocessing_stats['ps_std']
        processed_samples = processed_samples * ps_std + ps_mean
    
    return processed_samples


def inverse_preprocess_conditions(
    conditions: tf.Tensor,
    preprocessing_stats: Dict[str, tf.Tensor]
) -> tf.Tensor:
    """Apply inverse preprocessing to mass conditions"""
    
    processed_conditions = conditions
    
    # Check if standardization was applied
    if preprocessing_stats.get('standardize', True):
        # Inverse standardization
        mass_mean = preprocessing_stats['mass_mean']
        mass_std = preprocessing_stats['mass_std']
        processed_conditions = processed_conditions * mass_std + mass_mean
    
    # Check if log transform was applied
    if preprocessing_stats.get('log_transform_mass', True):
        # Inverse log transform
        processed_conditions = tf.exp(processed_conditions)
    
    return processed_conditions


def generate_mass_conditions(
    n_samples: int,
    mass_strategy: str = "uniform",
    mass_range: Optional[Tuple[float, float]] = None,
    target_mass: Optional[float] = None,
    seed: int = 42
) -> tf.Tensor:
    """Generate mass conditions for conditional sampling
    
    Args:
        n_samples: Number of samples to generate
        mass_strategy: Strategy for mass generation ('uniform', 'log_uniform', 'fixed', 'kroupa')
        mass_range: Tuple of (min_mass, max_mass) in solar masses
        target_mass: Fixed mass value for 'fixed' strategy
        seed: Random seed
        
    Returns:
        Mass conditions tensor of shape (n_samples, 1)
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    if mass_strategy == "fixed":
        if target_mass is None:
            target_mass = 1.0  # 1 solar mass default
        masses = tf.fill((n_samples, 1), target_mass)
        
    elif mass_strategy == "uniform":
        if mass_range is None:
            mass_range = (0.1, 100.0)  # Default range in solar masses
        min_mass, max_mass = mass_range
        masses = tf.random.uniform((n_samples, 1), min_mass, max_mass)
        
    elif mass_strategy == "log_uniform":
        if mass_range is None:
            mass_range = (0.01, 100.0)  # Default range in solar masses
        min_mass, max_mass = mass_range
        log_min = np.log10(min_mass)
        log_max = np.log10(max_mass)
        log_masses = tf.random.uniform((n_samples, 1), log_min, log_max)
        masses = tf.pow(10.0, log_masses)
        
    elif mass_strategy == "kroupa":
        # Simplified Kroupa IMF sampling
        if mass_range is None:
            mass_range = (0.08, 100.0)  # Main sequence stars
        min_mass, max_mass = mass_range
        
        # Generate random numbers
        u = tf.random.uniform((n_samples,))
        
        # Kroupa IMF power law slopes
        # Simplified version: single power law with alpha = -2.3
        alpha = -2.3
        masses_1d = min_mass * tf.pow(
            1 + u * (tf.pow(max_mass/min_mass, alpha+1) - 1),
            1/(alpha+1)
        )
        masses = tf.reshape(masses_1d, (n_samples, 1))
        
    else:
        raise ValueError(f"Unknown mass strategy: {mass_strategy}")
    
    print(f"✅ Generated {n_samples} mass conditions using '{mass_strategy}' strategy")
    print(f"   Mass range: [{tf.reduce_min(masses):.3e}, {tf.reduce_max(masses):.3e}] M☉")
    
    return masses


def apply_preprocessing_to_conditions(
    masses: tf.Tensor,
    preprocessing_stats: Dict[str, tf.Tensor]
) -> tf.Tensor:
    """Apply preprocessing to mass conditions"""
    
    processed_masses = masses
    
    # Apply log transform if it was used during training
    if preprocessing_stats.get('log_transform_mass', True):
        processed_masses = tf.math.log(processed_masses + 1e-10)
    
    # Apply standardization if it was used during training
    if preprocessing_stats.get('standardize', True):
        mass_mean = preprocessing_stats['mass_mean']
        mass_std = preprocessing_stats['mass_std']
        processed_masses = (processed_masses - mass_mean) / (mass_std + 1e-8)
    
    return processed_masses


def sample_conditional_flow(
    model_path: str,
    n_samples: int = 10000,
    mass_strategy: str = "log_uniform",
    mass_range: Optional[Tuple[float, float]] = None,
    target_mass: Optional[float] = None,
    output_dir: Optional[str] = None,
    seed: int = 42
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Generate samples from a conditional flow with specified mass conditions
    
    Args:
        model_path: Path to saved conditional flow model
        n_samples: Number of samples to generate
        mass_strategy: Strategy for mass generation
        mass_range: Mass range for sampling
        target_mass: Fixed mass for 'fixed' strategy
        output_dir: Directory to save samples (optional)
        seed: Random seed
        
    Returns:
        Tuple of (phase_space_samples, mass_conditions)
    """
    
    print(f"🎲 Generating {n_samples} conditional samples")
    print(f"   Mass strategy: {mass_strategy}")
    
    # Load the conditional flow model
    flow = load_conditional_flow(model_path)
    
    # Load preprocessing statistics
    preprocessing_path = model_path.replace('.npz', '_preprocessing.npz')
    if not Path(preprocessing_path).exists():
        raise FileNotFoundError(f"Preprocessing file not found: {preprocessing_path}")
    
    preprocessing_stats = load_preprocessing_stats(preprocessing_path)
    
    # Generate mass conditions
    mass_conditions_raw = generate_mass_conditions(
        n_samples=n_samples,
        mass_strategy=mass_strategy,
        mass_range=mass_range,
        target_mass=target_mass,
        seed=seed
    )
    
    # Preprocess mass conditions to match training format
    mass_conditions_processed = apply_preprocessing_to_conditions(
        mass_conditions_raw, preprocessing_stats
    )
    
    # Generate samples from the conditional flow
    print("🔄 Sampling from conditional flow...")
    phase_space_samples_processed = flow.sample(
        n_samples=n_samples,
        conditions=mass_conditions_processed,
        seed=seed
    )
    
    # Inverse preprocess the samples
    phase_space_samples = inverse_preprocess_samples(
        phase_space_samples_processed, preprocessing_stats
    )
    
    print(f"✅ Generated {n_samples} conditional samples")
    print(f"   Phase space shape: {phase_space_samples.shape}")
    print(f"   Mass conditions shape: {mass_conditions_raw.shape}")
    
    # Save samples if output directory is provided
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save samples and conditions
        samples_file = output_path / f"conditional_samples_{mass_strategy}_{n_samples}.npz"
        
        np.savez_compressed(
            samples_file,
            phase_space_samples=phase_space_samples.numpy(),
            mass_conditions=mass_conditions_raw.numpy(),
            mass_strategy=mass_strategy,
            n_samples=n_samples,
            seed=seed,
            mass_range=mass_range if mass_range else [None, None],
            target_mass=target_mass if target_mass else None
        )
        
        print(f"💾 Samples saved to {samples_file}")
        
        # Save metadata
        metadata = {
            'n_samples': int(n_samples),
            'mass_strategy': mass_strategy,
            'mass_range': list(mass_range) if mass_range else None,
            'target_mass': float(target_mass) if target_mass else None,
            'seed': int(seed),
            'model_path': str(model_path),
            'sample_statistics': {
                'phase_space_mean': tf.reduce_mean(phase_space_samples, axis=0).numpy().tolist(),
                'phase_space_std': tf.math.reduce_std(phase_space_samples, axis=0).numpy().tolist(),
                'mass_mean': float(tf.reduce_mean(mass_conditions_raw)),
                'mass_std': float(tf.math.reduce_std(mass_conditions_raw)),
                'mass_min': float(tf.reduce_min(mass_conditions_raw)),
                'mass_max': float(tf.reduce_max(mass_conditions_raw))
            }
        }
        
        metadata_file = output_path / f"conditional_samples_{mass_strategy}_{n_samples}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return phase_space_samples, mass_conditions_raw


def main():
    parser = argparse.ArgumentParser(description="Generate samples from conditional TFP flows")
    
    # Required arguments
    parser.add_argument("--model_path", required=True, help="Path to saved conditional flow model")
    
    # Sampling arguments
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--mass_strategy", default="log_uniform", 
                       choices=["uniform", "log_uniform", "fixed", "kroupa"],
                       help="Strategy for generating mass conditions")
    parser.add_argument("--mass_range", nargs=2, type=float, metavar=("MIN", "MAX"),
                       help="Mass range in solar masses (e.g., --mass_range 0.1 100)")
    parser.add_argument("--target_mass", type=float, help="Fixed mass for 'fixed' strategy")
    
    # Output arguments
    parser.add_argument("--output_dir", help="Directory to save generated samples")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Convert mass_range to tuple if provided
    mass_range = tuple(args.mass_range) if args.mass_range else None
    
    # Set default output directory if not provided
    if not args.output_dir:
        model_dir = Path(args.model_path).parent
        args.output_dir = str(model_dir / "conditional_samples")
    
    print("🎯 Conditional Flow Sampling")
    print("=" * 40)
    print(f"Model: {args.model_path}")
    print(f"Samples: {args.n_samples}")
    print(f"Mass strategy: {args.mass_strategy}")
    if mass_range:
        print(f"Mass range: {mass_range[0]:.3f} - {mass_range[1]:.3f} M☉")
    if args.target_mass:
        print(f"Target mass: {args.target_mass:.3f} M☉")
    print(f"Output: {args.output_dir}")
    print()
    
    # Set random seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        # Generate samples
        phase_space_samples, mass_conditions = sample_conditional_flow(
            model_path=args.model_path,
            n_samples=args.n_samples,
            mass_strategy=args.mass_strategy,
            mass_range=mass_range,
            target_mass=args.target_mass,
            output_dir=args.output_dir,
            seed=args.seed
        )
        
        print(f"✅ Conditional sampling completed!")
        print(f"   Generated {len(phase_space_samples)} samples")
        print(f"   Phase space dimensions: {phase_space_samples.shape[1]}")
        print(f"   Mass range: [{tf.reduce_min(mass_conditions):.3e}, {tf.reduce_max(mass_conditions):.3e}] M☉")
        
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
