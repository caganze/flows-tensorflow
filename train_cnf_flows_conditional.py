#!/usr/bin/env python3
"""
Conditional Continuous Normalizing Flows (CNFs) training script
Conditions the CNF on mass distribution to learn p(xi|mass) for each particle using Neural ODEs
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Configure TensorFlow environment before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
# Set CUDA paths for TensorFlow before importing
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0'

# Configure TensorFlow threading early to avoid conflicts
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import h5py

# Import our CNF flow implementation
from cnf_flows_solution import ConditionalCNFNormalizingFlow, ConditionalCNFFlowTrainer, load_cnf_flow
from kroupa_imf import sample_with_kroupa_imf
from optimized_io import save_samples_optimized
from comprehensive_logging import ComprehensiveLogger
from symlib_utils import load_particle_data, get_output_paths, validate_symlib_environment

# TFP aliases
tfd = tfp.distributions
tfb = tfp.bijectors


def setup_gpu():
    """Configure GPU settings for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Try to enable memory growth, but don't fail if already initialized
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled")
        except RuntimeError as e:
            if "Physical devices cannot be modified after being initialized" in str(e):
                print("⚠️ GPU already initialized, continuing with current settings")
            else:
                print(f"⚠️ GPU configuration warning: {e}")
        
        print(f"✅ GPU ready: {gpus[0]}")
        return True
    else:
        print("⚠️ No GPU found - using CPU")
        return False


def load_conditional_data(halo_id: str, particle_pid: int, suite: str = 'eden') -> Tuple[tf.Tensor, tf.Tensor, Dict[str, Any]]:
    """
    Load astrophysical data with mass conditioning from symlib simulation
    
    Args:
        halo_id: Halo ID (e.g., 'Halo268')
        particle_pid: Particle ID to extract
        suite: Simulation suite ('eden', 'mwest', 'symphony', 'symphony-hr')
    
    Returns:
        phase_space_data: TensorFlow tensor with shape (n_samples, 6) - 6D phase space
        mass_conditions: TensorFlow tensor with shape (n_samples, 1) - stellar masses
        metadata: Dictionary with dataset information
    """
    print(f"📊 Loading conditional data: {halo_id} PID {particle_pid} from {suite}")
    
    # Validate symlib environment
    if not validate_symlib_environment():
        raise RuntimeError("❌ Symlib environment not available")
    
    # Load particle data using symlib (returns 7D: position + velocity + mass)
    data, metadata = load_particle_data(halo_id, particle_pid, suite)
    
    # Phase space data (first 6 columns: position + velocity)
    phase_space = data[:, :6]  # Shape: (N, 6)
    
    # Mass data (7th column)
    masses = data[:, 6:7]  # Shape: (N, 1)
    
    print(f"✅ Loaded {len(masses)} individual particle masses from data")
    print(f"   Mass range: [{masses.min():.2e}, {masses.max():.2e}] M☉")
    print(f"   Mean mass: {masses.mean():.2e} M☉")
    
    # Convert to TensorFlow tensors
    phase_space_tensor = tf.constant(phase_space, dtype=tf.float32)
    mass_tensor = tf.constant(masses, dtype=tf.float32)
    
    print(f"✅ Loaded {phase_space.shape[0]:,} particles with 6D phase space + mass conditioning")
    print(f"   Phase space shape: {phase_space.shape}")
    print(f"   Mass conditions shape: {masses.shape}")
    print(f"   Stellar mass range: [{np.min(masses):.2e}, {np.max(masses):.2e}] M☉")
    
    # Update metadata with conditioning info
    # Handle zero masses safely for log calculation
    masses_nonzero = masses[masses > 0]
    if len(masses_nonzero) > 0:
        log_mass_min = float(np.log10(np.min(masses_nonzero)))
        log_mass_max = float(np.log10(np.max(masses_nonzero)))
    else:
        log_mass_min = log_mass_max = 0.0
    
    metadata.update({
        'has_mass_conditioning': True,
        'mass_range': [float(np.min(masses)), float(np.max(masses))],
        'log_mass_range': [log_mass_min, log_mass_max],
        'n_zero_masses': int(np.sum(masses == 0))
    })
    
    return phase_space_tensor, mass_tensor, metadata


def preprocess_conditional_data(
    phase_space_data: tf.Tensor,
    mass_conditions: tf.Tensor,
    standardize: bool = True,
    clip_outliers: float = 5.0,
    log_transform_mass: bool = True
) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
    """
    Preprocess astrophysical data and mass conditions for conditional CNF training
    
    Args:
        phase_space_data: Input phase space tensor
        mass_conditions: Mass conditioning tensor
        standardize: Whether to standardize (zero mean, unit variance)
        clip_outliers: Clip outliers beyond this many standard deviations
        log_transform_mass: Whether to log-transform masses for better conditioning
    
    Returns:
        processed_phase_space: Preprocessed phase space tensor
        processed_conditions: Preprocessed mass conditions
        preprocessing_stats: Statistics for inverse transform
    """
    print("Preprocessing conditional data...")
    
    # Preprocess phase space data
    ps_mean = tf.reduce_mean(phase_space_data, axis=0)
    ps_std = tf.math.reduce_std(phase_space_data, axis=0)
    
    processed_phase_space = phase_space_data
    
    if standardize:
        # Standardize phase space
        processed_phase_space = (processed_phase_space - ps_mean) / (ps_std + 1e-8)
        
        # Clip outliers
        if clip_outliers > 0:
            processed_phase_space = tf.clip_by_value(
                processed_phase_space, 
                -clip_outliers, 
                clip_outliers
            )
    
    # Preprocess mass conditions
    processed_conditions = mass_conditions
    
    if log_transform_mass:
        # Log transform masses for better numerical stability
        processed_conditions = tf.math.log(mass_conditions + 1e-10)
    
    # Standardize mass conditions
    mass_mean = tf.reduce_mean(processed_conditions, axis=0)
    mass_std = tf.math.reduce_std(processed_conditions, axis=0)
    
    if standardize:
        processed_conditions = (processed_conditions - mass_mean) / (mass_std + 1e-8)
    
    preprocessing_stats = {
        'ps_mean': ps_mean,
        'ps_std': ps_std,
        'mass_mean': mass_mean,
        'mass_std': mass_std,
        'standardize': standardize,
        'clip_outliers': clip_outliers,
        'log_transform_mass': log_transform_mass
    }
    
    print(f"✅ Conditional preprocessing complete")
    print(f"Phase space range: [{tf.reduce_min(processed_phase_space):.3f}, {tf.reduce_max(processed_phase_space):.3f}]")
    print(f"Mass conditions range: [{tf.reduce_min(processed_conditions):.3f}, {tf.reduce_max(processed_conditions):.3f}]")
    
    return processed_phase_space, processed_conditions, preprocessing_stats


def make_json_serializable(obj):
    """Convert NumPy/TensorFlow types to JSON-serializable Python types"""
    if hasattr(obj, 'numpy'):
        return float(obj.numpy())
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif hasattr(obj, 'dtype') and hasattr(obj, 'numpy'):
        # Handle TensorFlow tensors
        try:
            return float(obj.numpy())
        except:
            return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    else:
        return obj

def split_conditional_data(
    phase_space_data: tf.Tensor,
    mass_conditions: tf.Tensor,
    train_frac: float = 0.8, 
    val_frac: float = 0.1,
    seed: int = 42
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Split conditional data into train/validation/test sets"""
    
    tf.random.set_seed(seed)
    n_samples = tf.shape(phase_space_data)[0]
    
    # Create random indices
    indices = tf.random.shuffle(tf.range(n_samples))
    
    # Calculate split points
    n_samples_int = int(n_samples.numpy()) if hasattr(n_samples, 'numpy') else int(n_samples)
    n_train = int(n_samples_int * train_frac)
    n_val = int(n_samples_int * val_frac)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Split phase space data
    train_ps = tf.gather(phase_space_data, train_indices)
    val_ps = tf.gather(phase_space_data, val_indices)
    test_ps = tf.gather(phase_space_data, test_indices)
    
    # Split mass conditions
    train_mass = tf.gather(mass_conditions, train_indices)
    val_mass = tf.gather(mass_conditions, val_indices)
    test_mass = tf.gather(mass_conditions, test_indices)
    
    print(f"Conditional data split:")
    print(f"  Training: {len(train_ps):,} samples ({len(train_ps)/len(phase_space_data)*100:.1f}%)")
    print(f"  Validation: {len(val_ps):,} samples ({len(val_ps)/len(phase_space_data)*100:.1f}%)")
    print(f"  Test: {len(test_ps):,} samples ({len(test_ps)/len(phase_space_data)*100:.1f}%)")
    
    return train_ps, train_mass, val_ps, val_mass, test_ps, test_mass


def save_conditional_cnf_model_only(
    flow: ConditionalCNFNormalizingFlow,
    trainer: ConditionalCNFFlowTrainer,
    preprocessing_stats: Dict[str, tf.Tensor],
    metadata: Dict[str, Any],
    output_dir: str,
    model_name: str = "conditional_cnf_flow"
) -> str:
    """Save only the trained conditional CNF model and metadata (no sampling)"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save Conditional CNF model
    model_path = output_path / f"{model_name}.npz"
    flow.save(str(model_path))
    
    # Save preprocessing statistics
    preprocessing_path = output_path / f"{model_name}_preprocessing.npz"
    np.savez(
        preprocessing_path,
        **{k: v.numpy() if isinstance(v, tf.Tensor) else v 
           for k, v in preprocessing_stats.items()}
    )
    
    # Save training results
    
    results = {
        'train_losses': [float(loss) for loss in trainer.train_losses],
        'val_losses': [float(loss) for loss in trainer.val_losses],
        'metadata': make_json_serializable(metadata),
        'model_config': {
            'input_dim': int(flow.input_dim),
            'condition_dim': int(flow.condition_dim),
            'hidden_units': list(flow.hidden_units),
            'activation': str(flow.activation),
            'integration_time': float(flow.integration_time),
            'num_integration_steps': int(flow.num_integration_steps),
            'model_type': 'conditional_cnf'
        }
    }
    
    results_path = output_path / f"{model_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Conditional CNF Model saved to {output_path}")
    print(f"  Model: {model_path}")
    print(f"  Preprocessing: {preprocessing_path}")
    print(f"  Results: {results_path}")
    
    return str(model_path)


def train_and_save_conditional_cnf_flow(
    halo_id: str,
    particle_pid: int,
    suite: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    hidden_units: List[int] = [64, 64],
    activation: str = 'tanh',
    integration_time: float = 1.0,
    num_integration_steps: int = 10,
    generate_samples: bool = True,
    n_samples: int = 100000,
    validation_split: float = 0.2,
    early_stopping_patience: int = 50,
    reduce_lr_patience: int = 20
) -> Tuple[str, str]:
    """
    High-level function to train and save a conditional CNF for a specific particle.
    
    Returns:
        Tuple of (model_path, samples_path)
    """
    # Set up comprehensive logging
    log_dir = f"{output_dir}/logs"
    logger = ComprehensiveLogger(log_dir, "conditional_cnf_training", particle_pid)
    
    logger.info(f"🚀 Starting conditional CNF training for particle PID {particle_pid}")
    logger.info(f"📋 Training parameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Hidden units: {hidden_units}")
    logger.info(f"  Activation: {activation}")
    logger.info(f"  Integration time: {integration_time}")
    logger.info(f"  Integration steps: {num_integration_steps}")
    logger.info(f"  Mass conditioning: TRUE")
    
    try:
        # Set up GPU
        logger.info("🔧 Setting up GPU...")
        setup_gpu()
        
        # Load and preprocess conditional data
        logger.info(f"📊 Loading conditional data for PID {particle_pid}...")
        phase_space_data, mass_conditions, metadata = load_conditional_data(halo_id, particle_pid, suite)
        
        if len(phase_space_data) == 0:
            raise ValueError(f"No data found for particle PID {particle_pid}")
        
        # Preprocessing
        logger.info("🔧 Preprocessing conditional data...")
        processed_ps, processed_mass, preprocessing_stats = preprocess_conditional_data(
            phase_space_data, mass_conditions
        )
        
        # Split data
        logger.info("🔀 Splitting conditional data...")
        train_ps, train_mass, val_ps, val_mass, test_ps, test_mass = split_conditional_data(
            processed_ps, processed_mass, val_frac=validation_split
        )
        
        # Create conditional CNF
        logger.info(f"🏗️ Creating conditional continuous normalizing flow...")
        flow = ConditionalCNFNormalizingFlow(
            input_dim=6,  # 6D phase space
            condition_dim=1,  # 1D mass conditioning
            hidden_units=hidden_units,
            activation=activation,
            integration_time=integration_time,
            num_integration_steps=num_integration_steps,
            name=f'conditional_cnf_pid{particle_pid}'
        )
        
        # Create trainer
        logger.info("🎯 Creating conditional CNF trainer...")
        trainer = ConditionalCNFFlowTrainer(
            flow=flow,
            learning_rate=learning_rate
        )
        
        # Train
        logger.info(f"🏋️ Training conditional CNF for {epochs} epochs...")
        trainer.train(
            train_data=train_ps,
            train_conditions=train_mass,
            val_data=val_ps,
            val_conditions=val_mass,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Enhanced metadata
        logger.info("📋 Preparing enhanced metadata...")
        enhanced_metadata = {
            **metadata,
            'particle_pid': particle_pid,
            'training_epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'validation_split': validation_split,
            'model_type': 'conditional_cnf',
            'integration_time': integration_time,
            'num_integration_steps': num_integration_steps,
            'conditioning_variables': ['stellar_mass'],
            'n_samples_requested': n_samples if generate_samples else 0
        }
        
        # Save model
        logger.info("💾 Saving trained conditional CNF model...")
        model_path = save_conditional_cnf_model_only(
            flow=flow,
            trainer=trainer,
            preprocessing_stats=preprocessing_stats,
            metadata=enhanced_metadata,
            output_dir=output_dir,
            model_name=f"conditional_cnf_model_pid{particle_pid}"
        )
        
        logger.info(f"✅ Conditional CNF training completed for PID {particle_pid}")
        logger.log_metric("final_training_loss", trainer.train_losses[-1])
        if trainer.val_losses:
            logger.log_metric("final_validation_loss", trainer.val_losses[-1])
        
        logger.mark_completed(True, f"Successfully trained conditional CNF model for PID {particle_pid}")
        
        # Note: Conditional sampling would require specifying mass conditions
        # For now, just return the model path
        return model_path, None
        
    except Exception as e:
        logger.log_error_with_traceback(e, f"conditional CNF training PID {particle_pid}")
        logger.mark_completed(False, f"Conditional CNF training failed for PID {particle_pid}: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train conditional Continuous Normalizing Flows (CNFs)")
    
    # Data arguments (symlib)
    parser.add_argument("--halo_id", required=True, help="Halo ID (e.g., Halo268)")
    parser.add_argument("--particle_pid", type=int, required=True, help="Specific particle ID to process")
    parser.add_argument("--suite", default="eden", help="Simulation suite name (default: eden)")
    parser.add_argument("--output_dir", help="Output directory for model and results")
    
    # Model arguments
    parser.add_argument("--hidden_units", nargs='+', type=int, default=[64, 64], help="Hidden units per layer")
    parser.add_argument("--activation", default="tanh", help="Activation function (tanh recommended for CNFs)")
    parser.add_argument("--integration_time", type=float, default=1.0, help="Total integration time T")
    parser.add_argument("--num_integration_steps", type=int, default=10, help="Number of ODE solver steps")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--validation_freq", type=int, default=10, help="Validation frequency")
    
    # Data preprocessing
    parser.add_argument("--no_standardize", action="store_true", help="Skip data standardization")
    parser.add_argument("--clip_outliers", type=float, default=5.0, help="Outlier clipping threshold")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_name", default="conditional_cnf_flow", help="Model name for saving")
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if not args.output_dir:
        args.output_dir = "/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/cnf_output_conditional/"
    
    print("🚀 Conditional Continuous Normalizing Flow Training (Symlib)")
    print("=" * 70)
    print(f"Halo ID: {args.halo_id}")
    print(f"Particle PID: {args.particle_pid}")
    print(f"Suite: {args.suite}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model config: {args.hidden_units} hidden units, {args.activation} activation")
    print(f"Integration: T={args.integration_time}, steps={args.num_integration_steps}")
    print(f"Training config: {args.epochs} epochs, batch size {args.batch_size}")
    print("🌟 Conditioning: Mass distribution")
    print()
    
    # Configure TensorFlow threading early to avoid conflicts
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError:
        # TensorFlow already initialized, skip configuration
        pass
    
    # Set seeds for reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup GPU
    gpu_available = setup_gpu()
    
    # Train conditional CNF
    print(f"🎯 Training conditional CNF for particle PID {args.particle_pid}")
    model_path, samples_path = train_and_save_conditional_cnf_flow(
        halo_id=args.halo_id,
        particle_pid=args.particle_pid,
        suite=args.suite,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        activation=args.activation,
        integration_time=args.integration_time,
        num_integration_steps=args.num_integration_steps,
        generate_samples=False,  # Conditional sampling requires specific mass conditions
        validation_split=0.2,
        early_stopping_patience=50,
        reduce_lr_patience=20
    )
    
    print(f"✅ Conditional CNF training completed successfully!")
    print(f"Model saved to: {model_path}")
    print("💡 To generate samples, specify mass conditions for the conditional CNF")


if __name__ == "__main__":
    main()

