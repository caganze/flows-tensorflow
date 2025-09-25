#!/usr/bin/env python3
"""
Training script for Continuous Normalizing Flows (CNFs)
Compatible with GPU and designed for astrophysical data using Neural ODEs
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

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
from cnf_flows_solution import CNFNormalizingFlow, CNFFlowTrainer, load_cnf_flow
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


def load_symlib_astrophysical_data(halo_id: str, particle_pid: int, suite: str = 'eden') -> Tuple[tf.Tensor, Dict[str, Any]]:
    """
    Load astrophysical data from symlib simulation
    
    Args:
        halo_id: Halo ID (e.g., 'Halo268')
        particle_pid: Particle ID to extract
        suite: Simulation suite ('eden', 'mwest', 'symphony', 'symphony-hr')
    
    Returns:
        data: TensorFlow tensor with shape (n_samples, 6) - 6D phase space
        metadata: Dictionary with dataset information
    """
    print(f"📊 Loading symlib data: {halo_id} PID {particle_pid} from {suite}")
    
    # Validate symlib environment
    if not validate_symlib_environment():
        raise RuntimeError("❌ Symlib environment not available")
    
    # Load particle data using symlib
    data, metadata = load_particle_data(halo_id, particle_pid, suite)
    
    # Ensure we have at least 6D data (pos + vel), but handle 7D (pos + vel + mass)
    if data.shape[1] >= 6:
        data = data[:, :6]  # Take first 6 columns (pos + vel) for non-conditional training
    else:
        raise ValueError(f"Insufficient data dimensions for PID {particle_pid}: {data.shape}")
    
    # Convert to TensorFlow tensor
    data_tensor = tf.constant(data, dtype=tf.float32)
    
    print(f"✅ Loaded {data.shape[0]:,} particles with {data.shape[1]} features")
    print(f"   Data shape: {data.shape}")
    print(f"   Stellar mass: {metadata['stellar_mass']:.2e} M☉")
    
    return data_tensor, metadata


def load_particle_specific_data(filepath: str, particle_pid: int) -> Tuple[tf.Tensor, Dict[str, Any]]:
    """
    Load data for a specific particle PID from HDF5 file using symlib
    
    Args:
        filepath: Path to HDF5 file
        particle_pid: Specific particle ID to extract
        
    Returns:
        data: TensorFlow tensor with particle-specific data
        metadata: Dictionary with comprehensive metadata
    """
    print(f"🎯 Loading particle-specific data for PID {particle_pid}")
    
    # Use symlib functions
    from symlib_utils import load_particle_data
    
    try:
        # Extract halo_id and suite from filepath
        filename = os.path.basename(filepath)
        if '_' in filename:
            suite, halo_part = filename.replace('.h5', '').split('_', 1)
            halo_id = halo_part.replace('halo', 'Halo')
        else:
            raise ValueError(f"Could not parse halo_id and suite from filename: {filename}")
        
        print(f"📊 Parsed: suite={suite}, halo_id={halo_id}")
        
        # Use symlib to load particle data
        data, metadata = load_particle_data(halo_id, particle_pid, suite)
        
        if len(data) == 0:
            raise ValueError(f"❌ No data found for particle PID {particle_pid}")
        
        # Ensure we have at least 6D data (pos + vel), but handle 7D (pos + vel + mass)
        if data.shape[1] >= 6:
            data = data[:, :6]  # Take first 6 columns (pos + vel) for non-conditional training
        else:
            raise ValueError(f"Insufficient data dimensions for PID {particle_pid}: {data.shape}")
        
        print(f"✅ Extracted {len(data):,} particles for PID {particle_pid}")
        
        # Convert to TensorFlow tensor
        data_tensor = tf.constant(data, dtype=tf.float32)
        
        print(f"✅ Particle data loaded successfully: {data_tensor.shape}")
        return data_tensor, metadata
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR loading particle {particle_pid}: {e}")
        import traceback
        traceback.print_exc()
        raise


def preprocess_data(
    data: tf.Tensor, 
    standardize: bool = True,
    clip_outliers: float = 5.0
) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """
    Preprocess astrophysical data for training
    
    Args:
        data: Input data tensor
        standardize: Whether to standardize (zero mean, unit variance)
        clip_outliers: Clip outliers beyond this many standard deviations
    
    Returns:
        processed_data: Preprocessed tensor
        preprocessing_stats: Statistics for inverse transform
    """
    print("Preprocessing data...")
    
    # Compute statistics
    mean = tf.reduce_mean(data, axis=0)
    std = tf.math.reduce_std(data, axis=0)
    
    processed_data = data
    
    if standardize:
        # Standardize
        processed_data = (processed_data - mean) / (std + 1e-8)
        
        # Clip outliers
        if clip_outliers > 0:
            processed_data = tf.clip_by_value(
                processed_data, 
                -clip_outliers, 
                clip_outliers
            )
    
    preprocessing_stats = {
        'mean': mean,
        'std': std,
        'standardize': standardize,
        'clip_outliers': clip_outliers
    }
    
    print(f"✅ Preprocessing complete")
    print(f"Original range: [{tf.reduce_min(data):.3f}, {tf.reduce_max(data):.3f}]")
    print(f"Processed range: [{tf.reduce_min(processed_data):.3f}, {tf.reduce_max(processed_data):.3f}]")
    
    return processed_data, preprocessing_stats


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

def split_data(
    data: tf.Tensor, 
    train_frac: float = 0.8, 
    val_frac: float = 0.1,
    seed: int = 42
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Split data into train/validation/test sets"""
    
    tf.random.set_seed(seed)
    n_samples = tf.shape(data)[0]
    
    # Create random indices
    indices = tf.random.shuffle(tf.range(n_samples))
    
    # Calculate split points (convert TensorFlow tensor to Python int)
    n_samples_int = int(n_samples.numpy()) if hasattr(n_samples, 'numpy') else int(n_samples)
    n_train = int(n_samples_int * train_frac)
    n_val = int(n_samples_int * val_frac)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Split data
    train_data = tf.gather(data, train_indices)
    val_data = tf.gather(data, val_indices)
    test_data = tf.gather(data, test_indices)
    
    print(f"Data split:")
    print(f"  Training: {len(train_data):,} samples ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Validation: {len(val_data):,} samples ({len(val_data)/len(data)*100:.1f}%)")
    print(f"  Test: {len(test_data):,} samples ({len(test_data)/len(data)*100:.1f}%)")
    
    return train_data, val_data, test_data


def save_cnf_model_only(
    flow: CNFNormalizingFlow,
    trainer: CNFFlowTrainer,
    preprocessing_stats: Dict[str, tf.Tensor],
    metadata: Dict[str, Any],
    output_dir: str,
    model_name: str = "cnf_flow"
) -> str:
    """Save only the trained CNF model and metadata (no sampling)"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save CNF model
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
    def make_json_serializable(obj):
        """Convert NumPy/TensorFlow types to JSON-serializable Python types"""
        if hasattr(obj, 'numpy'):
            return float(obj.numpy())
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        else:
            return obj
    
    results = {
        'train_losses': [float(loss) for loss in trainer.train_losses],
        'val_losses': [float(loss) for loss in trainer.val_losses],
        'metadata': make_json_serializable(metadata),
        'model_config': {
            'input_dim': int(flow.input_dim),
            'hidden_units': list(flow.hidden_units),
            'activation': str(flow.activation),
            'integration_time': float(flow.integration_time),
            'num_integration_steps': int(flow.num_integration_steps),
            'model_type': 'cnf'
        }
    }
    
    results_path = output_path / f"{model_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ CNF Model saved to {output_path}")
    print(f"  Model: {model_path}")
    print(f"  Preprocessing: {preprocessing_path}")
    print(f"  Results: {results_path}")
    
    return str(model_path)


def generate_cnf_samples_separately(
    flow: Optional[CNFNormalizingFlow] = None,
    model_path: Optional[str] = None,
    preprocessing_stats: Optional[Dict[str, tf.Tensor]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    model_name: str = "cnf_flow",
    n_samples: Optional[int] = None
) -> str:
    """Generate samples separately from saved CNF model"""
    
    # Load flow from model_path if not provided
    if flow is None:
        if model_path is None:
            raise ValueError("Either flow or model_path must be provided")
        flow = load_cnf_flow(model_path)
    
    # Load preprocessing stats if not provided
    if preprocessing_stats is None:
        if model_path is None:
            raise ValueError("preprocessing_stats must be provided if model_path is not given")
        preprocessing_path = model_path.replace('.npz', '_preprocessing.npz')
        preprocessing_data = np.load(preprocessing_path, allow_pickle=True)
        preprocessing_stats = {k: tf.constant(v) for k, v in preprocessing_data.items()}
    
    # Load metadata if not provided
    if metadata is None:
        if model_path is None:
            raise ValueError("metadata must be provided if model_path is not given")
        # Try to load from results file
        results_path = model_path.replace('.npz', '_results.json')
        if Path(results_path).exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
                metadata = results.get('metadata', {})
        else:
            raise ValueError(f"Metadata not found at {results_path}")
    
    # Set output_dir from model_path if not provided
    if output_dir is None:
        if model_path is None:
            raise ValueError("output_dir must be provided if model_path is not given")
        output_dir = str(Path(model_path).parent)
    
    output_path = Path(output_dir)
    
    # MANDATORY: Use Kroupa IMF for all sample generation
    stellar_mass = metadata.get('stellar_mass', None)
    
    # Kroupa IMF is MANDATORY - validate requirements
    if stellar_mass is None:
        raise ValueError("❌ No stellar mass found in metadata. Kroupa IMF requires stellar mass.")
    
    # Use Kroupa IMF for realistic sampling - NO FALLBACKS
    print(f"🌟 Using MANDATORY Kroupa IMF sampling for stellar mass: {stellar_mass:.2e} M☉")
    
    # For CNF, we need to generate samples differently since it doesn't use masses directly
    # We'll generate from the CNF and then assign masses using Kroupa IMF
    if n_samples is None:
        # Calculate adaptive number of samples based on stellar mass
        base_samples = int(stellar_mass / 1e6 * 100000)  # 100k samples per 1M solar masses
        n_samples = max(10000, min(1000000, base_samples))
    
    print(f"🎲 Generating {n_samples:,} samples from CNF...")
    samples = flow.sample(n_samples, seed=42)
    
    # Inverse preprocess samples
    if preprocessing_stats.get('standardize', True):
        mean = preprocessing_stats['mean']
        std = preprocessing_stats['std']
        samples = samples * std + mean
    
    # Generate Kroupa masses separately
    from kroupa_imf import kroupa_masses
    masses = kroupa_masses(n_samples, seed=42)
    
    # Validate CNF sampling results
    if np.any(np.isnan(samples.numpy())):
        raise ValueError("❌ CNF sampling produced NaN samples")
    
    if np.any(np.isnan(masses)):
        raise ValueError("❌ Kroupa IMF sampling produced NaN masses")
    
    print(f"✅ CNF sampling successful: {n_samples:,} samples with valid masses")
    
    # Use optimized I/O strategy based on sample count
    samples_dir = Path(str(output_path).replace('trained_flows', 'samples'))
    
    try:
        # Prepare comprehensive metadata for samples
        samples_metadata = {
            **make_json_serializable(metadata),  # Include all existing metadata (converted to JSON-serializable)
            'n_samples': n_samples,
            'model_name': model_name,
            'model_type': 'cnf',
            'created_timestamp': time.time(),
            'created_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tensorflow_version': tf.__version__,
            'has_kroupa_masses': True
        }
        
        samples_metadata.update({
            'total_stellar_mass': float(np.sum(masses)),
            'mean_stellar_mass': float(np.mean(masses))
        })
        
        # Use optimized save strategy (automatically chooses HDF5 vs NPZ based on size)
        saved_files = save_samples_optimized(
            samples=samples,
            masses=masses,
            output_dir=str(samples_dir),
            model_name=model_name,
            metadata=samples_metadata
        )
        
        print(f"✅ Saved {n_samples:,} samples using optimized I/O:")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type.upper()}: {file_path}")
        print(f"  Kroupa masses: ✅ included")
        
        return str(samples_dir)
        
    except Exception as e:
        print(f"⚠️ Sample generation failed: {e}")
        raise


def train_and_save_cnf_flow(
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
    High-level function to train and save a CNF for a specific particle.
    
    Returns:
        Tuple of (model_path, samples_path)
    """
    # Set up comprehensive logging
    log_dir = f"{output_dir}/logs"
    logger = ComprehensiveLogger(log_dir, "cnf_training", particle_pid)
    
    logger.info(f"🚀 Starting CNF training for particle PID {particle_pid}")
    logger.info(f"📋 Training parameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Hidden units: {hidden_units}")
    logger.info(f"  Activation: {activation}")
    logger.info(f"  Integration time: {integration_time}")
    logger.info(f"  Integration steps: {num_integration_steps}")
    logger.info(f"  Generate samples: {generate_samples}")
    logger.info(f"  N samples: {n_samples}")
    logger.info(f"  Use Kroupa IMF: TRUE (MANDATORY)")
    
    try:
        # Set up GPU
        logger.info("🔧 Setting up GPU...")
        setup_gpu()
        
        # Load and preprocess data for specific particle
        logger.info(f"📊 Loading data for PID {particle_pid}...")
        data, metadata = load_symlib_astrophysical_data(halo_id, particle_pid, suite)
        
        if len(data) == 0:
            raise ValueError(f"No data found for particle PID {particle_pid}")
        
        # Preprocessing
        logger.info("🔧 Preprocessing data...")
        preprocessed_data, preprocessing_stats = preprocess_data(data)
        
        # Split data
        logger.info("🔀 Splitting data...")
        train_data, val_data, test_data = split_data(
            preprocessed_data, 
            val_frac=validation_split
        )
        
        # Create CNF
        logger.info(f"🏗️ Creating continuous normalizing flow...")
        flow = CNFNormalizingFlow(
            input_dim=6,
            hidden_units=hidden_units,
            activation=activation,
            integration_time=integration_time,
            num_integration_steps=num_integration_steps,
            name=f'cnf_pid{particle_pid}'
        )
        
        # Create trainer
        logger.info("🎯 Creating CNF trainer...")
        trainer = CNFFlowTrainer(
            flow=flow,
            learning_rate=learning_rate
        )
        
        # Train
        logger.info(f"🏋️ Training CNF for {epochs} epochs...")
        trainer.train(
            train_data=train_data,
            val_data=val_data,
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
            'model_type': 'cnf',
            'integration_time': integration_time,
            'num_integration_steps': num_integration_steps,
            'use_kroupa_imf': True,  # MANDATORY - always True
            'n_samples_requested': n_samples if generate_samples else 0
        }
        
        # Save model first (critical - don't lose trained model)
        logger.info("💾 Saving trained CNF model...")
        model_path = save_cnf_model_only(
            flow=flow,
            trainer=trainer,
            preprocessing_stats=preprocessing_stats,
            metadata=enhanced_metadata,
            output_dir=output_dir,
            model_name=f"cnf_model_pid{particle_pid}"
        )
        
        # Generate samples separately (can be done later if memory issues)
        samples_path = None
        if generate_samples:
            logger.info("🎲 Generating CNF samples...")
            try:
                samples_path = generate_cnf_samples_separately(
                    flow=flow,
                    preprocessing_stats=preprocessing_stats,
                    metadata=enhanced_metadata,
                    output_dir=output_dir,
                    model_name=f"cnf_model_pid{particle_pid}",
                    n_samples=n_samples
                )
            except Exception as e:
                logger.warning(f"⚠️ Sample generation failed: {e}")
                logger.info("📋 CNF model saved successfully - samples can be generated later")
                samples_path = None
        
        logger.info(f"✅ CNF training completed for PID {particle_pid}")
        logger.log_metric("final_training_loss", trainer.train_losses[-1])
        if trainer.val_losses:
            logger.log_metric("final_validation_loss", trainer.val_losses[-1])
        
        logger.mark_completed(True, f"Successfully trained CNF model for PID {particle_pid}")
        return model_path, samples_path
        
    except Exception as e:
        logger.log_error_with_traceback(e, f"CNF training PID {particle_pid}")
        logger.mark_completed(False, f"CNF training failed for PID {particle_pid}: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train Continuous Normalizing Flows (CNFs)")
    
    # Data arguments (symlib)
    parser.add_argument("--halo_id", required=True, help="Halo ID (e.g., Halo268)")
    parser.add_argument("--particle_pid", type=int, required=True, help="Specific particle ID to process")
    parser.add_argument("--suite", default="eden", help="Simulation suite name (default: eden)")
    parser.add_argument("--output_dir", help="Output directory for model and results (default: /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/cnf_output/)")
    
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
    
    # Sampling arguments
    parser.add_argument("--generate-samples", action="store_true", default=True, help="Generate samples after training")
    parser.add_argument("--no-generate-samples", dest="generate_samples", action="store_false", help="Skip sample generation")
    # MANDATORY: Kroupa IMF is always enabled - no option to disable
    parser.add_argument("--use_kroupa_imf", action="store_true", default=True, help="Use Kroupa IMF for sample generation (MANDATORY - always enabled)")
    parser.add_argument("--n_samples", type=int, default=100000, help="Number of samples to generate (overridden by Kroupa IMF)")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_name", default="cnf_flow", help="Model name for saving")
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if not args.output_dir:
        args.output_dir = "/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/cnf_output/"
    
    # MANDATORY: Enforce Kroupa IMF usage
    if not args.use_kroupa_imf:
        raise ValueError("❌ CRITICAL: Kroupa IMF is MANDATORY for all training runs. Remove --no-kroupa-imf if used.")
    
    print("🚀 Continuous Normalizing Flow Training (Symlib)")
    print("=" * 50)
    print(f"Halo ID: {args.halo_id}")
    print(f"Particle PID: {args.particle_pid}")
    print(f"Suite: {args.suite}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model config: {args.hidden_units} hidden units, {args.activation} activation")
    print(f"Integration: T={args.integration_time}, steps={args.num_integration_steps}")
    print(f"Training config: {args.epochs} epochs, batch size {args.batch_size}")
    print("🌟 Kroupa IMF: MANDATORY (enforced)")
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
    
    # If particle_pid is specified, use the high-level train_and_save_cnf_flow function
    if args.particle_pid:
        print(f"🎯 Training CNF for particle PID {args.particle_pid}")
        model_path, samples_path = train_and_save_cnf_flow(
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
            generate_samples=args.generate_samples,
            n_samples=args.n_samples,
            validation_split=0.2,
            early_stopping_patience=50,
            reduce_lr_patience=20
        )
        print(f"✅ CNF training completed successfully!")
        print(f"Model saved to: {model_path}")
        if samples_path:
            print(f"Samples saved to: {samples_path}")
        return
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    data, metadata = load_symlib_astrophysical_data(args.halo_id, args.particle_pid, args.suite)
    
    processed_data, preprocessing_stats = preprocess_data(
        data, 
        standardize=not args.no_standardize,
        clip_outliers=args.clip_outliers
    )
    
    # Split data
    train_data, val_data, test_data = split_data(processed_data, seed=args.seed)
    
    # Create CNF model
    print(f"\n🔄 Creating continuous normalizing flow...")
    flow = CNFNormalizingFlow(
        input_dim=6,  # Always 6D for non-conditional flows (pos + vel)
        hidden_units=args.hidden_units,
        activation=args.activation,
        integration_time=args.integration_time,
        num_integration_steps=args.num_integration_steps,
        name='main_cnf'
    )
    
    # Create trainer
    trainer = CNFFlowTrainer(flow, learning_rate=args.learning_rate)
    
    # Train the model
    print(f"\n🎯 Training the CNF...")
    start_time = time.time()
    
    trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_freq=args.validation_freq,
        verbose=True
    )
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    print(f"\n📊 Evaluating on test set...")
    test_log_probs = flow.log_prob(test_data)
    test_loss = float(-tf.reduce_mean(test_log_probs))
    
    print(f"Test loss: {test_loss:.4f}")
    
    # Generate samples for visualization
    print(f"\n🎲 Generating samples...")
    n_samples_viz = min(5000, len(test_data))
    generated_samples = flow.sample(n_samples_viz, seed=args.seed + 999)
    
    print(f"Generated {len(generated_samples)} samples")
    
    # Save results
    print(f"\n💾 Saving model and results...")
    
    # Add training results to metadata - ensure JSON serializable
    metadata.update({
        'training_time_seconds': float(training_time),
        'final_train_loss': float(trainer.train_losses[-1]),
        'final_val_loss': float(trainer.val_losses[-1]) if trainer.val_losses else None,
        'test_loss': float(test_loss),
        'gpu_used': bool(gpu_available),
        'model_type': 'cnf'
    })
    
    # Save CNF model
    model_path = save_cnf_model_only(
        flow=flow,
        trainer=trainer,
        preprocessing_stats=preprocessing_stats,
        metadata=metadata,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
    
    # Generate samples if requested
    if args.generate_samples:
        samples_path = generate_cnf_samples_separately(
            flow=flow,
            preprocessing_stats=preprocessing_stats,
            metadata=metadata,
            output_dir=args.output_dir,
            model_name=args.model_name,
            n_samples=args.n_samples
        )
    
    print(f"📊 Final training loss: {trainer.train_losses[-1]:.6f}")
    if trainer.val_losses:
        print(f"📊 Final validation loss: {trainer.val_losses[-1]:.6f}")
    
    # Final summary
    print(f"\n✅ CNF TRAINING COMPLETED")
    print("=" * 50)
    print(f"Training time: {training_time:.1f}s")
    print(f"Final train loss: {trainer.train_losses[-1]:.4f}")
    if trainer.val_losses:
        print(f"Final validation loss: {trainer.val_losses[-1]:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()

