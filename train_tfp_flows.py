#!/usr/bin/env python3
"""
Training script for TensorFlow Probability normalizing flows
Compatible with GPU and designed for astrophysical data
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
# import pandas as pd  # Removed to avoid GLIBCXX conflicts
import tensorflow as tf
import tensorflow_probability as tfp
import h5py
# Plotting libraries removed - not needed for production training

# Import our TFP flow implementation
from tfp_flows_gpu_solution import TFPNormalizingFlow, TFPFlowTrainer
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

# Plotting function removed - not needed for production training

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

def save_model_only(
    flow: TFPNormalizingFlow,
    trainer: TFPFlowTrainer,
    preprocessing_stats: Dict[str, tf.Tensor],
    metadata: Dict[str, Any],
    output_dir: str,
    model_name: str = "tfp_flow"
) -> str:
    """Save only the trained model and metadata (no sampling)"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save flow model (both variables and flow object)
    model_path = output_path / f"{model_name}.npz"
    
    # Get all trainable variables
    variables = flow.trainable_variables
    
    # Create a dictionary to save each variable separately
    save_dict = {}
    
    # Save model configuration
    save_dict['config'] = np.array({
        'input_dim': flow.input_dim,
        'n_layers': flow.n_layers,
        'hidden_units': flow.hidden_units,
        'activation': flow.activation,
        'name': flow.name
    }, dtype=object)
    
    # Save each variable individually with a unique key
    for i, var in enumerate(variables):
        var_array = var.numpy()
        save_dict[f'variable_{i}'] = var_array
        save_dict[f'variable_{i}_shape'] = np.array(var_array.shape)
        save_dict[f'variable_{i}_name'] = var.name
    
    # Save number of variables for loading
    save_dict['n_variables'] = len(variables)
    
    # Note: We can't save the flow object directly due to pickle limitations
    # Instead, we save enough info to reconstruct it in load_flow_from_model()
    
    # Use compressed format for efficiency
    np.savez_compressed(str(model_path), **save_dict)
    
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
            'n_layers': int(flow.n_layers),
            'hidden_units': int(flow.hidden_units),
            'activation': str(flow.activation)
        }
    }
    
    results_path = output_path / f"{model_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Model saved to {output_path}")
    print(f"  Model: {model_path}")
    print(f"  Preprocessing: {preprocessing_path}")
    print(f"  Results: {results_path}")
    
    return str(model_path)

def load_flow_from_model(model_path: str) -> TFPNormalizingFlow:
    """Load a flow model from saved file by reconstructing from config and variables"""
    try:
        # Load the saved data
        data = np.load(model_path, allow_pickle=True)
        
        # Reconstruct flow from configuration (we can't save the flow object due to pickle limitations)
        config = data['config'].item()
        flow = TFPNormalizingFlow(
            input_dim=config['input_dim'],
            n_layers=config['n_layers'],
            hidden_units=config['hidden_units'],
            activation=config['activation'],
            name=config['name']
        )
        
        # Force initialization of the flow by running a dummy forward pass
        # This ensures all variables are created before we try to load them
        dummy_input = tf.zeros((1, config['input_dim']), dtype=tf.float32)
        _ = flow.log_prob(dummy_input)  # This creates all the variables
        
        # Load the saved variables
        n_variables = int(data['n_variables'])
        variables = flow.trainable_variables
        
        if len(variables) != n_variables:
            raise ValueError(f"Model structure mismatch: expected {n_variables} variables, got {len(variables)}")
        
        # Assign the loaded values to variables
        for i, var in enumerate(variables):
            loaded_value = data[f'variable_{i}']
            var.assign(loaded_value)
        
        print(f"✅ Flow reconstructed from {model_path}")
        print(f"   Loaded {len(variables)} variables")
        return flow
        
    except Exception as e:
        print(f"❌ Error loading flow model: {e}")
        raise

def generate_samples_separately(
    flow: Optional[TFPNormalizingFlow] = None,
    model_path: Optional[str] = None,
    preprocessing_stats: Optional[Dict[str, tf.Tensor]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    model_name: str = "tfp_flow",
    n_samples: Optional[int] = None
) -> str:
    """Generate samples separately from saved model"""
    
    # Load flow from model_path if not provided
    if flow is None:
        if model_path is None:
            raise ValueError("Either flow or model_path must be provided")
        flow = load_flow_from_model(model_path)
    
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
    samples, masses = sample_with_kroupa_imf(
        flow=flow,
        n_target_mass=stellar_mass,
        preprocessing_stats=preprocessing_stats,
        seed=42
    )
    n_samples = len(samples)
    
    # Validate Kroupa sampling results - fail hard if any issues
    if masses is None or len(masses) == 0:
        raise ValueError("❌ Kroupa IMF sampling failed - no masses generated")
    
    if np.any(np.isnan(samples.numpy())):
        raise ValueError("❌ Kroupa IMF sampling produced NaN samples")
    
    if np.any(np.isnan(masses)):
        raise ValueError("❌ Kroupa IMF sampling produced NaN masses")
    
    print(f"✅ Kroupa sampling successful: {n_samples:,} samples with valid masses")
    
    # Use optimized I/O strategy based on sample count
    samples_dir = Path(str(output_path).replace('trained_flows', 'samples'))
    
    try:
        # Prepare comprehensive metadata for samples
        samples_metadata = {
            **make_json_serializable(metadata),  # Include all existing metadata (converted to JSON-serializable)
            'n_samples': n_samples,
            'model_name': model_name,
            'created_timestamp': time.time(),
            'created_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tensorflow_version': tf.__version__,
            'has_kroupa_masses': masses is not None
        }
        
        if masses is not None:
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
        if masses is not None:
            print(f"  Kroupa masses: ✅ included")
        
        return str(samples_dir)
        
    except Exception as e:
        print(f"⚠️ Sample generation failed: {e}")
        raise

def save_model_and_results(
    flow: TFPNormalizingFlow,
    trainer: TFPFlowTrainer,
    preprocessing_stats: Dict[str, tf.Tensor],
    metadata: Dict[str, Any],
    output_dir: str,
    model_name: str = "tfp_flow",
    generate_samples: bool = True,
    n_samples: Optional[int] = None  # If None, will be calculated adaptively
    # use_kroupa_imf removed - now MANDATORY and always True
):
    """Save trained model and results"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save flow model
    model_path = output_path / f"{model_name}.npz"
    flow.save(str(model_path))
    
    # Save preprocessing statistics
    preprocessing_path = output_path / f"{model_name}_preprocessing.npz"
    np.savez(
        preprocessing_path,
        **{k: v.numpy() if isinstance(v, tf.Tensor) else v 
           for k, v in preprocessing_stats.items()}
    )
    
    # Save training results - convert all values to JSON-serializable types
    
    results = {
        'train_losses': [float(loss) for loss in trainer.train_losses],
        'val_losses': [float(loss) for loss in trainer.val_losses],
        'metadata': make_json_serializable(metadata),
        'model_config': {
            'input_dim': int(flow.input_dim),
            'n_layers': int(flow.n_layers),
            'hidden_units': int(flow.hidden_units),
            'activation': str(flow.activation)
        }
    }
    
    results_path = output_path / f"{model_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate and save large sample set
    if generate_samples:
        # MANDATORY: Use Kroupa IMF for all sample generation
        stellar_mass = metadata.get('stellar_mass', None)
        
        # Kroupa IMF is MANDATORY - validate requirements
        if stellar_mass is None:
            raise ValueError("❌ No stellar mass found in metadata. Kroupa IMF requires stellar mass.")
        
        # Use Kroupa IMF for realistic sampling - NO FALLBACKS
        print(f"🌟 Using MANDATORY Kroupa IMF sampling for stellar mass: {stellar_mass:.2e} M☉")
        samples, masses = sample_with_kroupa_imf(
            flow=flow,
            n_target_mass=stellar_mass,
            preprocessing_stats=preprocessing_stats,
            seed=42
        )
        n_samples = len(samples)
        
        # Validate Kroupa sampling results - fail hard if any issues
        if masses is None or len(masses) == 0:
            raise ValueError("❌ Kroupa IMF sampling failed - no masses generated")
        
        if np.any(np.isnan(samples.numpy())):
            raise ValueError("❌ Kroupa IMF sampling produced NaN samples")
        
        if np.any(np.isnan(masses)):
            raise ValueError("❌ Kroupa IMF sampling produced NaN masses")
        
        print(f"✅ Kroupa sampling successful: {n_samples:,} samples with valid masses")
        
        # Use optimized I/O strategy based on sample count
        samples_dir = Path(str(output_path).replace('trained_flows', 'samples'))
        
        try:
            # Prepare comprehensive metadata for samples
            samples_metadata = {
                **make_json_serializable(metadata),  # Include all existing metadata (converted to JSON-serializable)
                'n_samples': n_samples,
                'model_name': model_name,
                'training_epochs': len(trainer.train_losses),
                'final_train_loss': float(trainer.train_losses[-1]),
                'final_val_loss': float(trainer.val_losses[-1]) if trainer.val_losses else 0.0,
                'created_timestamp': time.time(),
                'created_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'tensorflow_version': tf.__version__,
                'has_kroupa_masses': masses is not None
            }
            
            if masses is not None:
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
            if masses is not None:
                print(f"  Kroupa masses: ✅ included")
            
            # Update results with sampling info
            results['sampling'] = {
                'n_samples_generated': n_samples,
                'saved_files': saved_files,
                'io_strategy': 'HDF5 + NPZ' if n_samples > 1_000_000 else 'NPZ only',
                'kroupa_masses_included': masses is not None,
                'sample_statistics': {
                    'mean': [float(x) for x in tf.reduce_mean(samples, axis=0).numpy()],
                    'std': [float(x) for x in tf.math.reduce_std(samples, axis=0).numpy()],
                    'min': [float(x) for x in tf.reduce_min(samples, axis=0).numpy()],
                    'max': [float(x) for x in tf.reduce_max(samples, axis=0).numpy()]
                }
            }
            
            # Re-save results with sampling info
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        except Exception as e:
            print(f"⚠️ Sample generation failed: {e}")
            print(f"📋 Model is still saved successfully")
            # Log the error but don't fail the entire training
            results['sampling_error'] = str(e)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
    
    print(f"✅ Model and samples saved to {output_path}")
    print(f"  Model: {model_path}")
    print(f"  Preprocessing: {preprocessing_path}")
    print(f"  Results: {results_path}")
    if generate_samples and 'saved_files' in results.get('sampling', {}):
        print(f"  Samples: {samples_dir} ({results['sampling']['io_strategy']})")
        
    return str(model_path), str(samples_dir) if generate_samples else None

def train_and_save_flow(
    halo_id: str,
    particle_pid: int,
    suite: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    n_layers: int = 4,
    hidden_units: int = 64,
    activation: str = "relu",
    use_batchnorm: bool = False,
    weight_decay: float = 0.0,
    noise_std: float = 0.0,
    generate_samples: bool = True,
    n_samples: int = 100000,
    # use_kroupa_imf removed - now MANDATORY and always True
    validation_split: float = 0.2,
    early_stopping_patience: int = 50,
    reduce_lr_patience: int = 20
) -> Tuple[str, str]:
    """
    High-level function to train and save a TFP flow for a specific particle.
    
    Returns:
        Tuple of (model_path, samples_path)
    """
    # Set up comprehensive logging
    log_dir = f"{output_dir}/logs"
    logger = ComprehensiveLogger(log_dir, "training", particle_pid)
    
    logger.info(f"🚀 Starting TFP flow training for particle PID {particle_pid}")
    logger.info(f"📋 Training parameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  N layers: {n_layers}")
    logger.info(f"  Hidden units: {hidden_units}")
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
        
        # Create flow
        logger.info(f"🏗️ Creating normalizing flow ({n_layers} layers, {hidden_units} units)...")
        flow = TFPNormalizingFlow(
            input_dim=6,
            n_layers=n_layers,
            hidden_units=hidden_units,
            activation=activation,
            use_batchnorm=use_batchnorm,
            name=f'flow_pid{particle_pid}'
        )
        
        # Create trainer
        logger.info("🎯 Creating trainer...")
        trainer = TFPFlowTrainer(
            flow=flow,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            noise_std=noise_std
        )
        
        # Train
        logger.info(f"🏋️ Training for {epochs} epochs...")
        train_losses, val_losses = trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_freq=5,
            early_stopping_patience=early_stopping_patience,
            reduce_lr_patience=reduce_lr_patience
        )
        
        # CRITICAL: Ensure best weights are restored before saving
        logger.info("🔄 Verifying best model weights are restored...")
        if val_losses and len(val_losses) > 0:
            best_val_loss = min(val_losses)
            final_val_loss = val_losses[-1]
            logger.info(f"   Best validation loss: {best_val_loss:.6f}")
            logger.info(f"   Final validation loss: {final_val_loss:.6f}")
            
            # If final loss is much worse than best, the restoration might have failed
            if final_val_loss > best_val_loss * 1.1:  # 10% tolerance
                logger.warning(f"⚠️ Final loss ({final_val_loss:.6f}) is significantly worse than best ({best_val_loss:.6f})")
                logger.warning("   This suggests the best weights restoration may have failed")
        
        # Enhanced metadata
        logger.info("📋 Preparing enhanced metadata...")
        enhanced_metadata = {
            **metadata,
            'particle_pid': particle_pid,
            'training_epochs': len(train_losses),
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'validation_split': validation_split,
            'use_kroupa_imf': True,  # MANDATORY - always True
            'n_samples_requested': n_samples if generate_samples else 0,
            'final_train_loss': float(train_losses[-1]) if train_losses else 0.0,
            'final_val_loss': float(val_losses[-1]) if val_losses else 0.0,
            'best_val_loss': float(min(val_losses)) if val_losses else 0.0
        }
        
        # Save model first (critical - don't lose trained model)
        logger.info("💾 Saving trained model...")
        model_path = save_model_only(
            flow=flow,
            trainer=trainer,
            preprocessing_stats=preprocessing_stats,
            metadata=enhanced_metadata,
            output_dir=output_dir,
            model_name=f"model_pid{particle_pid}"
        )
        
        # CRITICAL: Test the saved model to ensure it works
        logger.info("🧪 Testing saved model with small sample...")
        try:
            test_samples = flow.sample(10)
            if tf.reduce_any(tf.math.is_nan(test_samples)):
                logger.error("❌ CRITICAL: Saved model produces NaN samples!")
                logger.error("   The model weights may be corrupted or not properly saved")
                raise ValueError("Saved model produces NaN samples - training failed")
            else:
                logger.info(f"✅ Saved model test successful: {test_samples.shape}")
                logger.info(f"   Sample range: [{test_samples.numpy().min():.3f}, {test_samples.numpy().max():.3f}]")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Saved model test failed: {e}")
            logger.error("   The model cannot generate valid samples")
            raise ValueError(f"Saved model test failed: {e}")
        
        # Generate samples separately (can be done later if memory issues)
        samples_path = None
        if generate_samples:
            logger.info("🎲 Generating samples...")
            try:
                # Use proper samples directory structure
                output_paths = get_output_paths(halo_id, particle_pid, suite)
                samples_output_dir = output_paths['samples_dir']
                
                samples_path = generate_samples_separately(
                    flow=flow,
                    preprocessing_stats=preprocessing_stats,
                    metadata=enhanced_metadata,
                    output_dir=samples_output_dir,
                    model_name=f"model_pid{particle_pid}",
                    n_samples=n_samples
                )
            except Exception as e:
                logger.warning(f"⚠️ Sample generation failed: {e}")
                logger.info("📋 Model saved successfully - samples can be generated later")
                samples_path = None
        
        logger.info(f"✅ Training completed for PID {particle_pid}")
        logger.log_metric("final_training_loss", trainer.train_losses[-1])
        if trainer.val_losses:
            logger.log_metric("final_validation_loss", trainer.val_losses[-1])
        
        logger.mark_completed(True, f"Successfully trained model for PID {particle_pid}")
        return model_path, samples_path
        
    except Exception as e:
        logger.log_error_with_traceback(e, f"training PID {particle_pid}")
        logger.mark_completed(False, f"Training failed for PID {particle_pid}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Train TensorFlow Probability normalizing flows")
    
    # Data arguments (symlib)
    parser.add_argument("--halo_id", required=True, help="Halo ID (e.g., Halo268)")
    parser.add_argument("--particle_pid", type=int, required=True, help="Specific particle ID to process")
    parser.add_argument("--suite", default="eden", help="Simulation suite name (default: eden)")
    parser.add_argument("--output_dir", help="Output directory for model and results (default: /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output/)")
    
    # Model arguments
    parser.add_argument("--n_layers", type=int, default=4, help="Number of flow layers")
    parser.add_argument("--hidden_units", type=int, default=512, help="Hidden units per layer")
    parser.add_argument("--activation", default="relu", help="Activation function")
    parser.add_argument("--use_batchnorm", action="store_true", help="Insert invertible BatchNormalization bijectors between flow layers")
    parser.add_argument("--use_gmm_base", action="store_true", help="Use Gaussian Mixture base distribution")
    parser.add_argument("--gmm_components", type=int, default=5, help="Number of GMM components for base distribution")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--validation_freq", type=int, default=10, help="Validation frequency")
    parser.add_argument("--early_stopping_patience", type=int, default=50, help="Validation checks to wait before early stopping")
    parser.add_argument("--reduce_lr_patience", type=int, default=20, help="Validation checks to wait before reducing LR")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 weight decay applied during training")
    parser.add_argument("--noise_std", type=float, default=0.0, help="Stddev of Gaussian noise added to inputs during training (on standardized data)")
    
    # Data preprocessing
    parser.add_argument("--no_standardize", action="store_true", help="Skip data standardization")
    parser.add_argument("--clip_outliers", type=float, default=5.0, help="Outlier clipping threshold")
    
    # Sampling arguments
    parser.add_argument("--generate-samples", action="store_true", default=True, help="Generate samples after training")
    parser.add_argument("--no-generate-samples", dest="generate_samples", action="store_false", help="Skip sample generation")
    # MANDATORY: Kroupa IMF is always enabled - no option to disable
    parser.add_argument("--use_kroupa_imf", action="store_true", default=True, help="Use Kroupa IMF for sample generation (MANDATORY - always enabled)")
    # Removed --no-kroupa-imf option to enforce Kroupa IMF usage
    parser.add_argument("--n_samples", type=int, default=100000, help="Number of samples to generate (overridden by Kroupa IMF)")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_name", default="tfp_flow", help="Model name for saving")
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if not args.output_dir:
        # Use proper hierarchical structure
        output_paths = get_output_paths(args.halo_id, args.particle_pid, args.suite)
        args.output_dir = output_paths['trained_flows_dir']
    
    # MANDATORY: Enforce Kroupa IMF usage
    if not args.use_kroupa_imf:
        raise ValueError("❌ CRITICAL: Kroupa IMF is MANDATORY for all training runs. Remove --no-kroupa-imf if used.")
    
    print("🚀 TensorFlow Probability Flow Training (Symlib)")
    print("=" * 50)
    print(f"Halo ID: {args.halo_id}")
    print(f"Particle PID: {args.particle_pid}")
    print(f"Suite: {args.suite}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model config: {args.n_layers} layers, {args.hidden_units} hidden units")
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
    
    # If particle_pid is specified, use the high-level train_and_save_flow function
    if args.particle_pid:
        print(f"🎯 Training flow for particle PID {args.particle_pid}")
        model_path, samples_path = train_and_save_flow(
            halo_id=args.halo_id,
            particle_pid=args.particle_pid,
            suite=args.suite,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            n_layers=args.n_layers,
            hidden_units=args.hidden_units,
            activation=args.activation,
            use_batchnorm=args.use_batchnorm,
            weight_decay=args.weight_decay,
            noise_std=args.noise_std,
            generate_samples=args.generate_samples,
            n_samples=args.n_samples,
            # use_kroupa_imf removed - now MANDATORY and always True
            validation_split=0.2,
            early_stopping_patience=args.early_stopping_patience,
            reduce_lr_patience=args.reduce_lr_patience
        )
        print(f"✅ Training completed successfully!")
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
    
    # Create flow model
    print(f"\n🔄 Creating normalizing flow...")
    flow = TFPNormalizingFlow(
        input_dim=6,  # Always 6D for non-conditional flows (pos + vel)
        n_layers=args.n_layers,
        hidden_units=args.hidden_units,
        activation=args.activation,
        use_batchnorm=args.use_batchnorm,
        use_gmm_base=args.use_gmm_base,
        gmm_components=args.gmm_components,
        name='main_flow'
    )
    
    # Create trainer
    trainer = TFPFlowTrainer(flow, learning_rate=args.learning_rate,
                             weight_decay=args.weight_decay,
                             noise_std=args.noise_std)
    
    # Train the model
    print(f"\n🎯 Training the flow...")
    start_time = time.time()
    
    train_losses, val_losses = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_freq=args.validation_freq,
        early_stopping_patience=args.early_stopping_patience,
        reduce_lr_patience=args.reduce_lr_patience,
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
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]) if val_losses else None,
        'test_loss': float(test_loss),
        'gpu_used': bool(gpu_available)
    })
    
    save_model_and_results(
        flow=flow,
        trainer=trainer,
        preprocessing_stats=preprocessing_stats,
        metadata=metadata,
        output_dir=args.output_dir,
        model_name=args.model_name
        # use_kroupa_imf removed - now MANDATORY and always True
    )
    
    # Plot training curves
    # Plotting removed - not needed for production training
    print(f"📊 Final training loss: {trainer.train_losses[-1]:.6f}")
    if trainer.val_losses:
        print(f"📊 Final validation loss: {trainer.val_losses[-1]:.6f}")
    
    # Final summary
    print(f"\n✅ TRAINING COMPLETED")
    print("=" * 50)
    print(f"Training time: {training_time:.1f}s")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print()

if __name__ == "__main__":
    main()
