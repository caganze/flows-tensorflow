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

# Set CUDA paths for TensorFlow before importing
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0'

import numpy as np
# import pandas as pd  # Removed to avoid GLIBCXX conflicts
import tensorflow as tf
import tensorflow_probability as tfp
import h5py
# Plotting libraries removed - not needed for production training

# Import our TFP flow implementation
from tfp_flows_gpu_solution import TFPNormalizingFlow, TFPFlowTrainer
from kroupa_imf import sample_with_kroupa_imf, get_stellar_mass_from_h5
from optimized_io import save_samples_optimized
from comprehensive_logging import ComprehensiveLogger

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

def load_astrophysical_data(filepath: str, particle_pid: int = None, features: list = None) -> Tuple[tf.Tensor, Dict[str, Any]]:
    """
    Load astrophysical data from HDF5 file
    
    Args:
        filepath: Path to HDF5 file
        features: List of feature names to load (default: all 6D phase space)
    
    Returns:
        data: TensorFlow tensor with shape (n_samples, n_features)
        metadata: Dictionary with dataset information
    """
    if features is None:
        features = ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']
    
    print(f"Loading data from: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        # Try different possible dataset structures
        data_arrays = []
        
        if 'pos3' in f and 'vel3' in f:
            # Format: separate pos3 and vel3 arrays
            pos3 = f['pos3'][:]  # Shape: (n_samples, 3)
            vel3 = f['vel3'][:]  # Shape: (n_samples, 3)
            data = np.concatenate([pos3, vel3], axis=1)  # Shape: (n_samples, 6)
            
        elif all(feat in f for feat in features):
            # Format: individual feature arrays
            arrays = [f[feat][:] for feat in features]
            data = np.column_stack(arrays)
            
        else:
            # Try to find any 6D data
            for key in f.keys():
                if f[key].shape[-1] == 6:
                    data = f[key][:]
                    break
            else:
                raise ValueError(f"Could not find 6D data in {filepath}")
        
        # Collect metadata
        metadata = {
            'n_samples': len(data),
            'n_features': data.shape[1],
            'feature_names': features[:data.shape[1]],
            'data_file': filepath
        }
        
        # Add any HDF5 attributes
        for attr_name in f.attrs:
            metadata[attr_name] = f.attrs[attr_name]
    
    print(f"✅ Loaded data: {data.shape}")
    print(f"Features: {metadata['feature_names']}")
    
    # Convert to TensorFlow tensor
    data_tf = tf.constant(data, dtype=tf.float32)
    
    return data_tf, metadata

def load_particle_specific_data(filepath: str, particle_pid: int) -> Tuple[tf.Tensor, Dict[str, Any]]:
    """
    Load data for a specific particle PID from HDF5 file
    
    Args:
        filepath: Path to HDF5 file
        particle_pid: Specific particle ID to extract
        
    Returns:
        data: TensorFlow tensor with particle-specific data
        metadata: Dictionary with comprehensive metadata
    """
    print(f"🎯 Loading particle-specific data for PID {particle_pid}")
    
    # Use the proper H5 reader that handles particle filtering
    from test_h5_read_single_particle import read_h5_to_dict, extract_particle_data
    
    try:
        # Read full H5 file to dictionary
        print(f"📂 Reading H5 file: {filepath}")
        data_dict = read_h5_to_dict(filepath)
        print(f"📊 H5 file contains {len(data_dict)} datasets: {list(data_dict.keys())[:5]}...")
        
        # Extract data for specific particle
        print(f"🔍 Extracting data for PID {particle_pid}...")
        particle_data = extract_particle_data(data_dict, particle_pid)
        
        if len(particle_data) == 0:
            raise ValueError(f"❌ No data found for particle PID {particle_pid}")
        
        # Ensure we have 6D data (pos + vel)
        if particle_data.shape[1] >= 6:
            data = particle_data[:, :6]  # Take first 6 columns (pos + vel)
        else:
            raise ValueError(f"Insufficient data dimensions for PID {particle_pid}: {particle_data.shape}")
        
        print(f"✅ Extracted {len(data):,} particles for PID {particle_pid}")
        
        # Get stellar mass for this particle
        stellar_mass = None
        try:
            from kroupa_imf import get_stellar_mass_from_h5
            stellar_mass = get_stellar_mass_from_h5(data_dict, particle_pid)
            print(f"⭐ Stellar mass for PID {particle_pid}: {stellar_mass:.2e} M☉")
        except Exception as e:
            print(f"⚠️  Could not get stellar mass, using default: {e}")
            stellar_mass = 1e8  # Default fallback
        
        # Create comprehensive metadata
        metadata = {
            'filepath': filepath,
            'particle_pid': particle_pid,
            'n_samples': len(data),
            'n_features': data.shape[1],
            'data_shape': data.shape,
            'data_min': data.min(axis=0),
            'data_max': data.max(axis=0),
            'data_mean': data.mean(axis=0),
            'data_std': data.std(axis=0),
            'stellar_mass': stellar_mass,
            'h5_datasets': list(data_dict.keys()),
            'file_size_mb': os.path.getsize(filepath) / (1024 * 1024),
            'load_timestamp': time.time()
        }
        
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

def save_model_and_results(
    flow: TFPNormalizingFlow,
    trainer: TFPFlowTrainer,
    preprocessing_stats: Dict[str, tf.Tensor],
    metadata: Dict[str, Any],
    output_dir: str,
    model_name: str = "tfp_flow",
    generate_samples: bool = True,
    n_samples: Optional[int] = None  # If None, will be calculated adaptively
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
    
    # Generate and save large sample set
    if generate_samples:
        # Use Kroupa IMF to determine realistic sample count based on stellar mass
        try:
            stellar_mass = metadata.get('stellar_mass', None)
            if stellar_mass is None:
                print("⚠️ No stellar mass in metadata, using adaptive sampling fallback")
                # Fallback: adaptive strategy
                n_training_data = metadata.get('n_training_data', 50000)
                if n_training_data < 10000:
                    multiplier = 5
                elif n_training_data < 50000:
                    multiplier = 3
                else:
                    multiplier = 2
                n_samples = min(500000, max(50000, n_training_data * multiplier))
                samples = flow.sample(n_samples, seed=42)
                masses = None  # No masses generated in fallback
            else:
                # Use Kroupa IMF for realistic sampling
                print(f"🌟 Using Kroupa IMF sampling for stellar mass: {stellar_mass:.2e} M☉")
                samples, masses = sample_with_kroupa_imf(
                    flow=flow,
                    n_target_mass=stellar_mass,
                    preprocessing_stats=preprocessing_stats,
                    seed=42
                )
                n_samples = len(samples)
                
        except Exception as e:
            print(f"⚠️ Kroupa sampling failed: {e}, using fallback")
            # Fallback to simple sampling
            if n_samples is None:
                n_samples = 100000
            samples = flow.sample(n_samples, seed=42)
            masses = None
        
        # Use optimized I/O strategy based on sample count
        samples_dir = Path(str(output_path).replace('trained_flows', 'samples'))
        
        try:
            # Prepare comprehensive metadata for samples
            samples_metadata = {
                **metadata,  # Include all existing metadata
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
                    'mean': tf.reduce_mean(samples, axis=0).numpy().tolist(),
                    'std': tf.math.reduce_std(samples, axis=0).numpy().tolist(),
                    'min': tf.reduce_min(samples, axis=0).numpy().tolist(),
                    'max': tf.reduce_max(samples, axis=0).numpy().tolist()
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
    h5_file: str,
    particle_pid: int,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    n_layers: int = 4,
    hidden_units: int = 64,
    generate_samples: bool = True,
    n_samples: int = 100000,
    use_kroupa_imf: bool = True,
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
    logger.info(f"  Use Kroupa IMF: {use_kroupa_imf}")
    
    try:
        # Set up GPU
        logger.info("🔧 Setting up GPU...")
        setup_gpu()
        
        # Load and preprocess data for specific particle
        logger.info(f"📊 Loading data for PID {particle_pid}...")
        data, metadata = load_particle_specific_data(h5_file, particle_pid)
        
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
            name=f'flow_pid{particle_pid}'
        )
        
        # Create trainer
        logger.info("🎯 Creating trainer...")
        trainer = TFPFlowTrainer(
            flow=flow,
            learning_rate=learning_rate
        )
        
        # Train
        logger.info(f"🏋️ Training for {epochs} epochs...")
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
            'use_kroupa_imf': use_kroupa_imf,
            'n_samples_requested': n_samples if generate_samples else 0
        }
        
        # Save model and generate samples
        logger.info("💾 Saving model and generating samples...")
        model_path, samples_path = save_model_and_results(
            flow=flow,
            trainer=trainer,
            preprocessing_stats=preprocessing_stats,
            metadata=enhanced_metadata,
            output_dir=output_dir,
            model_name=f"model_pid{particle_pid}",
            generate_samples=generate_samples,
            n_samples=n_samples
        )
        
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
    
    # Data arguments
    parser.add_argument("--data_path", required=True, help="Path to HDF5 data file")
    parser.add_argument("--particle_pid", type=int, help="Specific particle ID to process")
    parser.add_argument("--output_dir", required=True, help="Output directory for model and results")
    
    # Model arguments
    parser.add_argument("--n_layers", type=int, default=4, help="Number of flow layers")
    parser.add_argument("--hidden_units", type=int, default=512, help="Hidden units per layer")
    parser.add_argument("--activation", default="relu", help="Activation function")
    
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
    parser.add_argument("--model_name", default="tfp_flow", help="Model name for saving")
    
    args = parser.parse_args()
    
    print("🚀 TensorFlow Probability Flow Training")
    print("=" * 50)
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    if args.particle_pid:
        print(f"Particle PID: {args.particle_pid}")
    print(f"Model config: {args.n_layers} layers, {args.hidden_units} hidden units")
    print(f"Training config: {args.epochs} epochs, batch size {args.batch_size}")
    print()
    
    # Set seeds for reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup GPU
    gpu_available = setup_gpu()
    
    # If particle_pid is specified, use the high-level train_and_save_flow function
    if args.particle_pid:
        print(f"🎯 Training flow for particle PID {args.particle_pid}")
        model_path, samples_path = train_and_save_flow(
            h5_file=args.data_path,
            particle_pid=args.particle_pid,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            n_layers=args.n_layers,
            hidden_units=args.hidden_units,
            generate_samples=True,
            n_samples=100000,
            use_kroupa_imf=True,
            validation_split=0.2,
            early_stopping_patience=50,
            reduce_lr_patience=20
        )
        print(f"✅ Training completed successfully!")
        print(f"Model saved to: {model_path}")
        if samples_path:
            print(f"Samples saved to: {samples_path}")
        return
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    data, metadata = load_astrophysical_data(args.data_path)
    
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
        input_dim=int(data.shape[1]),
        n_layers=args.n_layers,
        hidden_units=args.hidden_units,
        activation=args.activation,
        name='main_flow'
    )
    
    # Create trainer
    trainer = TFPFlowTrainer(flow, learning_rate=args.learning_rate)
    
    # Train the model
    print(f"\n🎯 Training the flow...")
    start_time = time.time()
    
    train_losses, val_losses = trainer.train(
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
