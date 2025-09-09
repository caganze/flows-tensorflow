#!/usr/bin/env python3
"""
Conditional TensorFlow Probability normalizing flows training script
Conditions the flow on mass distribution to learn p(xi|mass) for each particle
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

# Import our TFP flow implementation
from tfp_flows_gpu_solution import TFPFlowTrainer
from kroupa_imf import sample_with_kroupa_imf
from optimized_io import save_samples_optimized
from comprehensive_logging import ComprehensiveLogger
from symlib_utils import load_particle_data, get_output_paths, validate_symlib_environment

# TFP aliases
tfd = tfp.distributions
tfb = tfp.bijectors


class ConditionalTFPNormalizingFlow:
    """TensorFlow Probability conditional normalizing flow that conditions on mass distribution"""
    
    def __init__(self, input_dim: int, condition_dim: int = 1, n_layers: int = 4, 
                 hidden_units: int = 64, activation: str = 'relu', name: str = 'conditional_tfp_flow'):
        """Initialize conditional normalizing flow
        
        Args:
            input_dim: Dimensionality of input data (6D phase space)
            condition_dim: Dimensionality of conditioning variables (mass)
            n_layers: Number of MAF layers
            hidden_units: Hidden units in autoregressive networks
            activation: Activation function
            name: Flow name
        """
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.activation = activation
        self.name = name
        
        # Build the conditional flow
        self._build_conditional_flow()
    
    def _build_conditional_flow(self):
        """Build the conditional normalizing flow architecture"""
        
        # Create base distribution (unconditional)
        self.base_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.input_dim, dtype=tf.float32),
            scale_diag=tf.ones(self.input_dim, dtype=tf.float32)
        )
        
        # Create conditional MAF layers
        bijectors = []
        for i in range(self.n_layers):
            # Create conditional autoregressive network
            # The network takes both the current data and the conditioning variable
            made = tfb.AutoregressiveNetwork(
                params=2,  # loc and log_scale
                hidden_units=[self.hidden_units] * 3,  # Deeper network for conditional
                event_shape=[self.input_dim],
                conditional_event_shape=[self.condition_dim],  # Conditioning on mass
                activation=self.activation,
                dtype=tf.float32,
                name=f'{self.name}_conditional_made_{i}'
            )
            
            # Create conditional MAF bijector
            maf = tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=made,
                name=f'{self.name}_conditional_maf_{i}'
            )
            bijectors.append(maf)
            
            # Add permutation layer (except for last layer)
            if i < self.n_layers - 1:
                permutation = tfb.Permute(
                    permutation=np.random.permutation(self.input_dim).astype(np.int32),
                    name=f'{self.name}_permute_{i}'
                )
                bijectors.append(permutation)
        
        # Chain bijectors (reverse order for forward pass)
        self.bijector = tfb.Chain(list(reversed(bijectors)), name=f'{self.name}_chain')
        
        # Create the conditional transformed distribution
        # Note: For conditional flows, we need to handle conditioning differently
        self.flow = tfd.TransformedDistribution(
            distribution=self.base_dist,
            bijector=self.bijector,
            name=self.name
        )
    
    @property
    def trainable_variables(self):
        """Get trainable variables from the flow"""
        return self.flow.trainable_variables
    
    def log_prob(self, x, conditions=None):
        """Compute conditional log probability of data given conditions
        
        Args:
            x: Input data tensor of shape (batch_size, input_dim)
            conditions: Conditioning variables of shape (batch_size, condition_dim)
        """
        if conditions is None:
            raise ValueError("Conditions must be provided for conditional flow")
        
        # For conditional flows, we need to pass conditions to the bijector
        # This requires a custom implementation since TFP's conditional support is limited
        return self._conditional_log_prob(x, conditions)
    
    def _conditional_log_prob(self, x, conditions):
        """Custom conditional log probability computation"""
        # Transform x through each bijector layer with conditioning
        log_det_jacobian = tf.zeros(tf.shape(x)[0], dtype=tf.float32)
        y = x
        
        # Apply bijectors in forward direction
        for bijector in reversed(self.bijector.bijectors):
            if hasattr(bijector, 'shift_and_log_scale_fn'):
                # This is a conditional MAF layer
                shift_and_log_scale = bijector.shift_and_log_scale_fn(y, conditional_input=conditions)
                shift = shift_and_log_scale[..., 0]
                log_scale = shift_and_log_scale[..., 1]
                
                # Apply the transformation
                y = (y - shift) * tf.exp(-log_scale)
                log_det_jacobian += tf.reduce_sum(-log_scale, axis=-1)
            else:
                # Non-conditional bijector (e.g., permutation)
                y = bijector.forward(y)
                log_det_jacobian += bijector.forward_log_det_jacobian(y, event_ndims=1)
        
        # Compute base distribution log prob
        base_log_prob = self.base_dist.log_prob(y)
        
        return base_log_prob + log_det_jacobian
    
    def sample(self, n_samples: int, conditions=None, seed: Optional[int] = None):
        """Generate conditional samples from the flow
        
        Args:
            n_samples: Number of samples to generate
            conditions: Conditioning variables of shape (n_samples, condition_dim)
            seed: Random seed
        """
        if conditions is None:
            raise ValueError("Conditions must be provided for conditional flow")
        
        # Sample from base distribution
        if seed is not None:
            tf.random.set_seed(seed)
        
        base_samples = self.base_dist.sample(n_samples)
        
        # Transform through inverse bijectors with conditioning
        return self._conditional_sample(base_samples, conditions)
    
    def _conditional_sample(self, base_samples, conditions):
        """Custom conditional sampling"""
        y = base_samples
        
        # Apply bijectors in reverse (inverse) direction
        for bijector in self.bijector.bijectors:
            if hasattr(bijector, 'shift_and_log_scale_fn'):
                # This is a conditional MAF layer
                shift_and_log_scale = bijector.shift_and_log_scale_fn(y, conditional_input=conditions)
                shift = shift_and_log_scale[..., 0]
                log_scale = shift_and_log_scale[..., 1]
                
                # Apply the inverse transformation
                y = y * tf.exp(log_scale) + shift
            else:
                # Non-conditional bijector (e.g., permutation)
                y = bijector.inverse(y)
        
        return y
    
    def save(self, filepath: str):
        """Save the conditional flow model"""
        # Get all trainable variables
        variables = self.trainable_variables
        
        # Create a dictionary to save each variable separately
        save_dict = {}
        
        # Save model configuration
        save_dict['config'] = np.array({
            'input_dim': self.input_dim,
            'condition_dim': self.condition_dim,
            'n_layers': self.n_layers,
            'hidden_units': self.hidden_units,
            'activation': self.activation,
            'name': self.name,
            'model_type': 'conditional'
        }, dtype=object)
        
        # Save each variable individually with a unique key
        for i, var in enumerate(variables):
            var_array = var.numpy()
            save_dict[f'variable_{i}'] = var_array
            save_dict[f'variable_{i}_shape'] = np.array(var_array.shape)
            save_dict[f'variable_{i}_name'] = var.name
        
        # Save number of variables for loading
        save_dict['n_variables'] = len(variables)
        
        # Use compressed format for efficiency
        np.savez_compressed(filepath, **save_dict)
        print(f"✅ Conditional flow saved to {filepath}")


class ConditionalTFPFlowTrainer:
    """Trainer for conditional TFP normalizing flows"""
    
    def __init__(self, flow: ConditionalTFPNormalizingFlow, learning_rate: float = 1e-3):
        self.flow = flow
        self.learning_rate = learning_rate
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Track losses
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(self, x, conditions):
        """Compute negative log likelihood loss for conditional flow"""
        log_probs = self.flow.log_prob(x, conditions)
        return -tf.reduce_mean(log_probs)
    
    @tf.function
    def train_step(self, x, conditions):
        """Single training step"""
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, conditions)
        
        gradients = tape.gradient(loss, self.flow.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))
        
        return loss
    
    def train(self, train_data, train_conditions, val_data=None, val_conditions=None,
              epochs: int = 100, batch_size: int = 512, validation_freq: int = 10,
              verbose: bool = True):
        """Train the conditional flow
        
        Args:
            train_data: Training phase space data
            train_conditions: Training mass conditions
            val_data: Validation phase space data  
            val_conditions: Validation mass conditions
            epochs: Number of training epochs
            batch_size: Batch size
            validation_freq: Frequency of validation evaluation
            verbose: Whether to print progress
        """
        
        n_batches = len(train_data) // batch_size
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data
            indices = tf.random.shuffle(tf.range(len(train_data)))
            train_data_shuffled = tf.gather(train_data, indices)
            train_conditions_shuffled = tf.gather(train_conditions, indices)
            
            # Training batches
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_data = train_data_shuffled[start_idx:end_idx]
                batch_conditions = train_conditions_shuffled[start_idx:end_idx]
                
                loss = self.train_step(batch_data, batch_conditions)
                epoch_losses.append(float(loss))
            
            avg_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_loss)
            
            # Validation
            if val_data is not None and val_conditions is not None and epoch % validation_freq == 0:
                val_loss = float(self.compute_loss(val_data, val_conditions))
                self.val_losses.append(val_loss)
                
                if verbose:
                    print(f"Epoch {epoch:4d}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")
            elif verbose and epoch % validation_freq == 0:
                print(f"Epoch {epoch:4d}: Train Loss = {avg_loss:.6f}")
        
        if verbose:
            print(f"✅ Training completed after {epochs} epochs")


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
    
    # Load particle data using symlib
    data, metadata = load_particle_data(halo_id, particle_pid, suite)
    
    # Extract phase space (position + velocity) and mass information
    if data.shape[1] < 7:  # Assuming columns are [x, y, z, vx, vy, vz, mass, ...]
        raise ValueError(f"Insufficient data dimensions. Expected at least 7 columns (pos+vel+mass), got {data.shape[1]}")
    
    # Phase space data (first 6 columns: position + velocity)
    phase_space = data[:, :6]
    
    # Mass data (7th column, assuming it's available in the symlib data)
    # If mass is not in the 7th column, we'll need to load it separately
    if data.shape[1] >= 7:
        masses = data[:, 6:7]  # Keep as 2D array for conditioning
    else:
        # Fallback: use stellar mass from metadata or create dummy masses
        stellar_mass = metadata.get('stellar_mass', 1e6)  # Default 1M solar masses
        n_particles = len(phase_space)
        # Create log-uniform distribution of masses around the stellar mass
        log_masses = np.random.normal(np.log10(stellar_mass/n_particles), 0.5, n_particles)
        masses = 10**log_masses.reshape(-1, 1)
        print("⚠️ Using generated mass distribution - consider loading actual particle masses")
    
    # Convert to TensorFlow tensors
    phase_space_tensor = tf.constant(phase_space, dtype=tf.float32)
    mass_tensor = tf.constant(masses, dtype=tf.float32)
    
    print(f"✅ Loaded {phase_space.shape[0]:,} particles with 6D phase space + mass conditioning")
    print(f"   Phase space shape: {phase_space.shape}")
    print(f"   Mass conditions shape: {masses.shape}")
    print(f"   Stellar mass range: [{np.min(masses):.2e}, {np.max(masses):.2e}] M☉")
    
    # Update metadata with conditioning info
    metadata.update({
        'has_mass_conditioning': True,
        'mass_range': [float(np.min(masses)), float(np.max(masses))],
        'log_mass_range': [float(np.log10(np.min(masses))), float(np.log10(np.max(masses)))]
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
    Preprocess astrophysical data and mass conditions for conditional training
    
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


def train_and_save_conditional_flow(
    halo_id: str,
    particle_pid: int,
    suite: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    n_layers: int = 4,
    hidden_units: int = 64,
    generate_samples: bool = True,
    n_samples: int = 100000,
    validation_split: float = 0.2,
    early_stopping_patience: int = 50,
    reduce_lr_patience: int = 20
) -> Tuple[str, str]:
    """
    High-level function to train and save a conditional TFP flow for a specific particle.
    
    Returns:
        Tuple of (model_path, samples_path)
    """
    # Set up comprehensive logging
    log_dir = f"{output_dir}/logs"
    logger = ComprehensiveLogger(log_dir, "conditional_training", particle_pid)
    
    logger.info(f"🚀 Starting conditional TFP flow training for particle PID {particle_pid}")
    logger.info(f"📋 Training parameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  N layers: {n_layers}")
    logger.info(f"  Hidden units: {hidden_units}")
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
        
        # Create conditional flow
        logger.info(f"🏗️ Creating conditional normalizing flow ({n_layers} layers, {hidden_units} units)...")
        flow = ConditionalTFPNormalizingFlow(
            input_dim=6,  # 6D phase space
            condition_dim=1,  # 1D mass conditioning
            n_layers=n_layers,
            hidden_units=hidden_units,
            name=f'conditional_flow_pid{particle_pid}'
        )
        
        # Create trainer
        logger.info("🎯 Creating conditional trainer...")
        trainer = ConditionalTFPFlowTrainer(
            flow=flow,
            learning_rate=learning_rate
        )
        
        # Train
        logger.info(f"🏋️ Training conditional flow for {epochs} epochs...")
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
            'model_type': 'conditional',
            'conditioning_variables': ['stellar_mass'],
            'n_samples_requested': n_samples if generate_samples else 0
        }
        
        # Save model
        logger.info("💾 Saving trained conditional model...")
        model_path = f"{output_dir}/conditional_model_pid{particle_pid}.npz"
        flow.save(model_path)
        
        # Save preprocessing statistics
        preprocessing_path = f"{output_dir}/conditional_model_pid{particle_pid}_preprocessing.npz"
        np.savez(
            preprocessing_path,
            **{k: v.numpy() if isinstance(v, tf.Tensor) else v 
               for k, v in preprocessing_stats.items()}
        )
        
        # Save training results
        results = {
            'train_losses': [float(loss) for loss in trainer.train_losses],
            'val_losses': [float(loss) for loss in trainer.val_losses],
            'metadata': enhanced_metadata,
            'model_config': {
                'input_dim': 6,
                'condition_dim': 1,
                'n_layers': n_layers,
                'hidden_units': hidden_units,
                'model_type': 'conditional'
            }
        }
        
        results_path = f"{output_dir}/conditional_model_pid{particle_pid}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Conditional training completed for PID {particle_pid}")
        logger.log_metric("final_training_loss", trainer.train_losses[-1])
        if trainer.val_losses:
            logger.log_metric("final_validation_loss", trainer.val_losses[-1])
        
        logger.mark_completed(True, f"Successfully trained conditional model for PID {particle_pid}")
        
        # Note: Conditional sampling would require specifying mass conditions
        # For now, just return the model path
        return model_path, None
        
    except Exception as e:
        logger.log_error_with_traceback(e, f"conditional training PID {particle_pid}")
        logger.mark_completed(False, f"Conditional training failed for PID {particle_pid}: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train conditional TensorFlow Probability normalizing flows")
    
    # Data arguments (symlib)
    parser.add_argument("--halo_id", required=True, help="Halo ID (e.g., Halo268)")
    parser.add_argument("--particle_pid", type=int, required=True, help="Specific particle ID to process")
    parser.add_argument("--suite", default="eden", help="Simulation suite name (default: eden)")
    parser.add_argument("--output_dir", help="Output directory for model and results")
    
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
    parser.add_argument("--model_name", default="conditional_tfp_flow", help="Model name for saving")
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if not args.output_dir:
        args.output_dir = "/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/conditional_tfp_output/"
    
    print("🚀 Conditional TensorFlow Probability Flow Training (Symlib)")
    print("=" * 60)
    print(f"Halo ID: {args.halo_id}")
    print(f"Particle PID: {args.particle_pid}")
    print(f"Suite: {args.suite}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model config: {args.n_layers} layers, {args.hidden_units} hidden units")
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
    
    # Train conditional flow
    print(f"🎯 Training conditional flow for particle PID {args.particle_pid}")
    model_path, samples_path = train_and_save_conditional_flow(
        halo_id=args.halo_id,
        particle_pid=args.particle_pid,
        suite=args.suite,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_layers=args.n_layers,
        hidden_units=args.hidden_units,
        generate_samples=False,  # Conditional sampling requires specific mass conditions
        validation_split=0.2,
        early_stopping_patience=50,
        reduce_lr_patience=20
    )
    
    print(f"✅ Conditional training completed successfully!")
    print(f"Model saved to: {model_path}")
    print("💡 To generate samples, specify mass conditions for the conditional flow")


if __name__ == "__main__":
    main()
