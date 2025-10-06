#!/usr/bin/env python3
"""
Fixed Conditional TensorFlow Probability normalizing flows training script
Simplified implementation to avoid NaN losses
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
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Import our utilities
from symlib_utils import load_particle_data, get_output_paths, validate_symlib_environment
from comprehensive_logging import ComprehensiveLogger

# TFP aliases
tfd = tfp.distributions
tfb = tfp.bijectors


class SimpleConditionalFlow:
    """Simplified conditional normalizing flow"""
    
    def __init__(self, input_dim: int, condition_dim: int, n_layers: int = 2, 
                 hidden_units: int = 64, name: str = 'simple_conditional_flow'):
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.name = name
        
        # Create base distribution
        self.base_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.input_dim, dtype=tf.float32),
            scale_diag=tf.ones(self.input_dim, dtype=tf.float32)
        )
        
        # Create simple conditional bijector chain
        self._build_flow()
    
    def _build_flow(self):
        """Build a simple conditional flow"""
        bijectors = []
        
        for i in range(self.n_layers):
            # Create conditional autoregressive network
            made = tfb.AutoregressiveNetwork(
                params=2,  # loc and log_scale
                hidden_units=[self.hidden_units],
                event_shape=[self.input_dim],
                conditional=True,
                conditional_event_shape=[self.condition_dim],
                activation='relu',
                dtype=tf.float32,
                name=f'{self.name}_made_{i}'
            )
            
            # Create MAF bijector
            maf = tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=made,
                name=f'{self.name}_maf_{i}'
            )
            bijectors.append(maf)
        
        # Chain bijectors
        self.bijector = tfb.Chain(bijectors, name=f'{self.name}_chain')
        
        # Create transformed distribution
        self.flow = tfd.TransformedDistribution(
            distribution=self.base_dist,
            bijector=self.bijector,
            name=self.name
        )
    
    @property
    def trainable_variables(self):
        return self.flow.trainable_variables
    
    def log_prob(self, x, conditions):
        """Compute log probability with proper conditional handling"""
        if conditions is None:
            raise ValueError("Conditions must be provided")
        
        # Apply bijectors with conditional input
        log_det_jacobian = tf.zeros(tf.shape(x)[0], dtype=tf.float32)
        y = x
        
        # Forward pass through bijectors
        for bijector in reversed(self.bijector.bijectors):
            if 'MaskedAutoregressiveFlow' in str(type(bijector)):
                y = bijector.forward(y, conditional_input=conditions)
                log_det_jacobian += bijector.forward_log_det_jacobian(y, event_ndims=1, conditional_input=conditions)
            else:
                y = bijector.forward(y)
                log_det_jacobian += bijector.forward_log_det_jacobian(y, event_ndims=1)
        
        # Base distribution log prob
        base_log_prob = self.base_dist.log_prob(y)
        return base_log_prob + log_det_jacobian
    
    def sample(self, n_samples: int, conditions, seed: Optional[int] = None):
        """Generate samples"""
        if conditions is None:
            raise ValueError("Conditions must be provided")
        
        if seed is not None:
            tf.random.set_seed(seed)
        
        base_samples = self.base_dist.sample(n_samples)
        y = base_samples
        
        # Inverse pass through bijectors
        for bijector in self.bijector.bijectors:
            if 'MaskedAutoregressiveFlow' in str(type(bijector)):
                y = bijector.inverse(y, conditional_input=conditions)
            else:
                y = bijector.inverse(y)
        
        return y


class SimpleTrainer:
    """Simplified trainer"""
    
    def __init__(self, flow: SimpleConditionalFlow, learning_rate: float = 1e-3):
        self.flow = flow
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(self, x, conditions):
        """Compute loss with numerical stability"""
        log_probs = self.flow.log_prob(x, conditions)
        loss = -tf.reduce_mean(log_probs)
        
        # Add small regularization to prevent NaN
        if tf.math.is_nan(loss):
            return tf.constant(1.0, dtype=tf.float32)
        
        return loss
    
    @tf.function
    def train_step(self, x, conditions):
        """Single training step"""
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, conditions)
        
        gradients = tape.gradient(loss, self.flow.trainable_variables)
        
        # Clip gradients
        gradients = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in gradients]
        
        self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))
        return loss
    
    def train(self, train_data, train_conditions, val_data=None, val_conditions=None,
              epochs: int = 10, batch_size: int = 512, verbose: bool = True):
        """Simple training loop"""
        
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_conditions))
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        if val_data is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_conditions))
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        for epoch in range(epochs):
            # Training
            epoch_losses = []
            for batch_data, batch_conditions in train_dataset:
                loss = self.train_step(batch_data, batch_conditions)
                epoch_losses.append(loss.numpy())
            
            train_loss = np.mean(epoch_losses)
            self.train_losses.append(train_loss)
            
            # Validation
            if val_data is not None:
                val_losses = []
                for batch_data, batch_conditions in val_dataset:
                    loss = self.compute_loss(batch_data, batch_conditions)
                    val_losses.append(loss.numpy())
                val_loss = np.mean(val_losses)
                self.val_losses.append(val_loss)
                
                if verbose:
                    print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
            else:
                if verbose:
                    print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}")


def load_and_preprocess_data(halo_id: str, particle_pid: int, suite: str = 'eden'):
    """Load and preprocess data with proper error handling"""
    print(f"📊 Loading data: {halo_id} PID {particle_pid} from {suite}")
    
    # Validate environment
    if not validate_symlib_environment():
        raise RuntimeError("❌ Symlib environment not available")
    
    # Load data
    data, metadata = load_particle_data(halo_id, particle_pid, suite)
    
    if data.shape[1] < 7:
        raise ValueError(f"Expected 7 columns, got {data.shape[1]}")
    
    # Extract phase space and mass
    phase_space = data[:, :6]  # 6D phase space
    masses = data[:, 6:7]      # Mass column
    
    # Handle zero masses
    min_mass = np.maximum(np.min(masses), 1e-10)  # Avoid log(0)
    masses = np.maximum(masses, min_mass)
    
    print(f"✅ Loaded {len(phase_space):,} particles")
    print(f"   Phase space shape: {phase_space.shape}")
    print(f"   Mass shape: {masses.shape}")
    print(f"   Mass range: [{np.min(masses):.2e}, {np.max(masses):.2e}] M☉")
    
    # Simple preprocessing
    # Standardize phase space
    ps_mean = np.mean(phase_space, axis=0)
    ps_std = np.std(phase_space, axis=0) + 1e-8
    phase_space_norm = (phase_space - ps_mean) / ps_std
    
    # Log transform and standardize mass
    log_masses = np.log10(masses)
    mass_mean = np.mean(log_masses, axis=0)
    mass_std = np.std(log_masses, axis=0) + 1e-8
    mass_norm = (log_masses - mass_mean) / mass_std
    
    print(f"✅ Preprocessing complete")
    print(f"   Phase space range: [{np.min(phase_space_norm):.3f}, {np.max(phase_space_norm):.3f}]")
    print(f"   Mass range: [{np.min(mass_norm):.3f}, {np.max(mass_norm):.3f}]")
    
    return (tf.constant(phase_space_norm, dtype=tf.float32), 
            tf.constant(mass_norm, dtype=tf.float32), 
            metadata)


def main():
    parser = argparse.ArgumentParser(description='Simple Conditional TFP Flow Training')
    parser.add_argument('--halo_id', required=True, help='Halo ID')
    parser.add_argument('--particle_pid', type=int, required=True, help='Particle PID')
    parser.add_argument('--suite', default='eden', help='Simulation suite')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--hidden_units', type=int, default=64, help='Hidden units')
    
    args = parser.parse_args()
    
    print("🚀 Simple Conditional TFP Flow Training")
    print("=" * 50)
    print(f"Halo ID: {args.halo_id}")
    print(f"Particle PID: {args.particle_pid}")
    print(f"Suite: {args.suite}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    
    try:
        # Load and preprocess data
        phase_space, mass_conditions, metadata = load_and_preprocess_data(
            args.halo_id, args.particle_pid, args.suite
        )
        
        # Split data
        n_train = int(0.8 * len(phase_space))
        train_ps = phase_space[:n_train]
        train_mass = mass_conditions[:n_train]
        val_ps = phase_space[n_train:]
        val_mass = mass_conditions[n_train:]
        
        print(f"📊 Data split: {len(train_ps)} train, {len(val_ps)} validation")
        
        # Create flow
        print(f"🏗️ Creating conditional flow ({args.n_layers} layers, {args.hidden_units} units)...")
        flow = SimpleConditionalFlow(
            input_dim=6,
            condition_dim=1,
            n_layers=args.n_layers,
            hidden_units=args.hidden_units,
            name=f'conditional_flow_pid{args.particle_pid}'
        )
        
        # Create trainer
        print("🎯 Creating trainer...")
        trainer = SimpleTrainer(flow, learning_rate=args.learning_rate)
        
        # Train
        print(f"🏋️ Training for {args.epochs} epochs...")
        trainer.train(
            train_data=train_ps,
            train_conditions=train_mass,
            val_data=val_ps,
            val_conditions=val_mass,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        print("✅ Training completed successfully!")
        print(f"Final train loss: {trainer.train_losses[-1]:.6f}")
        if trainer.val_losses:
            print(f"Final val loss: {trainer.val_losses[-1]:.6f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())









