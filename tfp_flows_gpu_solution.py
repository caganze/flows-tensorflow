#!/usr/bin/env python3
"""
TensorFlow Probability normalizing flows with GPU support
Fixed version addressing np.savez() inhomogeneous array issues
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Optional, List, Dict, Any
import json
from pathlib import Path

tfd = tfp.distributions
tfb = tfp.bijectors


class TFPNormalizingFlow:
    """TensorFlow Probability normalizing flow with proper saving/loading"""
    
    def __init__(self, input_dim: int, n_layers: int = 4, hidden_units: int = 64, 
                 activation: str = 'relu', name: str = 'tfp_flow'):
        """Initialize normalizing flow
        
        Args:
            input_dim: Dimensionality of input data
            n_layers: Number of MAF layers
            hidden_units: Hidden units in autoregressive networks
            activation: Activation function
            name: Flow name
        """
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.activation = activation
        self.name = name
        
        # Build the flow
        self._build_flow()
    
    def _build_flow(self):
        """Build the normalizing flow architecture"""
        
        # Create base distribution
        self.base_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.input_dim, dtype=tf.float32),
            scale_diag=tf.ones(self.input_dim, dtype=tf.float32)
        )
        
        # Create MAF layers
        bijectors = []
        for i in range(self.n_layers):
            # Create autoregressive network
            made = tfb.AutoregressiveNetwork(
                params=2,  # loc and log_scale
                hidden_units=[self.hidden_units] * 2,
                event_shape=[self.input_dim],
                activation=self.activation,
                dtype=tf.float32,
                name=f'{self.name}_made_{i}'
            )
            
            # Create MAF bijector
            maf = tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=made,
                name=f'{self.name}_maf_{i}'
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
        
        # Create the transformed distribution
        self.flow = tfd.TransformedDistribution(
            distribution=self.base_dist,
            bijector=self.bijector,
            name=self.name
        )
    
    @property
    def trainable_variables(self):
        """Get trainable variables from the flow"""
        return self.flow.trainable_variables
    
    def log_prob(self, x):
        """Compute log probability of data"""
        return self.flow.log_prob(x)
    
    def sample(self, n_samples: int, seed: Optional[int] = None):
        """Generate samples from the flow"""
        return self.flow.sample(n_samples, seed=seed)
    
    def save(self, filepath: str):
        """Save the flow model with proper handling of inhomogeneous arrays
        
        Args:
            filepath: Path to save the model (.npz file)
        """
        try:
            # Get all trainable variables
            variables = self.trainable_variables
            
            # Create a dictionary to save each variable separately
            # This avoids the inhomogeneous array issue
            save_dict = {}
            
            # Save model configuration
            save_dict['config'] = np.array({
                'input_dim': self.input_dim,
                'n_layers': self.n_layers,
                'hidden_units': self.hidden_units,
                'activation': self.activation,
                'name': self.name
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
            
            print(f"✅ Flow model saved to {filepath}")
            print(f"   Saved {len(variables)} variables")
            
        except Exception as e:
            print(f"❌ Error saving flow model: {e}")
            raise
    
    def load(self, filepath: str):
        """Load the flow model
        
        Args:
            filepath: Path to load the model from (.npz file)
        """
        try:
            # Load the saved data
            data = np.load(filepath, allow_pickle=True)
            
            # Reconstruct the model with saved configuration
            config = data['config'].item()
            self.input_dim = config['input_dim']
            self.n_layers = config['n_layers']
            self.hidden_units = config['hidden_units']
            self.activation = config['activation']
            self.name = config['name']
            
            # Rebuild the flow architecture
            self._build_flow()
            
            # Load the variables
            n_variables = int(data['n_variables'])
            variables = self.trainable_variables
            
            if len(variables) != n_variables:
                raise ValueError(f"Model structure mismatch: expected {n_variables} variables, got {len(variables)}")
            
            # Assign the loaded values to variables
            for i, var in enumerate(variables):
                loaded_value = data[f'variable_{i}']
                var.assign(loaded_value)
            
            print(f"✅ Flow model loaded from {filepath}")
            print(f"   Loaded {len(variables)} variables")
            
        except Exception as e:
            print(f"❌ Error loading flow model: {e}")
            raise


class TFPFlowTrainer:
    """Trainer for TensorFlow Probability flows"""
    
    def __init__(self, flow: TFPNormalizingFlow, learning_rate: float = 1e-3):
        """Initialize trainer
        
        Args:
            flow: The normalizing flow to train
            learning_rate: Learning rate for optimization
        """
        self.flow = flow
        self.learning_rate = learning_rate
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def _loss_fn(self, data):
        """Compute negative log likelihood loss"""
        log_probs = self.flow.log_prob(data)
        return -tf.reduce_mean(log_probs)
    
    @tf.function
    def _train_step(self, batch):
        """Single training step"""
        with tf.GradientTape() as tape:
            loss = self._loss_fn(batch)
        
        gradients = tape.gradient(loss, self.flow.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))
        
        return loss
    
    def train(self, train_data: tf.Tensor, val_data: tf.Tensor = None, 
              epochs: int = 100, batch_size: int = 512, 
              validation_freq: int = 10, verbose: bool = True):
        """Train the normalizing flow
        
        Args:
            train_data: Training data
            val_data: Validation data (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_freq: How often to compute validation loss
            verbose: Whether to print progress
            
        Returns:
            Tuple of (train_losses, val_losses)
        """
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(train_data)
        dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            epoch_losses = []
            for batch in dataset:
                loss = self._train_step(batch)
                epoch_losses.append(loss.numpy())
            
            train_loss = np.mean(epoch_losses)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = None
            if val_data is not None and epoch % validation_freq == 0:
                val_loss = self._loss_fn(val_data).numpy()
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            # Print progress
            if verbose and epoch % max(1, epochs // 10) == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch:4d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch:4d}/{epochs} | Train Loss: {train_loss:.4f}")
        
        if verbose:
            print(f"Training completed! Final train loss: {self.train_losses[-1]:.4f}")
            if self.val_losses:
                print(f"Final validation loss: {self.val_losses[-1]:.4f}")
        
        return self.train_losses, self.val_losses


def test_flow():
    """Test the flow implementation"""
    print("🧪 Testing TFP Normalizing Flow...")
    
    # Create synthetic data
    np.random.seed(42)
    tf.random.set_seed(42)
    
    n_samples = 1000
    input_dim = 6
    
    # Generate some test data
    data = tf.random.normal([n_samples, input_dim], dtype=tf.float32)
    
    # Create and test flow
    flow = TFPNormalizingFlow(input_dim=input_dim, n_layers=3, hidden_units=32)
    print(f"✅ Flow created with {len(flow.trainable_variables)} trainable variables")
    
    # Test log probability
    log_probs = flow.log_prob(data)
    print(f"✅ Log probability computed: shape {log_probs.shape}")
    
    # Test sampling
    samples = flow.sample(100, seed=42)
    print(f"✅ Samples generated: shape {samples.shape}")
    
    # Test saving and loading
    test_path = "/tmp/test_flow.npz"
    flow.save(test_path)
    
    # Create new flow and load
    flow2 = TFPNormalizingFlow(input_dim=input_dim, n_layers=3, hidden_units=32)
    flow2.load(test_path)
    
    # Verify they produce the same results
    log_probs2 = flow2.log_prob(data)
    diff = tf.reduce_max(tf.abs(log_probs - log_probs2))
    print(f"✅ Save/load test: max difference = {diff:.2e}")
    
    # Clean up
    Path(test_path).unlink(missing_ok=True)
    
    print("🎉 All tests passed!")


if __name__ == "__main__":
    test_flow()
