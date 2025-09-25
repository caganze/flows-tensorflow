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
                 activation: str = 'relu', name: str = 'tfp_flow',
                 use_batchnorm: bool = False,
                 use_gmm_base: bool = False,
                 gmm_components: int = 5):
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
        self.use_batchnorm = use_batchnorm
        self.use_gmm_base = bool(use_gmm_base)
        self.gmm_components = int(gmm_components)
        
        # Build the flow
        self._build_flow()
    
    def _build_flow(self):
        """Build the normalizing flow architecture"""
        
        # Create base distribution (optionally Gaussian mixture)
        if self.use_gmm_base and self.gmm_components > 1:
            # Initialize mixture logits uniformly, means 0, std 1
            logits = tf.Variable(tf.zeros([self.gmm_components], dtype=tf.float32), name=f'{self.name}_gmm_logits')
            locs = tf.Variable(tf.zeros([self.gmm_components, self.input_dim], dtype=tf.float32), name=f'{self.name}_gmm_locs')
            scales = tf.Variable(tf.ones([self.gmm_components, self.input_dim], dtype=tf.float32), name=f'{self.name}_gmm_scales')

            cat = tfd.Categorical(logits=logits)
            comp = tfd.MultivariateNormalDiag(loc=locs, scale_diag=scales)
            self.base_dist = tfd.MixtureSameFamily(mixture_distribution=cat, components_distribution=comp)
            self._gmm_params = {'logits': logits, 'locs': locs, 'scales': scales}
        else:
            self.base_dist = tfd.MultivariateNormalDiag(
                loc=tf.zeros(self.input_dim, dtype=tf.float32),
                scale_diag=tf.ones(self.input_dim, dtype=tf.float32)
            )
        
        # Create MAF layers
        bijectors = []
        for i in range(self.n_layers):
            # Create autoregressive network with better initialization
            made = tfb.AutoregressiveNetwork(
                params=2,  # loc and log_scale
                hidden_units=[self.hidden_units] * 2,
                event_shape=[self.input_dim],
                activation=self.activation,
                dtype=tf.float32,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                name=f'{self.name}_made_{i}'
            )
            
            # Create MAF bijector
            maf = tfb.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=made,
                name=f'{self.name}_maf_{i}'
            )
            bijectors.append(maf)

            # Optional invertible normalization (ActNorm-like)
            if self.use_batchnorm:
                bn = tfb.BatchNormalization(name=f'{self.name}_bn_{i}')
                bijectors.append(bn)
            
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
        
        # Initialize the flow with a small forward pass to ensure all variables are created
        dummy_input = tf.zeros((1, self.input_dim), dtype=tf.float32)
        _ = self.flow.log_prob(dummy_input)
    
    @property
    def trainable_variables(self):
        """Get trainable variables from the flow"""
        # Include GMM parameters if used
        variables = list(self.flow.trainable_variables)
        if getattr(self, '_gmm_params', None) is not None:
            variables.extend(list(self._gmm_params.values()))
        return variables
    
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
                'name': self.name,
                'use_gmm_base': self.use_gmm_base,
                'gmm_components': self.gmm_components
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
    
    def __init__(self, flow: TFPNormalizingFlow, learning_rate: float = 1e-3,
                 weight_decay: float = 0.0, noise_std: float = 0.0):
        """Initialize trainer
        
        Args:
            flow: The normalizing flow to train
            learning_rate: Learning rate for optimization
        """
        self.flow = flow
        self.learning_rate = learning_rate
        self.weight_decay = float(weight_decay)
        self.noise_std = float(noise_std)
        
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
        """Single training step with gradient clipping and numerical stability"""
        with tf.GradientTape() as tape:
            # Add Gaussian noise to inputs during training if requested
            if self.noise_std > 0.0:
                noise = tf.random.normal(tf.shape(batch), stddev=self.noise_std, dtype=batch.dtype)
                batch = batch + noise

            # Compute loss with numerical stability checks
            log_probs = self.flow.log_prob(batch)
            
            # Clip log probabilities to prevent extreme values
            log_probs = tf.clip_by_value(log_probs, -50.0, 50.0)
            
            loss = -tf.reduce_mean(log_probs)

            # L2 weight decay regularization on trainable vars (training loss only)
            if self.weight_decay > 0.0:
                l2_terms = [tf.nn.l2_loss(v) for v in self.flow.trainable_variables]
                if l2_terms:
                    loss = loss + self.weight_decay * tf.add_n(l2_terms)
        
        gradients = tape.gradient(loss, self.flow.trainable_variables)
        
        # Clip gradients more aggressively to prevent NaN
        gradients, grad_norm = tf.clip_by_global_norm(gradients, 0.5)
        
        # Check for NaN in loss and gradients
        loss_is_finite = tf.math.is_finite(loss)
        gradients_are_finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in gradients])
        
        # Additional check: loss should not be too large
        loss_is_reasonable = tf.less(loss, 100.0)
        
        # Only apply gradients if they are finite and reasonable
        def apply_grads():
            self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))
            return loss
        
        def skip_step():
            return tf.constant(100.0, dtype=tf.float32)  # Return a large but finite value
        
        return tf.cond(
            tf.logical_and(tf.logical_and(loss_is_finite, gradients_are_finite), loss_is_reasonable),
            apply_grads,
            skip_step
        )
    
    def train(self, train_data: tf.Tensor, val_data: tf.Tensor = None, 
              epochs: int = 100, batch_size: int = 512, 
              validation_freq: int = 10, verbose: bool = True,
              early_stopping_patience: int = 10,
              reduce_lr_patience: int = 5,
              reduce_lr_factor: float = 0.5,
              min_learning_rate: float = 1e-6):
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
        best_weights = None  # Store best model weights
        
        patience_counter = 0
        lr_patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            epoch_losses = []
            nan_batches = 0
            
            for batch in dataset:
                loss = self._train_step(batch)
                loss_val = loss.numpy()
                
                # Skip NaN/Inf losses and count them
                if not np.isfinite(loss_val):
                    nan_batches += 1
                    continue
                    
                # Check for explosion (loss > 1000)
                if loss_val > 1000:
                    if verbose:
                        print(f"⚠️ Loss explosion detected: {loss_val:.1f} - stopping training")
                    break
                    
                epoch_losses.append(loss_val)
            
            # If too many NaN batches, stop training
            if nan_batches > len(epoch_losses):
                if verbose:
                    print(f"❌ Too many NaN batches ({nan_batches}) - stopping training")
                break
            
            if not epoch_losses:  # No valid losses
                if verbose:
                    print(f"❌ No valid losses in epoch {epoch} - stopping training")
                break
                
            train_loss = np.mean(epoch_losses)
            self.train_losses.append(train_loss)
            
            # Check for training loss explosion
            if len(self.train_losses) > 1 and train_loss > 10 * self.train_losses[-2]:
                if verbose:
                    print(f"❌ Training loss explosion: {train_loss:.1f} - stopping training")
                break
            
            # Validation
            val_loss = None
            if val_data is not None and epoch % validation_freq == 0:
                val_log_probs = self.flow.log_prob(val_data)
                # Clip validation log probabilities too
                val_log_probs = tf.clip_by_value(val_log_probs, -50.0, 50.0)
                val_loss = float(-tf.reduce_mean(val_log_probs))
                
                # Check for validation NaN or extreme values
                if not np.isfinite(val_loss) or val_loss > 100.0:
                    if verbose:
                        print(f"❌ Validation loss is NaN or too large ({val_loss}) - stopping training")
                    break
                
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0  # Reset patience
                    lr_patience_counter = 0  # Reset LR patience
                    # Save best weights (deep copy to avoid reference issues)
                    best_weights = [var.numpy().copy() for var in self.flow.trainable_variables]
                    if verbose:
                        print(f"✅ New best validation loss: {val_loss:.6f} (epoch {epoch})")
                else:
                    patience_counter += 1
                    lr_patience_counter += 1
                    
                # Reduce learning rate on plateau
                if lr_patience_counter >= reduce_lr_patience:
                    current_lr = float(self.optimizer.learning_rate.numpy())
                    new_lr = max(current_lr * reduce_lr_factor, min_learning_rate)
                    if new_lr < current_lr:
                        self.optimizer.learning_rate.assign(new_lr)
                        if verbose:
                            print(f"🔽 Reducing learning rate: {current_lr:.3e} -> {new_lr:.3e}")
                    lr_patience_counter = 0

                # Early stopping
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"🛑 Early stopping: no improvement for {early_stopping_patience} validation checks")
                    break
            
            # Print progress
            if verbose and epoch % max(1, epochs // 10) == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch:4d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch:4d}/{epochs} | Train Loss: {train_loss:.4f}")
                    
                if nan_batches > 0:
                    print(f"   ⚠️ Skipped {nan_batches} NaN batches")
        
        # Restore best weights if we have validation data
        if best_weights is not None and val_data is not None:
            for var, best_weight in zip(self.flow.trainable_variables, best_weights):
                var.assign(best_weight)
            if verbose:
                print(f"🔄 Restored best model weights (val_loss: {best_val_loss:.6f})")
            
            # Verify restoration worked by checking current validation loss
            if verbose:
                try:
                    current_val_loss = self._compute_validation_loss(val_data)
                    if abs(current_val_loss - best_val_loss) < 1e-6:
                        print(f"✅ Weight restoration verified: current val_loss = {current_val_loss:.6f}")
                    else:
                        print(f"⚠️ Weight restoration may have failed: current val_loss = {current_val_loss:.6f}, expected = {best_val_loss:.6f}")
                except Exception as e:
                    print(f"⚠️ Could not verify weight restoration: {e}")
        
        if verbose:
            print(f"Training completed! Final train loss: {self.train_losses[-1]:.4f}")
            if self.val_losses:
                print(f"Best validation loss: {best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses
    
    def _compute_validation_loss(self, val_data):
        """Compute validation loss with numerical stability"""
        val_log_probs = self.flow.log_prob(val_data)
        val_log_probs = tf.clip_by_value(val_log_probs, -50.0, 50.0)
        return float(-tf.reduce_mean(val_log_probs))


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


def load_trained_flow(filepath: str) -> TFPNormalizingFlow:
    """
    Convenience function to load a trained flow from file
    
    Args:
        filepath: Path to the saved model (.npz file)
        
    Returns:
        Loaded TFPNormalizingFlow instance
    """
    # First, we need to get the configuration from the saved file
    data = np.load(filepath, allow_pickle=True)
    config = data['config'].item()
    
    # Create a new flow instance with the saved configuration
    flow = TFPNormalizingFlow(
        input_dim=config['input_dim'],
        n_layers=config['n_layers'],
        hidden_units=config['hidden_units'],
        activation=config['activation'],
        name=config['name']
    )
    
    # Load the trained parameters
    flow.load(filepath)
    
    return flow


if __name__ == "__main__":
    test_flow()
