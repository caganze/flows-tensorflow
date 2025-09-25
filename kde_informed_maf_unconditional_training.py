#!/usr/bin/env python3
"""
KDE-Informed Unconditional MAF Training Implementation

This script implements the approach suggested by Google Gemini to combine KDE and MAF training.
Instead of training MAF from scratch, we use KDE results to guide the MAF's learning process
through regularization in the loss function.

ARCHITECTURE: Unconditional MAF
- Learns p(x,y,z,vx,vy,vz,m) - joint distribution of phase space and mass
- Uses Masked Autoregressive Flow (MAF) without conditioning
- Single KDE model for the entire dataset
- MSE loss between MAF and KDE log probabilities

Key features:
1. KDE density evaluation for MAF training batches
2. Combined loss function: L = L_NLL + λ * L_KDE
3. MSE between MAF and KDE log probabilities as regularization
4. Toy dataset generation for testing
5. Flexible hyperparameter tuning (λ, learning rates, etc.)

Usage:
    python kde_informed_maf_unconditional_training.py --toy_dataset --epochs 50 --lambda_kde 0.1
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import h5py

# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# Import existing utilities
try:
    from train_tfp_flows_conditional import ConditionalTFPNormalizingFlow, ConditionalTFPFlowTrainer
    from train_tfp_flows_conditional import preprocess_conditional_data, split_conditional_data
except ImportError:
    print("Warning: Could not import conditional flow classes. Using simplified versions.")
    
    # Simplified fallback classes
    class ConditionalTFPNormalizingFlow:
        def __init__(self, input_dim, condition_dim, n_layers=3, hidden_units=64, name='conditional_flow'):
            self.input_dim = input_dim
            self.condition_dim = condition_dim
            self.name = name
            
            # Create a simple MAF
            self.maf = tfp.bijectors.MaskedAutoregressiveFlow(
                shift_and_log_scale_fn=tfp.bijectors.AutoregressiveNetwork(
                    params=2,
                    hidden_units=[hidden_units] * n_layers,
                    activation='relu'
                )
            )
            
            # Base distribution
            self.base_dist = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(input_dim),
                scale_diag=tf.ones(input_dim)
            )
            
            # Create the flow
            self.flow = tfp.distributions.TransformedDistribution(
                distribution=self.base_dist,
                bijector=self.maf
            )
        
        def log_prob(self, x, conditions=None):
            """Compute log probability of samples"""
            return self.flow.log_prob(x)
        
        def sample(self, n_samples, conditions=None, seed=None):
            """Generate samples from the flow"""
            if seed is not None:
                tf.random.set_seed(seed)
            return self.flow.sample(n_samples)
        
        def save(self, filepath):
            """Save model weights"""
            # Simplified saving - just save the flow parameters
            weights = self.flow.trainable_variables
            np.savez(filepath, weights=[w.numpy() for w in weights])
            print(f"✅ Model saved to {filepath}")
    
    class ConditionalTFPFlowTrainer:
        def __init__(self, flow, learning_rate=1e-3):
            self.flow = flow
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            self.train_losses = []
            self.val_losses = []
        
        def compute_loss(self, x, conditions=None):
            """Compute standard NLL loss with numerical stability"""
            log_probs = self.flow.log_prob(x, conditions)
            
            # Clip log probabilities to prevent extreme values
            log_probs = tf.clip_by_value(log_probs, -50.0, 50.0)
            
            return -tf.reduce_mean(log_probs)
        
        @tf.function
        def train_step(self, x, conditions=None):
            """Single training step with numerical stability"""
            with tf.GradientTape() as tape:
                loss = self.compute_loss(x, conditions)
            
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
        
        def train(self, train_data, train_conditions=None, val_data=None, val_conditions=None,
                  epochs=100, batch_size=512, validation_freq=10, verbose=True):
            """Train the flow"""
            dataset = tf.data.Dataset.from_tensor_slices((train_data, train_conditions) if train_conditions is not None else train_data)
            dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            for epoch in range(epochs):
                epoch_losses = []
                for batch in dataset:
                    if train_conditions is not None:
                        batch_data, batch_conditions = batch
                        loss = self.train_step(batch_data, batch_conditions)
                    else:
                        loss = self.train_step(batch)
                    epoch_losses.append(loss.numpy())
                
                avg_loss = np.mean(epoch_losses)
                self.train_losses.append(avg_loss)
                
                if verbose and (epoch + 1) % validation_freq == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


class KDEInformedMAFTrainer:
    """
    KDE-Informed MAF Trainer
    
    This trainer uses KDE density estimates to guide MAF training through regularization.
    The loss function combines negative log-likelihood with MSE between MAF and KDE
    log probabilities: L = L_NLL + λ * L_KDE
    """
    
    def __init__(self, flow: ConditionalTFPNormalizingFlow, kde_model: gaussian_kde,
                 learning_rate: float = 1e-3, lambda_kde: float = 0.1,
                 weight_decay: float = 0.0, noise_std: float = 0.0):
        """
        Initialize KDE-informed MAF trainer
        
        Parameters:
        -----------
        flow : ConditionalTFPNormalizingFlow
            The MAF model to train
        kde_model : gaussian_kde
            Pre-trained KDE model for density estimation
        learning_rate : float
            Learning rate for optimizer
        lambda_kde : float
            Regularization weight for KDE term (λ in loss function)
        weight_decay : float
            L2 weight decay regularization
        noise_std : float
            Standard deviation of Gaussian noise added during training
        """
        self.flow = flow
        self.kde_model = kde_model
        self.lambda_kde = lambda_kde
        self.weight_decay = weight_decay
        self.noise_std = noise_std
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Track losses
        self.train_losses = []
        self.val_losses = []
        self.nll_losses = []
        self.kde_losses = []
    
    def evaluate_kde_density(self, x: tf.Tensor) -> tf.Tensor:
        """
        Evaluate KDE density for a batch of samples
        
        Parameters:
        -----------
        x : tf.Tensor
            Batch of samples to evaluate
            
        Returns:
        --------
        kde_log_probs : tf.Tensor
            Log probabilities from KDE model
        """
        # Convert to numpy for KDE evaluation
        x_np = x.numpy()
        
        # Evaluate KDE density (returns log density)
        kde_log_probs_np = self.kde_model.logpdf(x_np.T)  # KDE expects (n_features, n_samples)
        
        # Convert back to tensor
        kde_log_probs = tf.constant(kde_log_probs_np, dtype=tf.float32)
        
        return kde_log_probs
    
    def compute_mse_log_probs(self, maf_log_probs: tf.Tensor, kde_log_probs: tf.Tensor) -> tf.Tensor:
        """
        Compute MSE between MAF and KDE log probabilities
        
        MSE = E[(log P_MAF - log P_KDE)^2]
        
        This is more computationally efficient and effective than KL divergence
        for making the MAF's density match the KDE's density.
        
        Parameters:
        -----------
        maf_log_probs : tf.Tensor
            Log probabilities from MAF
        kde_log_probs : tf.Tensor
            Log probabilities from KDE
            
        Returns:
        --------
        mse_loss : tf.Tensor
            MSE between log probabilities
        """
        # MSE between log probabilities
        mse_loss = tf.reduce_mean(tf.square(maf_log_probs - kde_log_probs))
        return mse_loss
    
    def compute_loss(self, x: tf.Tensor, conditions: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute combined loss: L = L_NLL + λ * L_KDE
        
        Where L_KDE is the MSE between MAF and KDE log probabilities:
        L_KDE = E[(log P_MAF - log P_KDE)^2]
        
        Parameters:
        -----------
        x : tf.Tensor
            Training batch
        conditions : tf.Tensor, optional
            Conditioning variables
            
        Returns:
        --------
        total_loss : tf.Tensor
            Combined loss
        nll_loss : tf.Tensor
            Negative log-likelihood loss
        kde_loss : tf.Tensor
            KDE regularization loss (MSE between log probabilities)
        """
        # Standard NLL loss with numerical stability
        maf_log_probs = self.flow.log_prob(x, conditions)
        
        # Clip log probabilities to prevent extreme values
        maf_log_probs = tf.clip_by_value(maf_log_probs, -50.0, 50.0)
        
        nll_loss = -tf.reduce_mean(maf_log_probs)
        
        # KDE regularization loss (MSE between log probabilities)
        kde_log_probs = self.evaluate_kde_density(x)
        kde_loss = self.compute_mse_log_probs(maf_log_probs, kde_log_probs)
        
        # Combined loss
        total_loss = nll_loss + self.lambda_kde * kde_loss
        
        # Add weight decay if specified
        if self.weight_decay > 0.0:
            l2_terms = [tf.nn.l2_loss(v) for v in self.flow.trainable_variables]
            if l2_terms:
                total_loss = total_loss + self.weight_decay * tf.add_n(l2_terms)
        
        return total_loss, nll_loss, kde_loss
    
    @tf.function
    def train_step(self, x: tf.Tensor, conditions: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Single training step with KDE guidance
        
        Parameters:
        -----------
        x : tf.Tensor
            Training batch
        conditions : tf.Tensor, optional
            Conditioning variables
            
        Returns:
        --------
        total_loss : tf.Tensor
            Combined loss
        nll_loss : tf.Tensor
            NLL loss
        kde_loss : tf.Tensor
            KDE loss
        """
        with tf.GradientTape() as tape:
            # Add noise if specified
            if self.noise_std > 0.0:
                noise = tf.random.normal(tf.shape(x), stddev=self.noise_std, dtype=x.dtype)
                x = x + noise
            
            total_loss, nll_loss, kde_loss = self.compute_loss(x, conditions)
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.flow.trainable_variables)
        
        # Clip gradients more aggressively to prevent NaN
        gradients, grad_norm = tf.clip_by_global_norm(gradients, 0.5)
        
        # Check for NaN in loss and gradients
        loss_is_finite = tf.math.is_finite(total_loss)
        gradients_are_finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g in gradients])
        
        # Additional check: loss should not be too large
        loss_is_reasonable = tf.less(total_loss, 100.0)
        
        # Only apply gradients if they are finite and reasonable
        def apply_grads():
            self.optimizer.apply_gradients(zip(gradients, self.flow.trainable_variables))
            return total_loss, nll_loss, kde_loss
        
        def skip_step():
            return tf.constant(100.0, dtype=tf.float32), tf.constant(100.0, dtype=tf.float32), tf.constant(100.0, dtype=tf.float32)
        
        return tf.cond(
            tf.logical_and(tf.logical_and(loss_is_finite, gradients_are_finite), loss_is_reasonable),
            apply_grads,
            skip_step
        )
    
    def train(self, train_data: tf.Tensor, train_conditions: Optional[tf.Tensor] = None,
              val_data: Optional[tf.Tensor] = None, val_conditions: Optional[tf.Tensor] = None,
              epochs: int = 100, batch_size: int = 512, validation_freq: int = 10,
              verbose: bool = True) -> Dict[str, list]:
        """
        Train the KDE-informed MAF
        
        Parameters:
        -----------
        train_data : tf.Tensor
            Training data
        train_conditions : tf.Tensor, optional
            Training conditions
        val_data : tf.Tensor, optional
            Validation data
        val_conditions : tf.Tensor, optional
            Validation conditions
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        validation_freq : int
            Frequency of validation evaluation
        verbose : bool
            Whether to print training progress
            
        Returns:
        --------
        training_history : Dict[str, list]
            Training history with losses
        """
        # Create dataset
        if train_conditions is not None:
            dataset = tf.data.Dataset.from_tensor_slices((train_data, train_conditions))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(train_data)
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        print(f"🚀 Starting KDE-informed MAF training")
        print(f"   Epochs: {epochs}, Batch size: {batch_size}")
        print(f"   λ (KDE weight): {self.lambda_kde}")
        print(f"   Learning rate: {self.optimizer.learning_rate.numpy():.2e}")
        print(f"   Loss: L = L_NLL + λ * MSE(log P_MAF, log P_KDE)")
        print("=" * 60)
        
        for epoch in range(epochs):
            epoch_total_losses = []
            epoch_nll_losses = []
            epoch_kde_losses = []
            
            # Training loop
            for batch in dataset:
                if train_conditions is not None:
                    batch_data, batch_conditions = batch
                    total_loss, nll_loss, kde_loss = self.train_step(batch_data, batch_conditions)
                else:
                    total_loss, nll_loss, kde_loss = self.train_step(batch)
                
                epoch_total_losses.append(total_loss.numpy())
                epoch_nll_losses.append(nll_loss.numpy())
                epoch_kde_losses.append(kde_loss.numpy())
            
            # Record average losses
            avg_total_loss = np.mean(epoch_total_losses)
            avg_nll_loss = np.mean(epoch_nll_losses)
            avg_kde_loss = np.mean(epoch_kde_losses)
            
            self.train_losses.append(avg_total_loss)
            self.nll_losses.append(avg_nll_loss)
            self.kde_losses.append(avg_kde_loss)
            
            # Validation
            if val_data is not None and (epoch + 1) % validation_freq == 0:
                val_loss, val_nll, val_kde = self.compute_loss(val_data, val_conditions)
                self.val_losses.append(val_loss.numpy())
                
                if verbose:
                    print(f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Train: {avg_total_loss:.4f} (NLL: {avg_nll_loss:.4f}, KDE: {avg_kde_loss:.4f}) | "
                          f"Val: {val_loss:.4f}")
            elif verbose and (epoch + 1) % validation_freq == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train: {avg_total_loss:.4f} (NLL: {avg_nll_loss:.4f}, KDE: {avg_kde_loss:.4f})")
        
        print("✅ Training completed!")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'nll_losses': self.nll_losses,
            'kde_losses': self.kde_losses
        }


def create_toy_dataset(n_samples: int = 10000, n_dim: int = 6, seed: int = 42) -> Tuple[np.ndarray, gaussian_kde]:
    """
    Create a toy dataset for testing KDE-informed MAF training
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_dim : int
        Dimensionality of the data
    seed : int
        Random seed
        
    Returns:
    --------
    data : np.ndarray
        Generated data samples
    kde_model : gaussian_kde
        Fitted KDE model
    """
    np.random.seed(seed)
    
    print(f"🎲 Creating toy dataset: {n_samples} samples, {n_dim}D")
    
    # Create a mixture of Gaussians for interesting density structure
    n_components = 3
    
    # Component parameters
    means = np.random.uniform(-5, 5, (n_components, n_dim))
    covs = []
    for i in range(n_components):
        # Create random positive definite covariance matrix
        A = np.random.randn(n_dim, n_dim)
        cov = A @ A.T + 0.1 * np.eye(n_dim)
        covs.append(cov)
    
    # Generate samples from mixture
    data = []
    component_weights = np.array([0.4, 0.3, 0.3])  # Mixing weights
    
    for i in range(n_samples):
        # Choose component
        component = np.random.choice(n_components, p=component_weights)
        
        # Sample from chosen component
        sample = np.random.multivariate_normal(means[component], covs[component])
        data.append(sample)
    
    data = np.array(data)
    
    # Fit KDE model
    print("🔧 Fitting KDE model...")
    kde_model = gaussian_kde(data.T)
    
    print(f"✅ Toy dataset created:")
    print(f"   Data shape: {data.shape}")
    print(f"   Data range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"   KDE model fitted with {len(data)} samples")
    
    return data, kde_model


def create_astrophysical_toy_dataset(n_samples: int = 10000, seed: int = 42) -> Tuple[np.ndarray, gaussian_kde]:
    """
    Create an astrophysically-motivated toy dataset
    
    Simulates 6D phase space (x, y, z, vx, vy, vz) with realistic correlations
    """
    np.random.seed(seed)
    
    print(f"🌟 Creating astrophysical toy dataset: {n_samples} samples")
    
    # Generate stellar masses (log-normal distribution)
    log_masses = np.random.normal(0, 1, n_samples)
    masses = 10**log_masses
    
    # Create mass-dependent phase space correlations
    # Higher mass stars: more central positions, higher velocities
    mass_factor = np.log10(masses + 1e-3)
    
    # Positions (kpc) - mass-dependent radius
    radius_scale = np.exp(-0.1 * mass_factor)  # More massive = more central
    positions = []
    for i in range(3):  # x, y, z
        pos = np.random.normal(0, radius_scale, n_samples)
        positions.append(pos)
    
    # Velocities (km/s) - mass-dependent velocity dispersion
    velocity_scale = 50 + 20 * mass_factor  # More massive = higher velocity
    velocities = []
    for i in range(3):  # vx, vy, vz
        vel = np.random.normal(0, velocity_scale, n_samples)
        velocities.append(vel)
    
    # Combine into 6D phase space
    phase_space = np.column_stack(positions + velocities)
    
    # Fit KDE model
    print("🔧 Fitting KDE model...")
    kde_model = gaussian_kde(phase_space.T)
    
    print(f"✅ Astrophysical toy dataset created:")
    print(f"   Phase space shape: {phase_space.shape}")
    print(f"   Position range: [{phase_space[:, :3].min():.2f}, {phase_space[:, :3].max():.2f}] kpc")
    print(f"   Velocity range: [{phase_space[:, 3:].min():.2f}, {phase_space[:, 3:].max():.2f}] km/s")
    print(f"   Mass range: [{masses.min():.3e}, {masses.max():.3e}] M☉")
    
    return phase_space, kde_model


def plot_training_results(history: Dict[str, list], save_path: Optional[str] = None):
    """
    Plot training results
    
    Parameters:
    -----------
    history : Dict[str, list]
        Training history
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    axes[0, 0].plot(history['train_losses'], label='Train', alpha=0.8)
    if history['val_losses']:
        axes[0, 0].plot(range(0, len(history['train_losses']), 10), history['val_losses'], 
                       label='Validation', marker='o', markersize=4)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # NLL loss
    axes[0, 1].plot(history['nll_losses'], label='NLL', color='orange', alpha=0.8)
    axes[0, 1].set_title('Negative Log-Likelihood Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('NLL Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KDE loss
    axes[1, 0].plot(history['kde_losses'], label='KDE MSE', color='green', alpha=0.8)
    axes[1, 0].set_title('KDE Regularization Loss (MSE)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KDE MSE Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss components comparison
    axes[1, 1].plot(history['nll_losses'], label='NLL', alpha=0.8)
    axes[1, 1].plot(history['kde_losses'], label='KDE MSE', alpha=0.8)
    axes[1, 1].set_title('Loss Components Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training plots saved to {save_path}")
    
    plt.show()


def compare_maf_vs_kde_samples(maf_flow: ConditionalTFPNormalizingFlow, kde_model: gaussian_kde,
                              n_samples: int = 5000, save_path: Optional[str] = None):
    """
    Compare samples from MAF and KDE models
    
    Parameters:
    -----------
    maf_flow : ConditionalTFPNormalizingFlow
        Trained MAF model
    kde_model : gaussian_kde
        KDE model
    n_samples : int
        Number of samples to generate
    save_path : str, optional
        Path to save comparison plots
    """
    print(f"🔍 Comparing MAF vs KDE samples ({n_samples} samples each)")
    
    # Generate samples
    maf_samples = maf_flow.sample(n_samples, seed=42).numpy()
    kde_samples = kde_model.resample(n_samples).T
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot first 3 dimensions
    for i in range(3):
        # MAF samples
        axes[0, i].hist(maf_samples[:, i], bins=50, alpha=0.7, label='MAF', density=True)
        axes[0, i].set_title(f'MAF - Dimension {i+1}')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Density')
        axes[0, i].grid(True, alpha=0.3)
        
        # KDE samples
        axes[1, i].hist(kde_samples[:, i], bins=50, alpha=0.7, label='KDE', density=True, color='orange')
        axes[1, i].set_title(f'KDE - Dimension {i+1}')
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Density')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plots saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print("\n📈 Sample Statistics:")
    for i in range(min(6, maf_samples.shape[1])):
        maf_mean, maf_std = maf_samples[:, i].mean(), maf_samples[:, i].std()
        kde_mean, kde_std = kde_samples[:, i].mean(), kde_samples[:, i].std()
        print(f"   Dim {i+1}: MAF({maf_mean:.3f}±{maf_std:.3f}) vs KDE({kde_mean:.3f}±{kde_std:.3f})")


def main():
    """Main function to run KDE-informed MAF training"""
    parser = argparse.ArgumentParser(description='KDE-Informed MAF Training')
    parser.add_argument('--toy_dataset', action='store_true', help='Use toy dataset')
    parser.add_argument('--astrophysical', action='store_true', help='Use astrophysical toy dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lambda_kde', type=float, default=0.1, help='KDE regularization weight')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of MAF layers')
    parser.add_argument('--hidden_units', type=int, default=64, help='Hidden units per layer')
    parser.add_argument('--output_dir', type=str, default='./kde_maf_output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🌟 KDE-Informed MAF Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Dataset: {'Astrophysical' if args.astrophysical else 'Toy' if args.toy_dataset else 'Default'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  λ (KDE weight): {args.lambda_kde}")
    print(f"  Samples: {args.n_samples}")
    print(f"  MAF layers: {args.n_layers}")
    print(f"  Hidden units: {args.hidden_units}")
    print("=" * 60)
    
    # Generate dataset
    if args.astrophysical:
        data, kde_model = create_astrophysical_toy_dataset(args.n_samples, args.seed)
        n_dim = 6
    elif args.toy_dataset:
        data, kde_model = create_toy_dataset(args.n_samples, n_dim=6, seed=args.seed)
        n_dim = 6
    else:
        # Default: simple 2D dataset for quick testing
        data, kde_model = create_toy_dataset(args.n_samples, n_dim=2, seed=args.seed)
        n_dim = 2
    
    # Convert to TensorFlow tensors
    data_tf = tf.constant(data, dtype=tf.float32)
    
    # Split data
    n_train = int(0.8 * len(data))
    train_data = data_tf[:n_train]
    val_data = data_tf[n_train:]
    
    print(f"📊 Data split: {len(train_data)} train, {len(val_data)} validation")
    
    # Create MAF model
    print("🏗️ Creating MAF model...")
    maf_flow = ConditionalTFPNormalizingFlow(
        input_dim=n_dim,
        condition_dim=0,  # No conditioning for this example
        n_layers=args.n_layers,
        hidden_units=args.hidden_units,
        name='kde_informed_maf'
    )
    
    # Create KDE-informed trainer
    print("🎯 Creating KDE-informed trainer...")
    trainer = KDEInformedMAFTrainer(
        flow=maf_flow,
        kde_model=kde_model,
        learning_rate=args.learning_rate,
        lambda_kde=args.lambda_kde,
        weight_decay=0.0,
        noise_std=0.0
    )
    
    # Train the model
    print("🏋️ Training KDE-informed MAF...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_freq=5,
        verbose=True
    )
    
    # Save model
    model_path = output_dir / "kde_informed_maf.npz"
    maf_flow.save(str(model_path))
    
    # Plot training results
    plot_path = output_dir / "training_results.png"
    plot_training_results(history, str(plot_path))
    
    # Compare MAF vs KDE samples
    comparison_path = output_dir / "maf_vs_kde_comparison.png"
    compare_maf_vs_kde_samples(maf_flow, kde_model, n_samples=5000, save_path=str(comparison_path))
    
    # Save training history
    history_path = output_dir / "training_history.npz"
    np.savez(history_path, **history)
    
    print(f"\n✅ KDE-informed MAF training completed!")
    print(f"📁 Outputs saved to: {output_dir}")
    print(f"   Model: {model_path}")
    print(f"   Training plots: {plot_path}")
    print(f"   Comparison plots: {comparison_path}")
    print(f"   History: {history_path}")
    
    # Print final statistics
    final_train_loss = history['train_losses'][-1]
    final_nll_loss = history['nll_losses'][-1]
    final_kde_loss = history['kde_losses'][-1]
    
    print(f"\n📈 Final Training Statistics:")
    print(f"   Total loss: {final_train_loss:.4f}")
    print(f"   NLL loss: {final_nll_loss:.4f}")
    print(f"   KDE MSE loss: {final_kde_loss:.4f}")
    print(f"   KDE contribution: {args.lambda_kde * final_kde_loss:.4f} ({100 * args.lambda_kde * final_kde_loss / final_train_loss:.1f}%)")
    print(f"\n💡 Hyperparameter tuning tips:")
    print(f"   - Increase λ (--lambda_kde) to make MAF follow KDE more closely")
    print(f"   - Decrease λ to let MAF learn more from raw data")
    print(f"   - Current λ = {args.lambda_kde} balances both objectives")


if __name__ == "__main__":
    main()
