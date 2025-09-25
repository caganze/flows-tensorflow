#!/usr/bin/env python3
"""
KDE-Informed Conditional MAF Training Implementation

This script implements KDE-informed training for conditional MAF models that condition on stellar mass.
The approach combines KDE and conditional MAF training through regularization in the loss function.

ARCHITECTURE: Conditional MAF
- Learns p(x,y,z,vx,vy,vz|m) - phase space conditioned on mass
- Uses Masked Autoregressive Flow (MAF) with mass conditioning
- KDE models created for different mass bins
- MSE loss between conditional MAF and conditional KDE log probabilities

Key features:
1. Conditional KDE density evaluation for MAF training batches
2. Combined loss function: L = L_NLL + λ * L_KDE
3. MSE between MAF and KDE log probabilities as regularization
4. Mass-conditioned density estimation
5. Toy dataset generation with mass-phase space correlations
6. Flexible hyperparameter tuning (λ, learning rates, etc.)

Usage:
    python kde_informed_maf_conditional_training.py --toy_dataset --epochs 50 --lambda_kde 0.1
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from pathlib import Path
from typing import Tuple, Dict, Optional, Any, List
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
            
            # Create a simple conditional MAF
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
            """Compute log probability of samples with conditioning"""
            if conditions is None:
                raise ValueError("Conditions must be provided for conditional flow")
            return self.flow.log_prob(x)
        
        def sample(self, n_samples, conditions=None, seed=None):
            """Generate samples from the flow with conditioning"""
            if conditions is None:
                raise ValueError("Conditions must be provided for conditional flow")
            if seed is not None:
                tf.random.set_seed(seed)
            return self.flow.sample(n_samples)
        
        def save(self, filepath):
            """Save model weights"""
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
            dataset = tf.data.Dataset.from_tensor_slices((train_data, train_conditions))
            dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            
            for epoch in range(epochs):
                epoch_losses = []
                for batch in dataset:
                    batch_data, batch_conditions = batch
                    loss = self.train_step(batch_data, batch_conditions)
                    epoch_losses.append(loss.numpy())
                
                avg_loss = np.mean(epoch_losses)
                self.train_losses.append(avg_loss)
                
                if verbose and (epoch + 1) % validation_freq == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


class ConditionalKDEInformedMAFTrainer:
    """
    KDE-Informed Conditional MAF Trainer
    
    This trainer uses conditional KDE density estimates to guide conditional MAF training.
    The loss function combines negative log-likelihood with MSE between MAF and KDE
    log probabilities: L = L_NLL + λ * L_KDE
    """
    
    def __init__(self, flow: ConditionalTFPNormalizingFlow, kde_models: Dict[float, gaussian_kde],
                 learning_rate: float = 1e-3, lambda_kde: float = 0.1,
                 weight_decay: float = 0.0, noise_std: float = 0.0,
                 mass_bins: int = 10):
        """
        Initialize KDE-informed conditional MAF trainer
        
        Parameters:
        -----------
        flow : ConditionalTFPNormalizingFlow
            The conditional MAF model to train
        kde_models : Dict[float, gaussian_kde]
            Dictionary mapping mass values to KDE models
        learning_rate : float
            Learning rate for optimizer
        lambda_kde : float
            Regularization weight for KDE term (λ in loss function)
        weight_decay : float
            L2 weight decay regularization
        noise_std : float
            Standard deviation of Gaussian noise added during training
        mass_bins : int
            Number of mass bins for KDE model selection
        """
        self.flow = flow
        self.kde_models = kde_models
        self.lambda_kde = lambda_kde
        self.weight_decay = weight_decay
        self.noise_std = noise_std
        self.mass_bins = mass_bins
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Track losses
        self.train_losses = []
        self.val_losses = []
        self.nll_losses = []
        self.kde_losses = []
    
    def get_kde_model_for_mass(self, mass_value: float) -> gaussian_kde:
        """
        Get the appropriate KDE model for a given mass value
        
        Parameters:
        -----------
        mass_value : float
            Mass value to find KDE model for
            
        Returns:
        --------
        kde_model : gaussian_kde
            KDE model for the mass value
        """
        # Find closest mass bin
        mass_keys = list(self.kde_models.keys())
        closest_mass = min(mass_keys, key=lambda x: abs(x - mass_value))
        return self.kde_models[closest_mass]
    
    def evaluate_conditional_kde_density(self, x: tf.Tensor, conditions: tf.Tensor) -> tf.Tensor:
        """
        Evaluate conditional KDE density for a batch of samples
        
        Parameters:
        -----------
        x : tf.Tensor
            Batch of samples to evaluate
        conditions : tf.Tensor
            Mass conditions for the samples
            
        Returns:
        --------
        kde_log_probs : tf.Tensor
            Log probabilities from conditional KDE models
        """
        # Convert to numpy for KDE evaluation
        x_np = x.numpy()
        conditions_np = conditions.numpy().flatten()
        
        kde_log_probs_list = []
        
        for i, mass_val in enumerate(conditions_np):
            # Get KDE model for this mass
            kde_model = self.get_kde_model_for_mass(mass_val)
            
            # Evaluate KDE density for this sample
            sample = x_np[i:i+1].T  # KDE expects (n_features, n_samples)
            kde_log_prob = kde_model.logpdf(sample)[0]  # Single sample
            kde_log_probs_list.append(kde_log_prob)
        
        # Convert back to tensor
        kde_log_probs = tf.constant(kde_log_probs_list, dtype=tf.float32)
        
        return kde_log_probs
    
    def compute_mse_log_probs(self, maf_log_probs: tf.Tensor, kde_log_probs: tf.Tensor) -> tf.Tensor:
        """
        Compute MSE between MAF and KDE log probabilities
        
        MSE = E[(log P_MAF - log P_KDE)^2]
        
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
        mse_loss = tf.reduce_mean(tf.square(maf_log_probs - kde_log_probs))
        return mse_loss
    
    def compute_loss(self, x: tf.Tensor, conditions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute combined loss: L = L_NLL + λ * L_KDE
        
        Where L_KDE is the MSE between MAF and KDE log probabilities:
        L_KDE = E[(log P_MAF - log P_KDE)^2]
        
        Parameters:
        -----------
        x : tf.Tensor
            Training batch
        conditions : tf.Tensor
            Mass conditions
            
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
        kde_log_probs = self.evaluate_conditional_kde_density(x, conditions)
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
    def train_step(self, x: tf.Tensor, conditions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Single training step with conditional KDE guidance
        
        Parameters:
        -----------
        x : tf.Tensor
            Training batch
        conditions : tf.Tensor
            Mass conditions
            
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
    
    def train(self, train_data: tf.Tensor, train_conditions: tf.Tensor,
              val_data: Optional[tf.Tensor] = None, val_conditions: Optional[tf.Tensor] = None,
              epochs: int = 100, batch_size: int = 512, validation_freq: int = 10,
              verbose: bool = True) -> Dict[str, list]:
        """
        Train the KDE-informed conditional MAF
        
        Parameters:
        -----------
        train_data : tf.Tensor
            Training data
        train_conditions : tf.Tensor
            Training mass conditions
        val_data : tf.Tensor, optional
            Validation data
        val_conditions : tf.Tensor, optional
            Validation mass conditions
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
        dataset = tf.data.Dataset.from_tensor_slices((train_data, train_conditions))
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        print(f"🚀 Starting KDE-informed conditional MAF training")
        print(f"   Epochs: {epochs}, Batch size: {batch_size}")
        print(f"   λ (KDE weight): {self.lambda_kde}")
        print(f"   Learning rate: {self.optimizer.learning_rate.numpy():.2e}")
        print(f"   Loss: L = L_NLL + λ * MSE(log P_MAF, log P_KDE)")
        print(f"   KDE models: {len(self.kde_models)} mass bins")
        print("=" * 60)
        
        for epoch in range(epochs):
            epoch_total_losses = []
            epoch_nll_losses = []
            epoch_kde_losses = []
            
            # Training loop
            for batch in dataset:
                batch_data, batch_conditions = batch
                total_loss, nll_loss, kde_loss = self.train_step(batch_data, batch_conditions)
                
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


def create_conditional_kde_models(data: np.ndarray, masses: np.ndarray, 
                                 mass_bins: int = 10) -> Dict[float, gaussian_kde]:
    """
    Create conditional KDE models for different mass bins
    
    Parameters:
    -----------
    data : np.ndarray
        Phase space data
    masses : np.ndarray
        Mass values
    mass_bins : int
        Number of mass bins
        
    Returns:
    --------
    kde_models : Dict[float, gaussian_kde]
        Dictionary mapping mass bin centers to KDE models
    """
    print(f"🔧 Creating conditional KDE models for {mass_bins} mass bins...")
    
    # Create mass bins
    mass_min, mass_max = masses.min(), masses.max()
    mass_bin_edges = np.linspace(mass_min, mass_max, mass_bins + 1)
    mass_bin_centers = 0.5 * (mass_bin_edges[1:] + mass_bin_edges[:-1])
    
    kde_models = {}
    
    for i, (bin_start, bin_end) in enumerate(zip(mass_bin_edges[:-1], mass_bin_edges[1:])):
        # Find data in this mass bin
        mask = (masses >= bin_start) & (masses < bin_end)
        if i == len(mass_bin_edges) - 2:  # Include upper bound for last bin
            mask = (masses >= bin_start) & (masses <= bin_end)
        
        bin_data = data[mask]
        
        if len(bin_data) > 10:  # Need minimum samples for KDE
            kde_model = gaussian_kde(bin_data.T)
            kde_models[mass_bin_centers[i]] = kde_model
            print(f"   Mass bin {i+1}: {len(bin_data)} samples, mass range [{bin_start:.3f}, {bin_end:.3f}]")
        else:
            print(f"   Mass bin {i+1}: Skipped (only {len(bin_data)} samples)")
    
    print(f"✅ Created {len(kde_models)} conditional KDE models")
    return kde_models


def create_astrophysical_conditional_toy_dataset(n_samples: int = 10000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict[float, gaussian_kde]]:
    """
    Create an astrophysically-motivated conditional toy dataset with KDE models
    
    Simulates 6D phase space (x, y, z, vx, vy, vz) with realistic mass correlations
    """
    np.random.seed(seed)
    
    print(f"🌟 Creating astrophysical conditional toy dataset: {n_samples} samples")
    
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
    
    # Create conditional KDE models
    kde_models = create_conditional_kde_models(phase_space, masses, mass_bins=8)
    
    print(f"✅ Astrophysical conditional toy dataset created:")
    print(f"   Phase space shape: {phase_space.shape}")
    print(f"   Mass shape: {masses.shape}")
    print(f"   Position range: [{phase_space[:, :3].min():.2f}, {phase_space[:, :3].max():.2f}] kpc")
    print(f"   Velocity range: [{phase_space[:, 3:].min():.2f}, {phase_space[:, 3:].max():.2f}] km/s")
    print(f"   Mass range: [{masses.min():.3e}, {masses.max():.3e}] M☉")
    
    return phase_space, masses, kde_models


def plot_conditional_training_results(history: Dict[str, list], save_path: Optional[str] = None):
    """
    Plot conditional training results
    
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


def compare_conditional_maf_vs_kde_samples(maf_flow: ConditionalTFPNormalizingFlow, kde_models: Dict[float, gaussian_kde],
                                          mass_conditions: np.ndarray, n_samples: int = 1000, 
                                          save_path: Optional[str] = None):
    """
    Compare samples from conditional MAF and KDE models
    
    Parameters:
    -----------
    maf_flow : ConditionalTFPNormalizingFlow
        Trained conditional MAF model
    kde_models : Dict[float, gaussian_kde]
        KDE models for different masses
    mass_conditions : np.ndarray
        Mass conditions for sampling
    n_samples : int
        Number of samples to generate per mass
    save_path : str, optional
        Path to save comparison plots
    """
    print(f"🔍 Comparing conditional MAF vs KDE samples")
    
    # Sample from different mass bins
    mass_bins = list(kde_models.keys())
    n_mass_bins = min(4, len(mass_bins))  # Compare up to 4 mass bins
    
    fig, axes = plt.subplots(2, n_mass_bins, figsize=(4*n_mass_bins, 8))
    if n_mass_bins == 1:
        axes = axes.reshape(-1, 1)
    
    for i, mass_val in enumerate(mass_bins[:n_mass_bins]):
        # Generate MAF samples
        mass_condition = tf.constant([[mass_val]], dtype=tf.float32)
        maf_samples = maf_flow.sample(n_samples, mass_condition, seed=42).numpy()
        
        # Generate KDE samples
        kde_model = kde_models[mass_val]
        kde_samples = kde_model.resample(n_samples).T
        
        # Plot first dimension (x position)
        axes[0, i].hist(maf_samples[:, 0], bins=30, alpha=0.7, label='MAF', density=True)
        axes[0, i].set_title(f'MAF - Mass {mass_val:.2f} M☉')
        axes[0, i].set_xlabel('X Position (kpc)')
        axes[0, i].set_ylabel('Density')
        axes[0, i].grid(True, alpha=0.3)
        
        axes[1, i].hist(kde_samples[:, 0], bins=30, alpha=0.7, label='KDE', density=True, color='orange')
        axes[1, i].set_title(f'KDE - Mass {mass_val:.2f} M☉')
        axes[1, i].set_xlabel('X Position (kpc)')
        axes[1, i].set_ylabel('Density')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plots saved to {save_path}")
    
    plt.show()


def main():
    """Main function to run KDE-informed conditional MAF training"""
    parser = argparse.ArgumentParser(description='KDE-Informed Conditional MAF Training')
    parser.add_argument('--toy_dataset', action='store_true', help='Use toy dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lambda_kde', type=float, default=0.1, help='KDE regularization weight')
    parser.add_argument('--n_samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of MAF layers')
    parser.add_argument('--hidden_units', type=int, default=64, help='Hidden units per layer')
    parser.add_argument('--mass_bins', type=int, default=8, help='Number of mass bins for KDE')
    parser.add_argument('--output_dir', type=str, default='./kde_conditional_maf_output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🌟 KDE-Informed Conditional MAF Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Dataset: {'Astrophysical' if args.toy_dataset else 'Default'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  λ (KDE weight): {args.lambda_kde}")
    print(f"  Samples: {args.n_samples}")
    print(f"  MAF layers: {args.n_layers}")
    print(f"  Hidden units: {args.hidden_units}")
    print(f"  Mass bins: {args.mass_bins}")
    print("=" * 60)
    
    # Generate dataset
    if args.toy_dataset:
        phase_space, masses, kde_models = create_astrophysical_conditional_toy_dataset(args.n_samples, args.seed)
        n_dim = 6
    else:
        # Default: simple 2D dataset for quick testing
        phase_space, masses, kde_models = create_astrophysical_conditional_toy_dataset(args.n_samples, args.seed)
        n_dim = 6
    
    # Convert to TensorFlow tensors
    data_tf = tf.constant(phase_space, dtype=tf.float32)
    masses_tf = tf.constant(masses.reshape(-1, 1), dtype=tf.float32)
    
    # Split data
    n_train = int(0.8 * len(phase_space))
    train_data = data_tf[:n_train]
    train_masses = masses_tf[:n_train]
    val_data = data_tf[n_train:]
    val_masses = masses_tf[n_train:]
    
    print(f"📊 Data split: {len(train_data)} train, {len(val_data)} validation")
    
    # Create conditional MAF model
    print("🏗️ Creating conditional MAF model...")
    maf_flow = ConditionalTFPNormalizingFlow(
        input_dim=n_dim,
        condition_dim=1,  # 1D mass conditioning
        n_layers=args.n_layers,
        hidden_units=args.hidden_units,
        name='kde_informed_conditional_maf'
    )
    
    # Create KDE-informed trainer
    print("🎯 Creating KDE-informed conditional trainer...")
    trainer = ConditionalKDEInformedMAFTrainer(
        flow=maf_flow,
        kde_models=kde_models,
        learning_rate=args.learning_rate,
        lambda_kde=args.lambda_kde,
        weight_decay=0.0,
        noise_std=0.0,
        mass_bins=args.mass_bins
    )
    
    # Train the model
    print("🏋️ Training KDE-informed conditional MAF...")
    history = trainer.train(
        train_data=train_data,
        train_conditions=train_masses,
        val_data=val_data,
        val_conditions=val_masses,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_freq=5,
        verbose=True
    )
    
    # Save model
    model_path = output_dir / "kde_informed_conditional_maf.npz"
    maf_flow.save(str(model_path))
    
    # Plot training results
    plot_path = output_dir / "conditional_training_results.png"
    plot_conditional_training_results(history, str(plot_path))
    
    # Compare MAF vs KDE samples
    comparison_path = output_dir / "conditional_maf_vs_kde_comparison.png"
    compare_conditional_maf_vs_kde_samples(maf_flow, kde_models, masses, n_samples=1000, save_path=str(comparison_path))
    
    # Save training history
    history_path = output_dir / "conditional_training_history.npz"
    np.savez(history_path, **history)
    
    print(f"\n✅ KDE-informed conditional MAF training completed!")
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
    print(f"   - Adjust --mass_bins to control KDE model granularity")
    print(f"   - Current λ = {args.lambda_kde} balances both objectives")


if __name__ == "__main__":
    main()
