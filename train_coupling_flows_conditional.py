#!/usr/bin/env python3
"""
Conditional Coupling Flows (RealNVP) training script
Uses RealNVP bijectors with mass conditioning for stellar particle data
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

# Import our utilities
from kroupa_imf import sample_with_kroupa_imf
from optimized_io import save_samples_optimized
from comprehensive_logging import ComprehensiveLogger
from symlib_utils import load_particle_data, get_output_paths, validate_symlib_environment

# TFP aliases
tfd = tfp.distributions
tfb = tfp.bijectors


class ConditionalCouplingFlow:
    """
    Conditional RealNVP coupling flow with mass conditioning
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        n_layers: int = 8,
        hidden_units: int = 128,
        n_cond_layers: int = 3,
        n_mass_bins: int = 8,
        embedding_dim: int = 4,
        n_gmm_components: int = 5,
        name: str = 'conditional_coupling_flow'
    ):
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.n_cond_layers = n_cond_layers
        self.n_mass_bins = n_mass_bins
        self.embedding_dim = embedding_dim
        self.n_gmm_components = n_gmm_components
        self.name = name
        
        # Build the flow
        self.flow = self._build_flow()
    
    def _build_coupling_conditional_net(
        self, input_dim_x: int, output_dim: int, layer_idx: int
    ):
        """
        Builds the conditional network for the scale and shift (s and t).
        It takes a split of the phase space (x_part) and the mass bin embedding.
        """
        
        # 1. The Embedding Layer (Converts integer bin index to float vector)
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.n_mass_bins,
            output_dim=self.embedding_dim,
            input_length=1,
            name=f'{self.name}_embedding_{layer_idx}'
        )
        
        # 2. Input Definitions with explicit shapes
        # Input 1: Phase Space portion (x_part). Shape: (None, D_PART)
        x_part_input = tf.keras.Input(shape=(input_dim_x,), dtype=tf.float32, name=f'{self.name}_x_part_input_{layer_idx}')
        
        # Input 2: Condition (mass bin index). Shape: (None, 1)
        c_input = tf.keras.Input(shape=(1,), dtype=tf.int32, name=f'{self.name}_c_input_{layer_idx}') 

        # 3. Process Conditional Input (c_input)
        c_embedding = embedding_layer(c_input) 
        # Reshape to ensure proper shape: (None, embedding_dim)
        c_flat = tf.reshape(c_embedding, [-1, self.embedding_dim])

        # 4. Concatenate using tf.concat instead of Keras Concatenate layer
        # Total input dimension for the dense layers: input_dim_x + embedding_dim
        combined_input = tf.concat([x_part_input, c_flat], axis=-1)

        # 5. Core Dense Network (The Sequential model of hidden layers)
        # Input shape must be D_PART + embedding_dim (e.g., 3 + 4 = 7)
        CORE_INPUT_DIM = input_dim_x + self.embedding_dim
        
        network = tf.keras.Sequential([], name=f'{self.name}_core_net_{layer_idx}')
        
        network.add(tf.keras.layers.Dense(
            self.hidden_units, 
            activation='relu', 
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            # 🌟 CRITICAL FIX: Explicitly set the input shape of the first dense layer 
            input_shape=(CORE_INPUT_DIM,), 
            name=f'{self.name}_dense_{layer_idx}_0'
        ))

        for i in range(1, self.n_cond_layers):
            network.add(tf.keras.layers.Dense(
                self.hidden_units, 
                activation='relu', 
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                name=f'{self.name}_dense_{layer_idx}_{i}'
            ))
            
        # Output layer
        network.add(tf.keras.layers.Dense(
            output_dim, 
            activation=None,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            name=f'{self.name}_output_{layer_idx}'
        ))

        # Apply the network to the combined input
        output = network(combined_input)  # <--- This layer call is now robust

        # 6. Full Model Definition
        full_conditional_model = tf.keras.Model(
            inputs=[x_part_input, c_input], 
            outputs=output, 
            name=f'{self.name}_full_conditional_net_{layer_idx}'
        )

        return full_conditional_model

    def _coupling_shift_and_log_scale_fn(self, conditional_net):
        """
        The callable required by tfp.bijectors.RealNVP.
        NOTE: This is intentionally simplified. Rank 2 input is guaranteed by log_prob/sample.
        """
        def shift_and_log_scale_fn(x, conditional_input=None):
            # 1. Cast conditional input to the required integer dtype
            c_input = tf.cast(conditional_input, dtype=tf.int32)
            
            # 2. Call the conditional Keras model. 
            # We assume x and c_input are already Rank 2 due to the log_prob fix.
            # This simplifies the function and avoids nested tf.cond issues.
            output = conditional_net([x, c_input])
            
            # Split the output into shift and log_scale (Must be D/2 dimension each)
            shift, log_scale = tf.split(output, 2, axis=-1)
            
            # Clip log_scale for stability (a standard practice)
            log_scale = tf.clip_by_value(log_scale, -5.0, 5.0)
            
            return shift, log_scale
        
        return shift_and_log_scale_fn

    def _build_flow(self):
        """Constructs the full TFP distribution (RealNVP + GMM base)."""
        
        # RealNVP splits the D-dim input into two halves (e.g., [r_x, r_y, r_z] and [v_x, v_y, v_z])
        D_PART = self.input_dim // 2
        
        bijectors = []
        
        for i in range(self.n_layers):
            # 1. Build the Conditional Network for the current layer
            # Input to the net is D_PART (the fixed part of the flow) + EMBEDDING_DIM
            # Output of the net is 2 * D_PART (for shift and scale of the other part)
            conditional_net = self._build_coupling_conditional_net(
                input_dim_x=D_PART,
                output_dim=2 * D_PART,
                layer_idx=i
            )
            
            # 2. Create the RealNVP Bijector
            real_nvp = tfb.RealNVP(
                num_masked=D_PART, # Fixed part (e.g., 3 dimensions)
                shift_and_log_scale_fn=self._coupling_shift_and_log_scale_fn(conditional_net),
                is_constant_jacobian=False,
                name=f'{self.name}_RealNVP_{i}'
            )
            bijectors.append(real_nvp)
            
            # 3. Add Permutation Bijector (Critical for coupling flows)
            # This shuffles the dimensions so a different part gets transformed next.
            if i < self.n_layers - 1:
                bijectors.append(tfb.Permute(permutation=list(np.roll(range(self.input_dim), shift=D_PART))))
                
        # Reverse the order of bijectors to define the forward pass (X -> Z)
        chain = tfb.Chain(list(reversed(bijectors)), name=f'{self.name}_Chain_RealNVP')

        # 4. Base Distribution (Trainable GMM)
        # The GMM parameters themselves are trainable variables
        base_dist = tfd.MixtureSameFamily(
            components_distribution=tfd.MultivariateNormalDiag(
                loc=tf.Variable(tf.random.normal([self.n_gmm_components, self.input_dim]), name=f'{self.name}_gmm_locs'),
                scale_diag=tf.Variable(tf.random.uniform([self.n_gmm_components, self.input_dim], minval=0.1, maxval=1.0), name=f'{self.name}_gmm_scales')
            ),
            mixture_distribution=tfd.Categorical(
                logits=tf.Variable(tf.random.normal([self.n_gmm_components]), name=f'{self.name}_gmm_logits')
            ),
            name=f'{self.name}_GMM_Base'
        )

        # 5. Final Transformed Distribution
        td = tfd.TransformedDistribution(
            distribution=base_dist, 
            bijector=chain,
            name=self.name
        )
        
        return td

    def log_prob(self, x, conditions):
        """
        Compute log probability of samples given conditions.
        
        🌟 FINAL ROBUST FIX: Use tf.reshape to explicitly force Rank 2 (Batch, Features)
        when the tensor is Rank 1. This resolves TFP's internal slicing bug where
        it fails to recognize the batch dimension added by tf.expand_dims.
        """
        
        # Check Rank of x (Phase Space)
        rank_x = tf.rank(x)
        x_expanded = tf.cond(
            tf.less(rank_x, 2), 
            # FIX: Use reshape to ensure the new batch dimension is registered in the graph's static shape
            lambda: tf.reshape(x, (1, -1)), 
            lambda: x 
        )
        
        # Check Rank of conditions (Mass Bins)
        rank_c = tf.rank(conditions)
        c_expanded = tf.cond(
            tf.less(rank_c, 2),
            # FIX: Use reshape to ensure the new batch dimension is registered
            lambda: tf.reshape(conditions, (1, 1)), 
            lambda: conditions 
        )
        
        # Pass the robustly shaped tensors to the internal flow
        return self.flow.log_prob(x_expanded, conditional_input=c_expanded)

    def sample(self, n_samples, conditions):
        """
        Sample from the conditional distribution.
        
        🌟 Apply the same Rank 2 enforcement fix for consistency.
        """
        # Check Rank of conditions (Mass Bins)
        rank_c = tf.rank(conditions)
        c_expanded = tf.cond(
            tf.less(rank_c, 2),
            # FIX: Use reshape to ensure the new batch dimension is registered
            lambda: tf.reshape(conditions, (1, 1)),
            lambda: conditions
        )
        
        return self.flow.sample(n_samples, conditional_input=c_expanded)

    @property
    def trainable_variables(self):
        """Get all trainable variables"""
        return self.flow.trainable_variables

    def save(self, filepath: str):
        """Save the trained model"""
        # Get all trainable variables
        variables = self.trainable_variables
        
        # Create save dictionary with configuration
        save_dict = {
            'config': np.array({
                'input_dim': self.input_dim,
                'n_layers': self.n_layers,
                'hidden_units': self.hidden_units,
                'n_cond_layers': self.n_cond_layers,
                'n_mass_bins': self.n_mass_bins,
                'embedding_dim': self.embedding_dim,
                'n_gmm_components': self.n_gmm_components,
                'name': self.name
            }, dtype=object)
        }
        
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
        print(f"✅ Coupling flow saved to {filepath}")


class ConditionalCouplingFlowTrainer:
    """Trainer for conditional coupling flows"""
    
    def __init__(self, flow, learning_rate=1e-3):
        self.flow = flow
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.train_losses = []
        self.val_losses = []
    
    @tf.function
    def compute_loss(self, x, conditions):
        """Compute negative log likelihood loss"""
        log_probs = self.flow.log_prob(x, conditions)
        # Clip log probabilities for numerical stability
        log_probs = tf.clip_by_value(log_probs, -50.0, 50.0)
        return -tf.reduce_mean(log_probs)
    
    @tf.function
    def train_step(self, x, conditions):
        """Single training step"""
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, conditions)
        
        # Check for finite loss
        loss_is_finite = tf.math.is_finite(loss)
        
        gradients = tape.gradient(loss, self.flow.trainable_variables)
        
        # Check for finite gradients
        gradients_are_finite = tf.reduce_all([
            tf.reduce_all(tf.math.is_finite(g)) for g in gradients if g is not None
        ])
        
        # Check for reasonable loss
        loss_is_reasonable = tf.less(loss, 100.0)
        
        # Only apply gradients if everything is finite and reasonable
        should_apply_gradients = tf.logical_and(
            tf.logical_and(loss_is_finite, gradients_are_finite),
            loss_is_reasonable
        )
        
        def apply_gradients():
            # Clip gradients for stability
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
            self.optimizer.apply_gradients(zip(clipped_gradients, self.flow.trainable_variables))
            return loss
        
        def skip_step():
            return tf.constant(100.0, dtype=tf.float32)
        
        final_loss = tf.cond(should_apply_gradients, apply_gradients, skip_step)
        return final_loss
    
    def train(self, train_data, train_conditions, val_data, val_conditions, epochs=50, validation_freq=5):
        """Train the coupling flow"""
        print(f"🏋️ Training coupling flow for {epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            # Training
            epoch_losses = []
            for batch_data, batch_conditions in zip(train_data, train_conditions):
                loss = self.train_step(batch_data, batch_conditions)
                epoch_losses.append(loss.numpy())
            
            train_loss = np.mean(epoch_losses)
            self.train_losses.append(train_loss)
            
            # Validation
            if epoch % validation_freq == 0 or epoch == epochs - 1:
                val_loss = self.compute_loss(val_data, val_conditions).numpy()
                self.val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"✅ New best validation loss: {best_val_loss:.6f} (epoch {epoch})")
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch}")
                    break
                
                # Check for numerical issues
                if not np.isfinite(val_loss) or val_loss > 100.0:
                    print(f"❌ Validation loss is NaN or too large - stopping training")
                    break
        
        print(f"✅ Training completed! Final train loss: {train_loss:.6f}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        return self.train_losses, self.val_losses


def setup_gpu():
    """Set up GPU configuration"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
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
    """Load astrophysical data with mass conditioning from symlib simulation"""
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
    if data.shape[1] >= 7:
        masses = data[:, 6]
    else:
        # If mass is not available, create dummy masses
        masses = np.ones(len(phase_space)) * 1.0  # 1 M☉ default
    
    # Convert to TensorFlow tensors
    phase_space_tensor = tf.convert_to_tensor(phase_space, dtype=tf.float32)
    mass_tensor = tf.convert_to_tensor(masses, dtype=tf.float32)
    
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
    n_mass_bins: int = 8
) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
    """Preprocess astrophysical data and mass conditions for conditional training with embedding"""
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
    
    # Convert masses to numpy for binning operations
    masses_np = mass_conditions.numpy()
    
    # Handle the RuntimeWarning by setting minimum mass for log10
    log_masses = np.log10(masses_np, where=masses_np > 0, out=np.full_like(masses_np, -10.0))
    log_mass_min = np.min(log_masses)
    log_mass_max = np.max(log_masses)
    
    # Create bins based on log-uniform distribution
    bins = np.linspace(log_mass_min, log_mass_max, n_mass_bins + 1)
    
    # Get the integer bin index for each particle
    # np.digitize returns indices [1, n_mass_bins]. Subtract 1 to get indices [0, n_mass_bins-1].
    mass_bin_indices = np.digitize(log_masses, bins[:-1]) - 1
    mass_bin_indices = np.clip(mass_bin_indices, 0, n_mass_bins - 1)
    
    # Convert to TensorFlow tensor of integer type
    mass_bin_indices = tf.convert_to_tensor(mass_bin_indices, dtype=tf.int32)
    mass_bin_indices = tf.expand_dims(mass_bin_indices, axis=-1)
    
    # Store preprocessing statistics
    preprocessing_stats = {
        'ps_mean': ps_mean,
        'ps_std': ps_std,
        'mass_bins': bins,
        'n_mass_bins': n_mass_bins,
        'log_mass_min': log_mass_min,
        'log_mass_max': log_mass_max,
        'standardize': standardize,
        'clip_outliers': clip_outliers
    }
    
    print(f"✅ Conditional preprocessing complete")
    print(f"Phase space range: [{tf.reduce_min(processed_phase_space):.3f}, {tf.reduce_max(processed_phase_space):.3f}]")
    print(f"Mass bin indices range: [{tf.reduce_min(mass_bin_indices):.0f}, {tf.reduce_max(mass_bin_indices):.0f}]")
    print(f"Number of mass bins: {n_mass_bins}")
    
    return processed_phase_space, mass_bin_indices, preprocessing_stats


def split_conditional_data(
    phase_space_data: tf.Tensor,
    mass_conditions: tf.Tensor,
    train_frac: float = 0.8, 
    val_frac: float = 0.1,
    seed: int = 42
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Split data into train/validation/test sets"""
    np.random.seed(seed)
    n_samples = len(phase_space_data)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    
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


def make_json_serializable(obj):
    """Convert NumPy/TensorFlow types to JSON-serializable Python types"""
    if hasattr(obj, 'numpy'):
        return float(obj.numpy())
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif hasattr(obj, 'dtype') and hasattr(obj, 'numpy'):
        return obj.numpy().tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


def train_and_save_conditional_coupling_flow(
    halo_id: str,
    particle_pid: int,
    suite: str,
    output_dir: str,
    epochs: int = 150,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    n_layers: int = 8,
    hidden_units: int = 128,
    n_cond_layers: int = 3,
    n_mass_bins: int = 8,
    embedding_dim: int = 4,
    n_gmm_components: int = 5,
    generate_samples: bool = True,
    n_samples: int = 100000,
    validation_split: float = 0.2,
    validation_freq: int = 5
) -> Tuple[str, str]:
    """High-level function to train and save a conditional coupling flow for a specific particle."""
    
    # Set up comprehensive logging
    log_dir = f"{output_dir}/logs"
    logger = ComprehensiveLogger(log_dir, "conditional_coupling_training", particle_pid)
    
    logger.info(f"🚀 Starting conditional coupling flow training for particle PID {particle_pid}")
    logger.info(f"📋 Training parameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  N layers: {n_layers}")
    logger.info(f"  Hidden units: {hidden_units}")
    logger.info(f"  N cond layers: {n_cond_layers}")
    logger.info(f"  N mass bins: {n_mass_bins}")
    logger.info(f"  Embedding dim: {embedding_dim}")
    logger.info(f"  N GMM components: {n_gmm_components}")
    
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
            phase_space_data, mass_conditions, n_mass_bins=n_mass_bins
        )
        
        # Split data
        logger.info("🔀 Splitting conditional data...")
        train_ps, train_mass, val_ps, val_mass, test_ps, test_mass = split_conditional_data(
            processed_ps, processed_mass, val_frac=validation_split
        )
        
        # Create conditional coupling flow
        logger.info(f"🏗️ Creating conditional coupling flow ({n_layers} layers, {hidden_units} units)...")
        flow = ConditionalCouplingFlow(
            input_dim=6,  # 6D phase space
            n_layers=n_layers,
            hidden_units=hidden_units,
            n_cond_layers=n_cond_layers,
            n_mass_bins=n_mass_bins,
            embedding_dim=embedding_dim,
            n_gmm_components=n_gmm_components,
            name=f'conditional_coupling_flow_pid{particle_pid}'
        )
        
        # Create trainer
        logger.info("🎯 Creating coupling flow trainer...")
        trainer = ConditionalCouplingFlowTrainer(flow, learning_rate=learning_rate)
        
        # Train the flow
        logger.info(f"🏋️ Training coupling flow for {epochs} epochs...")
        train_losses, val_losses = trainer.train(
            train_ps, train_mass, val_ps, val_mass, 
            epochs=epochs, validation_freq=validation_freq
        )
        
        # Save the trained model
        logger.info("💾 Saving trained model...")
        model_path = f"{output_dir}/model_pid{particle_pid}.npz"
        flow.save(model_path)
        
        # Save preprocessing statistics
        preprocessing_path = f"{output_dir}/model_pid{particle_pid}_preprocessing.npz"
        np.savez_compressed(preprocessing_path, **{k: v.numpy() if hasattr(v, 'numpy') else v 
                                                 for k, v in preprocessing_stats.items()})
        
        # Generate samples if requested
        samples_path = None
        if generate_samples:
            logger.info("🎲 Generating samples...")
            try:
                # Use Kroupa IMF for sampling
                stellar_mass = float(metadata.get('total_mass', 1e5))
                logger.info(f"🌟 Using Kroupa IMF sampling for stellar mass: {stellar_mass:.2e} M☉")
                
                # Generate samples using Kroupa IMF
                samples, sample_metadata = sample_with_kroupa_imf(
                    flow, preprocessing_stats, stellar_mass, n_samples=n_samples
                )
                
                # Save samples
                samples_path = f"{output_dir}/samples_pid{particle_pid}.h5"
                save_samples_optimized(samples, sample_metadata, samples_path)
                
                logger.info(f"✅ Samples saved to {samples_path}")
                
            except Exception as e:
                logger.warning(f"⚠️ Sample generation failed: {e}")
                samples_path = None
        
        # Save training results
        results_path = f"{output_dir}/model_pid{particle_pid}_results.json"
        results = {
            'training_losses': train_losses,
            'validation_losses': val_losses,
            'final_train_loss': float(train_losses[-1]) if train_losses else None,
            'final_val_loss': float(val_losses[-1]) if val_losses else None,
            'best_val_loss': float(min(val_losses)) if val_losses else None,
            'metadata': make_json_serializable(metadata),
            'preprocessing_stats': make_json_serializable(preprocessing_stats),
            'model_config': {
                'input_dim': 6,
                'n_layers': n_layers,
                'hidden_units': hidden_units,
                'n_cond_layers': n_cond_layers,
                'n_mass_bins': n_mass_bins,
                'embedding_dim': embedding_dim,
                'n_gmm_components': n_gmm_components
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Training completed for PID {particle_pid}")
        logger.info(f"📈 final_training_loss: {train_losses[-1] if train_losses else 'N/A'}")
        logger.info(f"📈 final_validation_loss: {val_losses[-1] if val_losses else 'N/A'}")
        
        return model_path, samples_path
        
    except Exception as e:
        logger.error(f"❌ ERROR coupling flow training PID {particle_pid}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Train conditional coupling flows for stellar particles')
    
    # Data parameters
    parser.add_argument('--halo_id', type=str, required=True, help='Halo ID (e.g., Halo268)')
    parser.add_argument('--particle_pid', type=int, required=True, help='Particle ID to train on')
    parser.add_argument('--suite', type=str, default='eden', choices=['eden', 'mwest', 'symphony', 'symphony-hr'],
                       help='Simulation suite')
    
    # Model parameters
    parser.add_argument('--n_layers', type=int, default=8, help='Number of coupling layers')
    parser.add_argument('--hidden_units', type=int, default=128, help='Hidden units in conditional networks')
    parser.add_argument('--n_cond_layers', type=int, default=3, help='Number of layers in conditional networks')
    parser.add_argument('--n_mass_bins', type=int, default=8, help='Number of mass bins for embedding')
    parser.add_argument('--embedding_dim', type=int, default=4, help='Embedding dimension for mass bins')
    parser.add_argument('--n_gmm_components', type=int, default=5, help='Number of GMM components in base distribution')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--validation_freq', type=int, default=5, help='Validation frequency')
    
    # Data preprocessing
    parser.add_argument('--clip_outliers', type=float, default=5.0, help='Clip outliers beyond this many std devs')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--generate_samples', action='store_true', help='Generate samples after training')
    parser.add_argument('--n_samples', type=int, default=100000, help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        args.output_dir = f"coupling_output_conditional/{args.suite}/{args.halo_id}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 Conditional Coupling Flow Training (RealNVP)")
    print("=" * 50)
    print(f"Halo ID: {args.halo_id}")
    print(f"Particle PID: {args.particle_pid}")
    print(f"Suite: {args.suite}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model config: {args.n_layers} layers, {args.hidden_units} units")
    print(f"Training config: {args.epochs} epochs, batch size {args.batch_size}")
    print(f"🌟 Conditioning: Mass distribution with {args.n_mass_bins} bins")
    
    # Train and save the flow
    try:
        model_path, samples_path = train_and_save_conditional_coupling_flow(
            halo_id=args.halo_id,
            particle_pid=args.particle_pid,
            suite=args.suite,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            n_layers=args.n_layers,
            hidden_units=args.hidden_units,
            n_cond_layers=args.n_cond_layers,
            n_mass_bins=args.n_mass_bins,
            embedding_dim=args.embedding_dim,
            n_gmm_components=args.n_gmm_components,
            generate_samples=args.generate_samples,
            n_samples=args.n_samples,
            validation_freq=args.validation_freq
        )
        
        print(f"✅ Training completed successfully!")
        print(f"Model saved to: {model_path}")
        if samples_path:
            print(f"Samples saved to: {samples_path}")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import traceback
    main()
