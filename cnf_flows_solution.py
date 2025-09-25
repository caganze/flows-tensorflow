#!/usr/bin/env python3
"""
Continuous Normalizing Flows (CNFs) with TensorFlow and Neural ODEs
Implementation of continuous-time normalizing flows for astrophysical data
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Optional, List, Dict, Any, Callable
import json
from pathlib import Path

tfd = tfp.distributions
tfb = tfp.bijectors


class NeuralODE(tf.keras.Model):
    """Neural ODE implementation for continuous normalizing flows"""
    
    def __init__(self, input_dim: int, hidden_units: List[int] = [64, 64], 
                 activation: str = 'tanh', name: str = 'neural_ode'):
        """Initialize Neural ODE
        
        Args:
            input_dim: Dimensionality of the state space
            hidden_units: List of hidden layer sizes
            activation: Activation function
            name: Model name
        """
        super().__init__(name=name)
        
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.activation = activation
        
        # Build the neural network for the ODE function
        self.layers_list = []
        
        # Input layer
        self.layers_list.append(tf.keras.layers.Dense(
            hidden_units[0], 
            activation=activation,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            name=f'{name}_dense_0'
        ))
        
        # Hidden layers
        for i, units in enumerate(hidden_units[1:], 1):
            self.layers_list.append(tf.keras.layers.Dense(
                units, 
                activation=activation,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                name=f'{name}_dense_{i}'
            ))
        
        # Output layer (same dimension as input for continuous flow)
        self.layers_list.append(tf.keras.layers.Dense(
            input_dim, 
            activation=None,  # Linear output
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            name=f'{name}_output'
        ))
    
    def call(self, x, t=None):
        """Forward pass of the Neural ODE
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            t: Time parameter (optional, for time-dependent ODEs)
            
        Returns:
            dx/dt: Time derivative of x
        """
        # If time-dependent, concatenate time to input
        if t is not None:
            # Broadcast time to match batch size
            t_expanded = tf.broadcast_to(
                tf.reshape(t, [1, 1]), 
                [tf.shape(x)[0], 1]
            )
            x_with_time = tf.concat([x, t_expanded], axis=-1)
            
            # Need to adjust first layer for time input
            h = x_with_time
        else:
            h = x
        
        # Forward pass through network
        for layer in self.layers_list:
            h = layer(h)
        
        return h


class ConditionalNeuralODE(tf.keras.Model):
    """Conditional Neural ODE for conditional continuous normalizing flows"""
    
    def __init__(self, input_dim: int, condition_dim: int, 
                 hidden_units: List[int] = [64, 64], 
                 activation: str = 'tanh', name: str = 'conditional_neural_ode'):
        """Initialize Conditional Neural ODE
        
        Args:
            input_dim: Dimensionality of the state space
            condition_dim: Dimensionality of conditioning variables
            hidden_units: List of hidden layer sizes
            activation: Activation function
            name: Model name
        """
        super().__init__(name=name)
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_units = hidden_units
        self.activation = activation
        
        # Build the neural network for the conditional ODE function
        self.layers_list = []
        
        # Input layer (takes state + conditions)
        input_size = input_dim + condition_dim
        self.layers_list.append(tf.keras.layers.Dense(
            hidden_units[0], 
            activation=activation,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            name=f'{name}_dense_0'
        ))
        
        # Hidden layers
        for i, units in enumerate(hidden_units[1:], 1):
            self.layers_list.append(tf.keras.layers.Dense(
                units, 
                activation=activation,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                name=f'{name}_dense_{i}'
            ))
        
        # Output layer (same dimension as input for continuous flow)
        self.layers_list.append(tf.keras.layers.Dense(
            input_dim, 
            activation=None,  # Linear output
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            name=f'{name}_output'
        ))
    
    def call(self, x, conditions, t=None):
        """Forward pass of the Conditional Neural ODE
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            conditions: Conditioning variables of shape (batch_size, condition_dim)
            t: Time parameter (optional)
            
        Returns:
            dx/dt: Time derivative of x
        """
        # Concatenate state and conditions
        x_with_conditions = tf.concat([x, conditions], axis=-1)
        
        # If time-dependent, add time as well
        if t is not None:
            t_expanded = tf.broadcast_to(
                tf.reshape(t, [1, 1]), 
                [tf.shape(x)[0], 1]
            )
            h = tf.concat([x_with_conditions, t_expanded], axis=-1)
        else:
            h = x_with_conditions
        
        # Forward pass through network
        for layer in self.layers_list:
            h = layer(h)
        
        return h


class CNFNormalizingFlow:
    """Continuous Normalizing Flow using Neural ODEs"""
    
    def __init__(self, input_dim: int, hidden_units: List[int] = [64, 64], 
                 activation: str = 'tanh', integration_time: float = 1.0,
                 num_integration_steps: int = 10, name: str = 'cnf_flow'):
        """Initialize CNF
        
        Args:
            input_dim: Dimensionality of input data
            hidden_units: Hidden units in Neural ODE
            activation: Activation function
            integration_time: Total integration time T
            num_integration_steps: Number of ODE solver steps
            name: Flow name
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.activation = activation
        self.integration_time = integration_time
        self.num_integration_steps = num_integration_steps
        self.name = name
        
        # Create base distribution
        self.base_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(input_dim, dtype=tf.float32),
            scale_diag=tf.ones(input_dim, dtype=tf.float32)
        )
        
        # Create Neural ODE
        self.neural_ode = NeuralODE(
            input_dim=input_dim,
            hidden_units=hidden_units,
            activation=activation,
            name=f'{name}_ode'
        )
        
        # Time points for integration
        self.time_points = tf.linspace(0.0, integration_time, num_integration_steps + 1)
    
    @property
    def trainable_variables(self):
        """Get trainable variables from the Neural ODE"""
        return self.neural_ode.trainable_variables
    
    def ode_func(self, t, x):
        """ODE function for tfp.math.ode_solve
        
        Args:
            t: Time tensor
            x: State tensor
            
        Returns:
            dx/dt: Time derivative
        """
        return self.neural_ode(x, t)
    
    def forward_transform(self, x):
        """Transform from data space to latent space
        
        Args:
            x: Input data tensor
            
        Returns:
            z: Latent space representation
            log_det_jacobian: Log determinant of Jacobian
        """
        batch_size = tf.shape(x)[0]
        
        # Solve ODE forward in time
        # We need to augment the state with log det jacobian
        # Initial log det jacobian is zero
        initial_state = tf.concat([
            x,  # Initial data
            tf.zeros([batch_size, 1], dtype=tf.float32)  # Initial log det J
        ], axis=-1)
        
        def augmented_ode_func(t, state):
            """Augmented ODE that tracks log det jacobian"""
            x_part = state[..., :self.input_dim]
            
            # Compute dx/dt
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_part)
                dx_dt = self.neural_ode(x_part, t)
            
            # Compute trace of Jacobian (divergence)
            trace_jac = tf.zeros(tf.shape(x_part)[0], dtype=tf.float32)
            for i in range(self.input_dim):
                grad_i = tape.gradient(dx_dt[..., i], x_part)
                if grad_i is not None:
                    trace_jac += grad_i[..., i]
            
            # Clean up persistent tape
            del tape
            
            # Negative trace for log det jacobian
            dlogdet_dt = -trace_jac
            
            return tf.concat([dx_dt, tf.expand_dims(dlogdet_dt, -1)], axis=-1)
        
        # Solve augmented ODE
        solver = tfp.math.ode.BDF(rtol=1e-5, atol=1e-8)
        solution = solver.solve(
            ode_fn=augmented_ode_func,
            initial_time=self.time_points[0],
            initial_state=initial_state,
            solution_times=self.time_points
        )
        
        # Extract final state
        final_state = solution.states[-1]  # Last time step
        z = final_state[..., :self.input_dim]
        log_det_jacobian = final_state[..., self.input_dim]
        
        return z, log_det_jacobian
    
    def inverse_transform(self, z):
        """Transform from latent space to data space
        
        Args:
            z: Latent space tensor
            
        Returns:
            x: Data space representation
        """
        batch_size = tf.shape(z)[0]
        
        # Solve ODE backward in time (reverse time points)
        reverse_time_points = tf.reverse(self.time_points, [0])
        
        # Initial state is z at final time
        initial_state = z
        
        def reverse_ode_func(t, x):
            """Reverse ODE function (negative time)"""
            return -self.neural_ode(x, self.integration_time - t)
        
        # Solve reverse ODE
        solver = tfp.math.ode.BDF(rtol=1e-5, atol=1e-8)
        solution = solver.solve(
            ode_fn=reverse_ode_func,
            initial_time=reverse_time_points[0],
            initial_state=initial_state,
            solution_times=reverse_time_points
        )
        
        # Extract final state (which is at t=0)
        x = solution.states[-1]
        
        return x
    
    def log_prob(self, x):
        """Compute log probability of data"""
        z, log_det_jacobian = self.forward_transform(x)
        base_log_prob = self.base_dist.log_prob(z)
        return base_log_prob + log_det_jacobian
    
    def sample(self, n_samples: int, seed: Optional[int] = None):
        """Generate samples from the CNF"""
        if seed is not None:
            tf.random.set_seed(seed)
        
        # Sample from base distribution
        z = self.base_dist.sample(n_samples)
        
        # Transform to data space
        x = self.inverse_transform(z)
        
        return x
    
    def save(self, filepath: str):
        """Save the CNF model"""
        # Get all trainable variables
        variables = self.trainable_variables
        
        # Create a dictionary to save each variable separately
        save_dict = {}
        
        # Save model configuration
        save_dict['config'] = np.array({
            'input_dim': self.input_dim,
            'hidden_units': self.hidden_units,
            'activation': self.activation,
            'integration_time': self.integration_time,
            'num_integration_steps': self.num_integration_steps,
            'name': self.name,
            'model_type': 'cnf'
        }, dtype=object)
        
        # Save each variable individually
        for i, var in enumerate(variables):
            var_array = var.numpy()
            save_dict[f'variable_{i}'] = var_array
            save_dict[f'variable_{i}_shape'] = np.array(var_array.shape)
            save_dict[f'variable_{i}_name'] = var.name
        
        # Save number of variables for loading
        save_dict['n_variables'] = len(variables)
        
        # Use compressed format for efficiency
        np.savez_compressed(filepath, **save_dict)
        print(f"✅ CNF saved to {filepath}")


class ConditionalCNFNormalizingFlow:
    """Conditional Continuous Normalizing Flow using Neural ODEs"""
    
    def __init__(self, input_dim: int, condition_dim: int = 1, 
                 hidden_units: List[int] = [64, 64], activation: str = 'tanh',
                 integration_time: float = 1.0, num_integration_steps: int = 10,
                 name: str = 'conditional_cnf_flow'):
        """Initialize Conditional CNF
        
        Args:
            input_dim: Dimensionality of input data
            condition_dim: Dimensionality of conditioning variables
            hidden_units: Hidden units in Neural ODE
            activation: Activation function
            integration_time: Total integration time T
            num_integration_steps: Number of ODE solver steps
            name: Flow name
        """
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_units = hidden_units
        self.activation = activation
        self.integration_time = integration_time
        self.num_integration_steps = num_integration_steps
        self.name = name
        
        # Create base distribution
        self.base_dist = tfd.MultivariateNormalDiag(
            loc=tf.zeros(input_dim, dtype=tf.float32),
            scale_diag=tf.ones(input_dim, dtype=tf.float32)
        )
        
        # Create Conditional Neural ODE
        self.neural_ode = ConditionalNeuralODE(
            input_dim=input_dim,
            condition_dim=condition_dim,
            hidden_units=hidden_units,
            activation=activation,
            name=f'{name}_ode'
        )
        
        # Time points for integration
        self.time_points = tf.linspace(0.0, integration_time, num_integration_steps + 1)
    
    @property
    def trainable_variables(self):
        """Get trainable variables from the Conditional Neural ODE"""
        return self.neural_ode.trainable_variables
    
    def forward_transform(self, x, conditions):
        """Transform from data space to latent space with conditioning
        
        Args:
            x: Input data tensor
            conditions: Conditioning variables
            
        Returns:
            z: Latent space representation
            log_det_jacobian: Log determinant of Jacobian
        """
        batch_size = tf.shape(x)[0]
        
        # Augment state with log det jacobian
        initial_state = tf.concat([
            x,  # Initial data
            tf.zeros([batch_size, 1], dtype=tf.float32)  # Initial log det J
        ], axis=-1)
        
        def augmented_ode_func(t, state):
            """Augmented ODE that tracks log det jacobian"""
            x_part = state[..., :self.input_dim]
            
            # Compute dx/dt with conditioning
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_part)
                dx_dt = self.neural_ode(x_part, conditions, t)
            
            # Compute trace of Jacobian (divergence)
            trace_jac = tf.zeros(tf.shape(x_part)[0], dtype=tf.float32)
            for i in range(self.input_dim):
                grad_i = tape.gradient(dx_dt[..., i], x_part)
                if grad_i is not None:
                    trace_jac += grad_i[..., i]
            
            # Clean up persistent tape
            del tape
            
            # Negative trace for log det jacobian
            dlogdet_dt = -trace_jac
            
            return tf.concat([dx_dt, tf.expand_dims(dlogdet_dt, -1)], axis=-1)
        
        # Solve augmented ODE
        solver = tfp.math.ode.BDF(rtol=1e-5, atol=1e-8)
        solution = solver.solve(
            ode_fn=augmented_ode_func,
            initial_time=self.time_points[0],
            initial_state=initial_state,
            solution_times=self.time_points
        )
        
        # Extract final state
        final_state = solution.states[-1]
        z = final_state[..., :self.input_dim]
        log_det_jacobian = final_state[..., self.input_dim]
        
        return z, log_det_jacobian
    
    def inverse_transform(self, z, conditions):
        """Transform from latent space to data space with conditioning
        
        Args:
            z: Latent space tensor
            conditions: Conditioning variables
            
        Returns:
            x: Data space representation
        """
        # Solve ODE backward in time
        reverse_time_points = tf.reverse(self.time_points, [0])
        
        def reverse_ode_func(t, x):
            """Reverse ODE function with conditioning"""
            return -self.neural_ode(x, conditions, self.integration_time - t)
        
        # Solve reverse ODE
        solver = tfp.math.ode.BDF(rtol=1e-5, atol=1e-8)
        solution = solver.solve(
            ode_fn=reverse_ode_func,
            initial_time=reverse_time_points[0],
            initial_state=z,
            solution_times=reverse_time_points
        )
        
        # Extract final state (at t=0)
        x = solution.states[-1]
        
        return x
    
    def log_prob(self, x, conditions):
        """Compute conditional log probability of data"""
        z, log_det_jacobian = self.forward_transform(x, conditions)
        base_log_prob = self.base_dist.log_prob(z)
        return base_log_prob + log_det_jacobian
    
    def sample(self, n_samples: int, conditions, seed: Optional[int] = None):
        """Generate conditional samples from the CNF"""
        if seed is not None:
            tf.random.set_seed(seed)
        
        # Sample from base distribution
        z = self.base_dist.sample(n_samples)
        
        # Transform to data space with conditioning
        x = self.inverse_transform(z, conditions)
        
        return x
    
    def save(self, filepath: str):
        """Save the Conditional CNF model"""
        variables = self.trainable_variables
        save_dict = {}
        
        # Save model configuration
        save_dict['config'] = np.array({
            'input_dim': self.input_dim,
            'condition_dim': self.condition_dim,
            'hidden_units': self.hidden_units,
            'activation': self.activation,
            'integration_time': self.integration_time,
            'num_integration_steps': self.num_integration_steps,
            'name': self.name,
            'model_type': 'conditional_cnf'
        }, dtype=object)
        
        # Save variables
        for i, var in enumerate(variables):
            var_array = var.numpy()
            save_dict[f'variable_{i}'] = var_array
            save_dict[f'variable_{i}_shape'] = np.array(var_array.shape)
            save_dict[f'variable_{i}_name'] = var.name
        
        save_dict['n_variables'] = len(variables)
        
        np.savez_compressed(filepath, **save_dict)
        print(f"✅ Conditional CNF saved to {filepath}")


class CNFFlowTrainer:
    """Trainer for CNF normalizing flows"""
    
    def __init__(self, flow: CNFNormalizingFlow, learning_rate: float = 1e-3):
        self.flow = flow
        self.learning_rate = learning_rate
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Track losses
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(self, x):
        """Compute negative log likelihood loss with numerical stability"""
        log_probs = self.flow.log_prob(x)
        
        # Clip log probabilities to prevent extreme values
        log_probs = tf.clip_by_value(log_probs, -50.0, 50.0)
        
        return -tf.reduce_mean(log_probs)
    
    @tf.function
    def train_step(self, x):
        """Single training step with numerical stability"""
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        
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
    
    def train(self, train_data, val_data=None, epochs: int = 100, 
              batch_size: int = 512, validation_freq: int = 10, verbose: bool = True):
        """Train the CNF"""
        
        n_batches = len(train_data) // batch_size
        best_val_loss = float('inf')
        best_weights = None  # Store best model weights
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle data
            indices = tf.random.shuffle(tf.range(len(train_data)))
            train_data_shuffled = tf.gather(train_data, indices)
            
            # Training batches
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_data = train_data_shuffled[start_idx:end_idx]
                loss = self.train_step(batch_data)
                epoch_losses.append(float(loss))
            
            avg_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_loss)
            
            # Validation
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
                
                # Save best weights
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = [var.numpy().copy() for var in self.flow.trainable_variables]
                    if verbose:
                        print(f"✅ New best validation loss: {val_loss:.6f} (epoch {epoch})")
                
                if verbose:
                    print(f"Epoch {epoch:4d}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")
            elif verbose and epoch % validation_freq == 0:
                print(f"Epoch {epoch:4d}: Train Loss = {avg_loss:.6f}")
        
        # Restore best weights if we have validation data
        if best_weights is not None and val_data is not None:
            for var, best_weight in zip(self.flow.trainable_variables, best_weights):
                var.assign(best_weight)
            if verbose:
                print(f"🔄 Restored best CNF model weights (val_loss: {best_val_loss:.6f})")
        
        if verbose:
            print(f"✅ CNF training completed after {epochs} epochs")


class ConditionalCNFFlowTrainer:
    """Trainer for Conditional CNF normalizing flows"""
    
    def __init__(self, flow: ConditionalCNFNormalizingFlow, learning_rate: float = 1e-3):
        self.flow = flow
        self.learning_rate = learning_rate
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Track losses
        self.train_losses = []
        self.val_losses = []
    
    def compute_loss(self, x, conditions):
        """Compute negative log likelihood loss for conditional flow with numerical stability"""
        log_probs = self.flow.log_prob(x, conditions)
        
        # Clip log probabilities to prevent extreme values
        log_probs = tf.clip_by_value(log_probs, -50.0, 50.0)
        
        return -tf.reduce_mean(log_probs)
    
    @tf.function
    def train_step(self, x, conditions):
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
    
    def train(self, train_data, train_conditions, val_data=None, val_conditions=None,
              epochs: int = 100, batch_size: int = 512, validation_freq: int = 10,
              verbose: bool = True):
        """Train the conditional CNF"""
        
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
                val_log_probs = self.flow.log_prob(val_data, val_conditions)
                # Clip validation log probabilities too
                val_log_probs = tf.clip_by_value(val_log_probs, -50.0, 50.0)
                val_loss = float(-tf.reduce_mean(val_log_probs))
                
                # Check for validation NaN or extreme values
                if not np.isfinite(val_loss) or val_loss > 100.0:
                    if verbose:
                        print(f"❌ Validation loss is NaN or too large ({val_loss}) - stopping training")
                    break
                
                self.val_losses.append(val_loss)
                
                if verbose:
                    print(f"Epoch {epoch:4d}: Train Loss = {avg_loss:.6f}, Val Loss = {val_loss:.6f}")
            elif verbose and epoch % validation_freq == 0:
                print(f"Epoch {epoch:4d}: Train Loss = {avg_loss:.6f}")
        
        if verbose:
            print(f"✅ Conditional CNF training completed after {epochs} epochs")


def load_cnf_flow(model_path: str):
    """Load a CNF model from saved file"""
    try:
        data = np.load(model_path, allow_pickle=True)
        config = data['config'].item()
        
        if config.get('model_type') == 'conditional_cnf':
            # Load conditional CNF
            flow = ConditionalCNFNormalizingFlow(
                input_dim=config['input_dim'],
                condition_dim=config['condition_dim'],
                hidden_units=config['hidden_units'],
                activation=config['activation'],
                integration_time=config['integration_time'],
                num_integration_steps=config['num_integration_steps'],
                name=config['name']
            )
            
            # Initialize with dummy data
            dummy_input = tf.zeros((1, config['input_dim']), dtype=tf.float32)
            dummy_conditions = tf.zeros((1, config['condition_dim']), dtype=tf.float32)
            _ = flow.log_prob(dummy_input, dummy_conditions)
            
        else:
            # Load standard CNF
            flow = CNFNormalizingFlow(
                input_dim=config['input_dim'],
                hidden_units=config['hidden_units'],
                activation=config['activation'],
                integration_time=config['integration_time'],
                num_integration_steps=config['num_integration_steps'],
                name=config['name']
            )
            
            # Initialize with dummy data
            dummy_input = tf.zeros((1, config['input_dim']), dtype=tf.float32)
            _ = flow.log_prob(dummy_input)
        
        # Load variables
        n_variables = int(data['n_variables'])
        variables = flow.trainable_variables
        
        if len(variables) != n_variables:
            raise ValueError(f"Model structure mismatch: expected {n_variables} variables, got {len(variables)}")
        
        for i, var in enumerate(variables):
            loaded_value = data[f'variable_{i}']
            var.assign(loaded_value)
        
        print(f"✅ CNF loaded from {model_path}")
        return flow
        
    except Exception as e:
        print(f"❌ Error loading CNF model: {e}")
        raise

