#!/usr/bin/env python3
"""
Flow Model Loading and Sampling Utilities

This module provides functions to load and sample from trained normalizing flow models
including regular flows, conditional flows, and continuous flows.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import h5py
import os
from scipy.spatial.distance import cdist

tfd = tfp.distributions
tfb = tfp.bijectors


def debug_and_load_flow_model(halo_id="Halo939", parent_id=2, suite="eden", flow_type="regular"):
    """
    Debug and load flow model - compact version.
    
    Parameters:
    -----------
    halo_id : str
        Halo identifier
    parent_id : int  
        Parent ID
    suite : str
        Suite name
    flow_type : str
        Type of flow: "regular", "conditional", "continuous"
        
    Returns:
    --------
    model : loaded model (if successful)
    model_data : dict of model data
    samples_data : existing samples for reference
    """
    
    # Define base directories based on flow type
    if flow_type == "regular":
        base_dir = f"/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output"
    elif flow_type == "conditional":
        base_dir = f"/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output_conditional"
    elif flow_type == "continuous":
        base_dir = f"/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output_conditional"
    else:
        raise ValueError(f"Unknown flow_type: {flow_type}")
    
    halo_num = halo_id.replace('Halo', '')
    model_path = f"{base_dir}/trained_flows/{suite}/halo{halo_num}/model_pid{parent_id}.npz"
    samples_path = f"{base_dir}/samples/{suite}/halo{halo_num}/model_pid{parent_id}_samples.npz"
    model_dir = os.path.dirname(model_path)
    
    print(f"🔍 Loading {halo_id} PID {parent_id} ({suite}) - {flow_type} flow")
    
    # 1. Check directory contents
    try:
        files = [f for f in os.listdir(model_dir) if f.startswith(f"model_pid{parent_id}")]
        print(f"📁 Found files: {files[:5]}{'...' if len(files) > 5 else ''}")
    except:
        print("❌ Model directory not found")
        return None, None, None
    
    # 2. Try loading different formats
    model, model_data, samples_data = None, None, None
    
    # Try .npz model file
    if os.path.exists(model_path):
        try:
            model_data = np.load(model_path, allow_pickle=True)
            print(f"✅ Model .npz: {list(model_data.keys())[:5]}{'...' if len(model_data.keys()) > 5 else ''}")
            
            # Show key parameters
            if 'config' in model_data:
                config = model_data['config'].item()
                print(f"  Config: {config}")
            
            print(f"  Variables: {model_data.get('n_variables', 'unknown').item() if hasattr(model_data.get('n_variables', 0), 'item') else 'unknown'}")
            
        except Exception as e:
            print(f"❌ Model .npz failed: {e}")
    
    # Try alternative model formats
    alt_paths = [
        (f"{model_dir}/model_pid{parent_id}.h5", lambda p: tf.keras.models.load_model(p)),
        (f"{model_dir}/model_pid{parent_id}", lambda p: tf.saved_model.load(p)),
    ]
    
    for path, loader in alt_paths:
        if os.path.exists(path):
            try:
                model = loader(path)
                print(f"✅ Loaded model from {os.path.basename(path)}")
                print(f"  Type: {type(model)}")
                print(f"  Has sample(): {hasattr(model, 'sample')}")
                break
            except Exception as e:
                print(f"❌ Failed {os.path.basename(path)}: {e}")
    
    # Try samples file
    if os.path.exists(samples_path):
        try:
            samples_data = np.load(samples_path, allow_pickle=True)
            print(f"✅ Samples: {list(samples_data.keys())}")
            for k, v in samples_data.items():
                print(f"  {k}: {v.shape}, range=[{v.min():.2f}, {v.max():.2f}]")
        except Exception as e:
            print(f"❌ Samples failed: {e}")
    
    # 3. Test and recommend
    if model and hasattr(model, 'sample'):
        try:
            test = model.sample(2)
            print(f"✅ Sampling works! Shape: {test.shape}")
        except Exception as e:
            print(f"⚠️ Sampling failed: {e}")
    elif model_data:
        print("💡 Use reconstruct_and_load_model() to rebuild architecture")
    elif samples_data:
        print("💡 Use existing samples_data directly")
    else:
        print("❌ Nothing found")
    
    # Try to load preprocessing stats
    preprocessing_stats = None
    preprocessing_path = f"{model_dir}/model_pid{parent_id}_preprocessing.npz"
    if os.path.exists(preprocessing_path):
        try:
            preprocessing_stats = dict(np.load(preprocessing_path, allow_pickle=True))
            print(f"✅ Preprocessing stats: {list(preprocessing_stats.keys())}")
            if 'mean' in preprocessing_stats and 'std' in preprocessing_stats:
                mean_val = preprocessing_stats['mean']
                std_val = preprocessing_stats['std']
                print(f"  Mean: {mean_val}")
                print(f"  Std: {std_val}")
        except Exception as e:
            print(f"❌ Failed to load preprocessing stats: {e}")
    else:
        print(f"⚠️ No preprocessing stats found at {preprocessing_path}")
    
    return model, model_data, samples_data, preprocessing_stats


def reconstruct_and_load_model(model_data):
    """
    Reconstruct the flow model from saved weights and config.
    
    Parameters:
    -----------
    model_data : dict
        Model data loaded from .npz file
        
    Returns:
    --------
    flow : TensorFlow Probability flow model
    """
    
    # Extract config
    config = model_data['config'].item()
    print(f"Model config: {config}")
    
    # Extract parameters from config
    input_dim = config['input_dim']
    num_layers = config['n_layers'] 
    hidden_units = config['hidden_units']  # This is an integer like 512
    
    # Convert to list format expected by TFP
    hidden_units_list = [hidden_units]  # Make it [512] instead of 512
    
    # Create bijectors
    bijectors = []
    
    for layer in range(num_layers):
        # Create autoregressive network with list format
        made = tfb.AutoregressiveNetwork(
            params=2,  # shift and log_scale
            hidden_units=hidden_units_list,  # Use list format
            activation='relu'
        )
        
        # Create MAF bijector
        maf = tfb.MaskedAutoregressiveFlow(
            shift_and_log_scale_fn=made
        )
        
        bijectors.append(maf)
        
        # Add permutation (except for last layer)
        if layer < num_layers - 1:
            bijectors.append(tfb.Permute(permutation=list(range(input_dim-1, -1, -1))))
    
    # Create the flow
    flow = tfd.TransformedDistribution(
        distribution=tfd.MultivariateNormalDiag(loc=tf.zeros(input_dim)),
        bijector=tfb.Chain(bijectors)
    )
    
    # Build the model to initialize variables
    _ = flow.sample(1)
    
    # Now set the weights with better error handling
    variables = flow.trainable_variables
    print(f"Flow has {len(variables)} trainable variables")
    print(f"Saved weights has {model_data['n_variables'].item()} variables")
    
    # Debug: Print variable shapes first
    print("\n🔍 Variable shape comparison:")
    for i, var in enumerate(variables):
        if i < model_data['n_variables'].item():
            saved_shape = tuple(model_data[f'variable_{i}_shape'])
            print(f"  Variable {i}: Expected {var.shape} vs Saved {saved_shape}")
    
    # Map saved weights to model variables with smarter matching
    loaded_count = 0
    unmatched_vars = []
    
    for i, var in enumerate(variables):
        matched = False
        
        # Try exact index match first
        if i < model_data['n_variables'].item():
            saved_weight = model_data[f'variable_{i}']
            expected_shape = tuple(model_data[f'variable_{i}_shape'])
            
            if var.shape == expected_shape:
                var.assign(saved_weight)
                print(f"✅ Loaded variable_{i}: {var.shape}")
                loaded_count += 1
                matched = True
            else:
                # Try to find a matching shape in other saved variables
                for j in range(model_data['n_variables'].item()):
                    alt_shape = tuple(model_data[f'variable_{j}_shape'])
                    if var.shape == alt_shape:
                        alt_weight = model_data[f'variable_{j}']
                        var.assign(alt_weight)
                        print(f"✅ Loaded variable_{i} from saved variable_{j}: {var.shape}")
                        loaded_count += 1
                        matched = True
                        break
        
        if not matched:
            unmatched_vars.append((i, var.shape))
            print(f"❌ Could not match variable_{i}: {var.shape}")
    
    print(f"\n📊 Loading summary:")
    print(f"   ✅ Loaded: {loaded_count}/{len(variables)} variables")
    print(f"   ❌ Unmatched: {len(unmatched_vars)} variables")
    
    if len(unmatched_vars) > len(variables) // 2:
        print("⚠️  Warning: More than half the variables couldn't be loaded!")
        print("   The model may not work correctly.")
    
    return flow


def sample_from_loaded_model(model_data, n_samples=10000, preprocessing_stats=None):
    """
    Complete workflow: reconstruct model and sample.
    
    Parameters:
    -----------
    model_data : dict
        Model data from .npz file
    n_samples : int
        Number of samples to generate
    preprocessing_stats : dict, optional
        Preprocessing statistics for unstandardization (mean, std)
        
    Returns:
    --------
    samples : tf.Tensor
        Generated samples (unstandardized to original physical units)
    flow : TensorFlow Probability flow model
    """
    
    # Validate sample count
    max_samples = 1_000_000  # 1 million max
    if n_samples > max_samples:
        print(f"⚠️  Warning: {n_samples:,} samples requested, limiting to {max_samples:,}")
        n_samples = max_samples
    elif n_samples <= 0:
        print(f"⚠️  Warning: Invalid sample count {n_samples}, using 10,000")
        n_samples = 10000
    
    print(f"🎯 Will generate {n_samples:,} samples")
    
    # Reconstruct model
    flow = reconstruct_and_load_model(model_data)
    
    # Test with small sample first
    print(f"\n🧪 Testing with 5 samples...")
    try:
        test_samples = flow.sample(5)
        print(f"✅ Test successful: {test_samples.shape}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("   Model may not be correctly loaded.")
        return None, flow
    
    # Generate full samples with iterative validation to ensure target count
    print(f"\n🎯 Generating {n_samples:,} samples...")
    try:
        max_attempts = 5
        samples_list = []
        total_valid_samples = 0
        remaining_samples = n_samples
        
        for attempt in range(max_attempts):
            if remaining_samples <= 0:
                break
                
            # Generate samples (with some buffer for invalid samples)
            buffer_factor = 1.2 if attempt == 0 else 1.5  # Increase buffer on retries
            batch_size = int(remaining_samples * buffer_factor)
            
            print(f"   Attempt {attempt + 1}: Generating {batch_size:,} samples (need {remaining_samples:,} more)")
            batch_samples = flow.sample(batch_size)
            
            # Check for valid samples in this batch
            finite_mask = tf.reduce_all(tf.math.is_finite(batch_samples), axis=1)
            valid_batch = tf.boolean_mask(batch_samples, finite_mask)
            n_valid_batch = valid_batch.shape[0]
            
            if n_valid_batch > 0:
                # Take only what we need (don't exceed target)
                n_to_take = min(n_valid_batch, remaining_samples)
                samples_list.append(valid_batch[:n_to_take])
                total_valid_samples += n_to_take
                remaining_samples -= n_to_take
                
                print(f"   ✅ Batch {attempt + 1}: {n_valid_batch:,} valid samples, took {n_to_take:,}")
            else:
                print(f"   ❌ Batch {attempt + 1}: No valid samples generated")
                
            # Check if we have enough samples
            if total_valid_samples >= n_samples:
                break
        
        # Combine all valid samples
        if samples_list:
            samples = tf.concat(samples_list, axis=0)
            print(f"✅ Final result: {samples.shape[0]:,}/{n_samples:,} samples ({100.0 * samples.shape[0] / n_samples:.1f}%)")
            
            # Warn if we couldn't get the full count
            if samples.shape[0] < n_samples:
                shortage = n_samples - samples.shape[0]
                print(f"⚠️  Warning: {shortage:,} samples short of target ({100.0 * shortage / n_samples:.1f}% shortfall)")
                print(f"   This indicates the flow model has numerical stability issues")
        else:
            print("❌ Failed to generate any valid samples")
            return None, flow
        
        # Unstandardize samples if preprocessing stats are provided
        if preprocessing_stats is not None and samples.shape[0] > 0:
            print(f"\n🔄 Unstandardizing samples to original physical units...")
            
            # Extract mean and std from preprocessing stats
            if 'mean' in preprocessing_stats and 'std' in preprocessing_stats:
                mean = tf.constant(preprocessing_stats['mean'], dtype=tf.float32)
                std = tf.constant(preprocessing_stats['std'], dtype=tf.float32)
                
                # Unstandardize: x_original = x_standardized * std + mean
                samples = samples * std + mean
                
                print(f"✅ Samples unstandardized using mean={mean.numpy()} and std={std.numpy()}")
            else:
                print(f"⚠️ Preprocessing stats missing 'mean' or 'std' keys")
                print(f"   Available keys: {list(preprocessing_stats.keys())}")
        
        if samples.shape[0] > 0:
            pos = samples[:, :3].numpy()
            vel = samples[:, 3:].numpy()
            
            if preprocessing_stats is not None:
                print(f"🌟 UNSTANDARDIZED samples:")
                print(f"   Position range: [{pos.min():.3f}, {pos.max():.3f}] kpc")
                print(f"   Velocity range: [{vel.min():.3f}, {vel.max():.3f}] km/s")
                print(f"   Position std: {pos.std():.3f} kpc")
                print(f"   Velocity std: {vel.std():.3f} km/s")
            else:
                print(f"⚠️ STANDARDIZED samples (no preprocessing stats provided):")
                print(f"   Position range: [{pos.min():.3f}, {pos.max():.3f}] (normalized)")
                print(f"   Velocity range: [{vel.min():.3f}, {vel.max():.3f}] (normalized)")
                print(f"   Position std: {pos.std():.3f}")
                print(f"   Velocity std: {vel.std():.3f}")
        
        return samples, flow
        
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        print("   Try with fewer samples or check model loading.")
        return None, flow


def load_regular_flow(halo_id="Halo939", parent_id=2, suite="eden"):
    """
    Load regular flow model (non-conditional).
    
    Parameters:
    -----------
    halo_id : str
        Halo identifier
    parent_id : int
        Parent ID
    suite : str
        Suite name
        
    Returns:
    --------
    model, model_data, samples_data, preprocessing_stats : tuple
        Loaded model components including preprocessing statistics
    """
    return debug_and_load_flow_model(halo_id, parent_id, suite, flow_type="regular")


def load_conditional_flow(halo_id="Halo939", parent_id=2, suite="eden"):
    """
    Load conditional flow model.
    
    Parameters:
    -----------
    halo_id : str
        Halo identifier
    parent_id : int
        Parent ID
    suite : str
        Suite name
        
    Returns:
    --------
    model, model_data, samples_data : tuple
        Loaded model components
    """
    return debug_and_load_flow_model(halo_id, parent_id, suite, flow_type="conditional")


def load_continuous_flow(halo_id="Halo939", parent_id=2, suite="eden"):
    """
    Load continuous flow model.
    
    Parameters:
    -----------
    halo_id : str
        Halo identifier
    parent_id : int
        Parent ID
    suite : str
        Suite name
        
    Returns:
    --------
    model, model_data, samples_data : tuple
        Loaded model components
    """
    return debug_and_load_flow_model(halo_id, parent_id, suite, flow_type="continuous")


def load_and_sample_with_unstandardization(halo_id="Halo939", parent_id=2, suite="eden", 
                                          flow_type="regular", n_samples=10000):
    """
    Complete workflow: load model, preprocessing stats, and generate unstandardized samples.
    
    Parameters:
    -----------
    halo_id : str
        Halo identifier
    parent_id : int
        Parent ID
    suite : str
        Suite name
    flow_type : str
        Type of flow: "regular", "conditional", "continuous"
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    samples : tf.Tensor
        Generated samples in original physical units (kpc, km/s)
    flow : TensorFlow Probability flow model
    """
    
    print(f"🚀 COMPLETE SAMPLING WORKFLOW - {flow_type.upper()} FLOW")
    print("=" * 70)
    
    # Load model and preprocessing stats
    model, model_data, samples_data, preprocessing_stats = debug_and_load_flow_model(
        halo_id, parent_id, suite, flow_type
    )
    
    if model and hasattr(model, 'sample'):
        # Direct sampling from loaded model
        print(f"\n🎯 Sampling from loaded model...")
        samples = model.sample(n_samples)
        
        # Apply unstandardization if available
        if preprocessing_stats is not None:
            print(f"\n🔄 Unstandardizing samples...")
            if 'mean' in preprocessing_stats and 'std' in preprocessing_stats:
                mean = tf.constant(preprocessing_stats['mean'], dtype=tf.float32)
                std = tf.constant(preprocessing_stats['std'], dtype=tf.float32)
                samples = samples * std + mean
                print(f"✅ Samples unstandardized to original physical units")
        
        return samples, model
        
    elif model_data:
        # Reconstruct and sample with unstandardization
        print(f"\n🔧 Reconstructing model from weights...")
        samples, flow = sample_from_loaded_model(model_data, n_samples, preprocessing_stats)
        return samples, flow
        
    else:
        print("❌ Could not load model for sampling")
        return None, None


def conditional_sampler(flow, m_target, m_train, x_train, preprocessing_stats=None, 
                       use_binning=True, n_bins=50, n_samples=1000, bandwidth_factor=0.1):
    """
    Sample from p(x|m) using a regular flow p(x) and mass binning.
    
    Args:
        flow: Trained regular flow model
        m_target: Target mass for conditioning (scalar)
        m_train: Training masses array (N,) or (N,1)
        x_train: Training phase space data (N,6) 
        preprocessing_stats: Dict with 'mean', 'std' for unstandardization
        use_binning: Whether to use mass binning
        n_bins: Number of mass bins
        n_samples: Number of samples to generate
        bandwidth_factor: Bandwidth for mass weighting
        
    Returns:
        conditional_samples: Generated samples conditioned on m_target
    """
    
    # Flatten masses
    masses_flat = m_train.numpy().flatten() if hasattr(m_train, 'numpy') else m_train.flatten()
    x_data = x_train.numpy() if hasattr(x_train, 'numpy') else x_train
    
    if use_binning:
        # Create mass bins
        mass_min, mass_max = masses_flat.min(), masses_flat.max()
        bins = np.linspace(mass_min, mass_max, n_bins + 1)
        
        # Find which bin contains target mass
        bin_idx = np.digitize(m_target, bins) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        
        # Select particles in target mass bin
        bin_mask = (masses_flat >= bins[bin_idx]) & (masses_flat < bins[bin_idx + 1])
        
        if not np.any(bin_mask):
            # Fallback to nearest mass particles
            mass_diff = np.abs(masses_flat - m_target)
            nearest_indices = np.argsort(mass_diff)[:min(100, len(masses_flat))]
            bin_mask = np.zeros(len(masses_flat), dtype=bool)
            bin_mask[nearest_indices] = True
            
        selected_x = x_data[bin_mask]
        selected_masses = masses_flat[bin_mask]
        
    else:
        # Use distance-based weighting instead of binning
        mass_distances = np.abs(masses_flat - m_target)
        bandwidth = bandwidth_factor * np.std(masses_flat)
        weights = np.exp(-0.5 * (mass_distances / bandwidth) ** 2)
        weights /= weights.sum()
        
        # Sample indices based on weights
        n_reference = min(1000, len(masses_flat))
        selected_indices = np.random.choice(len(masses_flat), size=n_reference, 
                                          replace=True, p=weights)
        selected_x = x_data[selected_indices]
        selected_masses = masses_flat[selected_indices]
    
    # Generate samples from flow (these are in standardized space)
    flow_samples_standardized = flow.sample(n_samples)
    flow_samples_std_np = flow_samples_standardized.numpy()
    
    # Standardize the selected training data to match flow space
    if preprocessing_stats and 'mean' in preprocessing_stats:
        mean = preprocessing_stats['mean']
        std = preprocessing_stats['std']
        
        # Standardize selected training data
        selected_x_standardized = (selected_x - mean) / std
    else:
        selected_x_standardized = selected_x
        mean = np.zeros(6)
        std = np.ones(6)
    
    # Compute covariance of selected training data (in standardized space)
    if len(selected_x_standardized) > 6:  # Need enough samples for covariance
        train_cov = np.cov(selected_x_standardized.T)
        
        # Apply covariance-preserving transformation (in standardized space)
        # Center the flow samples
        flow_mean = np.mean(flow_samples_std_np, axis=0)
        flow_centered = flow_samples_std_np - flow_mean
        
        # Compute flow covariance
        flow_cov = np.cov(flow_centered.T)
        
        try:
            # Transform to match target covariance
            L_train = np.linalg.cholesky(train_cov + 1e-6 * np.eye(6))
            L_flow = np.linalg.cholesky(flow_cov + 1e-6 * np.eye(6))
            L_flow_inv = np.linalg.inv(L_flow)
            
            # Transform samples
            transform_matrix = L_train @ L_flow_inv
            transformed_samples = (transform_matrix @ flow_centered.T).T
            
            # Recenter to target distribution (in standardized space)
            target_mean_std = np.mean(selected_x_standardized, axis=0)
            conditional_samples_std = transformed_samples + target_mean_std
            
        except np.linalg.LinAlgError:
            # Fallback: just shift to target mean (in standardized space)
            target_mean_std = np.mean(selected_x_standardized, axis=0)
            conditional_samples_std = flow_samples_std_np - flow_mean + target_mean_std
    else:
        # Not enough data for covariance - just use flow samples
        conditional_samples_std = flow_samples_std_np
    
    # Now unstandardize to get final samples in physical units
    conditional_samples = conditional_samples_std * std + mean
    
    print(f"✅ Generated {len(conditional_samples)} conditional samples for mass {m_target:.3f}")
    print(f"   Used {len(selected_x)} reference particles")
    print(f"   Mass range in reference: [{selected_masses.min():.3f}, {selected_masses.max():.3f}]")
    
    return tf.constant(conditional_samples, dtype=tf.float32)


def save_samples_to_file(samples, output_path, halo_id="Halo939", parent_id=2, method="manual_sampling"):
    """
    Save generated samples to HDF5 file.
    
    Parameters:
    -----------
    samples : tf.Tensor or np.array
        Generated samples (n_samples, 6)
    output_path : str
        Output file path
    halo_id : str
        Halo identifier
    parent_id : int
        Parent ID
    method : str
        Sampling method description
    """
    print(f"Saving samples to: {output_path}")
    
    # Convert to numpy if needed
    if hasattr(samples, 'numpy'):
        samples = samples.numpy()
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('positions', data=samples[:, :3])
        f.create_dataset('velocities', data=samples[:, 3:])
        f.attrs['halo_id'] = halo_id
        f.attrs['parent_id'] = parent_id
        f.attrs['n_samples'] = len(samples)
        f.attrs['method'] = method
    
    print(f"✅ Saved {len(samples)} samples to {output_path}")


def quick_sample_workflow(halo_id="Halo939", parent_id=2, suite="eden", flow_type="regular", n_samples=10000):
    """
    Complete workflow: load any flow type and generate samples.
    
    Parameters:
    -----------
    halo_id : str
        Halo identifier
    parent_id : int
        Parent ID
    suite : str
        Suite name
    flow_type : str
        Type of flow: "regular", "conditional", "continuous"
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    samples : tf.Tensor
        Generated samples
    flow : TensorFlow Probability flow model
    """
    
    print(f"🚀 QUICK SAMPLE WORKFLOW - {flow_type.upper()} FLOW")
    print("=" * 60)
    
    # Load model
    model, model_data, samples_data = debug_and_load_flow_model(halo_id, parent_id, suite, flow_type)
    
    if model and hasattr(model, 'sample'):
        # Direct sampling from loaded model
        print(f"\n🎯 Sampling from loaded model...")
        samples = model.sample(n_samples)
        return samples, model
        
    elif model_data:
        # Reconstruct and sample
        print(f"\n🔧 Reconstructing model from weights...")
        samples, flow = sample_from_loaded_model(model_data, n_samples)
        return samples, flow
        
    else:
        print("❌ Could not load model for sampling")
        return None, None


# Example usage functions
def example_usage():
    """Show example usage of the module functions."""
    
    print("📋 EXAMPLE USAGE:")
    print("=" * 40)
    
    print("\n1. Load regular flow:")
    print("model, model_data, samples = load_regular_flow('Halo939', 2, 'eden')")
    
    print("\n2. Load conditional flow:")
    print("model, model_data, samples = load_conditional_flow('Halo939', 2, 'eden')")
    
    print("\n3. Quick sampling workflow:")
    print("samples, flow = quick_sample_workflow('Halo939', 2, 'eden', 'regular', 5000)")
    
    print("\n4. Manual reconstruction and sampling:")
    print("flow = reconstruct_and_load_model(model_data)")
    print("samples = flow.sample(10000)")
    
    print("\n5. Save samples:")
    print("save_samples_to_file(samples, 'my_samples.h5', 'Halo939', 2)")


if __name__ == "__main__":
    example_usage()
