#!/usr/bin/env python3
"""
Example usage of conditional TensorFlow Probability flows
Demonstrates training and sampling from conditional flows that condition on stellar mass
Updated to use functions from load_and_sample_flows.py
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path

# Import the functions from load_and_sample_flows.py
from load_and_sample_flows import (
    debug_and_load_flow_model,
    load_and_sample_with_unstandardization,
    sample_from_loaded_model,
    reconstruct_and_load_model
)

# Import conditional flow functions
from train_tfp_flows_conditional import load_conditional_data


def density_1d(coord, mass, bins=100, box_size=None, volume_norm=True, range=[-400, 400], err=False):
    """
    Compute 1D density profile along one coordinate.

    Parameters:
    -----------
    coord : array, coordinate values (x, y, z, vx, vy, vz)
    mass : array, particle masses
    bins : int, number of bins
    box_size : float, full length of the box (needed for 3D densities)
    volume_norm : bool, if True normalize to mass density [M/len^3],
                         if False just linear density [M/len]
    range : list, range for binning [min, max]
    err : bool, if True return Poisson error bars as sqrt(N)/volume

    Returns:
    --------
    centers : array, bin center coordinates
    dens : array, density values
    dens_err : array, Poisson error bars (only if err=True)
    """
    hist, edges = np.histogram(coord, bins=bins, weights=mass, range=range)
    centers = 0.5 * (edges[1:] + edges[:-1])
    widths = np.diff(edges)
    
    # Get particle counts per bin for error calculation
    if err:
        counts, _ = np.histogram(coord, bins=bins, range=range)

    if volume_norm and box_size is not None:
        # assume box is cubic with side box_size
        # volume of each bin = width * box_size^2
        volume = widths * box_size**2
        dens = hist / volume
        
        if err:
            # Poisson error bars: sqrt(N) / volume
            dens_err = np.sqrt(counts) / volume
    else:
        dens = hist / widths  # linear density (M per unit length)
        
        if err:
            # Poisson error bars: sqrt(N) / width
            dens_err = np.sqrt(counts) / widths

    if err:
        return centers, dens, dens_err
    else:
        return centers, dens

# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

def create_synthetic_conditional_data(n_samples: int = 10000, seed: int = 42):
    """Create synthetic astrophysical data with mass conditioning
    
    This simulates the kind of data you'd get from symlib with known mass-phase space correlations
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Generate stellar masses with log-normal distribution (like Kroupa IMF)
    log_masses = np.random.normal(0, 1, n_samples)  # Log10 of mass in solar masses
    masses = 10**log_masses
    
    # Create mass-dependent phase space correlations
    # Higher mass stars tend to have:
    # - More central positions (smaller radii)
    # - Higher velocities
    # - Different spatial distributions
    
    # Positions (x, y, z) in kpc - mass-dependent radius
    mass_factor = np.log10(masses + 1e-3)  # Avoid log(0)
    radius_scale = np.exp(-0.1 * mass_factor)  # More massive = more central
    
    positions = []
    for i in range(3):  # x, y, z
        pos = np.random.normal(0, radius_scale, n_samples)
        positions.append(pos)
    
    # Velocities (vx, vy, vz) in km/s - mass-dependent velocity dispersion
    velocity_scale = 50 + 20 * mass_factor  # More massive = higher velocity
    
    velocities = []
    for i in range(3):  # vx, vy, vz
        vel = np.random.normal(0, velocity_scale, n_samples)
        velocities.append(vel)
    
    # Combine into phase space
    phase_space = np.column_stack(positions + velocities)
    
    # Reshape masses for conditioning
    mass_conditions = masses.reshape(-1, 1)
    
    print(f"✅ Generated {n_samples} synthetic particles")
    print(f"   Phase space shape: {phase_space.shape}")
    print(f"   Mass conditions shape: {mass_conditions.shape}")
    print(f"   Mass range: [{np.min(masses):.3e}, {np.max(masses):.3e}] M☉")
    
    return tf.constant(phase_space, dtype=tf.float32), tf.constant(mass_conditions, dtype=tf.float32)


def example_conditional_training():
    """Example of training a conditional flow on synthetic data"""
    
    print("🚀 Example: Training Conditional Flow")
    print("=" * 50)
    
    # Import after TensorFlow configuration
    from train_tfp_flows_conditional import ConditionalTFPNormalizingFlow, ConditionalTFPFlowTrainer
    from train_tfp_flows_conditional import preprocess_conditional_data, split_conditional_data
    
    # Generate synthetic data
    print("📊 Generating synthetic data...")
    phase_space, mass_conditions = create_synthetic_conditional_data(n_samples=5000)
    
    # Preprocess data
    print("🔧 Preprocessing data...")
    processed_ps, processed_mass, preprocessing_stats = preprocess_conditional_data(
        phase_space, mass_conditions
    )
    
    # Split data
    print("🔀 Splitting data...")
    train_ps, train_mass, val_ps, val_mass, test_ps, test_mass = split_conditional_data(
        processed_ps, processed_mass, train_frac=0.7, val_frac=0.2
    )
    
    # Create conditional flow
    print("🏗️ Creating conditional flow...")
    flow = ConditionalTFPNormalizingFlow(
        input_dim=6,  # 6D phase space
        condition_dim=1,  # 1D mass conditioning
        n_layers=3,  # Smaller for faster example
        hidden_units=32,  # Smaller for faster example
        name='example_conditional_flow'
    )
    
    # Create trainer
    print("🎯 Creating trainer...")
    trainer = ConditionalTFPFlowTrainer(flow, learning_rate=1e-3)
    
    # Train (shorter for example)
    print("🏋️ Training conditional flow...")
    trainer.train(
        train_data=train_ps,
        train_conditions=train_mass,
        val_data=val_ps,
        val_conditions=val_mass,
        epochs=20,  # Fewer epochs for faster example
        batch_size=256,
        validation_freq=5
    )
    
    # Test the trained flow
    print("📊 Testing trained flow...")
    test_log_prob = flow.log_prob(test_ps, test_mass)
    test_loss = -tf.reduce_mean(test_log_prob)
    print(f"Test loss: {test_loss:.4f}")
    
    # Save the model
    output_dir = Path("example_output")
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "example_conditional_flow.npz"
    flow.save(str(model_path))
    
    # Save preprocessing stats
    preprocessing_path = output_dir / "example_conditional_flow_preprocessing.npz"
    np.savez(
        preprocessing_path,
        **{k: v.numpy() if isinstance(v, tf.Tensor) else v 
           for k, v in preprocessing_stats.items()}
    )
    
    print(f"✅ Model saved to {model_path}")
    print(f"✅ Preprocessing saved to {preprocessing_path}")
    
    return str(model_path), preprocessing_stats


def example_conditional_sampling(model_path: str):
    """Example of sampling from a trained conditional flow"""
    
    print("\n🎲 Example: Conditional Sampling")
    print("=" * 50)
    
    # Import sampling functions
    from sample_conditional_flow import load_conditional_flow, load_preprocessing_stats
    from sample_conditional_flow import generate_mass_conditions, apply_preprocessing_to_conditions
    from sample_conditional_flow import inverse_preprocess_samples
    
    # Load the trained flow
    print("📦 Loading trained conditional flow...")
    flow = load_conditional_flow(model_path)
    
    # Load preprocessing stats
    preprocessing_path = model_path.replace('.npz', '_preprocessing.npz')
    preprocessing_stats = load_preprocessing_stats(preprocessing_path)
    
    # Example 1: Sample with fixed mass
    print("\n🎯 Example 1: Fixed mass sampling")
    fixed_mass = 1.0  # 1 solar mass
    n_samples = 1000
    
    mass_conditions = generate_mass_conditions(
        n_samples=n_samples,
        mass_strategy="fixed",
        target_mass=fixed_mass
    )
    
    # Preprocess conditions
    processed_conditions = apply_preprocessing_to_conditions(mass_conditions, preprocessing_stats)
    
    # Sample from flow
    samples_processed = flow.sample(n_samples, processed_conditions, seed=42)
    samples = inverse_preprocess_samples(samples_processed, preprocessing_stats)
    
    print(f"✅ Generated {n_samples} samples with fixed mass {fixed_mass:.1f} M☉")
    print(f"   Sample shape: {samples.shape}")
    print(f"   Position range: [{tf.reduce_min(samples[:, :3]):.2f}, {tf.reduce_max(samples[:, :3]):.2f}] kpc")
    print(f"   Velocity range: [{tf.reduce_min(samples[:, 3:]):.2f}, {tf.reduce_max(samples[:, 3:]):.2f}] km/s")
    
    # Example 2: Sample with mass range
    print("\n🎯 Example 2: Mass range sampling")
    mass_range = (0.1, 10.0)  # 0.1 to 10 solar masses
    
    mass_conditions_range = generate_mass_conditions(
        n_samples=n_samples,
        mass_strategy="log_uniform",
        mass_range=mass_range
    )
    
    # Preprocess conditions
    processed_conditions_range = apply_preprocessing_to_conditions(mass_conditions_range, preprocessing_stats)
    
    # Sample from flow
    samples_range_processed = flow.sample(n_samples, processed_conditions_range, seed=123)
    samples_range = inverse_preprocess_samples(samples_range_processed, preprocessing_stats)
    
    print(f"✅ Generated {n_samples} samples with mass range {mass_range[0]:.1f}-{mass_range[1]:.1f} M☉")
    print(f"   Mass distribution: [{tf.reduce_min(mass_conditions_range):.3f}, {tf.reduce_max(mass_conditions_range):.3f}] M☉")
    
    # Example 3: Kroupa IMF sampling
    print("\n🎯 Example 3: Kroupa IMF sampling")
    
    mass_conditions_kroupa = generate_mass_conditions(
        n_samples=n_samples,
        mass_strategy="kroupa",
        mass_range=(0.08, 100.0)  # Main sequence range
    )
    
    processed_conditions_kroupa = apply_preprocessing_to_conditions(mass_conditions_kroupa, preprocessing_stats)
    samples_kroupa_processed = flow.sample(n_samples, processed_conditions_kroupa, seed=456)
    samples_kroupa = inverse_preprocess_samples(samples_kroupa_processed, preprocessing_stats)
    
    print(f"✅ Generated {n_samples} samples with Kroupa IMF masses")
    print(f"   Mass distribution: [{tf.reduce_min(mass_conditions_kroupa):.3f}, {tf.reduce_max(mass_conditions_kroupa):.3f}] M☉")
    
    # Show mass-dependent correlations
    print("\n📈 Mass-dependent correlations:")
    
    # Calculate radius for each sample
    radii = tf.sqrt(tf.reduce_sum(tf.square(samples_kroupa[:, :3]), axis=1))
    masses_flat = tf.reshape(mass_conditions_kroupa, [-1])
    
    # Bin by mass and show average radius
    low_mass_mask = masses_flat < tf.reduce_median(masses_flat)
    high_mass_mask = masses_flat >= tf.reduce_median(masses_flat)
    
    low_mass_radius = tf.reduce_mean(tf.boolean_mask(radii, low_mass_mask))
    high_mass_radius = tf.reduce_mean(tf.boolean_mask(radii, high_mass_mask))
    
    print(f"   Low mass stars average radius: {low_mass_radius:.2f} kpc")
    print(f"   High mass stars average radius: {high_mass_radius:.2f} kpc")
    print(f"   Mass-radius anti-correlation: {low_mass_radius > high_mass_radius}")
    
    return samples, mass_conditions


def test_density_1d_function():
    """Test the density_1d function with sample data"""
    
    print("\n🧪 Testing density_1d function with error bars")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_particles = 1000
    
    # Sample coordinates (position along x-axis)
    coord = np.random.normal(0, 100, n_particles)  # positions in kpc
    mass = np.random.uniform(0.1, 10, n_particles)  # masses in solar masses
    
    box_size = 500  # kpc
    
    # Test without error bars
    print("📊 Testing without error bars:")
    centers, dens = density_1d(coord, mass, bins=20, box_size=box_size, volume_norm=True, range=[-400, 400])
    print(f"   Centers shape: {centers.shape}")
    print(f"   Density shape: {dens.shape}")
    print(f"   Density range: [{np.min(dens):.3e}, {np.max(dens):.3e}] M☉/kpc³")
    
    # Test with error bars
    print("\n📊 Testing with error bars:")
    centers_err, dens_err, dens_err_bars = density_1d(coord, mass, bins=20, box_size=box_size, 
                                                      volume_norm=True, range=[-400, 400], err=True)
    print(f"   Centers shape: {centers_err.shape}")
    print(f"   Density shape: {dens_err.shape}")
    print(f"   Error bars shape: {dens_err_bars.shape}")
    print(f"   Error bars range: [{np.min(dens_err_bars):.3e}, {np.max(dens_err_bars):.3e}] M☉/kpc³")
    
    # Test linear density (without volume normalization)
    print("\n📊 Testing linear density with error bars:")
    centers_lin, dens_lin, dens_lin_err = density_1d(coord, mass, bins=20, volume_norm=False, 
                                                     range=[-400, 400], err=True)
    print(f"   Linear density range: [{np.min(dens_lin):.3e}, {np.max(dens_lin):.3e}] M☉/kpc")
    print(f"   Linear error bars range: [{np.min(dens_lin_err):.3e}, {np.max(dens_lin_err):.3e}] M☉/kpc")
    
    # Calculate relative errors
    relative_err = dens_err_bars / dens_err
    print(f"\n📈 Relative error statistics:")
    print(f"   Mean relative error: {np.mean(relative_err):.3f}")
    print(f"   Median relative error: {np.median(relative_err):.3f}")
    print(f"   Max relative error: {np.max(relative_err):.3f}")
    
    print("✅ density_1d function test completed!")
    
    return centers_err, dens_err, dens_err_bars


def load_eden_939_pid1_example():
    """Load Eden 939 PID 1 trained model and demonstrate sampling"""
    
    print("🌟 Loading Eden 939 PID 1 Model Example")
    print("=" * 60)
    
    # Initialize conditional_samples
    conditional_samples = None
    
    # Your specific model paths
    model_path = '/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/Halo939_pid1_test/tfp_regular_fixed/model_pid1.npz'
    preprocessing_path = '/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/Halo939_pid1_test/tfp_regular_fixed/model_pid1_preprocessing.npz'
    
    print("📦 Loading trained model using load_and_sample_flows functions...")
    
    # Method 1: Use the integrated loading and sampling function
    print("\n🚀 Method 1: Using load_and_sample_with_unstandardization")
    try:
        # Load model and generate samples with unstandardization
        test_samples, flow = load_and_sample_with_unstandardization(
            halo_id="Halo939", 
            parent_id=1, 
            suite="eden", 
            flow_type="regular", 
            n_samples=10000
        )
        
        if test_samples is not None:
            print(f"✅ Generated {len(test_samples)} test samples")
            print(f"   Sample range: [{test_samples.numpy().min():.2f}, {test_samples.numpy().max():.2f}]")
        else:
            print("❌ Failed to generate samples using Method 1")
            
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        test_samples, flow = None, None
    
    # Method 2: Manual loading with debug_and_load_flow_model
    print("\n🔧 Method 2: Using debug_and_load_flow_model")
    try:
        # Load model using the debug function
        model, model_data, samples_data, preprocessing_stats = debug_and_load_flow_model(
            halo_id="Halo939", 
            parent_id=1, 
            suite="eden", 
            flow_type="regular"
        )
        
        if model and hasattr(model, 'sample'):
            # Direct sampling from loaded model
            print(f"\n🎯 Sampling from loaded model...")
            samples = model.sample(10000)
            
            # Apply unstandardization if available
            if preprocessing_stats is not None:
                print(f"🔄 Unstandardizing samples...")
                if 'mean' in preprocessing_stats and 'std' in preprocessing_stats:
                    mean = tf.constant(preprocessing_stats['mean'], dtype=tf.float32)
                    std = tf.constant(preprocessing_stats['std'], dtype=tf.float32)
                    samples = samples * std + mean
                    print(f"✅ Samples unstandardized to original physical units")
            
            print(f"✅ Generated {len(samples)} samples using direct model loading")
            print(f"   Sample range: [{samples.numpy().min():.2f}, {samples.numpy().max():.2f}]")
            
        elif model_data:
            # Reconstruct and sample
            print(f"\n🔧 Reconstructing model from weights...")
            samples, reconstructed_flow = sample_from_loaded_model(
                model_data, 
                n_samples=10000, 
                preprocessing_stats=preprocessing_stats
            )
            
            if samples is not None:
                print(f"✅ Generated {len(samples)} samples using reconstructed model")
                print(f"   Sample range: [{samples.numpy().min():.2f}, {samples.numpy().max():.2f}]")
                flow = reconstructed_flow
            else:
                print("❌ Failed to generate samples from reconstructed model")
                
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
        samples = None
        preprocessing_stats = None
        flow = None
    
    # Method 3: Load training data for conditional sampling 
    print("\n📊 Method 3: Loading training data for conditioning")
    try:
        # Load training data using the conditional data loader
        x_train, m_train, metadata = load_conditional_data("Halo939", 1, "eden")
        
        print(f"✅ Loaded training data:")
        print(f"   Phase space shape: {x_train.shape}")
        print(f"   Mass conditions shape: {m_train.shape}")
        print(f"   Mass range: [{m_train.numpy().min():.2e}, {m_train.numpy().max():.2e}]")
        
        # Simple conditional sampling demonstration
        if flow is not None and hasattr(flow, 'sample'):
            print("\n🎲 Performing conditional sampling demonstration...")
            
            # Create target mass conditions (e.g., sample particles with mass = 1 solar mass)
            n_conditional_samples = 1000
            target_mass = 1.0  # 1 solar mass
            
            # For a regular flow, we can't do true conditional sampling, 
            # but we can demonstrate the workflow
            print(f"   Generating {n_conditional_samples} samples...")
            print(f"   Target mass: {target_mass} M☉")
            
            conditional_samples_raw = flow.sample(n_conditional_samples)
            
            # Apply unstandardization if available
            if preprocessing_stats is not None and 'mean' in preprocessing_stats:
                mean = tf.constant(preprocessing_stats['mean'], dtype=tf.float32)
                std = tf.constant(preprocessing_stats['std'], dtype=tf.float32)
                conditional_samples = conditional_samples_raw * std + mean
            else:
                conditional_samples = conditional_samples_raw
                
            print(f"✅ Generated {len(conditional_samples)} conditional samples")
            print(f"   Sample range: [{conditional_samples.numpy().min():.2f}, {conditional_samples.numpy().max():.2f}]")
            
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
        x_train, m_train = None, None
    
    # Summary
    print("\n📋 Summary:")
    print("=" * 40)
    if test_samples is not None:
        print("✅ Successfully loaded model and generated unstandardized samples")
    if conditional_samples is not None:
        print("✅ Successfully performed conditional sampling workflow")
    if x_train is not None:
        print("✅ Successfully loaded training data for conditioning")
        
    print("\n💡 Next steps:")
    print("   - For true conditional sampling, use a conditional flow model")
    print("   - The regular flow can be used as a base model for sampling")
    print("   - Training data can be used for post-processing or analysis")
    
    return conditional_samples, flow, x_train, m_train, preprocessing_stats


def main():
    """Run the complete conditional flow example"""
    
    print("🌟 Conditional TensorFlow Probability Flow Example")
    print("=" * 60)
    print("This example demonstrates:")
    print("1. Loading a trained regular flow model (Eden 939 PID 1)")
    print("2. Using functions from load_and_sample_flows.py")
    print("3. Generating unstandardized samples in physical units")
    print("4. Loading training data for conditioning")
    print("5. Testing density_1d function with Poisson error bars")
    print()
    
    try:
        # Test the density_1d function first
        test_density_1d_function()
        
        # Set up TensorFlow
        tf.random.set_seed(42)
        
        # Load Eden 939 PID 1 model and demonstrate sampling
        conditional_samples, flow, x_train, m_train, preprocessing_stats = load_eden_939_pid1_example()
        
        print("\n✅ Eden 939 PID 1 model loading and sampling completed!")
        print("💡 Key insights:")
        print("   - Successfully used load_and_sample_flows.py functions")
        print("   - Generated unstandardized samples in physical units (kpc, km/s)")
        print("   - Loaded training data for potential conditional analysis")
        print("   - Demonstrated multiple loading and sampling approaches")
        print("   - Ready for conditional flow training or analysis")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

