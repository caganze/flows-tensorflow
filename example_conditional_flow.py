#!/usr/bin/env python3
"""
Example usage of conditional TensorFlow Probability flows
Demonstrates training and sampling from conditional flows that condition on stellar mass
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path

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


def main():
    """Run the complete conditional flow example"""
    
    print("🌟 Conditional TensorFlow Probability Flow Example")
    print("=" * 60)
    print("This example demonstrates:")
    print("1. Training a conditional flow on synthetic data")
    print("2. Conditioning the flow on stellar mass")
    print("3. Sampling with different mass conditions")
    print("4. Analyzing mass-dependent correlations")
    print()
    
    try:
        # Set up TensorFlow
        tf.random.set_seed(42)
        
        # Part 1: Training
        model_path, preprocessing_stats = example_conditional_training()
        
        # Part 2: Sampling
        samples, mass_conditions = example_conditional_sampling(model_path)
        
        print("\n✅ Conditional flow example completed successfully!")
        print("💡 Key insights:")
        print("   - Conditional flows can learn p(phase_space | mass)")
        print("   - Different mass conditions produce different phase space distributions")
        print("   - This enables mass-aware stellar population synthesis")
        print("   - The approach can be extended to multiple conditioning variables")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
