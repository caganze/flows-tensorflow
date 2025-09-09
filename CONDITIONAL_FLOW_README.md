# Conditional TensorFlow Probability Flows

This directory contains an implementation of conditional normalizing flows that learn `p(xi | mass)` for stellar particles, where the flow is conditioned on the mass distribution of particles.

## Overview

The conditional flow extends the original `train_tfp_flows.py` to incorporate mass information as a conditioning variable. This allows the flow to learn mass-dependent phase space distributions, enabling more realistic stellar population synthesis.

## Key Features

- **Mass Conditioning**: Flows learn `p(phase_space | stellar_mass)` instead of just `p(phase_space)`
- **Flexible Mass Strategies**: Support for fixed mass, uniform distribution, log-uniform, and Kroupa IMF sampling
- **Mass-Aware Generation**: Generate particles with specific mass conditions
- **Symlib Integration**: Load actual particle masses from symlib simulations

## Files

1. **`train_tfp_flows_conditional.py`** - Main training script for conditional flows
2. **`sample_conditional_flow.py`** - Utility for generating samples with mass conditions  
3. **`example_conditional_flow.py`** - Complete example with synthetic data
4. **`CONDITIONAL_FLOW_README.md`** - This documentation

## Usage

### 1. Training a Conditional Flow

Train a conditional flow on a specific particle from symlib data:

```bash
python train_tfp_flows_conditional.py \
    --halo_id Halo268 \
    --particle_pid 12345 \
    --suite eden \
    --output_dir ./conditional_output \
    --epochs 100 \
    --batch_size 512 \
    --n_layers 4 \
    --hidden_units 64
```

This will create:
- `conditional_model_pid12345.npz` - The trained conditional flow
- `conditional_model_pid12345_preprocessing.npz` - Preprocessing statistics
- `conditional_model_pid12345_results.json` - Training metrics and metadata

### 2. Generating Conditional Samples

Generate samples with specific mass conditions:

```bash
# Fixed mass sampling (all particles have same mass)
python sample_conditional_flow.py \
    --model_path ./conditional_output/conditional_model_pid12345.npz \
    --n_samples 10000 \
    --mass_strategy fixed \
    --target_mass 1.0 \
    --output_dir ./samples

# Mass range sampling (uniform distribution)
python sample_conditional_flow.py \
    --model_path ./conditional_output/conditional_model_pid12345.npz \
    --n_samples 10000 \
    --mass_strategy log_uniform \
    --mass_range 0.1 100.0 \
    --output_dir ./samples

# Kroupa IMF sampling
python sample_conditional_flow.py \
    --model_path ./conditional_output/conditional_model_pid12345.npz \
    --n_samples 10000 \
    --mass_strategy kroupa \
    --mass_range 0.08 100.0 \
    --output_dir ./samples
```

### 3. Running the Example

Try the complete workflow with synthetic data:

```bash
python example_conditional_flow.py
```

This will:
- Generate synthetic mass-dependent phase space data
- Train a conditional flow
- Demonstrate different sampling strategies
- Show mass-dependent correlations

## Conditional Flow Architecture

The conditional flow extends the standard normalizing flow architecture:

```python
class ConditionalTFPNormalizingFlow:
    def __init__(self, input_dim=6, condition_dim=1, n_layers=4, hidden_units=64):
        # input_dim: Phase space dimensions (x, y, z, vx, vy, vz)
        # condition_dim: Mass conditioning (log stellar mass)
        # Architecture uses conditional autoregressive networks
```

### Key Differences from Standard Flows

1. **Conditional Autoregressive Networks**: Each MAF layer receives both phase space data and mass conditions
2. **Mass Preprocessing**: Log-transform and standardize masses for numerical stability
3. **Conditional Sampling**: Requires mass conditions to generate samples
4. **Enhanced Metadata**: Tracks mass ranges and conditioning information

## Mass Conditioning Strategies

### 1. Fixed Mass
All generated particles have the same specified mass:
```python
masses = tf.fill((n_samples, 1), target_mass)
```

### 2. Uniform Distribution
Masses uniformly distributed over a specified range:
```python
masses = tf.random.uniform((n_samples, 1), min_mass, max_mass)
```

### 3. Log-Uniform Distribution
Masses log-uniformly distributed (better for wide mass ranges):
```python
log_masses = tf.random.uniform((n_samples, 1), log_min, log_max)
masses = tf.pow(10.0, log_masses)
```

### 4. Kroupa IMF
Masses follow the Kroupa Initial Mass Function:
```python
# Simplified power law with α = -2.3
masses = min_mass * tf.pow(1 + u * (tf.pow(max_mass/min_mass, α+1) - 1), 1/(α+1))
```

## Data Format

### Input Data
- **Phase Space**: `(N, 6)` array with `[x, y, z, vx, vy, vz]`
- **Mass Conditions**: `(N, 1)` array with stellar masses in solar masses

### Output Samples
- **Phase Space Samples**: `(N, 6)` array in physical units (kpc, km/s)  
- **Mass Conditions**: `(N, 1)` array in solar masses

### Preprocessing
- **Phase Space**: Standardized (zero mean, unit variance), outliers clipped
- **Masses**: Log-transformed, then standardized for numerical stability

## Benefits of Conditional Flows

1. **Mass-Aware Distributions**: Learn how phase space depends on stellar mass
2. **Realistic Populations**: Generate stellar populations with proper mass-velocity correlations
3. **Targeted Sampling**: Create particles with specific mass properties
4. **Physical Consistency**: Respect mass-dependent dynamical processes

## Limitations and Considerations

1. **Training Data**: Requires sufficient particles across the mass range of interest
2. **Mass Information**: Needs actual or estimated particle masses from simulations
3. **Computational Cost**: Slightly more expensive than unconditional flows
4. **Conditioning Quality**: Performance depends on quality of mass-phase space correlations in training data

## Extending to Multiple Conditions

The framework can be extended to condition on multiple variables:

```python
# Example: condition on mass, age, and metallicity
ConditionalTFPNormalizingFlow(
    input_dim=6,           # Phase space
    condition_dim=3,       # [mass, age, metallicity]
    n_layers=4,
    hidden_units=64
)
```

## Integration with Symlib

The conditional flow integrates with your existing symlib workflow:

1. **Data Loading**: `load_conditional_data()` extracts phase space + mass from symlib
2. **Preprocessing**: Handles both phase space and mass conditioning
3. **Training**: Uses existing symlib particle data with mass information
4. **Sampling**: Generates realistic particles for stellar population models

## Troubleshooting

### Common Issues

1. **Missing Mass Data**: Ensure symlib data includes particle masses
2. **Numerical Instability**: Use log-transform for masses spanning many orders of magnitude
3. **Poor Conditioning**: Check that training data has sufficient mass-phase space correlations
4. **Memory Issues**: Reduce batch size or number of layers for large datasets

### Performance Tips

1. **Mass Range**: Train on the mass range you plan to sample from
2. **Batch Size**: Larger batches generally improve conditional flow training
3. **Architecture**: More layers/units help capture complex mass dependencies
4. **Preprocessing**: Proper standardization is crucial for stable training

---

For questions or issues, refer to the example script or the original TFP flows documentation.
