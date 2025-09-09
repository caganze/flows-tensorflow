# Continuous Normalizing Flows (CNFs) with Neural ODEs

This directory contains implementations of Continuous Normalizing Flows using Neural ODEs for astrophysical data, including both standard and conditional versions.

## Overview

Continuous Normalizing Flows (CNFs) use Neural Ordinary Differential Equations (ODEs) to model continuous-time transformations instead of discrete bijective layers like MAF. This provides several advantages:

- **Continuous transformations**: Smooth, infinitesimal transformations via ODEs
- **Memory efficiency**: Can trade compute for memory using adaptive ODE solvers
- **Theoretical elegance**: Exact likelihood computation via the instantaneous change of variables
- **Flexible architectures**: Neural networks define the dynamics of the transformation

## Key Features

- **Neural ODE Integration**: Uses TensorFlow Probability's ODE solver for continuous transformations
- **Exact Likelihood**: Computes exact log-likelihood via augmented ODE system tracking log-det-Jacobian
- **Conditional Flows**: CNFs that condition on stellar mass for mass-aware distributions
- **GPU Optimized**: Efficient GPU implementation with memory management
- **Symlib Integration**: Direct integration with symlib astrophysical simulations

## Files

### Core Implementation
1. **`cnf_flows_solution.py`** - Core CNF classes and Neural ODE implementations
2. **`train_cnf_flows.py`** - Training script for standard CNFs
3. **`train_cnf_flows_conditional.py`** - Training script for conditional CNFs
4. **`sample_cnf_flow.py`** - Sampling utilities for standard CNFs
5. **`sample_conditional_cnf_flow.py`** - Sampling utilities for conditional CNFs
6. **`CNF_README.md`** - This documentation

## Usage

### 1. Training a Standard CNF

Train a CNF on a specific particle from symlib data:

```bash
python train_cnf_flows.py \
    --halo_id Halo268 \
    --particle_pid 12345 \
    --suite eden \
    --output_dir ./cnf_output \
    --epochs 100 \
    --batch_size 512 \
    --hidden_units 64 64 \
    --activation tanh \
    --integration_time 1.0 \
    --num_integration_steps 10
```

### 2. Training a Conditional CNF

Train a conditional CNF that conditions on stellar mass:

```bash
python train_cnf_flows_conditional.py \
    --halo_id Halo268 \
    --particle_pid 12345 \
    --suite eden \
    --output_dir ./conditional_cnf_output \
    --epochs 100 \
    --batch_size 512 \
    --hidden_units 64 64 \
    --activation tanh \
    --integration_time 1.0 \
    --num_integration_steps 10
```

### 3. Sampling from Standard CNF

Generate samples from a trained CNF:

```bash
# Basic sampling
python sample_cnf_flow.py \
    --model_path ./cnf_output/cnf_model_pid12345.npz \
    --n_samples 10000 \
    --output_dir ./cnf_samples

# With Kroupa IMF masses
python sample_cnf_flow.py \
    --model_path ./cnf_output/cnf_model_pid12345.npz \
    --n_samples 10000 \
    --with_masses \
    --stellar_mass_total 1e6 \
    --analyze \
    --output_dir ./cnf_samples
```

### 4. Sampling from Conditional CNF

Generate samples with specific mass conditions:

```bash
# Fixed mass
python sample_conditional_cnf_flow.py \
    --model_path ./conditional_cnf_output/conditional_cnf_model_pid12345.npz \
    --n_samples 10000 \
    --mass_strategy fixed \
    --target_mass 1.0 \
    --analyze \
    --output_dir ./conditional_cnf_samples

# Mass range with Kroupa IMF
python sample_conditional_cnf_flow.py \
    --model_path ./conditional_cnf_output/conditional_cnf_model_pid12345.npz \
    --n_samples 10000 \
    --mass_strategy kroupa \
    --mass_range 0.08 100.0 \
    --analyze \
    --output_dir ./conditional_cnf_samples
```

## CNF Architecture

### Standard CNF

```python
class CNFNormalizingFlow:
    def __init__(self, input_dim=6, hidden_units=[64, 64], activation='tanh', 
                 integration_time=1.0, num_integration_steps=10):
        # Neural ODE defining dx/dt = f(x, t; θ)
        # Augmented system tracks log-det-Jacobian
        # Integration from t=0 to t=T transforms data to latent space
```

### Conditional CNF

```python
class ConditionalCNFNormalizingFlow:
    def __init__(self, input_dim=6, condition_dim=1, hidden_units=[64, 64], 
                 activation='tanh', integration_time=1.0, num_integration_steps=10):
        # Conditional Neural ODE: dx/dt = f(x, c, t; θ)
        # where c are the conditioning variables (e.g., mass)
```

## Neural ODE Implementation

### Forward Transform (Data → Latent)

The CNF transforms data `x` to latent space `z` by solving:
```
dz/dt = f(z, t; θ)
d(log|det J|)/dt = -∇ · f(z, t; θ)
```

Starting from `z(0) = x`, we integrate to `z(T)` where T is the integration time.

### Inverse Transform (Latent → Data)

To generate samples, we solve the reverse ODE:
```
dx/dt = -f(x, T-t; θ)
```

Starting from `x(0) ~ N(0, I)`, we integrate to get `x(T)`.

### Likelihood Computation

The log-likelihood is computed as:
```
log p(x) = log p(z) + log|det J|
```

where `z` is the forward transform result and `log|det J|` is tracked via the augmented ODE.

## Hyperparameters

### Architecture Parameters

- **`hidden_units`**: List of hidden layer sizes for Neural ODE (default: `[64, 64]`)
- **`activation`**: Activation function (recommend `tanh` for CNFs, default: `tanh`)
- **`integration_time`**: Total integration time T (default: `1.0`)
- **`num_integration_steps`**: Number of ODE solver steps (default: `10`)

### Training Parameters

- **`learning_rate`**: Adam learning rate (default: `1e-3`)
- **`batch_size`**: Training batch size (default: `512`)
- **`epochs`**: Number of training epochs (default: `100`)

### ODE Solver Parameters

- **`rtol`**: Relative tolerance for ODE solver (default: `1e-5`)
- **`atol`**: Absolute tolerance for ODE solver (default: `1e-8`)

## CNF vs MAF Comparison

| Aspect | CNF (Neural ODE) | MAF (Discrete) |
|--------|------------------|----------------|
| **Transformations** | Continuous, smooth | Discrete bijections |
| **Memory** | Trade compute for memory | Fixed memory cost |
| **Training** | More complex (ODE solving) | Simpler autoregressive |
| **Flexibility** | High (any neural network) | Constrained (autoregressive) |
| **Speed** | Slower (ODE integration) | Faster (direct forward) |
| **Theoretical** | Exact continuous dynamics | Exact discrete transformations |

## Conditional Flow Benefits

1. **Mass-Aware Distributions**: Learn `p(phase_space | mass)` instead of `p(phase_space)`
2. **Targeted Generation**: Create particles with specific mass properties
3. **Physical Realism**: Incorporate mass-dependent dynamical processes
4. **Flexible Conditioning**: Easy to extend to multiple conditioning variables

## Performance Tips

### Training Optimization

1. **Start Small**: Begin with fewer integration steps (`num_integration_steps=5`)
2. **Activation Choice**: Use `tanh` activation for better ODE stability
3. **Learning Rate**: Start with `1e-3`, reduce if training is unstable
4. **Batch Size**: Larger batches often help with CNF stability

### Memory Management

1. **Integration Steps**: Fewer steps = less memory but potentially less accurate
2. **Hidden Units**: Smaller networks use less memory
3. **Gradient Checkpointing**: Enable for very deep integrations (not implemented)

### Numerical Stability

1. **ODE Tolerances**: Adjust `rtol` and `atol` for accuracy vs speed tradeoff
2. **Integration Time**: Values around 1.0 work well; avoid very large values
3. **Preprocessing**: Proper normalization is crucial for ODE stability

## Output Formats

### Model Files
- **`cnf_model_pid{PID}.npz`** - Trained CNF model weights and config
- **`cnf_model_pid{PID}_preprocessing.npz`** - Preprocessing statistics
- **`cnf_model_pid{PID}_results.json`** - Training metrics and metadata

### Sample Files
- **`cnf_samples_{N}.npz`** - Generated samples in NPZ format
- **`cnf_samples_{N}_metadata.json`** - Sample metadata and statistics
- **`cnf_analysis_{N}.json`** - Analysis results (if `--analyze` used)

## Integration with Existing Workflow

### Symlib Compatibility
CNFs work with your existing symlib workflow:
- Same data loading (`load_particle_data`)
- Same preprocessing pipeline
- Same output formats for downstream analysis

### Comparison with MAF
You can train both MAF and CNF models on the same data:
```bash
# Train MAF
python train_tfp_flows.py --particle_pid 12345 --output_dir ./maf_output

# Train CNF  
python train_cnf_flows.py --particle_pid 12345 --output_dir ./cnf_output

# Compare results
python sample_cnf_flow.py --model_path ./cnf_output/model.npz --analyze
```

## Troubleshooting

### Common Issues

1. **ODE Integration Failures**
   - Reduce `integration_time` or increase `num_integration_steps`
   - Try smaller learning rates or batch sizes
   - Ensure proper data preprocessing

2. **Training Instability**
   - Use `tanh` activation instead of `relu`
   - Reduce learning rate
   - Check for NaN values in data

3. **Memory Issues**
   - Reduce `num_integration_steps`
   - Use smaller `hidden_units`
   - Reduce `batch_size`

4. **Slow Training**
   - CNFs are inherently slower than MAFs due to ODE solving
   - Reduce `num_integration_steps` for faster training
   - Use GPU acceleration

### Performance Expectations

- **Training Time**: 2-5x slower than equivalent MAF
- **Sampling Time**: 3-10x slower than MAF (depends on integration steps)
- **Memory Usage**: Similar to MAF, controllable via integration steps
- **Quality**: Often comparable or superior likelihood modeling

## Advanced Usage

### Custom Neural ODE Architectures

Modify the `NeuralODE` class in `cnf_flows_solution.py`:

```python
class CustomNeuralODE(NeuralODE):
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, **kwargs)
        # Add residual connections, attention, etc.
```

### Multiple Conditioning Variables

Extend conditional CNFs to condition on age, metallicity, etc.:

```python
flow = ConditionalCNFNormalizingFlow(
    input_dim=6,           # Phase space
    condition_dim=3,       # [mass, age, metallicity]
    hidden_units=[128, 128],
    integration_time=1.0
)
```

### Adaptive ODE Solving

For production use, consider adaptive solvers:
- Adjust tolerances based on data complexity
- Use different solvers (Runge-Kutta, Adams methods)
- Implement gradient checkpointing for memory efficiency

---

For questions or issues with CNF implementation, refer to the TensorFlow Probability documentation on Neural ODEs or the original CNF papers.
