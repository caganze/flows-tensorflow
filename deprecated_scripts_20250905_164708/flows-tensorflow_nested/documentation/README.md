# TensorFlow Probability Normalizing Flows

A complete solution for training and sampling normalizing flows using TensorFlow Probability, designed to work with available CUDA/cuDNN versions on Stanford Sherlock.

## Why TensorFlow Probability?

- ✅ **Mature CUDA support** - Works with cuDNN 9.4.0 (available on Sherlock)
- ✅ **No version conflicts** - Bypasses JAX/FlowJAX compatibility issues
- ✅ **GPU acceleration** - Full GPU support for training and sampling
- ✅ **Production ready** - Stable API with extensive documentation
- ✅ **Flexible architecture** - Easy to customize flow architectures

## Quick Start

### 1. Create Fresh Environment (GPU Node)

```bash
# Complete setup: Fresh environment + TF/TFP installation + testing
./create_fresh_bosque_env.sh
```

### 2. Verify Success

Look for this output:
```
🎉 SUCCESS: Fresh bosque environment ready!
✅ TensorFlow 2.16.1 + TFP 0.23.0 + Keras 2.15.0
✅ Python 3.11 + GPU support working
```

### 3. Train a Flow

```bash
# Update paths in submit_training.sh
vim submit_training.sh

# Submit training job
sbatch submit_training.sh
```

### 4. Generate Samples

```bash
# Update paths in submit_sampling.sh
vim submit_sampling.sh

# Submit sampling job
sbatch submit_sampling.sh
```

## File Overview

### Core Implementation
- **`tfp_flows_gpu_solution.py`** - Main TFP flow classes with GPU support
- **`train_tfp_flows.py`** - Training script for astrophysical data
- **`sample_tfp_flows.py`** - Large-scale sampling with memory management

### Installation
- **`install_tfp_gpu.sh`** - GPU-compatible TensorFlow Probability installation

### Job Scripts
- **`submit_training.sh`** - SLURM script for flow training
- **`submit_sampling.sh`** - SLURM script for sample generation

## Features

### Training Features
- **Automatic GPU detection** and configuration
- **Data preprocessing** with standardization and outlier clipping
- **Train/validation/test splits** with proper evaluation
- **Real-time monitoring** with loss curves and metrics
- **Flexible architecture** - configurable layers, hidden units, activations
- **Robust training** with gradient clipping and regularization

### Sampling Features
- **Chunked sampling** for memory efficiency
- **GPU memory management** with automatic fallback to CPU
- **Kroupa IMF mass generation** for astrophysical applications
- **Inverse preprocessing** to original data scale
- **HDF5 output** with compression and metadata
- **Progress monitoring** with memory usage tracking

### Architecture Features
- **Masked Autoregressive Flows (MAF)** with alternating orderings
- **Permutation layers** for better mixing
- **Flexible hidden layer sizes** and activations
- **Batch normalization** support
- **Custom bijector chains** for complex transformations

## Usage Examples

### Basic Training

```python
from tfp_flows_gpu_solution import TFPNormalizingFlow, TFPFlowTrainer

# Create flow
flow = TFPNormalizingFlow(
    input_dim=6,           # 6D phase space
    n_layers=4,            # 4 flow layers
    hidden_units=512,      # 512 hidden units per layer
    activation='relu',     # Activation function
    name='astro_flow'      # Flow name for tf.Module
)

# Train flow
trainer = TFPFlowTrainer(flow, learning_rate=1e-3)
train_losses, val_losses = trainer.train(
    train_data=train_data,
    val_data=val_data,
    epochs=100,
    batch_size=512
)
```

### Command Line Training

```bash
python train_tfp_flows.py \
    --data_path /path/to/data.h5 \
    --output_dir /path/to/output \
    --n_layers 4 \
    --hidden_units 512 \
    --epochs 200 \
    --batch_size 1024 \
    --learning_rate 1e-3
```

### Command Line Sampling

```bash
python sample_tfp_flows.py \
    --model_dir /path/to/trained/model \
    --output_path /path/to/samples.h5 \
    --n_samples 1000000 \
    --chunk_size 100000 \
    --generate_masses \
    --inverse_transform
```

## Data Format

### Input Data (HDF5)
The training script expects HDF5 files with:
- **6D phase space data**: `pos3` (3D positions) + `vel3` (3D velocities)
- **Alternative format**: Individual arrays (`pos_x`, `pos_y`, `pos_z`, `vel_x`, `vel_y`, `vel_z`)
- **Shape**: `(n_samples, 6)` for 6D phase space

### Output Data (HDF5)
Generated samples are saved with:
- **`samples`**: Flow-generated samples `(n_samples, 6)`
- **`masses`**: Kroupa IMF masses `(n_samples,)` (if requested)
- **`preprocessing/`**: Preprocessing statistics for inverse transform
- **Metadata**: Training info, sampling parameters, statistics

## Performance

### Training Performance
- **GPU training**: ~1000-5000 samples/sec (depends on batch size, architecture)
- **Memory usage**: ~2-8GB GPU memory (depends on batch size, model size)
- **Typical training time**: 1-4 hours for 100-200 epochs

### Sampling Performance
- **GPU sampling**: ~10,000-50,000 samples/sec
- **Memory management**: Chunked processing for large sample counts
- **Scalability**: Can generate millions to billions of samples

## Configuration Options

### Model Architecture
```python
flow = TFPNormalizingFlow(
    input_dim=6,              # Dimensionality of data
    n_layers=4,               # Number of flow layers (2-8 recommended)
    hidden_units=512,         # Hidden units per layer (128-1024)
    activation='relu',        # Activation function
    use_batch_norm=True,      # Batch normalization
    dtype=tf.float32          # Precision (float32/float64)
)
```

### Training Parameters
```bash
--epochs 200                  # Training epochs
--batch_size 1024            # Batch size (larger = faster, more memory)
--learning_rate 1e-3         # Learning rate
--validation_freq 10         # Validation frequency
--clip_outliers 5.0          # Outlier clipping threshold
```

### Sampling Parameters
```bash
--n_samples 1000000          # Number of samples to generate
--chunk_size 100000          # Chunk size for memory management
--generate_masses            # Generate Kroupa IMF masses
--inverse_transform          # Apply inverse preprocessing
--force_cpu                  # Force CPU usage (for bigmem nodes)
```

## Memory Management

### GPU Memory
- **Training**: Adjust `--batch_size` based on GPU memory
- **Sampling**: Use `--chunk_size` to control memory usage
- **Automatic fallback**: Falls back to CPU if GPU memory exhausted

### System Memory
- **Large datasets**: Use data streaming and chunked processing
- **BigMem nodes**: Use `--force_cpu` for sampling jobs on bigmem partition
- **Memory monitoring**: Built-in memory usage tracking

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check CUDA/cuDNN installation
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
   ```

2. **Out of Memory**
   ```bash
   # Reduce batch size or chunk size
   --batch_size 256
   --chunk_size 50000
   ```

3. **Slow Training**
   ```bash
   # Check GPU utilization
   nvidia-smi
   # Increase batch size if GPU not fully utilized
   ```

4. **NaN Losses**
   ```bash
   # Reduce learning rate
   --learning_rate 1e-4
   # Check data preprocessing
   --clip_outliers 3.0
   ```

### Performance Tips

1. **Optimize batch size**: Largest that fits in GPU memory
2. **Use mixed precision**: Set `dtype=tf.float16` for faster training
3. **Monitor GPU usage**: Use `nvidia-smi` to check utilization
4. **Preprocess data**: Standardization is crucial for stable training
5. **Use appropriate chunk sizes**: Balance memory usage and efficiency

## Integration with Existing Workflows

### Converting from FlowJAX
```python
# FlowJAX equivalent
from flowjax.flows import masked_autoregressive_flow
from flowjax.distributions import Normal

# TFP equivalent
from tfp_flows_gpu_solution import TFPNormalizingFlow

# Similar API, GPU compatible
flow = TFPNormalizingFlow(input_dim=6, n_layers=4, name='example_flow')
samples = flow.sample(1000)
log_probs = flow.log_prob(data)
```

### Cluster Integration
- **Sherlock compatibility**: Works with available cuDNN 9.4.0
- **SLURM integration**: Ready-to-use job scripts
- **Module system**: Compatible with nvhpc/24.7 + cudnn/9.4.0
- **Scalable**: From single GPU to multi-node sampling

## Next Steps

1. **Test installation**: Run `python tfp_flows_gpu_solution.py`
2. **Prepare your data**: Convert to HDF5 format with 6D phase space
3. **Train first model**: Use `submit_training.sh` with your data
4. **Generate samples**: Use `submit_sampling.sh` with trained model
5. **Scale up**: Increase model size and sample counts as needed

This solution provides a complete, production-ready alternative to FlowJAX that works with your existing cluster infrastructure!
