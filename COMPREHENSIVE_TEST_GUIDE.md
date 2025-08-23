# Comprehensive GPU Test Guide

## 🎯 Purpose
This comprehensive test validates the complete TensorFlow Probability flows pipeline with all new features:
- ✅ **Kroupa IMF**: Realistic stellar mass-based sampling
- ✅ **Optimized I/O**: Smart format selection (NPZ vs HDF5)
- ✅ **GPU Training**: Full TensorFlow GPU support
- ✅ **No JAX Dependencies**: Pure TensorFlow implementation

## 🚀 Quick Start on Sherlock

### Option 1: Submit SLURM Job (Recommended)
```bash
# Transfer files to Sherlock
rsync -av . sherlock:/path/to/flows-tensorflow/

# Submit comprehensive test job
sbatch run_comprehensive_gpu_test.sh
```

### Option 2: Interactive GPU Node
```bash
# Request interactive GPU node
srun --pty --partition=gpu --gres=gpu:1 --time=2:00:00 bash

# Load modules and environment
module purge
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0
conda activate bosque

# Run comprehensive test
python comprehensive_gpu_test.py --h5_file /path/to/your/data.h5 --particle_pids 1 2 200
```

## 📋 Test Components

### 1. Environment Checks
- ✅ TensorFlow GPU availability
- ✅ Required package versions
- ✅ No JAX conflicts

### 2. Component Tests
- ✅ Kroupa IMF sampling
- ✅ Optimized I/O methods
- ✅ TensorFlow serialization

### 3. Single Particle Training
- ✅ Complete training pipeline
- ✅ Kroupa mass generation
- ✅ Optimized sample saving
- ✅ Model weight saving

### 4. Multiple Particle Training
- ✅ Parallel-ready workflow
- ✅ Error handling
- ✅ Performance metrics

## 🔧 Configuration Options

### Environment Variables
```bash
export H5_FILE="/path/to/your/data.h5"           # HDF5 data file
export PARTICLE_PIDS="1 2 200"                   # PIDs to test
export N_SAMPLES="50000"                         # Samples per particle
export EPOCHS="5"                                # Training epochs
```

### Command Line Arguments
```bash
python comprehensive_gpu_test.py \
    --h5_file /path/to/data.h5 \
    --particle_pids 1 2 200 \
    --n_samples 50000 \
    --epochs 5 \
    --no_cleanup  # Keep test outputs
```

## 📊 Expected Output

### Success Example
```
🚀 COMPREHENSIVE GPU TEST STARTING
======================================

1️⃣ ENVIRONMENT CHECKS
✅ Required packages:
  tensorflow: 2.13.0
  tensorflow_probability: 0.21.0
  numpy: 1.24.3
  h5py: 3.9.0
✅ Physical GPU devices: 1
✅ GPU computation test passed

2️⃣ COMPONENT TESTS
✅ Kroupa IMF test passed
✅ Optimized I/O test passed

3️⃣ SINGLE PARTICLE TEST
✅ Training completed
✅ Loaded 50,000 samples from test_flow_samples.npz
✅ Kroupa masses included: 50,000 stars

4️⃣ MULTIPLE PARTICLE TEST
✅ PID 1: Success (45.2s)
✅ PID 2: Success (38.7s)
✅ PID 200: Success (52.1s)

📋 COMPREHENSIVE TEST RESULTS
======================================
  dependencies        : ✅ PASS
  gpu                  : ✅ PASS
  kroupa_imf           : ✅ PASS
  optimized_io         : ✅ PASS
  single_particle      : ✅ PASS
  multiple_particles   : ✅ PASS

🎯 OVERALL RESULT: ✅ ALL TESTS PASSED

🚀 Pipeline is ready for full deployment!
```

## 📁 Generated Test Files

After successful test, you'll find:
```
test_output/comprehensive_test_pid*/
├── model_weights.h5              # Trained model
├── preprocessing_params.npz      # Data preprocessing
├── training_results.json         # Training metrics
└── samples/
    ├── *_samples.npz            # Small datasets
    ├── *_samples.h5             # Large datasets (>1M samples)
    └── *_metadata.json          # Human-readable metadata
```

## ⚠️ Troubleshooting

### Common Issues

**GPU Not Detected**
```bash
# Check CUDA modules
module list
nvidia-smi

# Verify TensorFlow sees GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**JAX Conflicts**
```bash
# Check for JAX
python -c "import sys; print('jax' in sys.modules)"

# Remove JAX if present
conda remove jax jaxlib
```

**Memory Issues**
```bash
# Reduce test size
export N_SAMPLES="10000"
export EPOCHS="2"
```

**File Not Found**
```bash
# Check H5 file exists
ls -la /path/to/your/data.h5

# Use absolute path
export H5_FILE="/scratch/groups/aganze/christianaganze/symphony_mocks/all_in_one.h5"
```

## 🚀 Next Steps After Success

1. **Full Deployment**: Use the parallel scripts for all particles
2. **Production Runs**: Scale up sample counts and epochs
3. **Monitoring**: Set up job arrays with proper logging

### Deploy All Particles
```bash
# Generate all submission scripts
python generate_all_submission_scripts.py

# Submit job array
sbatch submit_flows_array.sh
```

## 📝 Test Results Log

The test automatically saves detailed logs to:
- `comprehensive_test_[job_id].out` - Standard output
- `comprehensive_test_[job_id].err` - Error output
- `test_output/` - Generated models and samples

Review these files to understand any issues or verify success.
