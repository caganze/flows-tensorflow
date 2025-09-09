#!/bin/bash
# Setup script for TensorFlow Probability on Sherlock
# Run this after transferring files to /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow

echo "🚀 Setting up TensorFlow Probability on Sherlock"
echo "==============================================="
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "Date: $(date)"
echo ""

# Make all scripts executable
echo "📋 Making scripts executable..."
chmod +x *.sh
chmod +x *.py

echo "✅ Scripts made executable"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p test_training_output
mkdir -p test_sampling_output
mkdir -p models
mkdir -p data
mkdir -p samples

echo "✅ Directories created"

# Create a quick environment test
echo "🧪 Creating environment test script..."
cat > test_environment.py << 'EOF'
#!/usr/bin/env python3
"""Quick environment test for TensorFlow Probability"""

import sys
import os

def test_basic_imports():
    """Test basic Python imports"""
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        
        import tensorflow_probability as tfp
        print(f"✅ TensorFlow Probability {tfp.__version__}")
        
        import h5py
        print(f"✅ h5py {h5py.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_gpu_availability():
    """Test GPU availability"""
    try:
        import tensorflow as tf
        
        print(f"GPU available: {tf.test.is_gpu_available()}")
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"GPU devices: {gpu_devices}")
            return True
        else:
            print("No GPU devices found")
            return False
            
    except Exception as e:
        print(f"GPU test error: {e}")
        return False

def test_tfp_flows():
    """Test basic TFP flow functionality"""
    try:
        import tensorflow as tf
        import tensorflow_probability as tfp
        
        tfd = tfp.distributions
        tfb = tfp.bijectors
        
        # Create simple flow
        base_dist = tfd.Normal(0., 1.)
        bijector = tfb.Shift(1.0)
        flow = tfd.TransformedDistribution(distribution=base_dist, bijector=bijector)
        
        # Test sampling and log_prob
        samples = flow.sample(10)
        log_probs = flow.log_prob(samples)
        
        print(f"✅ TFP flow test passed")
        print(f"   Samples shape: {samples.shape}")
        print(f"   Log probs shape: {log_probs.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ TFP flow test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Environment Test")
    print("=" * 30)
    
    print("\n📦 Testing imports...")
    imports_ok = test_basic_imports()
    
    print("\n🖥️ Testing GPU...")
    gpu_ok = test_gpu_availability()
    
    print("\n🔄 Testing TFP flows...")
    flows_ok = test_tfp_flows()
    
    print("\n📋 Summary:")
    print(f"Imports: {'✅' if imports_ok else '❌'}")
    print(f"GPU: {'✅' if gpu_ok else '❌'}")
    print(f"Flows: {'✅' if flows_ok else '❌'}")
    
    if imports_ok and flows_ok:
        print("\n🎉 Environment ready for TFP flows!")
        if gpu_ok:
            print("💫 GPU acceleration available")
        else:
            print("⚠️ Using CPU only")
    else:
        print("\n❌ Environment setup incomplete")
        sys.exit(1)
EOF

chmod +x test_environment.py
echo "✅ Created test_environment.py"

# Update the submit scripts with correct paths
echo "🔧 Updating script paths..."

# Update training script
if [ -f "submit_training.sh" ]; then
    sed -i.bak "s|/path/to/your/training/data.h5|$PWD/data/training_data.h5|g" submit_training.sh
    sed -i.bak "s|/path/to/your/output/models|$PWD/models|g" submit_training.sh
    sed -i.bak "s|/Users/christianaganze/research/flows-tensorflow/|$PWD/|g" submit_training.sh
    echo "✅ Updated submit_training.sh"
fi

# Update sampling script
if [ -f "submit_sampling.sh" ]; then
    sed -i.bak "s|/path/to/your/trained/model|$PWD/models|g" submit_sampling.sh
    sed -i.bak "s|/path/to/your/output/samples.h5|$PWD/samples/samples.h5|g" submit_sampling.sh
    sed -i.bak "s|/Users/christianaganze/research/flows-tensorflow/|$PWD/|g" submit_sampling.sh
    echo "✅ Updated submit_sampling.sh"
fi

# Create example data preparation script
echo "📊 Creating example data script..."
cat > create_example_data.py << 'EOF'
#!/usr/bin/env python3
"""Create example data for testing TFP flows"""

import numpy as np
import h5py
import argparse

def create_example_data(n_samples=50000, output_file="data/example_data.h5"):
    """Create synthetic 6D phase space data"""
    print(f"Creating {n_samples:,} synthetic particles...")
    
    np.random.seed(42)
    
    # Positions (kpc) - spherical distribution
    r = np.random.exponential(scale=8.0, size=n_samples)  # Exponential disk
    theta = np.random.uniform(0, np.pi, n_samples)
    phi = np.random.uniform(0, 2*np.pi, n_samples)
    
    pos_x = r * np.sin(theta) * np.cos(phi)
    pos_y = r * np.sin(theta) * np.sin(phi)
    pos_z = r * np.cos(theta) * 0.5  # Flattened disk
    
    # Velocities (km/s) - with realistic dispersions
    vel_dispersion = 30.0 + 40.0 * np.exp(-r/5.0)  # Higher dispersion in center
    
    vel_x = np.random.normal(0, vel_dispersion, n_samples)
    vel_y = np.random.normal(10.0, vel_dispersion, n_samples)  # Rotation
    vel_z = np.random.normal(0, vel_dispersion * 0.5, n_samples)  # Lower z-dispersion
    
    # Save to HDF5
    pos3 = np.column_stack([pos_x, pos_y, pos_z])
    vel3 = np.column_stack([vel_x, vel_y, vel_z])
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('pos3', data=pos3, compression='gzip')
        f.create_dataset('vel3', data=vel3, compression='gzip')
        
        # Add metadata
        f.attrs['n_samples'] = n_samples
        f.attrs['description'] = 'Synthetic galactic disk particles'
        f.attrs['position_units'] = 'kpc'
        f.attrs['velocity_units'] = 'km/s'
        f.attrs['created_by'] = 'create_example_data.py'
    
    print(f"✅ Created {output_file}")
    print(f"   Samples: {n_samples:,}")
    print(f"   Position range: [{pos3.min():.1f}, {pos3.max():.1f}] kpc")
    print(f"   Velocity range: [{vel3.min():.1f}, {vel3.max():.1f}] km/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--output", default="data/example_data.h5")
    args = parser.parse_args()
    
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    create_example_data(args.n_samples, args.output)
EOF

chmod +x create_example_data.py
echo "✅ Created create_example_data.py"

# Create a comprehensive README for Sherlock
echo "📚 Creating Sherlock-specific README..."
cat > README_SHERLOCK.md << 'EOF'
# TensorFlow Probability on Sherlock

Quick start guide for running TFP normalizing flows on Stanford Sherlock.

## Setup Steps

1. **Test environment**:
   ```bash
   python test_environment.py
   ```

2. **Install TensorFlow Probability** (if needed):
   ```bash
   sbatch install_tfp_gpu.sh
   ```

3. **Quick GPU test** (on GPU node):
   ```bash
   bash test_tfp_gpu_quick.sh
   ```

4. **Create example data**:
   ```bash
   python create_example_data.py --n_samples 10000
   ```

5. **Test training pipeline**:
   ```bash
   sbatch test_tfp_training.sh
   ```

6. **Test sampling pipeline**:
   ```bash
   sbatch test_tfp_sampling.sh
   ```

## Directory Structure
```
flows-tensorflow/
├── data/                  # Training data
├── models/               # Trained models  
├── samples/              # Generated samples
├── logs/                 # Job logs
└── test_*/              # Test outputs
```

## Key Files
- `train_tfp_flows.py` - Main training script
- `sample_tfp_flows.py` - Main sampling script
- `submit_training.sh` - SLURM training job
- `submit_sampling.sh` - SLURM sampling job

## Common Issues
- **GPU not detected**: Check `module load cudnn/9.4.0`
- **Import errors**: Run `install_tfp_gpu.sh`
- **Path errors**: Update paths in submit scripts
EOF

echo "✅ Created README_SHERLOCK.md"

echo ""
echo "✅ SHERLOCK SETUP COMPLETED"
echo "=========================="
echo ""
echo "📋 Next steps:"
echo "1. Test environment: python test_environment.py"
echo "2. Install TFP (if needed): sbatch install_tfp_gpu.sh"
echo "3. Quick GPU test: bash test_tfp_gpu_quick.sh"
echo "4. Create test data: python create_example_data.py"
echo "5. Run training test: sbatch test_tfp_training.sh"
echo "6. Run sampling test: sbatch test_tfp_sampling.sh"
echo ""
echo "📚 See README_SHERLOCK.md for detailed instructions"
echo ""
