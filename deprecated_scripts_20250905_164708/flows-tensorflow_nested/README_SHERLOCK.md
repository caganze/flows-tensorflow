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
