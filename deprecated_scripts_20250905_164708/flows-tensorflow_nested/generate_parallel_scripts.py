#!/usr/bin/env python3
"""
Generate parallel SLURM scripts for each PID in the halo data
This creates individual training jobs that can run simultaneously
"""

import argparse
import os
import sys

def create_training_script(halo_id, particle_pid, base_dir="/oak/stanford/orgs/kipac/users/caganze", test_mode=False):
    """Create a training script for a specific halo and particle PID"""
    
    # Job name using our tested naming convention: h023p001
    job_name = f"h{halo_id}p{particle_pid:03d}"
    
    # Define training parameters based on test mode (avoid backslashes in f-strings)
    if test_mode:
        training_params = " \\\n    --n_subsample 1000 \\\n    --n_epochs 10 \\\n    --patience 5 \\\n    --batch_size 128"
    else:
        training_params = " \\\n    --max_epochs 200 \\\n    --patience 20 \\\n    --batch_size 512"
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_h{halo_id}_p{particle_pid:03d}_%j.out
#SBATCH --error=logs/train_h{halo_id}_p{particle_pid:03d}_%j.err

# Comprehensive logging for debugging
LOG_FILE="complete_training_h{halo_id}_p{particle_pid:03d}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "🚀 COMPREHENSIVE TRAINING LOG - HALO {halo_id} PARTICLE {particle_pid}"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo "Halo ID: {halo_id}"
echo "Particle PID: {particle_pid}"
echo "=========================================="

echo "📋 System Information:"
echo "  Hostname: $(hostname)"
echo "  User: $(whoami)"
echo "  Working directory: $(pwd)"
echo "  Available memory: $(free -h | grep '^Mem:' | awk '{{print $2}}')"
echo "  Available disk: $(df -h . | tail -1 | awk '{{print $4}}')"
echo "  Load average: $(uptime | awk -F'load average:' '{{print $2}}')"

# Load modules - use working combination (CPU-only is fine)
echo "🔌 Loading modules..."
set -x
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131
set +x

echo "📋 Loaded modules:"
module list

# Activate conda environment
echo "🐍 Activating conda environment..."
source {base_dir}/anaconda3/etc/profile.d/conda.sh
conda activate bosque

# Set CUDA paths for TensorFlow
echo "🔧 Setting CUDA paths for TensorFlow..."
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0

# Verify environment
echo "📋 Environment Check:"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo "  Python location: $(which python)"
echo "  Python: $(python --version)"

echo "📋 Package versions:"
echo "  TensorFlow: $(python -c 'import tensorflow; print(tensorflow.__version__)' 2>&1)"
echo "  TFP: $(python -c 'import tensorflow_probability; print(tensorflow_probability.__version__)' 2>&1)"
echo "  NumPy: $(python -c 'import numpy; print(numpy.__version__)' 2>&1)"
echo "  Keras: $(python -c 'import keras; print(keras.__version__)' 2>&1)"

echo "📋 GPU Detection:"
python -c "
import tensorflow as tf
print(f'  GPU available: {{tf.test.is_gpu_available()}}')
print(f'  GPU devices: {{tf.config.list_physical_devices(\"GPU\")}}')
"

# Navigate to working directory
cd {base_dir}/flows-tensorflow

# Define paths - using actual file naming patterns (will search multiple locations)
INPUT_HALO_FILE_PATTERNS=(
    "../milkyway-eden-mocks/eden_scaled_Halo{halo_id}_sunrot0_0kpc200kpcoriginal_particles.h5"
    "../milkyway-eden-mocks/eden_scaled_Halo{halo_id}_m_sunrot0_0kpc200kpcoriginal_particles.h5"
    "../milkywaymocks/symphony_scaled_Halo{halo_id}_sunrot0_0kpc200kpcoriginal_particles.h5"
    "../milkyway-hr-mocks/symphonyHR_scaled_Halo{halo_id}_sunrot0_0kpc200kpcoriginal_particles.h5"
)
OUTPUT_DIR="trained_flows/pid_{pid}"
METRICS_DIR="metrics/pid_{pid}"

# Create output directories
mkdir -p "$OUTPUT_DIR" "$METRICS_DIR" logs

# Find the correct halo file from multiple possible locations
echo "🔍 Searching for halo file for Halo{halo_id}..."
INPUT_HALO_FILE=""
for pattern in "${{INPUT_HALO_FILE_PATTERNS[@]}}"; do
    if [ -f "$pattern" ]; then
        INPUT_HALO_FILE="$pattern"
        echo "✅ Found input file: $INPUT_HALO_FILE"
        break
    fi
done

if [ -z "$INPUT_HALO_FILE" ]; then
    echo "❌ No halo file found for Halo{halo_id}. Searched:"
    for pattern in "${{INPUT_HALO_FILE_PATTERNS[@]}}"; do
        echo "   $pattern"
    done
    echo "💡 Available halo files in directories:"
    ls -la ../milkyway*/eden_scaled_Halo{halo_id}*.h5 ../milkyway*/symphony*.h5 2>/dev/null || echo "No files found"
    exit 1
fi

# Run training for this specific PID
echo "🧠 Training normalizing flow for PID {pid}..."
echo "  Input file: $INPUT_HALO_FILE"
echo "  Output directory: $OUTPUT_DIR"  
echo "  Metrics directory: $METRICS_DIR"

set -x
python train_tfp_flows.py \\
    --input_file "$INPUT_HALO_FILE" \\
    --output_dir "$OUTPUT_DIR" \\
    --metrics_dir "$METRICS_DIR" \\
    --pid {particle_pid} \\
    --halo_id {halo_id} \\
    --particle_pid {particle_pid} \\{training_params} \\
    --learning_rate 1e-3 \\
    --flow_layers 8 \\
    --hidden_units 64 \\
    --use_gpu
TRAIN_EXIT_CODE=$?
set +x

echo "📋 Training completed with exit code: $TRAIN_EXIT_CODE"

# Check if training completed successfully
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training successful for Halo {halo_id}, PID {particle_pid}"
    echo "📁 Check outputs in: $OUTPUT_DIR"
    echo "📊 Check samples in: $(echo $OUTPUT_DIR | sed 's/trained_flows/samples/')"
else
    echo "❌ Training failed for Halo {halo_id}, PID {particle_pid}"
    
    # Log the failure
    python track_failures.py log {halo_id} {particle_pid} "training_failed" "Training script exited with code $TRAIN_EXIT_CODE"
    
    echo "📋 Failure logged to failed_particles/ directory"
fi

echo "📋 Output file details:"
ls -la "$OUTPUT_DIR/" 2>/dev/null || echo "Output directory: $OUTPUT_DIR (not found)"
ls -la "$METRICS_DIR/" 2>/dev/null || echo "Metrics directory: $METRICS_DIR (not found)"

echo "=========================================="
echo "🏁 TRAINING COMPLETE - PID {pid}"
echo "Finished at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Log file: $LOG_FILE"
echo "=========================================="
"""
    
    return script_content

def create_sampling_script(pid, base_dir="/oak/stanford/orgs/kipac/users/caganze"):
    """Create a sampling script for a specific PID"""
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name=sample_flow_pid_{pid}
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sample_pid_{pid}_%j.out
#SBATCH --error=logs/sample_pid_{pid}_%j.err

echo "🎲 Starting sampling for PID {pid}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"

# Load modules
module load math devel

# Activate conda environment
source {base_dir}/anaconda3/etc/profile.d/conda.sh
conda activate bosque

# Verify environment
echo "📋 Environment Check:"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo "  Python: $(python --version)"
echo "  Available memory: $(free -h | grep '^Mem:' | awk '{{print $2}}')"

# Navigate to working directory
cd {base_dir}/flows-tensorflow

# Define paths
TRAINED_MODEL_DIR="trained_flows/pid_{pid}"
OUTPUT_DIR="samples/pid_{pid}"
METRICS_DIR="sampling_metrics/pid_{pid}"

# Create output directories
mkdir -p "$OUTPUT_DIR" "$METRICS_DIR" logs

# Check if trained model exists
if [ ! -f "$TRAINED_MODEL_DIR/flow_model.h5" ]; then
    echo "❌ No trained model found for PID {pid} at $TRAINED_MODEL_DIR/flow_model.h5"
    echo "💡 Run training job first: sbatch submit_training_pid_{pid}.sh"
    exit 1
fi

# Run sampling for this specific PID
echo "🎲 Generating samples from trained flow for PID {pid}..."
python sample_tfp_flows.py \\
    --trained_model_dir "$TRAINED_MODEL_DIR" \\
    --output_dir "$OUTPUT_DIR" \\
    --metrics_dir "$METRICS_DIR" \\
    --pid {pid} \\
    --n_samples 1000000 \\
    --batch_size 10000 \\
    --save_format "hdf5" \\
    --compress

# Check if sampling completed successfully
if [ -f "$OUTPUT_DIR/samples.hdf5" ]; then
    echo "✅ Sampling completed successfully for PID {pid}"
    echo "📊 Samples saved to: $OUTPUT_DIR/samples.hdf5"
    echo "📈 Metrics saved to: $METRICS_DIR/"
    
    # Print sample statistics
    echo "📊 Sample Statistics:"
    python -c "
import h5py
import numpy as np
with h5py.File('$OUTPUT_DIR/samples.hdf5', 'r') as f:
    samples = f['samples'][:]
    print(f'  Shape: {{samples.shape}}')
    print(f'  Memory: {{samples.nbytes / 1024**3:.2f}} GB')
    print(f'  Range: [{{samples.min():.3f}}, {{samples.max():.3f}}]')
"
else
    echo "❌ Sampling failed for PID {pid}"
    exit 1
fi

echo "🏁 Finished sampling for PID {pid} at: $(date)"
"""
    
    return script_content

def create_submit_all_script(pids):
    """Create a script to submit all training jobs"""
    
    script_content = f"""#!/bin/bash
# Submit all training jobs in parallel

echo "🚀 Submitting {len(pids)} training jobs..."

# Create logs directory
mkdir -p logs

# Submit all training jobs
"""
    
    for pid in pids:
        script_content += f"""
echo "📤 Submitting training job for PID {pid}..."
TRAIN_JOB_{pid}=$(sbatch --parsable submit_training_pid_{pid}.sh)
echo "  Job ID: $TRAIN_JOB_{pid}"
"""

    script_content += f"""
echo ""
echo "✅ All {len(pids)} training jobs submitted!"
echo "📊 Monitor with: squeue -u $USER"
echo "📋 Check logs in: logs/"
echo ""
echo "🎯 After training completes, submit sampling jobs with:"
echo "   ./submit_all_sampling.sh"
"""
    
    return script_content

def create_submit_sampling_script(pids):
    """Create a script to submit all sampling jobs"""
    
    script_content = f"""#!/bin/bash
# Submit all sampling jobs (after training completes)

echo "🎲 Submitting {len(pids)} sampling jobs..."

# Create logs directory
mkdir -p logs

# Submit all sampling jobs
"""
    
    for pid in pids:
        script_content += f"""
echo "📤 Submitting sampling job for PID {pid}..."
SAMPLE_JOB_{pid}=$(sbatch --parsable submit_sampling_pid_{pid}.sh)
echo "  Job ID: $SAMPLE_JOB_{pid}"
"""

    script_content += f"""
echo ""
echo "✅ All {len(pids)} sampling jobs submitted!"
echo "📊 Monitor with: squeue -u $USER"
echo "📋 Check logs in: logs/"
echo ""
echo "🎯 Results will be in:"
echo "   trained_flows/pid_*/ (models)"
echo "   samples/pid_*/ (generated samples)"
echo "   metrics/pid_*/ (training metrics)"
echo "   sampling_metrics/pid_*/ (sampling metrics)"
"""
    
    return script_content

def main():
    parser = argparse.ArgumentParser(description="Generate parallel SLURM scripts for flow training")
    parser.add_argument("--pids", nargs="+", type=int, 
                       default=[23, 88, 188, 268, 327, 364, 415, 440, 469, 530, 570, 641, 718, 800, 829, 852, 939],
                       help="List of PIDs to create scripts for (default: common halo PIDs)")
    parser.add_argument("--base_dir", default="/oak/stanford/orgs/kipac/users/caganze",
                       help="Base directory path on Sherlock")
    parser.add_argument("--output_dir", default=".",
                       help="Directory to save generated scripts")
    
    args = parser.parse_args()
    
    print(f"🏭 Generating parallel scripts for {len(args.pids)} PIDs...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate individual scripts for each PID
    for pid in args.pids:
        # Training script
        train_script = create_training_script(pid, args.base_dir)
        train_file = os.path.join(args.output_dir, f"submit_training_pid_{pid}.sh")
        with open(train_file, 'w') as f:
            f.write(train_script)
        os.chmod(train_file, 0o755)
        print(f"✅ Created: {train_file}")
        
        # Sampling script
        sample_script = create_sampling_script(pid, args.base_dir)
        sample_file = os.path.join(args.output_dir, f"submit_sampling_pid_{pid}.sh")
        with open(sample_file, 'w') as f:
            f.write(sample_script)
        os.chmod(sample_file, 0o755)
        print(f"✅ Created: {sample_file}")
    
    # Generate batch submission scripts
    submit_all_train = create_submit_all_script(args.pids)
    submit_train_file = os.path.join(args.output_dir, "submit_all_training.sh")
    with open(submit_train_file, 'w') as f:
        f.write(submit_all_train)
    os.chmod(submit_train_file, 0o755)
    print(f"✅ Created: {submit_train_file}")
    
    submit_all_sample = create_submit_sampling_script(args.pids)
    submit_sample_file = os.path.join(args.output_dir, "submit_all_sampling.sh")
    with open(submit_sample_file, 'w') as f:
        f.write(submit_all_sample)
    os.chmod(submit_sample_file, 0o755)
    print(f"✅ Created: {submit_sample_file}")
    
    print(f"""
🎉 SUCCESS: Generated parallel pipeline for {len(args.pids)} PIDs!

📁 Files created:
  • submit_training_pid_*.sh     ({len(args.pids)} files)
  • submit_sampling_pid_*.sh     ({len(args.pids)} files)  
  • submit_all_training.sh       (batch submission)
  • submit_all_sampling.sh       (batch submission)

🚀 Usage:
  1. Transfer to Sherlock: scp * sherlock:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/
  2. Set up environment: ./create_working_gpu_env.sh
  3. Submit training: ./submit_all_training.sh
  4. After training: ./submit_all_sampling.sh

📊 Monitor: squeue -u $USER
""")

if __name__ == "__main__":
    main()
