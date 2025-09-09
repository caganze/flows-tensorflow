# 🚀 Brute Force GPU Job Usage Guide

This guide explains how to use the brute force GPU job system that systematically processes all halos and particle IDs (PIDs) for training and sampling.

## 📋 Overview

The brute force system consists of:
- **`brute_force_gpu_job.sh`**: Main SLURM array job script
- **`monitor_brute_force.sh`**: Monitoring and progress tracking script
- **Comprehensive logging**: Success/failure tracking with detailed error reporting

## 🎯 Success Criteria

A job is considered successful ONLY if ALL of the following are present:
1. ✅ **Trained model file**: `trained_flows/model_pidX/model_pidX.npz`
2. ✅ **Sample files**: `.npz` and/or `.h5` files in `samples/` directory
3. ✅ **Results file**: `model_pidX_results.json` with training metrics
4. ✅ **No critical errors** in the logs

## 🚀 How to Submit the Job on Sherlock

### 1. Setup and Deployment
```bash
# From your local flows-tensorflow directory, sync all files to Sherlock
rsync -av * caganze@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/

# SSH to Sherlock
ssh caganze@login.sherlock.stanford.edu

# Navigate to your flows directory
cd /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow

# Run deployment script to set up environment
./deploy_to_sherlock.sh

# Check that your files are there
ls -la brute_force_gpu_job.sh monitor_brute_force.sh
```

### 2. Basic Submission
```bash
# Submit the brute force job array
sbatch brute_force_gpu_job.sh
```

### 3. Custom Array Range
```bash
# Process only first 100 combinations
sbatch --array=1-100%5 brute_force_gpu_job.sh

# Process specific range
sbatch --array=50-150%3 brute_force_gpu_job.sh
```

### 4. Test Run (Single Job)
```bash
# Test with just one array task
sbatch --array=1 brute_force_gpu_job.sh
```

## 📊 Monitoring Progress

### Quick Status Check
```bash
./monitor_brute_force.sh
```

### Detailed Analysis
```bash
./monitor_brute_force.sh --detailed
```

### Live Monitoring (Updates every 30s)
```bash
./monitor_brute_force.sh --live
```

### Manual SLURM Monitoring
```bash
# Check job status
squeue -u $(whoami) --name="brute_force_all_halos_pids"

# Check specific job details
scontrol show job JOBID

# Cancel all brute force jobs if needed
scancel --name="brute_force_all_halos_pids"
```

## 📁 Output Structure

```
/oak/stanford/orgs/kipac/users/caganze/tfp_flows_output/
├── trained_flows/
│   ├── model_pid1/
│   │   ├── model_pid1.npz                 # ✅ Required
│   │   ├── model_pid1_results.json        # ✅ Required
│   │   └── model_pid1_preprocessing.npz
│   ├── model_pid2/
│   └── ...
├── samples/
│   ├── model_pid1_samples.npz             # ✅ Required (small datasets)
│   ├── model_pid1_samples.h5              # ✅ Required (large datasets)
│   ├── model_pid1_samples_quick.npz       # Optional (quick access)
│   └── ...
└── metrics/
    ├── pid_1/
    └── ...
```

## 📋 Log Files

### SLURM Logs
- **Output**: `logs/brute_force_JOBID_TASKID.out`
- **Errors**: `logs/brute_force_JOBID_TASKID.err`

### Success/Failure Tracking
- **Successes**: `success_logs/brute_force_success.log`
- **Failures**: `failed_jobs/brute_force_failures.log`

### Example Success Log Entry
```
2024-01-15 14:30:25 SUCCESS halo:023 pid:4 model:/path/to/model.npz samples:/path/to/samples.npz
```

### Example Failure Log Entry
```
2024-01-15 14:25:10 FAILED halo:088 pid:7 error_type:validation_failed msg:missing sample files
```

## 🔧 How It Works

### 1. Job Array Logic
- Each array task processes ONE halo-PID combination
- Array indexing: `task_id = (file_index * num_pids) + pid_index + 1`
- Automatically discovers available halo files and PIDs
- Skips combinations that exceed available data

### 2. File Discovery
```bash
# Finds all halo files matching pattern
find /oak/stanford/orgs/kipac/users/caganze/ -name '*Halo*.h5'

# Samples PID ranges from multiple files for efficiency
# Uses first 3 files to get representative PID ranges
```

### 3. Success Validation
For each halo-PID combination, validates:
- Model file exists and is non-empty
- At least one sample file exists (.npz or .h5)
- Results JSON file contains training metrics
- No critical errors in processing

### 4. Error Handling
- **File not found**: Logs and skips gracefully
- **Training failure**: Captures exit codes and logs
- **Validation failure**: Identifies missing output files
- **Already processed**: Skips completed combinations

## 🎛️ Configuration Options

### Resource Allocation
```bash
#SBATCH --time=72:00:00        # 3 days max per job
#SBATCH --mem=128GB            # High memory for large datasets
#SBATCH --gres=gpu:1           # 1 GPU per task
#SBATCH --array=1-500%5        # Max 5 jobs running simultaneously
```

### Training Parameters
```bash
python train_tfp_flows.py \
    --epochs 100 \              # 100 training epochs
    --batch_size 512 \          # Batch size for training
    --learning_rate 1e-3 \      # Learning rate
    --n_layers 6 \              # Number of flow layers
    --hidden_units 1024         # Hidden units per layer
```

## 🚨 Troubleshooting

### Common Issues

1. **No jobs starting**
   ```bash
   # Check partition availability
   sinfo -p gpu
   
   # Check job queue
   squeue -u $(whoami)
   ```

2. **Jobs failing immediately**
   ```bash
   # Check recent error logs
   tail -50 logs/brute_force_*.err
   
   # Check environment setup
   module list
   conda env list
   ```

3. **Files not found**
   ```bash
   # Verify halo file paths
   find /oak/stanford/orgs/kipac/users/caganze/ -name '*Halo*.h5' | head -5
   
   # Check permissions
   ls -la /oak/stanford/orgs/kipac/users/caganze/
   ```

4. **Memory issues**
   ```bash
   # Reduce batch size in script
   sed -i 's/--batch_size 512/--batch_size 256/' brute_force_gpu_job.sh
   ```

### Recovery from Failures

1. **Resubmit failed combinations only**
   ```bash
   # Check failure log
   cat failed_jobs/brute_force_failures.log
   
   # Manually resubmit specific ranges if needed
   sbatch --array=75-85 brute_force_gpu_job.sh
   ```

2. **Clean restart**
   ```bash
   # Cancel all jobs
   scancel --name="brute_force_all_halos_pids"
   
   # Clean logs (optional)
   rm -rf logs/brute_force_*
   rm -rf success_logs/* failed_jobs/*
   
   # Resubmit
   sbatch brute_force_gpu_job.sh
   ```

## 📈 Expected Performance

- **Timeline**: 24-72 hours depending on dataset size
- **Success Rate**: >95% with proper setup
- **Per Combination**: ~10-20 minutes (training + sampling)
- **Resource Usage**: ~60-80GB RAM, 1 GPU per job
- **Output Size**: ~10-100MB per halo-PID combination

## 🎯 Next Steps After Completion

1. **Verify completeness**
   ```bash
   ./monitor_brute_force.sh --detailed
   ```

2. **Analyze results**
   ```bash
   # Count successful models
   find /oak/stanford/orgs/kipac/users/caganze/tfp_flows_output/trained_flows -name "*.npz" | wc -l
   
   # Count sample files
   find /oak/stanford/orgs/kipac/users/caganze/tfp_flows_output/samples -name "*samples*" | wc -l
   ```

3. **Generate summary statistics**
   ```bash
   python -c "
   import json
   import glob
   
   results = []
   for f in glob.glob('/oak/stanford/orgs/kipac/users/caganze/tfp_flows_output/trained_flows/*/model_pid*_results.json'):
       with open(f) as fp:
           results.append(json.load(fp))
   
   print(f'Total models: {len(results)}')
   if results:
       avg_loss = sum(r.get('final_train_loss', 0) for r in results) / len(results)
       print(f'Average final loss: {avg_loss:.4f}')
   "
   ```

---

🌟 **The brute force system ensures comprehensive coverage of all halo-PID combinations with robust error handling and progress tracking!**
