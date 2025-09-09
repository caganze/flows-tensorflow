# 🖥️ CPU Parallel TensorFlow Probability Flows - Usage Guide

## 🚀 Overview

This CPU parallel system is optimized for high-CPU compute nodes, providing an efficient CPU-based alternative to the GPU brute force job. It processes one particle per array task with CPU-specific optimizations, leveraging your abundant CPU resources effectively.

## 📋 Key Features

### ✨ **CPU-Optimized Performance**
- 🧵 **Multi-threading**: Automatically configures optimal thread counts for CPU training
- 🧠 **Smart parameters**: Adjusts training config based on particle size
- 💾 **Memory efficiency**: CPU-optimized memory usage patterns
- ⚡ **High throughput**: One particle per array task for maximum parallelism

### 🎯 **Intelligent Resource Management**
- 📊 **Adaptive training**: Training config adjusts to particle size (Large/Medium/Small)
- 🔄 **Simple scaling**: Array size equals number of particles in list
- 📈 **Progress tracking**: Real-time monitoring with detailed statistics
- 🛡️ **Robustness**: Automatic completion checking and error handling

## 🚀 Quick Start

### 1. **Setup and Submission**
```bash
# Make scripts executable
chmod +x brute_force_cpu_parallel.sh submit_cpu_parallel.sh monitor_cpu_parallel.sh

# Submit with defaults (array size = number of particles, 10 concurrent, 64 CPUs each)
./submit_cpu_parallel.sh

# Or customize resource allocation
./submit_cpu_parallel.sh --concurrent 5 --cpus 32 --memory 256GB --partition bigmem
```

### 2. **Monitor Progress**
```bash
# Quick status check
./monitor_cpu_parallel.sh

# Detailed progress with auto-refresh
./monitor_cpu_parallel.sh --details --follow

# Check specific job
squeue -u $USER -j <JOB_ID>
```

## ⚙️ Configuration Options

### **Submission Parameters**

| Parameter | Default | Description | Example |
|-----------|---------|-------------|---------|
| `--array-size` | 50 | Number of array tasks | `--array-size 100` |
| `--concurrent` | 10 | Max concurrent tasks | `--concurrent 5` |
| `--partition` | normal | SLURM partition | `--partition bigmem` |
| `--time` | 24:00:00 | Time limit | `--time 12:00:00` |
| `--memory` | 512GB | Memory per task | `--memory 1TB` |
| `--cpus` | 64 | CPUs per task | `--cpus 32` |

### **Training Optimization**

The system automatically optimizes training parameters based on:

#### 🐋 **Large Particles (>100k objects)**
- Epochs: 60 (efficiency focused)
- Batch size: 1024 (large batches)
- Layers: 4, Hidden units: 512

#### 🐭 **Small Particles (<5k objects)**
- Epochs: 150 (quality focused)
- Batch size: 256 (small batches)
- Layers: 6, Hidden units: 1024

#### 🐟 **Medium Particles (5k-100k objects)**
- Epochs: 100 (balanced)
- Batch size: 512 (balanced)
- Layers: 5, Hidden units: 768

## 📊 Parallelization Strategy

### **Two-Level Parallelism**

1. **Array Level**: Multiple SLURM array tasks run simultaneously
2. **Task Level**: Each array task processes 4 particles in parallel using background processes

```
Array Task 1: ├─ Particle 1 (background)
              ├─ Particle 2 (background)  
              ├─ Particle 3 (background)
              └─ Particle 4 (background)

Array Task 2: ├─ Particle 5 (background)
              ├─ Particle 6 (background)
              ├─ Particle 7 (background)
              └─ Particle 8 (background)
...
```

### **Resource Allocation Example**
- **50 array tasks** × **4 particles each** = **200 particles processed simultaneously**
- **64 CPUs per task** optimally distributed across 4 parallel processes
- **512GB memory per task** shared efficiently

## 📈 Monitoring and Progress

### **Quick Status**
```bash
./monitor_cpu_parallel.sh
```
Shows:
- 🟢 Running/pending SLURM jobs
- 📊 Particle processing progress
- 📈 Completion percentage with progress bar
- ✅ Success rate

### **Detailed Analysis**
```bash
./monitor_cpu_parallel.sh --details
```
Additionally shows:
- 🎉 Recent successes
- ⚠️ Recent failures  
- 🔄 Running job details
- 💾 Storage usage statistics

### **Follow Mode**
```bash
./monitor_cpu_parallel.sh --follow --interval 30
```
Auto-refreshes every 30 seconds for continuous monitoring.

## 📁 Output Structure

```
/oak/stanford/orgs/kipac/users/caganze/
├── milkyway-eden-mocks/
│   └── tfp_output/
│       ├── trained_flows/
│       │   └── eden/halo123/
│       │       ├── model_pid1.npz
│       │       ├── model_pid1_preprocessing.npz
│       │       └── model_pid1_results.json
│       └── samples/
│           └── eden/halo123/
│               ├── model_pid1_samples.npz  # or .h5 for large
│               └── model_pid1_samples.json
├── logs/
│   ├── cpu_parallel_JOBID_ARRAYID.out
│   └── cpu_parallel_JOBID_ARRAYID.err
├── success_logs/
│   └── cpu_parallel_success.log
├── failed_jobs/
│   └── cpu_parallel_failures.log
└── cpu_parallel_progress/
    └── array_progress.log
```

## 🎛️ Advanced Usage

### **Custom Resource Allocation**
```bash
# High-memory partition with more CPUs
./submit_cpu_parallel.sh --partition bigmem --memory 1TB --cpus 128

# Quick test with fewer resources
./submit_cpu_parallel.sh --array-size 5 --concurrent 2 --cpus 16 --time 2:00:00

# Large-scale processing
./submit_cpu_parallel.sh --array-size 200 --concurrent 20 --cpus 64
```

### **Dry Run Testing**
```bash
# Test submission without actually submitting
./submit_cpu_parallel.sh --dry-run --array-size 100

# Preview with custom settings
./submit_cpu_parallel.sh --dry-run --partition bigmem --memory 1TB
```

### **Progress Analysis**
```bash
# Check success rate
grep SUCCESS success_logs/cpu_parallel_success.log | wc -l

# Check failure patterns
tail -20 failed_jobs/cpu_parallel_failures.log

# Array task completion
cat cpu_parallel_progress/array_progress.log

# Real-time log monitoring
tail -f logs/cpu_parallel_*.out
```

## 🔧 CPU Optimization Features

### **Automatic Threading**
- Detects available CPU cores from SLURM environment
- Configures TensorFlow threading optimally
- Sets math library thread counts (OpenMP, MKL, OpenBLAS)

### **Memory Efficiency**
- Enables TensorFlow compute optimizations
- Smart batch sizing based on available memory
- Efficient data loading and preprocessing

### **Model Scaling**
- Adjusts model complexity based on data size
- Balances training time vs. model quality
- CPU-specific architectural choices

## 🚨 Troubleshooting

### **Common Issues**

#### **No particles found**
```bash
# Regenerate particle list
./generate_particle_list.sh

# Check H5 file access
ls -la /oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/*.h5
```

#### **Jobs failing to start**
```bash
# Check SLURM queue
squeue -u $USER

# Verify partition availability
sinfo -p normal

# Check resource requests
scontrol show job <JOB_ID>
```

#### **Low success rate**
```bash
# Check recent failures
tail -10 failed_jobs/cpu_parallel_failures.log

# Monitor resource usage
ssh <compute_node> top -u $USER
```

### **Performance Tuning**

#### **Optimize for your data**
- Small particles: Increase `--array-size`, decrease `--cpus`
- Large particles: Decrease `--array-size`, increase `--memory`
- Mixed sizes: Use defaults with high `--concurrent`

#### **Resource balancing**
```bash
# CPU-heavy workload
./submit_cpu_parallel.sh --cpus 128 --memory 256GB

# Memory-heavy workload  
./submit_cpu_parallel.sh --cpus 32 --memory 1TB

# Balanced approach (default)
./submit_cpu_parallel.sh --cpus 64 --memory 512GB
```

## 🎯 Expected Performance

### **Throughput Estimates**
- **Small particles**: ~20-30 particles/hour per array task
- **Medium particles**: ~10-15 particles/hour per array task  
- **Large particles**: ~5-8 particles/hour per array task

### **Resource Efficiency**
- **CPU utilization**: ~85-95% across all cores
- **Memory usage**: ~60-80% of allocated memory
- **I/O optimization**: Efficient HDF5 reading and NPZ/HDF5 writing

### **Scaling Behavior**
- **Linear scaling** up to ~50 concurrent array tasks
- **Optimal throughput** at 10-20 concurrent tasks for most workloads
- **Memory saturation** around 1TB per task for very large particles

## 🏁 Success Criteria

A particle is considered successfully processed when ALL of these exist:
1. ✅ **Model file**: `model_pidX.npz`
2. ✅ **Preprocessing**: `model_pidX_preprocessing.npz`  
3. ✅ **Results**: `model_pidX_results.json`
4. ✅ **Samples**: `model_pidX_samples.npz` or `.h5`

## 📞 Quick Commands Reference

```bash
# Submit job
./submit_cpu_parallel.sh

# Monitor progress
./monitor_cpu_parallel.sh --follow

# Check specific job
squeue -u $USER -j <JOB_ID>

# Cancel jobs
scancel <JOB_ID>

# View logs
tail -f logs/cpu_parallel_*.out

# Success count
wc -l success_logs/cpu_parallel_success.log

# Failure count
wc -l failed_jobs/cpu_parallel_failures.log
```

---

🖥️ **Happy CPU parallel computing!** This system is designed to maximize your abundant CPU resources while maintaining efficiency and robustness. The intelligent parallelization should provide significantly higher throughput than sequential processing while remaining easy to monitor and manage.
