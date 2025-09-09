# 💪 Brute Force Particle Processing - Usage Guide

## 🚀 **Simple Sequential Processing for 12 Hours**

The brute force script now reads from the `particle_list.txt` and processes particles **sequentially** for the full 12-hour time limit. This is perfect when you want:

- ✅ **Continuous processing** without array job complexity
- ✅ **Simple monitoring** - one job, one log
- ✅ **Automatic completion checking** - skips already processed particles
- ✅ **Graceful time management** - stops before hitting time limit

## 🔧 **SETUP**

### **1. Generate Particle List**
```bash
./generate_particle_list.sh
```

### **2. Submit Brute Force Job**
```bash
sbatch brute_force_gpu_job.sh
```

## 📊 **WHAT IT DOES**

### **⚙️ Configuration:**
- 🕐 **Time limit**: 12 hours (with 30-minute buffer)
- 🎮 **GPUs**: 4 GPUs requested
- 💾 **Memory**: 128GB
- 🔄 **Mode**: Sequential processing (one particle at a time)

### **🔄 Processing Loop:**
```bash
while reading particle_list.txt:
    for each particle:
        1. Check if already completed → skip if yes
        2. Determine optimal parameters based on size
        3. Run train_tfp_flows.py
        4. Verify completion (model + samples)
        5. Log success/failure
        6. Check time limit
```

### **⚡ Adaptive Parameters:**
- **Large particles (>100k objects)**: 
  - Epochs: 80 (fewer for efficiency)
  - Batch size: 768 (larger batches)
- **Small particles (<10k objects)**:
  - Epochs: 120 (more for better training)
- **Normal particles**:
  - Epochs: 100, Batch size: 512

## 📈 **MONITORING**

### **📊 Progress Reports Every 10 Particles:**
```bash
📊 PROGRESS REPORT (Sun Aug 24 16:30:00 PDT 2025)
  ⏱️ Elapsed time: 2h 45m
  📈 Processed: 23/156
  ✅ Completed: 18
  ⏭️ Skipped: 3
  ❌ Failed: 2
```

### **📁 Log Files:**
- `logs/brute_force_<job_id>.out` - Main job output
- `success_logs/brute_force_success.log` - Success entries
- `failed_jobs/brute_force_failures.log` - Failure entries

## 🎯 **ADVANTAGES**

### **✅ Simplicity:**
- Single job submission
- No array job complexity
- Easy to monitor and debug

### **✅ Efficiency:**
- Skips already completed particles
- Adaptive parameters for different particle sizes
- Graceful time management

### **✅ Robustness:**
- Comprehensive completion checking
- Detailed logging
- Automatic recovery (just resubmit)

### **✅ Resource Management:**
- Uses full 12-hour allocation
- Stops gracefully before timeout
- Detailed statistics and reporting

## 🔄 **TYPICAL WORKFLOW**

### **First Run:**
```bash
# 1. Generate particle list
./generate_particle_list.sh

# 2. Submit brute force job
sbatch brute_force_gpu_job.sh

# 3. Monitor progress
tail -f logs/brute_force_<job_id>.out
```

### **Subsequent Runs:**
```bash
# Just resubmit - it will skip completed particles automatically
sbatch brute_force_gpu_job.sh
```

### **Check Results:**
```bash
# See what was completed
tail -20 success_logs/brute_force_success.log

# Check for failures
cat failed_jobs/brute_force_failures.log

# Quick status
./scan_and_resubmit.sh
```

## 📊 **EXPECTED PERFORMANCE**

### **Processing Time per Particle:**
- **Small particles** (<10k): ~5-15 minutes
- **Medium particles** (10k-100k): ~15-45 minutes  
- **Large particles** (>100k): ~1-3 hours

### **12-Hour Capacity:**
- **Small particles**: ~50-100 particles
- **Mixed sizes**: ~20-40 particles
- **Large particles**: ~5-10 particles

## 🆚 **BRUTE FORCE vs ARRAY JOBS**

| Feature | Brute Force | Array Jobs |
|---------|-------------|------------|
| **Complexity** | Simple | More complex |
| **Monitoring** | Single log | Multiple logs |
| **Parallelism** | Sequential | Parallel |
| **Resource Use** | Full 12h on 1 node | Distributed |
| **Best For** | Continuous processing | High throughput |

## 🚀 **USAGE COMMANDS**

```bash
# Quick start
./generate_particle_list.sh && sbatch brute_force_gpu_job.sh

# Monitor live
squeue -u $(whoami)
tail -f logs/brute_force_*.out

# Check status
./scan_and_resubmit.sh

# Restart after completion
sbatch brute_force_gpu_job.sh  # Will continue where it left off
```

## 🎉 **READY TO GO!**

The brute force script is **rock-solid** and will continuously process your particles for 12 hours straight. Perfect for making steady progress through your particle list! 💪
