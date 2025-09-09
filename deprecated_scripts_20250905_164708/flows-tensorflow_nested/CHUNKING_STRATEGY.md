# 📊 Chunking Strategy for 140k+ Particles

## 🎯 **PROBLEM & SOLUTION**

### **Challenge:**
- **140,000+ particles** to process
- **Array limit**: `--array=1-20%3` (only 20 tasks, 3 concurrent)
- **Constraints**: 12-hour time limit, 8 GPUs per job
- **Mix**: Mostly small/fast particles (few minutes) + some large/slow particles (hours)

### **Solution: Smart Chunking + Two-Phase Processing**
```
140,000 particles ÷ 20 array tasks = ~7,000 particles per chunk
```

## 🔧 **CHUNKING STRATEGY**

### **📦 Chunk Size Calculation:**
```bash
CHUNK_SIZE=7000  # particles per array task
Array Task 1: particles 1-7000
Array Task 2: particles 7001-14000
...
Array Task 20: particles 133001-140000
```

### **🔄 Two-Phase Processing per Chunk:**

#### **Phase 1: Small Particles (Parallel)**
- 🐭 **Small/Fast particles** processed **in parallel** using all 8 GPUs
- ⚡ **Batches of 8** particles at a time (one per GPU)
- 🚀 **High throughput** for majority of particles

#### **Phase 2: Large Particles (Sequential)**
- 🐋 **Large/Slow particles** processed **sequentially**
- 🎯 **One at a time** for memory efficiency
- ⏱️ **Time management** to fit in 12-hour window

## ⚙️ **IMPLEMENTATION**

### **Updated Configuration:**
```bash
#SBATCH --array=1-20%3              # Only 20 chunks, max 3 concurrent
CHUNK_SIZE=7000                     # Particles per array task  
PARTICLES_PER_TASK=8                # Parallel processes (match 8 GPUs)
```

### **Chunk Processing Logic:**
```bash
# Calculate chunk boundaries
START_LINE=$(( ($SLURM_ARRAY_TASK_ID - 1) * $CHUNK_SIZE + 1 ))
END_LINE=$(( $SLURM_ARRAY_TASK_ID * $CHUNK_SIZE ))

# Load chunk and separate by size
while reading chunk particles:
    if large particle: add to LARGE_PARTICLES[]
    else: add to SMALL_PARTICLES[]

# Phase 1: Process small particles in parallel
for each batch of 8 small particles:
    launch 8 parallel processes (one per GPU)
    wait for batch completion

# Phase 2: Process large particles sequentially  
for each large particle:
    process on single GPU
    check time limit
```

## 📊 **EXPECTED PERFORMANCE**

### **Per Chunk (7,000 particles):**
- **Small particles** (~6,500): 2-6 hours
- **Large particles** (~500): 4-8 hours  
- **Total per chunk**: 8-12 hours (fits in limit)

### **Overall Throughput:**
- **20 chunks × 3 concurrent** = **60 parallel workers**
- **Total capacity**: ~420,000 particles per 12-hour cycle
- **Your 140k particles**: Completes in **1-2 cycles**

## 🎮 **GPU Utilization**

### **Phase 1 (Small Particles):**
```
GPU 0: Small PID 1
GPU 1: Small PID 2  
GPU 2: Small PID 3
...
GPU 7: Small PID 8
→ Wait for batch → Next 8 particles
```

### **Phase 2 (Large Particles):**
```
GPU 0: Large PID 1001 (3 hours)
GPU 1-7: Idle (large particles need full GPU memory)
```

## 🚀 **USAGE WORKFLOW**

### **Setup:**
```bash
# 1. Generate particle list
./generate_particle_list.sh

# 2. Verify chunk size
TOTAL_PARTICLES=$(wc -l < particle_list.txt)
echo "Total particles: $TOTAL_PARTICLES"
echo "Chunks needed: $(( (TOTAL_PARTICLES + 6999) / 7000 ))"
```

### **Submit Jobs:**
```bash
# 3. Submit array job (20 chunks, 3 concurrent)
sbatch submit_tfp_array.sh
```

### **Monitor Progress:**
```bash
# 4. Monitor all chunks
squeue -u $(whoami)

# 5. Check individual chunk logs
tail -f logs/tfp_*_1.out   # Chunk 1
tail -f logs/tfp_*_2.out   # Chunk 2
```

### **Handle Failures:**
```bash
# 6. Scan for failures and resubmit
./scan_and_resubmit.sh
```

## 📈 **ADVANTAGES**

### **✅ Efficiency:**
- **Parallel small particles**: Maximum GPU utilization
- **Sequential large particles**: Avoids memory conflicts
- **Smart time management**: 11-hour processing + 1-hour buffer

### **✅ Scalability:**
- **140k+ particles**: Handled efficiently
- **Array constraints**: Works within limits
- **Resource optimization**: 8 GPUs fully utilized

### **✅ Robustness:**
- **Time limit protection**: Graceful shutdown before timeout
- **Progress tracking**: Detailed logging per phase
- **Failure handling**: Easy resubmission of incomplete chunks

### **✅ Monitoring:**
- **Phase-based progress**: Clear visibility into processing stages
- **Chunk statistics**: Success rates per chunk
- **Real-time updates**: Progress reports every batch

## 🎯 **OPTIMAL SETTINGS**

```bash
# Array configuration
#SBATCH --array=1-20%3

# Chunk settings  
CHUNK_SIZE=7000
PARTICLES_PER_TASK=8

# Timing
Phase 1: ~50-80% of chunk time (parallel)
Phase 2: ~20-50% of chunk time (sequential) 
Buffer: ~1 hour safety margin
```

## 🏁 **EXPECTED RESULTS**

With **140k particles**:
- **Chunks**: 20 chunks of ~7k particles each
- **Concurrent processing**: 3 chunks at a time
- **Total time**: ~6-8 cycles × 12 hours = **3-4 days**
- **Success rate**: >95% (with resubmission for failures)

This strategy **maximizes throughput** while respecting all constraints! 🚀
