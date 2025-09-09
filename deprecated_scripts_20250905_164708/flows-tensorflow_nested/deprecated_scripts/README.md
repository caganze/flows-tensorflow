# Deprecated Submission Scripts

These scripts have been replaced by the unified `submit_tfp_array.sh` system.

## 🗑️ Deprecated Scripts (Moved on $(date))

- `submit_flows_array.sh` → **Replaced by `submit_tfp_array.sh`**
- `submit_flows_batch1.sh` → **Replaced by `submit_tfp_array.sh`**  
- `submit_flows_batch.sh` → **Replaced by `submit_tfp_array.sh`**
- `submit_small_test.sh` → **Replaced by `submit_tfp_array.sh`**
- `auto_submit_flows.sh` → **Replaced by `scan_and_resubmit.sh`**
- `run_auto_submit_nohup.sh` → **No longer needed**

## ❌ Why These Were Deprecated

### **Issues Fixed:**
1. **Resource mismatches**: Old scripts requested 8 GPUs but only used 1
2. **No failure detection**: No intelligent resubmission capabilities  
3. **Race conditions**: Directory creation conflicts
4. **No particle size awareness**: All particles treated equally
5. **Manual resubmission**: Required manual intervention for failures

## ✅ New Unified System

### **Main Scripts:**
- `submit_tfp_array.sh` - Unified submission with intelligent GPU allocation
- `scan_and_resubmit.sh` - Comprehensive failure detection and resubmission
- `brute_force_gpu_job.sh` - Kept for specialized brute force testing

### **Key Improvements:**
- ✅ **Proper GPU utilization**: Uses all 8 requested GPUs in parallel
- ✅ **Particle size detection**: >100k objects get extended runtime
- ✅ **Smart resubmission**: Only retries actually failed particles  
- ✅ **File locking**: Prevents directory race conditions
- ✅ **Adaptive parameters**: Training parameters adjust to particle size

## 🚀 Migration Guide

### **Old Command → New Command**

```bash
# OLD: Multiple different scripts
sbatch submit_flows_array.sh
sbatch submit_flows_batch1.sh  
sbatch submit_small_test.sh

# NEW: Single unified script
sbatch submit_tfp_array.sh
```

### **Old Resubmission → New Resubmission**

```bash
# OLD: Manual identification and resubmission
# (required manual log analysis)

# NEW: Automatic failure detection
./scan_and_resubmit.sh
./resubmit_failed_YYYYMMDD_HHMMSS.sh  # Auto-generated
```

## 🔧 Emergency Recovery

If you need to reference old configurations:

```bash
# View old SLURM parameters
grep "#SBATCH" deprecated_scripts/submit_flows_array.sh

# View old module loading
grep "module load" deprecated_scripts/submit_flows_array.sh

# View old training parameters  
grep "train_tfp_flows.py" deprecated_scripts/submit_flows_array.sh
```

## 📁 Safe to Delete

These scripts can be safely deleted after confirming the new system works correctly.
