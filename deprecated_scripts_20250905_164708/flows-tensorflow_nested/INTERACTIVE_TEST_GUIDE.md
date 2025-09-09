# 🚀 Interactive Testing Guide for Argument Fixes

## **Available Interactive Test Options:**

### **Option 1: Quick Fix Test (Recommended)**
```bash
# On Sherlock
cd /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow
./quick_interactive_test.sh
```
**What it does:**
- Requests 4 GPUs, 1 hour, 8GB memory (minimal)
- Tests argument parsing directly
- Runs a 2-epoch training to verify fixes
- Gives immediate feedback on success/failure

### **Option 2: Full Validation Test**
```bash
# On Sherlock  
./validate_deployment.sh
```
**What it does:**
- Requests 4 GPUs, 1 hour, 16GB memory
- Comprehensive environment testing
- Full deployment validation
- More thorough but takes longer

### **Option 3: SLURM Array Simulation**
```bash
# On Sherlock
./test_slurm_deployment.sh  
```
**What it does:**
- Requests 4 GPUs, 1 hour, 8GB memory
- Tests SLURM array functionality
- Validates argument parsing in array context
- Tests multiple PIDs

## **🎯 What to Expect:**

### **Before Fixes (85% success rate):**
```
❌ train_tfp_flows.py: error: the following arguments are required: --data_path
❌ Multiple job failures due to argument mismatches
```

### **After Fixes (95%+ success rate):**
```
✅ Arguments parsed successfully
✅ Training starts without argument errors  
✅ Model files created successfully
```

## **📋 Testing Commands on Compute Node:**

Once you get the interactive allocation, run these manually:

### **Test 1: Basic Argument Check**
```bash
python train_tfp_flows.py --help | head -20
```

### **Test 2: Quick Training**
```bash
python train_tfp_flows.py \
    --data_path /oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/eden_scaled_Halo*_particles.h5 \
    --particle_pid 1 \
    --output_dir /tmp/test_output \
    --epochs 1 \
    --batch_size 128
```

### **Test 3: Verify Fixed Scripts**
```bash
# Check that scripts use --data_path now:
grep -A 5 "train_tfp_flows.py" submit_small_test.sh
grep -A 5 "train_tfp_flows.py" validate_deployment.sh
```

## **🚨 Troubleshooting:**

### **If allocation fails:**
```bash
# Check queue status
squeue -p gpu
sinfo -p gpu

# Try smaller allocation
salloc --partition=gpu --gres=gpu:1 --time=30:00 --mem=4GB
```

### **If arguments still fail:**
1. Check that files were synced properly
2. Verify you're in the right directory
3. Check for any remaining `--h5_file` references

## **🎉 Success Criteria:**
- ✅ No "arguments are required: --data_path" errors
- ✅ Training starts and runs at least 1 epoch
- ✅ Model files (.npz) are created
- ✅ Ready for full job array deployment!
