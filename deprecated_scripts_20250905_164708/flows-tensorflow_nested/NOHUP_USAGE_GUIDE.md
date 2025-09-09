# 🚀 NOHUP Auto-Submission Guide

## **Quick Start (Recommended):**

```bash
# On Sherlock login node:
cd /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow

# Start auto-submission in background:
./run_auto_submit_nohup.sh

# Disconnect from Sherlock - it keeps running!
# Check back later with:
tail -f logs/auto_submit_YYYYMMDD_HHMMSS.log
```

## **Manual nohup (Alternative):**

```bash
# Direct nohup command:
nohup ./auto_submit_flows.sh > logs/auto_submit_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Get process ID:
echo $! 

# Save PID for later:
echo $! > auto_submit.pid
```

## **🔍 Monitoring Commands:**

### **Check if still running:**
```bash
ps aux | grep auto_submit_flows
# OR
pgrep -f auto_submit_flows
```

### **Watch live progress:**
```bash
# Find the latest log file:
ls -la logs/auto_submit_*.log

# Monitor live:
tail -f logs/auto_submit_YYYYMMDD_HHMMSS.log
```

### **Check queue status:**
```bash
squeue --me
watch 'squeue --me'
```

## **🛑 Stopping the Process:**

### **Stop by name:**
```bash
pkill -f auto_submit_flows.sh
```

### **Stop by PID:**
```bash
# If you saved the PID:
kill $(cat auto_submit.pid)

# Or find and kill:
ps aux | grep auto_submit_flows
kill [PID]
```

## **📊 Expected Behavior:**

### **The script will:**
- ✅ **Run continuously** in the background
- ✅ **Submit jobs in batches** (respecting SLURM limits)
- ✅ **Wait intelligently** when hitting submission limits
- ✅ **Resume automatically** when limits reset
- ✅ **Log all activity** to timestamped files
- ✅ **Continue even if you disconnect** from Sherlock

### **Typical workflow:**
1. **Start:** Submit first batch of jobs
2. **Monitor:** Check queue status, wait for slots
3. **Submit:** New batches as capacity becomes available  
4. **Repeat:** Until all PIDs are processed

## **🎯 Success Indicators:**

### **In the logs:**
```
✅ Found X completed particles
🎯 Needed particles: Y
🚀 Submitting batch: PIDs A-B
📊 Queue status: C running, D pending
⏰ Waiting for submission limits to reset...
```

### **In queue status:**
```bash
squeue --me
# Should show jobs actively running/pending
```

## **⚠️ Troubleshooting:**

### **If process stops unexpectedly:**
```bash
# Check the log for errors:
tail -20 logs/auto_submit_*.log

# Common issues:
# - SLURM submission limits exceeded (normal, script waits)
# - File permissions issues (check output directories)
# - Network connectivity issues (restart)
```

### **If no new jobs are submitting:**
```bash
# Check if all work is done:
grep "needed particles" logs/auto_submit_*.log | tail -1

# If it says "0 needed particles" - all done! ✅
```

## **🎉 Expected Timeline:**

With the improved 95%+ success rate from argument fixes:
- **~478 total particles** to process
- **~5-20 jobs concurrent** (depending on limits)
- **~2-6 hours per large halo** (with 12-hour time limits)
- **~1-2 days total** for complete processing

**The auto-submission handles everything automatically!** 🚀
