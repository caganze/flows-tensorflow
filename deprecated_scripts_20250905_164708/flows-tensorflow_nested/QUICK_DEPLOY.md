# 🚀 Quick Deployment to Sherlock

## One-Command Sync
```bash
# From your local flows-tensorflow directory:
rsync -av * caganze@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/
```

## Then on Sherlock:
```bash
ssh caganze@login.sherlock.stanford.edu
cd /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow
./deploy_to_sherlock.sh          # Setup environment
sbatch test_single_job.sh        # Test (optional)
sbatch brute_force_gpu_job.sh    # Submit main job
./monitor_brute_force.sh --live  # Monitor progress
```

## Files Synced:
- ✅ brute_force_gpu_job.sh - Main SLURM job
- ✅ monitor_brute_force.sh - Progress monitor  
- ✅ deploy_to_sherlock.sh - Environment setup
- ✅ BRUTE_FORCE_USAGE_GUIDE.md - Full documentation
- ✅ All other project files

**Total**: All halos × all PIDs will be processed with full error tracking and success validation!
