#!/bin/bash

# 🔍 Debug Array Job Issues
# Check what happened with your SLURM array job

echo "🔍 DEBUGGING ARRAY JOB ISSUES"
echo "============================="

echo "📋 Commands to run on Sherlock to debug:"
echo ""

echo "1️⃣ CHECK RECENT SLURM JOBS:"
echo "   squeue -u \$USER"
echo "   sacct -u \$USER --starttime=today --format=JobID,JobName,State,ExitCode,Elapsed"
echo ""

echo "2️⃣ CHECK ARRAY JOB LOGS:"
echo "   ls -la logs/tfp_*"
echo "   ls -la logs/slurm_*"
echo ""

echo "3️⃣ CHECK SPECIFIC ARRAY TASK LOGS:"
echo "   # If your job ID was XXXXX, check individual tasks:"
echo "   cat logs/tfp_XXXXX_1.out   # Array task 1"
echo "   cat logs/tfp_XXXXX_2.out   # Array task 2"
echo "   cat logs/tfp_XXXXX_1.err   # Check for errors"
echo ""

echo "4️⃣ CHECK OUTPUT STRUCTURE:"
echo "   ls -la /oak/stanford/orgs/kipac/users/\$USER/milkyway-eden-mocks/tfp_output/trained_flows/"
echo "   ls -la /oak/stanford/orgs/kipac/users/\$USER/milkyway-eden-mocks/tfp_output/samples/"
echo ""

echo "5️⃣ TEST SINGLE PARTICLE MANUALLY:"
echo "   # Test if PID 2 works:"
echo "   python train_tfp_flows.py \\"
echo "       --data_path /path/to/halo023.h5 \\"
echo "       --particle_pid 2 \\"
echo "       --output_dir test_output \\"
echo "       --epochs 5 \\"
echo "       --batch_size 512 \\"
echo "       --generate-samples"
echo ""

echo "6️⃣ CHECK PARTICLE LIST:"
echo "   head -10 particle_list.txt"
echo "   wc -l particle_list.txt"
echo ""

echo "🎯 LIKELY ISSUES:"
echo "================"
echo "❌ Array tasks 2-1000 failed or were cancelled"
echo "❌ PID mapping logic error (only processing PID 1)"
echo "❌ Resource limits hit (memory/time)"
echo "❌ File access issues for other particles"
echo "❌ SLURM QOS limits preventing job submission"
echo ""

echo "💡 QUICK FIX TO TEST:"
echo "===================="
echo "# Submit a small test array job with just 3 tasks:"
echo "sbatch --array=1-3%3 submit_tfp_array.sh"
echo ""
echo "# Or test CPU brute force which processes sequentially:"
echo "./brute_force_cpu_parallel.sh"
echo ""

echo "🔧 TO GET MORE PARTICLES RUNNING:"
echo "================================"
echo "1. Check why other array tasks failed"
echo "2. Fix any resource/path issues"
echo "3. Resubmit with smaller array for testing"
echo "4. Use CPU jobs as backup (they work sequentially)"

