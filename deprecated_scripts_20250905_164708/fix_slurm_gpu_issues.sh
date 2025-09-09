#!/bin/bash

# 🔧 COMPREHENSIVE SLURM GPU SUBMISSION FIXES
# This script fixes all GPU submission issues:
# 1. Updates --gres=gpu syntax to --gpus
# 2. Adds --qos=owners for maximum resources  
# 3. Ensures proper partition usage
# 4. Provides commands to check GPU limits

set -e

echo "🔧 SLURM GPU SUBMISSION FIXES"
echo "============================="
echo "Fixing:"
echo "1. GPU request syntax (--gres=gpu → --gpus)"
echo "2. Adding --qos=owners for maximum resources"
echo "3. Ensuring proper partition usage"
echo ""

# 1. Fix submit_gpu_chunked.sh: Update GPU request syntax and add QOS
echo "1️⃣ Fixing submit_gpu_chunked.sh..."

# Replace --gres=gpu with --gpus in dry run display
sed -i 's/--gres=gpu:\$GPUS/--gpus=\$GPUS/' submit_gpu_chunked.sh

# Replace --gres=gpu with --gpus in actual sbatch command and add --qos
sed -i '/--gres=gpu:\$GPUS \\/c\            --gpus=$GPUS \\' submit_gpu_chunked.sh

# Add --qos=owners after --partition line
if ! grep -q "qos=owners" submit_gpu_chunked.sh; then
    sed -i '/--partition=\$PARTITION \\/a\            --qos=owners \\' submit_gpu_chunked.sh
fi

echo "✅ submit_gpu_chunked.sh updated"

# 2. Check and fix brute_force_gpu_job.sh SLURM directives
echo ""
echo "2️⃣ Checking brute_force_gpu_job.sh..."

# Ensure it has proper SLURM directives for GPU jobs
if ! grep -q "#SBATCH --gpus" brute_force_gpu_job.sh; then
    # Add --gpus directive if not present
    if grep -q "#SBATCH --partition=owners" brute_force_gpu_job.sh; then
        sed -i '/#SBATCH --partition=owners/a #SBATCH --gpus=4' brute_force_gpu_job.sh
    fi
fi

# Add --qos=owners if not present
if ! grep -q "#SBATCH --qos=owners" brute_force_gpu_job.sh; then
    if grep -q "#SBATCH --partition=owners" brute_force_gpu_job.sh; then
        sed -i '/#SBATCH --partition=owners/a #SBATCH --qos=owners' brute_force_gpu_job.sh
    fi
fi

echo "✅ brute_force_gpu_job.sh checked"

# 3. Verify the fixes
echo ""
echo "3️⃣ Verifying fixes..."

echo "📋 submit_gpu_chunked.sh GPU syntax:"
grep -n "gpus\|qos" submit_gpu_chunked.sh || echo "  No GPU/QOS directives found"

echo ""
echo "📋 brute_force_gpu_job.sh SLURM directives:"
grep -n "#SBATCH.*\(partition\|gpus\|qos\)" brute_force_gpu_job.sh || echo "  No relevant SLURM directives found"

echo ""
echo "🎉 ALL SLURM FIXES APPLIED!"
echo "=========================="
echo "✅ Updated --gres=gpu syntax to --gpus"
echo "✅ Added --qos=owners for maximum resources"
echo "✅ Verified SLURM directives"
echo ""

# 4. Provide commands to check GPU limits
echo "🔍 CHECK YOUR GPU LIMITS:"
echo "========================"
echo "1. Check available partitions and GPU limits:"
echo "   sh_part"
echo ""
echo "2. Check your account limits:"
echo "   sacctmgr show user \$USER -s"
echo ""
echo "3. Check association limits (includes GPU limits):"
echo "   sacctmgr show assoc user=\$USER -s"
echo ""
echo "4. See GPU nodes in owners partition:"
echo "   sinfo -p owners -o \"%N %c %m %f %G %T\""
echo ""
echo "🚀 READY TO SUBMIT:"
echo "==================="
echo "Submit with explicit settings:"
echo "   sbatch submit_gpu_smart.sh --partition owners --gpus 4"
echo ""
echo "Or use defaults (owners partition, QOS enabled):"
echo "   sbatch submit_gpu_smart.sh"
echo ""
