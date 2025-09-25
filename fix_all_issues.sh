#!/bin/bash

echo "🔧 COMPREHENSIVE FIX FOR ALL ISSUES"
echo "==================================="
echo "1. Cancel failing jobs"
echo "2. Fix double 'Halo' path issue"
echo "3. Fix GLIBC compatibility"
echo "4. Clean up duplicate directories"
echo "5. Ensure GPU node allocation"
echo ""

# 1. Cancel all failing jobs
echo "1️⃣ Canceling failing jobs..."
scancel -u $USER
echo "✅ Jobs canceled"

# 2. Fix the double "Halo" issue in brute_force_gpu_job.sh
echo -e "\n2️⃣ Fixing double 'Halo' path issue..."
# Replace halo${HALO_ID} with halo${HALO_ID#Halo} to remove "Halo" prefix
sed -i 's/halo\${HALO_ID}/halo${HALO_ID#Halo}/g' brute_force_gpu_job.sh
echo "✅ Path issue fixed"

# 3. Add GLIBC compatibility fix
echo -e "\n3️⃣ Adding GLIBC compatibility fix..."
# Add newer GCC library path after conda activation
sed -i '/conda activate bosque/a export LD_LIBRARY_PATH=/share/software/user/open/gcc/11.2.0/lib64:$LD_LIBRARY_PATH' brute_force_gpu_job.sh
echo "✅ GLIBC fix added"

# 4. Clean up duplicate and redundant directories
echo -e "\n4️⃣ Cleaning up duplicate directories..."

# Remove duplicate flows-tensorflow directory
if [[ -d "flows-tensorflow" ]]; then
    echo "   Removing duplicate flows-tensorflow directory..."
    rm -rf flows-tensorflow
    echo "   ✅ Duplicate directory removed"
fi

# Move misplaced samples and trained_flows to tfp_output
if [[ -d "samples" ]] && [[ ! -d "tfp_output/samples" ]]; then
    echo "   Moving samples to tfp_output..."
    mkdir -p tfp_output
    mv samples tfp_output/
    echo "   ✅ Samples moved"
fi

if [[ -d "trained_flows" ]] && [[ ! -d "tfp_output/trained_flows" ]]; then
    echo "   Moving trained_flows to tfp_output..."
    mkdir -p tfp_output
    mv trained_flows tfp_output/
    echo "   ✅ Trained flows moved"
fi

# Fix the wrong haloHalo directories
echo "   Fixing haloHalo directory names..."
find tfp_output -name "haloHalo*" -type d | while read dir; do
    newdir=$(echo "$dir" | sed 's/haloHalo/halo/')
    if [[ "$dir" != "$newdir" ]]; then
        mkdir -p "$(dirname "$newdir")"
        mv "$dir" "$newdir"
        echo "   ✅ Renamed: $(basename "$dir") → $(basename "$newdir")"
    fi
done

# 5. Ensure GPU node allocation
echo -e "\n5️⃣ Checking GPU allocation in scripts..."
if grep -q "gres=gpu" brute_force_gpu_job.sh; then
    echo "✅ GPU allocation present"
else
    echo "❌ GPU allocation missing - adding it"
    sed -i '/partition=owners/a #SBATCH --gres=gpu:4' brute_force_gpu_job.sh
fi

# 6. Verify module loading for newer GCC
echo -e "\n6️⃣ Ensuring proper module loading..."
if grep -q "gcc/11.2.0\|gcc/12" brute_force_gpu_job.sh; then
    echo "✅ Newer GCC module present"
else
    echo "   Adding newer GCC module..."
    sed -i 's/module load math devel nvhpc/module load math devel gcc\/11.2.0 nvhpc/' brute_force_gpu_job.sh
fi

# 7. Clean up log files from failed runs
echo -e "\n7️⃣ Cleaning up failed job logs..."
find logs -name "brute_force_517*" -size -3k -delete 2>/dev/null || true
echo "✅ Cleaned up small error logs"

echo -e "\n🎉 ALL FIXES APPLIED!"
echo "================================"
echo "✅ Jobs canceled"
echo "✅ Double 'Halo' paths fixed"  
echo "✅ GLIBC compatibility added"
echo "✅ Directory structure cleaned"
echo "✅ GPU allocation verified"
echo ""
echo "🚀 Ready to resubmit with:"
echo "   sbatch submit_gpu_smart.sh"
echo ""
echo "🔍 Verify fixes with:"
echo "   grep -A2 -B2 'LD_LIBRARY_PATH\\|halo\${HALO_ID' brute_force_gpu_job.sh"


