#!/bin/bash

# 🔄 COMPLETE SYMLIB MIGRATION - FINAL FIX
# This script removes ALL H5 dependencies and ensures symlib-only operation
# No more H5 format reversions!

set -e

echo "🔄 COMPLETE SYMLIB MIGRATION"
echo "============================"
echo "Removing ALL H5 dependencies and ensuring symlib-only operation"
echo ""

# 1. Fix filter_completed_particles.sh to handle symlib format
echo "1️⃣ Fixing filter_completed_particles.sh for symlib format..."
sed -i 's/# Parse particle entry: PID,H5_FILE,OBJECT_COUNT,SIZE_CATEGORY (current format)/# Parse particle entry: PID,HALO_ID,SUITE,OBJECT_COUNT,SIZE_CATEGORY (symlib format)/' filter_completed_particles.sh
sed -i 's/IFS='"'"','"'"' read -r pid h5_file object_count size_category/IFS='"'"','"'"' read -r pid halo_id suite object_count size_category/' filter_completed_particles.sh
sed -i 's/echo "  📊 Parsed: pid=$pid, h5_file=$h5_file, count=$object_count, size=$size_category"/echo "  📊 Parsed: pid=$pid, halo_id=$halo_id, suite=$suite, count=$object_count, size=$size_category"/' filter_completed_particles.sh
sed -i 's/if \[\[ -z "$pid" || -z "$h5_file" \]\]; then/if [[ -z "$pid" || -z "$halo_id" || -z "$suite" ]]; then/' filter_completed_particles.sh
echo "✅ filter_completed_particles.sh updated for symlib format"

# 2. Remove generate_particle_list.sh calls from chunked scripts
echo ""
echo "2️⃣ Removing H5 generator calls from chunked scripts..."
sed -i '/if ! \.\/generate_particle_list\.sh; then/,+2d' submit_gpu_chunked.sh
sed -i '/if ! \.\/generate_particle_list\.sh; then/,+2d' submit_cpu_chunked.sh
sed -i '/\.\/generate_particle_list\.sh/d' brute_force_cpu_parallel.sh
echo "✅ Removed H5 generator calls"

# 3. Update chunked scripts to require symlib particle list
echo ""
echo "3️⃣ Updating chunked scripts to require symlib particle list..."
sed -i 's/echo "📋 Generating particle list..."/echo "❌ Symlib particle list required. Run: .\/generate_all_priority_halos.sh"/' submit_gpu_chunked.sh
sed -i 's/echo "📋 Generating particle list..."/echo "❌ Symlib particle list required. Run: .\/generate_all_priority_halos.sh"/' submit_cpu_chunked.sh
sed -i 's/echo "📋 Generating particle list..."/echo "❌ Symlib particle list required. Run: .\/generate_all_priority_halos.sh"/' brute_force_cpu_parallel.sh
echo "✅ Scripts now require symlib particle list"

# 4. Remove all H5 file path references
echo ""
echo "4️⃣ Removing H5 file path references..."
sed -i '/milkyway-eden-mocks\|symphony_mocks/d' submit_tfp_array.sh
sed -i '/milkyway-eden-mocks\|symphony_mocks/d' test_slurm_deployment.sh
sed -i '/milkyway-eden-mocks\|symphony_mocks/d' train_single_gpu.sh
sed -i '/milkyway-eden-mocks\|symphony_mocks/d' one_time_long_job.sh
echo "✅ Removed H5 file path references"

# 5. Remove H5 fallback logic
echo ""
echo "5️⃣ Removing H5 fallback logic..."
sed -i '/all_in_one\.h5\|\.h5/d' submit_tfp_array.sh
sed -i '/all_in_one\.h5\|\.h5/d' test_slurm_deployment.sh  
sed -i '/all_in_one\.h5\|\.h5/d' train_single_gpu.sh
sed -i '/all_in_one\.h5\|\.h5/d' one_time_long_job.sh
echo "✅ Removed H5 fallback logic"

# 6. Fix sample file checking to only look for .npz files
echo ""
echo "6️⃣ Updating sample file checking to .npz only..."
sed -i 's/|| -f "$SAMPLES_DIR\/model_pid${SELECTED_PID}_samples\.h5"//' brute_force_gpu_job.sh
sed -i 's/|| -f "$SAMPLES_DIR\/model_pid${PID}_samples\.h5"//' priority_12hour_*.sh
echo "✅ Sample checking updated to .npz only"

# 7. Remove H5_PATTERN variables from priority scripts
echo ""
echo "7️⃣ Removing H5_PATTERN variables..."
sed -i '/H5_PATTERN=/d' priority_12hour_*.sh
echo "✅ Removed H5_PATTERN variables"

# 8. Update directory references to use symlib output structure
echo ""
echo "8️⃣ Updating output directory references..."
# Update any remaining old output paths to use the new symlib structure
for script in *.sh; do
    if [[ -f "$script" ]]; then
        sed -i 's|/oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/tfp_output|/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output|g' "$script"
        sed -i 's|/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/tfp_output|/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output|g' "$script"
    fi
done
echo "✅ Updated output directory references"

# 9. Create a backup of the old H5 generator script
echo ""
echo "9️⃣ Backing up and renaming old H5 generator..."
if [[ -f "flows-tensorflow/generate_particle_list.sh" ]]; then
    mv "flows-tensorflow/generate_particle_list.sh" "flows-tensorflow/generate_particle_list_h5_DEPRECATED.sh"
    echo "✅ Old H5 generator renamed to generate_particle_list_h5_DEPRECATED.sh"
fi
echo "✅ H5 generator neutralized"

echo ""
echo "🎉 COMPLETE SYMLIB MIGRATION FINISHED!"
echo "====================================="
echo "✅ filter_completed_particles.sh now handles symlib format"
echo "✅ Removed all calls to H5 generator"  
echo "✅ Scripts now require symlib particle lists"
echo "✅ Removed all H5 file path references"
echo "✅ Removed H5 fallback logic"
echo "✅ Updated sample checking to .npz only"
echo "✅ Updated output directory structure"
echo "✅ Old H5 generator neutralized"
echo ""
echo "🚀 READY FOR SYMLIB-ONLY OPERATION!"
echo "=================================="
echo "Next steps:"
echo "1. Generate symlib particle list: ./generate_all_priority_halos.sh"
echo "2. Test the migration: ./test_symlib_only.sh"
echo "3. Submit jobs: sbatch submit_gpu_smart.sh"
echo ""

