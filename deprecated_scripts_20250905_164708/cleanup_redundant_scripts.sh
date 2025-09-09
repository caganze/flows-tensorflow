#!/bin/bash

# 🗑️ Clean up redundant shell scripts
# Based on audit results - keeping only essential scripts

set -e

echo "🗑️ CLEANING UP REDUNDANT SHELL SCRIPTS"
echo "======================================"
echo "This will remove 27 redundant scripts, keeping only 10 essential ones"
echo ""

# Essential scripts to KEEP (DO NOT DELETE)
ESSENTIAL_SCRIPTS=(
    "brute_force_cpu_parallel.sh"
    "brute_force_gpu_job.sh" 
    "filter_completed_particles.sh"
    "generate_particle_list.sh"
    "kroupa_samples.sh"
    "meta_test_full_pipeline.sh"
    "run_comprehensive_gpu_test.sh"
    "submit_cpu_smart.sh"
    "submit_tfp_array.sh"
    "test_pipeline_robustness.sh"
)

# Scripts to remove (redundant)
REDUNDANT_SCRIPTS=(
    "check_completion_status.sh"
    "check_sherlock_paths.sh"
    "create_working_gpu_env.sh"
    "deploy_to_sherlock.sh"
    "fix_cuda_paths.sh"
    "job_summary.sh"
    "migrate_output_structure.sh"
    "monitor_brute_force.sh"
    "monitor_cpu_parallel.sh"
    "one_time_long_job.sh"
    "quick_interactive_test.sh"
    "quick_test_tfp.sh"
    "scan_and_resubmit.sh"
    "setup_git_repo.sh"
    "setup_sherlock.sh"
    "submit_cpu_chunked.sh"
    "submit_cpu_parallel.sh"
    "test_advanced_robustness.sh"
    "test_edge_cases.sh"
    "test_file_search_patterns.sh"
    "test_kroupa_samples.sh"
    "test_slurm_deployment.sh"
    "test_submit_tfp_array.sh"
    "train_single_gpu.sh"
    "validate_deployment.sh"
    "verify_chunking_logic.sh"
    "verify_fixes.sh"
)

# Interactive confirmation
echo "📋 ESSENTIAL SCRIPTS (will be kept):"
for script in "${ESSENTIAL_SCRIPTS[@]}"; do
    if [[ -f "$script" ]]; then
        echo "  ✅ $script"
    else
        echo "  ❌ $script (not found)"
    fi
done

echo ""
echo "🗑️ REDUNDANT SCRIPTS (will be removed):"
for script in "${REDUNDANT_SCRIPTS[@]}"; do
    if [[ -f "$script" ]]; then
        echo "  🗑️ $script"
    else
        echo "  ❓ $script (already missing)"
    fi
done

echo ""
read -p "❓ Continue with cleanup? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled"
    exit 0
fi

# Create backup directory
BACKUP_DIR="redundant_scripts_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo ""
echo "🔄 Creating backup and removing redundant scripts..."

REMOVED_COUNT=0
BACKED_UP_COUNT=0

for script in "${REDUNDANT_SCRIPTS[@]}"; do
    if [[ -f "$script" ]]; then
        # Move to backup first
        cp "$script" "$BACKUP_DIR/"
        ((BACKED_UP_COUNT++))
        
        # Remove original
        rm "$script"
        ((REMOVED_COUNT++))
        echo "  🗑️ Removed: $script (backed up)"
    fi
done

echo ""
echo "✅ CLEANUP COMPLETE"
echo "=================="
echo "  📁 Backup directory: $BACKUP_DIR"
echo "  🗑️ Scripts removed: $REMOVED_COUNT"
echo "  💾 Scripts backed up: $BACKED_UP_COUNT"
echo "  ✅ Essential scripts preserved: ${#ESSENTIAL_SCRIPTS[@]}"

echo ""
echo "📋 REMAINING SCRIPTS:"
ls -1 *.sh | sort

echo ""
echo "🎯 Next steps:"
echo "  1. Test essential scripts still work"
echo "  2. Remove backup directory if satisfied: rm -rf $BACKUP_DIR"
echo "  3. Commit changes to git"

