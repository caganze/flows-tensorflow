#!/bin/bash

# 🧹 CLEANUP FOR FRESH SHERLOCK START
# Removes redundant/incorrect scripts, keeps only essential ones
# Prepares for clean rsync to wiped Sherlock directory

set -e

echo "🧹 CLEANUP FOR FRESH SHERLOCK START"
echo "==================================="
echo "Removing redundant scripts, keeping only essentials"
echo ""

# List of ESSENTIAL scripts to KEEP
KEEP_SCRIPTS=(
    # Core submission scripts
    "submit_gpu_smart.sh"
    "submit_cpu_smart.sh" 
    "submit_gpu_chunked.sh"
    "submit_cpu_chunked.sh"
    
    # Core training scripts
    "brute_force_gpu_job.sh"
    "brute_force_cpu_parallel.sh"
    
    # Symlib infrastructure
    "generate_all_priority_halos.sh"
    "filter_completed_particles.sh"
    
    # Essential fixes and utilities
    "complete_symlib_migration.sh"
    "test_symlib_only.sh"
    "fix_all_issues.sh"
    "rsync_to_sherlock.sh"
    
    # Monitoring and resubmission
    "monitor_brute_force.sh"
    "monitor_cpu_parallel.sh"
    "scan_and_resubmit.sh"
    
    # Setup and validation
    "deploy_to_sherlock.sh"
    "validate_deployment.sh"
)

echo "📋 ESSENTIAL SCRIPTS TO KEEP:"
echo "============================="
for script in "${KEEP_SCRIPTS[@]}"; do
    if [[ -f "$script" ]]; then
        echo "  ✅ $script"
    else
        echo "  ❌ $script (missing)"
    fi
done
echo ""

# Create backup directory
BACKUP_DIR="deprecated_scripts_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 MOVING REDUNDANT SCRIPTS TO: $BACKUP_DIR"
echo "============================================="

MOVED_COUNT=0

# Move all .sh files that are NOT in the keep list
for script in *.sh; do
    if [[ -f "$script" ]]; then
        # Check if this script should be kept
        SHOULD_KEEP=false
        for keep_script in "${KEEP_SCRIPTS[@]}"; do
            if [[ "$script" == "$keep_script" ]]; then
                SHOULD_KEEP=true
                break
            fi
        done
        
        # Move redundant scripts
        if [[ "$SHOULD_KEEP" == false ]]; then
            mv "$script" "$BACKUP_DIR/"
            echo "  📦 Moved: $script"
            ((MOVED_COUNT++))
        fi
    fi
done

echo ""
echo "📊 CLEANUP SUMMARY"
echo "=================="
echo "✅ Kept essential scripts: ${#KEEP_SCRIPTS[@]}"
echo "📦 Moved redundant scripts: $MOVED_COUNT"
echo "📁 Backup location: $BACKUP_DIR"
echo ""

# Also clean up the nested flows-tensorflow directory
if [[ -d "flows-tensorflow" ]]; then
    echo "🗂️  Cleaning up nested flows-tensorflow directory..."
    mv "flows-tensorflow" "$BACKUP_DIR/flows-tensorflow_nested"
    echo "  📦 Moved: flows-tensorflow/ → $BACKUP_DIR/flows-tensorflow_nested"
    echo ""
fi

# Show final script list
echo "🎯 FINAL SCRIPT LIST FOR SHERLOCK"
echo "=================================="
echo "Core submission:"
for script in submit_*_smart.sh submit_*_chunked.sh; do
    [[ -f "$script" ]] && echo "  📤 $script"
done

echo ""
echo "Core training:"
for script in brute_force_*.sh; do
    [[ -f "$script" ]] && echo "  🚀 $script"
done

echo ""
echo "Symlib infrastructure:"
for script in generate_all_priority_halos.sh filter_completed_particles.sh; do
    [[ -f "$script" ]] && echo "  🔧 $script"
done

echo ""
echo "Utilities & monitoring:"
for script in monitor_*.sh scan_and_resubmit.sh rsync_to_sherlock.sh; do
    [[ -f "$script" ]] && echo "  📊 $script"
done

echo ""
echo "Setup & fixes:"
for script in complete_symlib_migration.sh test_symlib_only.sh fix_all_issues.sh; do
    [[ -f "$script" ]] && echo "  🔧 $script"
done

echo ""
echo "🎉 CLEANUP COMPLETE!"
echo "==================="
echo "✅ Redundant scripts moved to backup"
echo "✅ Essential scripts ready for Sherlock"
echo "✅ Clean slate prepared"
echo ""
echo "🚀 NEXT STEPS:"
echo "=============="
echo "1. Wipe Sherlock directory"
echo "2. Run: ./rsync_to_sherlock.sh"
echo "3. On Sherlock: ./complete_symlib_migration.sh"
echo "4. On Sherlock: ./test_symlib_only.sh"
echo "5. Submit jobs: sbatch submit_gpu_smart.sh"
echo ""
