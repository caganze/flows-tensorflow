#!/bin/bash

# 🧹 Cleanup Script for Symlib Migration
# Removes obsolete H5-based files and one-time scripts

echo "🧹 CLEANING UP FOR SYMLIB MIGRATION"
echo "===================================="
echo "This will remove obsolete H5-based and one-time scripts"
echo ""

# Files to delete
FILES_TO_DELETE=(
    # H5 File Processing (Obsolete)
    "create_example_data.py"
    "debug_h5_structure.py" 
    "test_h5_read_single_particle.py"
    "generate_particle_list.sh"
    "improved_particle_detection.py"
    
    # Legacy/Deprecated Scripts
    "analyze_kroupa_sampling.py"
    "comprehensive_consistency_check.py"
    "fix_fstring_error.py"
    "final_fstring_fix.py"
    "fix_remaining_scripts.py"
    "manual_fstring_fix.py"
    "generate_parallel_scripts.py"
    "migrate_output_structure.sh"
    
    # Old Testing/Debugging
    "audit_shell_scripts.py"
    "trace_execution_paths.py"
    "track_failures.py"
    "test_multiple_particles.py"
    "verify_no_jax.py"
)

echo "📋 Files to be deleted:"
for file in "${FILES_TO_DELETE[@]}"; do
    if [[ -f "$file" ]]; then
        echo "  ❌ $file"
    else
        echo "  ⚠️  $file (not found)"
    fi
done

echo ""
read -p "Proceed with deletion? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Deleting obsolete files..."
    for file in "${FILES_TO_DELETE[@]}"; do
        if [[ -f "$file" ]]; then
            rm "$file"
            echo "  ✅ Deleted $file"
        fi
    done
    
    echo ""
    echo "✅ Cleanup completed!"
    echo "📊 Next steps:"
    echo "  1. Create generate_symlib_particle_list.py"
    echo "  2. Update train_tfp_flows.py for symlib"
    echo "  3. Update all submission scripts"
else
    echo "❌ Cleanup cancelled"
fi
