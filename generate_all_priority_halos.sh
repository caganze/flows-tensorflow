#!/bin/bash

# Simple script to generate particle lists for all priority halos
# Uses the existing working generate_symlib_particle_list.py script

# Removed set -e to allow script to continue if individual halos fail
set -u  # Exit on undefined variables

echo "🎯 GENERATING PRIORITY HALO PARTICLE LISTS"
echo "=========================================="
echo "Using existing working script: generate_symlib_particle_list.py"
echo "Priority halos: 718, 939, 925"
echo ""

# Priority halos
PRIORITY_HALOS=("718" "939" "925")
SUITES=("eden" "symphony")

# Backup existing particle list
if [[ -f "particle_list.txt" ]]; then
    cp particle_list.txt particle_list_backup.txt
    echo "📋 Backed up existing particle_list.txt"
fi

# Clear the main file to start fresh
> particle_list.txt

TOTAL_SUCCESS=0
TOTAL_FAILED=0

# Generate for each halo+suite combination
for HALO in "${PRIORITY_HALOS[@]}"; do
    for SUITE in "${SUITES[@]}"; do
        echo ""
        echo "🔧 Processing: $SUITE Halo$HALO"
        echo "================================"
        
        # Create temporary output file
        TEMP_FILE="temp_${SUITE}_halo${HALO}.txt"
        
        # Try to generate particle list
        if python generate_symlib_particle_list.py "Halo${HALO}" --suite "$SUITE" --output "$TEMP_FILE"; then
            echo "✅ Generated: $SUITE Halo$HALO"
            
            # Append to main particle list if file was created and is not empty
            if [[ -f "$TEMP_FILE" ]] && [[ -s "$TEMP_FILE" ]]; then
                cat "$TEMP_FILE" >> particle_list.txt
                echo "   Added $(wc -l < "$TEMP_FILE") particles to main list"
                ((TOTAL_SUCCESS++))
                rm "$TEMP_FILE"
            else
                echo "   ⚠️  File empty or not created"
                ((TOTAL_FAILED++))
            fi
        else
            echo "❌ Failed: $SUITE Halo$HALO"
            ((TOTAL_FAILED++))
            # Clean up temp file if it exists
            [[ -f "$TEMP_FILE" ]] && rm "$TEMP_FILE"
        fi
    done
done

echo ""
echo "🏁 SUMMARY"
echo "==========="
echo "✅ Successful: $TOTAL_SUCCESS halo+suite combinations"
echo "❌ Failed: $TOTAL_FAILED halo+suite combinations"

if [[ -f "particle_list.txt" ]] && [[ -s "particle_list.txt" ]]; then
    TOTAL_PARTICLES=$(wc -l < particle_list.txt)
    echo "📊 Total particles in list: $TOTAL_PARTICLES"
    
    echo ""
    echo "📋 Particle list breakdown:"
    echo "Suite breakdown:"
    cut -d',' -f3 particle_list.txt | sort | uniq -c
    echo ""
    echo "Halo breakdown:"
    cut -d',' -f2 particle_list.txt | sort | uniq -c
    echo ""
    echo "Size breakdown:"
    cut -d',' -f5 particle_list.txt | sort | uniq -c
    
    echo ""
    echo "🎉 SUCCESS! Priority halo particle list ready"
    echo "📁 File: particle_list.txt"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. python test_priority_particles.py"
    echo "   2. sbatch brute_force_gpu_job.sh"
else
    echo ""
    echo "❌ No particles generated - all halos failed"
    echo "⚠️  Check if priority halos exist in symlib"
    
    # Restore backup if we failed completely
    if [[ -f "particle_list_backup.txt" ]]; then
        mv particle_list_backup.txt particle_list.txt
        echo "🔄 Restored backup particle list"
    fi
fi
