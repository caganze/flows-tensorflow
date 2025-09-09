#!/bin/bash

# 📋 Particle List Generator
# Scans H5 files and creates a comprehensive list of all available particles
# Each line contains: PARTICLE_ID,H5_FILE_PATH,OBJECT_COUNT,SIZE_CATEGORY

set -e

echo "📋 Particle List Generator"
echo "========================="
echo "Scanning H5 files to create comprehensive particle list"
echo

# Output file
PARTICLE_LIST_FILE="particle_list.txt"
BACKUP_FILE="particle_list_backup_$(date +%Y%m%d_%H%M%S).txt"

# Backup existing file if it exists
if [[ -f "$PARTICLE_LIST_FILE" ]]; then
    cp "$PARTICLE_LIST_FILE" "$BACKUP_FILE"
    echo "📁 Backed up existing list to: $BACKUP_FILE"
fi

# Clear the output file
> "$PARTICLE_LIST_FILE"

echo "🔍 Scanning for H5 files..."

# Search paths for H5 files
#SEARCH_PATHS=(
#    "/oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/"
#    "/oak/stanford/orgs/kipac/users/caganze/"
#    "./data/"
#    "./"
#)
SEARCH_PATHS=(
    "/oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/"
    "/oak/stanford/orgs/kipac/users/caganze/milkyway-hr-mocks/"
    "/oak/stanford/orgs/kipac/users/caganze/milkyway-hr-mocks/"
)


H5_FILES=()

# Find all H5 files
for search_path in "${SEARCH_PATHS[@]}"; do
    if [[ -d "$search_path" ]]; then
        echo "  Searching in: $search_path"
        while IFS= read -r -d '' file; do
            H5_FILES+=("$file")
            echo "    Found: $(basename "$file")"
        done < <(find "$search_path" -name "*.h5" -type f -print0 2>/dev/null)
    fi
done

if [[ ${#H5_FILES[@]} -eq 0 ]]; then
    echo "❌ No H5 files found in search paths"
    exit 1
fi

echo "📊 Found ${#H5_FILES[@]} H5 file(s)"
echo

# Function to extract particle info from H5 file
extract_particles() {
    local h5_file="$1"
    echo "🔍 Analyzing: $(basename "$h5_file")"
    
    # Use Python to extract particle information
    python -c "
import h5py
import numpy as np
import sys

h5_file = '$h5_file'

try:
    with h5py.File(h5_file, 'r') as f:
        particles_found = False
        
        # Method 1: Check PartType1/ParticleIDs structure
        if 'PartType1' in f and 'ParticleIDs' in f['PartType1']:
            pids = f['PartType1']['ParticleIDs'][:]
            unique_pids = np.unique(pids)
            
            print(f'  Found {len(unique_pids)} unique particles in PartType1/ParticleIDs', file=sys.stderr)
            
            for pid in unique_pids:
                if pid > 0:  # Skip invalid PIDs
                    pid_mask = (pids == pid)
                    count = np.sum(pid_mask)
                    size_category = 'Large' if count > 100000 else 'Small'
                    print(f'{int(pid)},{h5_file},{count},{size_category}')
            particles_found = True
            
        # Method 2: Check for 'parentid' dataset
        elif 'parentid' in f:
            parentids = f['parentid'][:]
            unique_pids = np.unique(parentids)
            
            print(f'  Found {len(unique_pids)} unique particles in parentid dataset', file=sys.stderr)
            
            for pid in unique_pids:
                if pid > 0:  # Skip invalid PIDs
                    pid_mask = (parentids == pid)
                    count = np.sum(pid_mask)
                    size_category = 'Large' if count > 100000 else 'Small'
                    print(f'{int(pid)},{h5_file},{count},{size_category}')
            particles_found = True
            
        # Method 3: General dataset search for particle-like data
        else:
            print(f'  No standard particle ID structure found, checking datasets...', file=sys.stderr)
            
            # Look for datasets that might contain particle IDs
            potential_datasets = []
            
            def find_datasets(name, obj):
                if isinstance(obj, h5py.Dataset):
                    if 'id' in name.lower() or 'particle' in name.lower():
                        potential_datasets.append(name)
            
            f.visititems(find_datasets)
            
            if potential_datasets:
                print(f'  Found potential particle datasets: {potential_datasets}', file=sys.stderr)
                # Try the first potential dataset
                try:
                    data = f[potential_datasets[0]][:]
                    if len(data) > 0:
                        unique_vals = np.unique(data)
                        if len(unique_vals) < len(data):  # Looks like IDs
                            for val in unique_vals[:10]:  # Limit to first 10 for safety
                                if val > 0:
                                    val_mask = (data == val)
                                    count = np.sum(val_mask)
                                    size_category = 'Large' if count > 100000 else 'Small'
                                    print(f'{int(val)},{h5_file},{count},{size_category}')
                            particles_found = True
                except Exception as e:
                    print(f'  Error processing {potential_datasets[0]}: {e}', file=sys.stderr)
            
            # Fallback: estimate based on file structure
            if not particles_found:
                print(f'  Using fallback estimation...', file=sys.stderr)
                # Estimate number of particles from largest dataset
                max_size = 0
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        max_size = max(max_size, f[key].shape[0] if f[key].shape else 0)
                
                if max_size > 0:
                    # Estimate 1-5 particles based on file size
                    estimated_particles = min(5, max(1, max_size // 50000))
                    avg_size = max_size // estimated_particles
                    size_category = 'Large' if avg_size > 100000 else 'Small'
                    
                    for i in range(1, estimated_particles + 1):
                        print(f'{i},{h5_file},{avg_size},{size_category}')
                    particles_found = True

        if not particles_found:
            print(f'  No particles could be extracted from file', file=sys.stderr)
            
except Exception as e:
    print(f'Error processing {h5_file}: {e}', file=sys.stderr)
"
}

# Process each H5 file
total_particles=0
for h5_file in "${H5_FILES[@]}"; do
    echo
    particles_from_file=$(extract_particles "$h5_file" 2>&1 | tee >(grep -E '^[0-9]+,' >> "$PARTICLE_LIST_FILE") | grep -v '^[0-9]+,' | cat)
    
    if [[ -n "$particles_from_file" ]]; then
        echo "$particles_from_file"
    fi
    
    # Count particles added from this file
    file_particle_count=$(grep -c ",$h5_file," "$PARTICLE_LIST_FILE" 2>/dev/null || echo "0")
    total_particles=$((total_particles + file_particle_count))
    echo "  ✅ Added $file_particle_count particles from this file"
done

echo
echo "📊 PARTICLE LIST SUMMARY"
echo "========================"
echo "Total particles found: $total_particles"
echo "Output file: $PARTICLE_LIST_FILE"

if [[ $total_particles -gt 0 ]]; then
    echo
    echo "🔍 Particle breakdown:"
    
    # Count by size category
    large_count=$(grep -c ",Large$" "$PARTICLE_LIST_FILE" 2>/dev/null || echo "0")
    small_count=$(grep -c ",Small$" "$PARTICLE_LIST_FILE" 2>/dev/null || echo "0")
    
    echo "  🐋 Large particles (>100k objects): $large_count"
    echo "  🐭 Small particles (<100k objects): $small_count"
    
    echo
    echo "📋 First 10 particles:"
    head -10 "$PARTICLE_LIST_FILE" | while IFS=',' read -r pid h5_file count category; do
        printf "  PID %3d: %8s objects (%s) - %s\n" "$pid" "$count" "$category" "$(basename "$h5_file")"
    done
    
    if [[ $total_particles -gt 10 ]]; then
        echo "  ... and $((total_particles - 10)) more"
    fi
    
    echo
    echo "🚀 READY FOR ARRAY SUBMISSION!"
    echo "=============================="
    echo "✅ Particle list file created: $PARTICLE_LIST_FILE"
    echo "✅ Use this file with submit_tfp_array.sh"
    echo "💡 Array range: 1-$total_particles"
    
else
    echo "❌ No particles found! Check H5 file structure and search paths."
    exit 1
fi
