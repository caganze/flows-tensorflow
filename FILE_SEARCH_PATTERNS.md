# H5 File Search Patterns Documentation

## Overview

The pipeline uses flexible search patterns to locate H5 files across multiple directories with various naming conventions. This document describes the updated search logic implemented across all scripts.

## Search Pattern Updates

### Before (Restrictive)
```bash
# Old patterns - too restrictive
find /path/to/eden/ -name "eden_scaled_Halo*_particles.h5"
find /path/to/data/ -name '*Halo*_*orig*.h5'
```

### After (Flexible)
```bash
# New patterns - handles all variations
find /path/to/eden/ -name "eden_scaled_Halo*"           # Catches all Eden files
find /path/to/data/ -name '*Halo*.h5'                  # Catches all Halo files
find /path/to/any/ -name "*.h5"                        # Fallback to any H5
```

## Supported File Patterns

### Eden Files
```
eden_scaled_Halo570_sunrot90_0kpc200kpcoriginal_particles.h5
eden_scaled_Halo203_sunrot0_0kpc200kpcoriginal_particles.h5
eden_scaled_Halo203_m_sunrot0_0kpc200kpcoriginal_particles.h5
eden_scaled_Halo088_sunrot45_0kpc200kpcoriginal_particles.h5
```
- Pattern: `eden_scaled_Halo*`
- Captures all sunrot angles (0°, 45°, 90°, 180°, etc.)
- Handles optional parameters like `_m_`
- Works with any distance parameters

### Symphony Files
```
symphony_scaled_Halo088_sunrot0_0kpc200kpcoriginal_particles.h5
symphony_scaled_Halo023_sunrot90_0kpc200kpcoriginal_particles.h5
```
- Pattern: `symphony_scaled_Halo*` or `*Halo*.h5`

### Symphony HR Files
```
symphonyHR_scaled_Halo188_sunrot0_0kpc200kpcoriginal_particles.h5
symphonyHR_scaled_Halo999_sunrot180_0kpc200kpcoriginal_particles.h5
```
- Pattern: `symphonyHR_scaled_Halo*` or `*Halo*.h5`

## Search Directory Priority

### 1. Primary Eden Search
```bash
find /oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/ -name "eden_scaled_Halo*" -type f
```

### 2. Fallback Directories (in order)
```bash
/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/
/oak/stanford/orgs/kipac/users/caganze/milkyway-hr-mocks/
/oak/stanford/orgs/kipac/users/caganze/milkywaymocks/
/oak/stanford/orgs/kipac/users/caganze/
```

### 3. Final Fallback
```bash
/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/all_in_one.h5
```

## HALO_ID Extraction

### Updated Regex (Compatible)
```bash
# Old (incompatible with some sed versions)
HALO_ID=$(echo "$filename" | sed 's/.*Halo\([0-9]\+\).*/\1/')

# New (universally compatible)
HALO_ID=$(echo "$filename" | sed 's/.*Halo\([0-9][0-9]*\).*/\1/')
```

### Extraction Examples
```
eden_scaled_Halo570_sunrot90_0kpc200kpcoriginal_particles.h5 → HALO_ID="570"
symphony_scaled_Halo088_sunrot0_0kpc200kpcoriginal_particles.h5 → HALO_ID="088"  
symphonyHR_scaled_Halo188_sunrot0_0kpc200kpcoriginal_particles.h5 → HALO_ID="188"
all_in_one.h5 → HALO_ID="000" (fallback)
no_halo_pattern.h5 → HALO_ID="000" (fallback)
```

## Data Source Detection

### Detection Logic
```bash
if [[ "$filename" == *"eden_scaled"* ]]; then
    DATA_SOURCE="eden"
elif [[ "$filename" == *"symphonyHR_scaled"* ]]; then
    DATA_SOURCE="symphony-hr"
elif [[ "$filename" == *"symphony_scaled"* ]]; then
    DATA_SOURCE="symphony"
else
    DATA_SOURCE="unknown"
fi

# Fallback for non-standard files
if [[ "$filename" == "all_in_one.h5" ]] || [[ "$HALO_ID" == "$filename" ]]; then
    HALO_ID="000"
    if [[ "$DATA_SOURCE" == "unknown" ]]; then
        DATA_SOURCE="symphony"
    fi
fi
```

## Output Structure Mapping

### Hierarchical Organization
```
/tfp_flows_output/
├── trained_flows/
│   ├── eden/
│   │   ├── halo570/model_pid1.npz
│   │   └── halo203/model_pid2.npz
│   ├── symphony/
│   │   ├── halo088/model_pid3.npz
│   │   └── halo000/model_pid4.npz    # fallback files
│   └── symphony-hr/
│       └── halo188/model_pid5.npz
└── samples/
    ├── eden/halo570/model_pid1_samples.npz
    ├── symphony/halo088/model_pid3_samples.h5
    └── symphony-hr/halo188/model_pid5_samples.npz
```

## Script Implementation

### find_h5_file Function
```bash
find_h5_file() {
    # Step 1: Look for Eden files (highest priority)
    local eden_files=$(find /oak/.../milkyway-eden-mocks/ -name "eden_scaled_Halo*" -type f 2>/dev/null | head -1)
    if [[ -n "$eden_files" ]]; then
        echo "$eden_files"
        return 0
    fi
    
    # Step 2: Search other directories
    local search_paths=(
        "/oak/.../symphony_mocks/"
        "/oak/.../milkyway-hr-mocks/"
        "/oak/.../milkywaymocks/"
        "/oak/.../"
    )
    
    for path in "${search_paths[@]}"; do
        if [[ -d "$path" ]]; then
            h5_file=$(find "$path" -name "*.h5" -type f 2>/dev/null | head -1)
            if [[ -n "$h5_file" ]]; then
                echo "$h5_file"
                return 0
            fi
        fi
    done
    
    # Step 3: Hardcoded fallback
    echo "/oak/.../symphony_mocks/all_in_one.h5"
}
```

### Brute Force Discovery
```bash
# Discover all Halo files across directories
H5_FILES=($(find /oak/stanford/orgs/kipac/users/caganze -name '*Halo*.h5' -type f 2>/dev/null | sort))
```

## Benefits

✅ **Flexibility**: Handles all sunrot angles and parameter variations  
✅ **Robustness**: Multiple fallback mechanisms  
✅ **Compatibility**: Works across different sed versions  
✅ **Comprehensive**: Searches all relevant directories  
✅ **Maintainable**: Clear priority order and consistent patterns  

## Testing

Run the search pattern test to verify functionality:
```bash
./test_file_search_patterns.sh
```

This will create mock files and test all search patterns and extraction logic.
