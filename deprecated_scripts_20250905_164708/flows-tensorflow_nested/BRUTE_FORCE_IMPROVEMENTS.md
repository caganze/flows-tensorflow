# 🎯 Brute Force Script Improvements

## Key Changes Based on Existing Script Analysis

### 1. **Resource Configuration** (As Requested)
```bash
# BEFORE:
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# AFTER:
#SBATCH --time=12:00:00        # ✅ Limited to 12 hours
#SBATCH --gres=gpu:4           # ✅ Increased to 4 GPUs
# (removed --cpus-per-task)     # ✅ Removed CPU constraint
```

### 2. **H5 File Discovery** (Following Proven Patterns)
```bash
# BEFORE: Naive search
find "$BASE_DIR" -name '*Halo*_*orig*.h5'

# AFTER: Multi-tier search matching existing scripts
1. eden_scaled files: "$BASE_DIR/milkyway-eden-mocks/eden_scaled_Halo*_particles.h5"
2. Symphony files: "$BASE_DIR/milkywaymocks/symphony_scaled_Halo*_particles.h5"  
3. HR mocks: "$BASE_DIR/milkyway-hr-mocks/symphonyHR_scaled_Halo*_particles.h5"
4. Fallback: Broad search for any Halo files
```

### 3. **PID Discovery** (Smarter Sampling)
```bash
# BEFORE: Complex multi-file sampling
for h5_file in "${SAMPLE_FILES[@]}"; do...

# AFTER: Single-file sampling with robust fallbacks
- Sample first available file (representative)
- Select top 20 most populous PIDs (better success rate)
- Fallback to proven PID list from existing scripts: [1,2,3,4,5,23,88,188...]
```

### 4. **Halo ID Extraction** (Multiple Pattern Support)
```bash
# BEFORE: Single regex pattern
sed -n 's/.*Halo\([0-9]\+\).*/\1/p'

# AFTER: Multiple patterns matching real file names
1. Pattern: eden_scaled_Halo023_...
2. Pattern: symphony_scaled_Halo023_...  
3. Pattern: any Halo[numbers]
4. Fallback: extract any numbers from filename
```

### 5. **Directory Structure** (Simplified)
```bash
# BEFORE: Complex BASE_DIR validation
if [[ ! -d "$SCRIPT_DIR" ]]; then...

# AFTER: Use current working directory + create as needed
mkdir -p "$OUTPUT_DIR/trained_flows"
mkdir -p "$OUTPUT_DIR/samples" 
mkdir -p "$OUTPUT_DIR/metrics"
```

## 📋 **Proven Search Patterns Used**

Based on analysis of working scripts:

### **Primary Search Locations:**
1. `/oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/`
2. `/oak/stanford/orgs/kipac/users/caganze/milkywaymocks/`
3. `/oak/stanford/orgs/kipac/users/caganze/milkyway-hr-mocks/`

### **File Name Patterns:**
1. `eden_scaled_Halo*_particles.h5` (newer format)
2. `symphony_scaled_Halo*_particles.h5` 
3. `symphonyHR_scaled_Halo*_particles.h5`
4. `*Halo*_*orig*.h5` (fallback)

### **Tested PID Lists:**
- From `generate_parallel_scripts.py`: `[23, 88, 188, 268, 327, 364, 415, 440, 469, 530, 570, 641, 718, 800, 829, 852, 939]`
- From test scripts: `[1, 2, 3, 4, 5]`
- Dynamic: Top 20 most populous PIDs from actual data

## 🎯 **Expected Behavior**

### **File Discovery:**
1. ✅ Finds eden_scaled files first (preferred)
2. ✅ Falls back to symphony files if no eden files
3. ✅ Falls back to HR mocks if no symphony files  
4. ✅ Ultimate fallback to any Halo files found

### **PID Selection:**
1. ✅ Samples representative file to get actual PIDs
2. ✅ Selects most populous PIDs (higher success rate)
3. ✅ Limits to 20 PIDs for manageable job array
4. ✅ Falls back to proven PID list if sampling fails

### **Resource Usage:**
1. ✅ 12-hour time limit (reasonable for most training)
2. ✅ 4 GPUs per job (parallel processing)
3. ✅ No CPU constraints (let SLURM decide)
4. ✅ 128GB RAM (sufficient for large datasets)

## 🔧 **Error Handling Improvements**

1. **Graceful Fallbacks**: Every step has a fallback strategy
2. **Proven Patterns**: Uses file discovery methods from working scripts
3. **Robust Parsing**: Multiple regex patterns for file name extraction
4. **Comprehensive Logging**: Detailed output for debugging

---

🌟 **The script now follows the exact patterns used by existing working scripts, ensuring higher compatibility and success rates!**
