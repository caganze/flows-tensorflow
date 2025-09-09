# 🔧 Output Structure Fixes Applied

## Problem Fixed
All .sh scripts were using flat output structure instead of hierarchical structure organized by data source and halo ID.

## New Hierarchical Structure
```
/oak/stanford/orgs/kipac/users/caganze/tfp_flows_output/
├── trained_flows/
│   ├── eden/
│   │   ├── halo023/
│   │   │   ├── model_pid1.npz
│   │   │   ├── model_pid1_results.json
│   │   │   └── model_pid1_preprocessing.npz
│   │   └── halo088/
│   ├── symphony/
│   │   └── halo188/
│   └── symphony-hr/
│       └── halo570/
└── samples/
    ├── eden/
    │   ├── halo023/
    │   │   ├── model_pid1_samples.npz
    │   │   └── model_pid1_samples.h5
    │   └── halo088/
    ├── symphony/
    └── symphony-hr/
```

## Scripts Fixed

### ✅ 1. brute_force_gpu_job.sh
- **Added**: Data source detection from filename
- **Added**: Hierarchical directory creation
- **Changed**: Output paths to use `$DATA_SOURCE/halo$HALO_ID/`

### ✅ 2. auto_submit_flows.sh  
- **Fixed**: Completed particle detection to search hierarchical structure
- **Changed**: Searches `source/haloXXX/model_pid*.npz` instead of flat `model_pidX/`

### ✅ 3. submit_flows_array.sh
- **Added**: Data source and halo ID extraction from H5_FILE
- **Changed**: Output directory to use hierarchical structure
- **Changed**: Completion check to use new paths

### ✅ 4. submit_flows_batch1.sh  
- **Added**: Same data source detection as submit_flows_array.sh
- **Changed**: All output paths to hierarchical structure

### ✅ 5. monitor_brute_force.sh
- **Updated**: Model counting to search hierarchical structure
- **Added**: Breakdown by data source (eden, symphony, symphony-hr)
- **Updated**: Sample file counting for new structure

### ✅ 6. quick_interactive_test.sh
- **Fixed**: Model file detection to search all data sources and halos
- **Changed**: Success verification logic

### ✅ 7. validate_deployment.sh
- **Added**: Data source detection for test runs
- **Changed**: Output directory creation to use hierarchical structure
- **Updated**: File verification paths

### ✅ 8. test_slurm_deployment.sh
- **Added**: Data source detection for directory creation tests
- **Changed**: Directory structure to match new hierarchy

## Data Source Detection Logic
```bash
# Determine data source from filename
if [[ "$FILENAME" == *"eden_scaled"* ]]; then
    DATA_SOURCE="eden"
elif [[ "$FILENAME" == *"symphonyHR_scaled"* ]]; then
    DATA_SOURCE="symphony-hr"
elif [[ "$FILENAME" == *"symphony_scaled"* ]]; then
    DATA_SOURCE="symphony"
else
    DATA_SOURCE="unknown"
fi
```

## Migration Script Created
- **`migrate_output_structure.sh`**: Migrates existing flat structure to hierarchical
- **Safely moves**: Model directories and sample files
- **Preserves**: All existing data
- **Organizes**: By data source and halo ID

## Benefits
1. **Organization**: Clear separation by data source (eden, symphony, symphony-hr)
2. **Scalability**: Easy to find files for specific halos
3. **Maintenance**: Easier to manage and clean up outputs
4. **Analysis**: Simple to compare results across data sources

## Usage
1. **New jobs**: Automatically use hierarchical structure
2. **Existing data**: Run `./migrate_output_structure.sh` to reorganize
3. **Monitoring**: Use `./monitor_brute_force.sh` to see breakdown by source

## Files NOT Modified (as requested)
- **Python files**: No .py files were changed
- **Training logic**: Core functionality unchanged
- **Data processing**: No impact on actual training/sampling

---

🎯 **All .sh scripts now consistently use the hierarchical output structure organized by data source and halo ID!**
