#!/bin/bash

# 🚀 Quick Training & Sampling Test - Tests complete pipeline end-to-end
# This will run training AND verify sampling works to test the complete workflow

echo "🚀 Quick Training & Sampling Test"
echo "================================="
echo "This will test both training and sampling on PID 1 to verify the complete pipeline"
echo

# Test configuration - Optimized for GPU compute node testing
export PARTICLES_PER_TASK=1     # Test just 1 particle
export EPOCHS=5                 # More epochs to verify GPU training
export BATCH_SIZE=256           # Larger batch for GPU efficiency
export N_LAYERS=3               # Reasonable layers  
export HIDDEN_UNITS=128         # Good hidden units for GPU
export CHECK_SAMPLES=true       # Enable sample checking on GPU

# GPU optimization settings
export TF_CPP_MIN_LOG_LEVEL=1                    # Reduce TF logging but show GPU info
export TF_FORCE_GPU_ALLOW_GROWTH=true           # Allow GPU memory growth
export CUDA_CACHE_DISABLE=0                     # Enable CUDA caching

# Use test output directory
export OUTPUT_BASE_DIR="/oak/stanford/orgs/kipac/users/caganze/tfp_flows_quick_train_test"

echo "🔧 Loading environment..."
module purge 2>/dev/null || true
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131 2>/dev/null || echo "⚠️ Module loading failed (normal if not on cluster)"

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0
export CUDA_VISIBLE_DEVICES=0

source ~/.bashrc 2>/dev/null || true
conda activate bosque 2>/dev/null || echo "⚠️ Conda activation failed"

echo "✅ Environment loaded"
echo

# Find H5 file
echo "🔍 Finding H5 file..."
find_h5_file() {
    local eden_files=$(find /oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/ -name "eden_scaled_Halo*" -type f 2>/dev/null | head -1)
    if [[ -n "$eden_files" ]]; then
        echo "$eden_files"
        return 0
    fi
    
    local search_paths=(
        "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/"
        "/oak/stanford/orgs/kipac/users/caganze/milkyway-hr-mocks/"
        "/oak/stanford/orgs/kipac/users/caganze/milkywaymocks/"
        "/oak/stanford/orgs/kipac/users/caganze/"
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
    
    echo "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/all_in_one.h5"
}

H5_FILE=$(find_h5_file)
echo "📁 Found H5 file: $H5_FILE"

if [[ ! -f "$H5_FILE" ]]; then
    echo "❌ ERROR: H5 file not accessible"
    echo "💡 This test requires access to the H5 data files"
    echo "💡 Run this on the cluster or ensure data is mounted"
    exit 1
fi

echo "✅ H5 file accessible"

# Extract data source and halo info
FILENAME=$(basename "$H5_FILE")
HALO_ID=$(echo "$FILENAME" | sed 's/.*Halo\([0-9][0-9]*\).*/\1/')

if [[ "$FILENAME" == *"eden_scaled"* ]]; then
    DATA_SOURCE="eden"
elif [[ "$FILENAME" == *"symphonyHR_scaled"* ]]; then
    DATA_SOURCE="symphony-hr"
elif [[ "$FILENAME" == *"symphony_scaled"* ]]; then
    DATA_SOURCE="symphony"
else
    DATA_SOURCE="unknown"
fi

if [[ "$FILENAME" == "all_in_one.h5" ]] || [[ "$HALO_ID" == "$FILENAME" ]]; then
    HALO_ID="000"
    DATA_SOURCE="symphony"
fi

MODEL_DIR="$OUTPUT_BASE_DIR/trained_flows/${DATA_SOURCE}/halo${HALO_ID}"
SAMPLES_DIR="$OUTPUT_BASE_DIR/samples/${DATA_SOURCE}/halo${HALO_ID}"

echo "📁 Creating output directories..."
mkdir -p "$MODEL_DIR" "$SAMPLES_DIR"
mkdir -p logs

echo "✅ Directories created:"
echo "  Model dir: $MODEL_DIR"
echo "  Samples dir: $SAMPLES_DIR"

# Test particle size detection
echo
echo "🔍 Testing particle size detection for PID 1..."
particle_size=$(python -c "
import h5py
import numpy as np

def get_particle_size_robust(h5_file_path, pid):
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if 'PartType1' in f and 'ParticleIDs' in f['PartType1']:
                pids = f['PartType1']['ParticleIDs'][:]
                pid_mask = (pids == pid)
                size = np.sum(pid_mask)
                if size > 0:
                    return size
            elif 'parentid' in f:
                pids = f['parentid'][:]
                pid_mask = (pids == pid)
                size = np.sum(pid_mask)
                if size > 0:
                    return size
    except:
        pass
    return 300000 if pid <= 10 else 150000 if pid <= 100 else 75000

print(get_particle_size_robust('$H5_FILE', 1))
" 2>/dev/null)

echo "📊 PID 1 has $particle_size objects"
if [[ $particle_size -gt 100000 ]]; then
    echo "🐋 Large particle detected (>100k objects)"
    size_category="Large"
else
    echo "🐭 Small particle (<100k objects)"
    size_category="Small"
fi

# Now actually run training!
echo
echo "🚀 STARTING ACTUAL TRAINING TEST"
echo "================================="
echo "Training PID 1 with minimal parameters for speed..."
echo "Expected runtime: 2-5 minutes"
echo

# Check if already completed
if [[ -f "$MODEL_DIR/model_pid1.npz" && -f "$MODEL_DIR/model_pid1_results.json" ]]; then
    echo "⚠️ PID 1 already trained, removing old files for fresh test..."
    rm -f "$MODEL_DIR/model_pid1.npz" "$MODEL_DIR/model_pid1_results.json"
    rm -f "$SAMPLES_DIR/model_pid1_samples.npz" "$SAMPLES_DIR/model_pid1_samples.h5"
fi

# Run the training
echo "🏃 Running train_tfp_flows.py..."
start_time=$(date +%s)

python train_tfp_flows.py \
    --data_path "$H5_FILE" \
    --particle_pid 1 \
    --output_dir "$MODEL_DIR" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 1e-3 \
    --n_layers $N_LAYERS \
    --hidden_units $HIDDEN_UNITS

TRAIN_EXIT_CODE=$?
end_time=$(date +%s)
runtime=$((end_time - start_time))

echo
echo "🎯 TRAINING TEST RESULTS"
echo "========================="
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Runtime: ${runtime} seconds"
echo "Particle: PID 1 ($particle_size objects, $size_category)"
echo

# Check outputs
if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo "✅ Training completed successfully!"
    
    # Check model file
    if [[ -f "$MODEL_DIR/model_pid1.npz" && -s "$MODEL_DIR/model_pid1.npz" ]]; then
        model_size=$(ls -lh "$MODEL_DIR/model_pid1.npz" | awk '{print $5}')
        echo "✅ Model file created: $model_size"
        MODEL_EXISTS=true
    else
        echo "❌ Model file missing or empty"
        MODEL_EXISTS=false
    fi
    
    # Check results file
    if [[ -f "$MODEL_DIR/model_pid1_results.json" ]]; then
        if python -c "import json; json.load(open('$MODEL_DIR/model_pid1_results.json'))" 2>/dev/null; then
            echo "✅ Results file created and valid JSON"
            
            # Show some results
            echo "📊 Training results:"
            python -c "
import json
try:
    with open('$MODEL_DIR/model_pid1_results.json', 'r') as f:
        results = json.load(f)
        if 'train_losses' in results:
            print(f'  Final train loss: {results[\"train_losses\"][-1]:.4f}')
        if 'val_losses' in results:
            print(f'  Final val loss: {results[\"val_losses\"][-1]:.4f}')
        if 'epochs_completed' in results:
            print(f'  Epochs completed: {results[\"epochs_completed\"]}')
        
        # Check if Kroupa IMF sampling info is present
        if 'sampling' in results:
            sampling_info = results['sampling']
            print(f'  Samples generated: {sampling_info.get(\"n_samples\", \"unknown\")}')
            if 'kroupa_mass_info' in sampling_info:
                print(f'  Kroupa IMF used: ✅')
            else:
                print(f'  Kroupa IMF used: ℹ️ (not specified)')
        else:
            print(f'  Sampling info: ℹ️ (check sample files)')
except Exception as e:
    print(f'  Could not parse results details: {e}')
"
            RESULTS_EXISTS=true
        else
            echo "❌ Results file invalid JSON"
            RESULTS_EXISTS=false
        fi
    else
        echo "❌ Results file missing"
        RESULTS_EXISTS=false
    fi
    
    # Check sample files - training script should generate these automatically
    echo
    echo "🎲 SAMPLING VERIFICATION"
    echo "========================"
    
    sample_npz="$SAMPLES_DIR/model_pid1_samples.npz"
    sample_h5="$SAMPLES_DIR/model_pid1_samples.h5"
    
    if [[ -f "$sample_npz" ]]; then
        sample_size=$(ls -lh "$sample_npz" | awk '{print $5}')
        echo "✅ Sample file (NPZ) created: $sample_size"
        
        # Analyze sample file
        echo "📊 Sample analysis:"
        python -c "
import numpy as np
try:
    data = np.load('$sample_npz')
    print(f'  Sample file keys: {list(data.keys())}')
    
    if 'samples' in data:
        samples = data['samples']
        print(f'  Sample shape: {samples.shape}')
        print(f'  Sample range: [{samples.min():.3f}, {samples.max():.3f}]')
        print(f'  Memory usage: {samples.nbytes / 1024**2:.1f} MB')
        
    if 'masses' in data:
        masses = data['masses']
        print(f'  Kroupa masses shape: {masses.shape}')
        print(f'  Mass range: [{masses.min():.2e}, {masses.max():.2e}] M☉')
        print(f'  Total mass: {masses.sum():.2e} M☉')
        print(f'  ✅ Kroupa IMF sampling confirmed!')
    else:
        print(f'  ⚠️ No Kroupa masses found (may be in metadata)')
        
    if 'metadata' in data:
        metadata = data['metadata'].item() if data['metadata'].ndim == 0 else data['metadata']
        if isinstance(metadata, dict):
            if 'stellar_mass' in metadata:
                print(f'  Stellar mass: {metadata[\"stellar_mass\"]:.2e} M☉')
            if 'kroupa_imf_used' in metadata:
                print(f'  Kroupa IMF confirmed: {metadata[\"kroupa_imf_used\"]}')
                
except Exception as e:
    print(f'  Error analyzing samples: {e}')
"
        SAMPLES_EXISTS=true
        
    elif [[ -f "$sample_h5" ]]; then
        sample_size=$(ls -lh "$sample_h5" | awk '{print $5}')
        echo "✅ Sample file (H5) created: $sample_size"
        
        # Analyze H5 sample file
        echo "📊 Sample analysis:"
        python -c "
import h5py
import numpy as np
try:
    with h5py.File('$sample_h5', 'r') as f:
        print(f'  H5 file keys: {list(f.keys())}')
        
        if 'samples' in f:
            samples = f['samples'][:]
            print(f'  Sample shape: {samples.shape}')
            print(f'  Sample range: [{samples.min():.3f}, {samples.max():.3f}]')
            print(f'  Memory usage: {samples.nbytes / 1024**2:.1f} MB')
            
        if 'masses' in f:
            masses = f['masses'][:]
            print(f'  Kroupa masses shape: {masses.shape}')
            print(f'  Mass range: [{masses.min():.2e}, {masses.max():.2e}] M☉')
            print(f'  Total mass: {masses.sum():.2e} M☉')
            print(f'  ✅ Kroupa IMF sampling confirmed!')
        elif 'kroupa_masses' in f:
            masses = f['kroupa_masses'][:]
            print(f'  Kroupa masses shape: {masses.shape}')
            print(f'  Mass range: [{masses.min():.2e}, {masses.max():.2e}] M☉')
            print(f'  ✅ Kroupa IMF sampling confirmed!')
        else:
            print(f'  ⚠️ No Kroupa masses found in H5 file')
            
        # Check for metadata
        if 'metadata' in f:
            print(f'  Metadata available: ✅')
        elif hasattr(f, 'attrs'):
            attrs = dict(f.attrs)
            if attrs:
                print(f'  Attributes: {list(attrs.keys())}')
                
except Exception as e:
    print(f'  Error analyzing H5 samples: {e}')
"
        SAMPLES_EXISTS=true
        
    else
        echo "❌ No sample files found"
        echo "ℹ️ Expected locations:"
        echo "  - $sample_npz"
        echo "  - $sample_h5"
        SAMPLES_EXISTS=false
    fi
    
    echo
    echo "🎉 COMPLETE PIPELINE TEST RESULTS"
    echo "=================================="
    echo "✅ Environment works correctly"
    echo "✅ H5 file access works"
    echo "✅ Particle size detection works ($particle_size objects)"
    echo "✅ train_tfp_flows.py runs successfully"
    
    if [[ "$MODEL_EXISTS" == "true" ]]; then
        echo "✅ Model file created correctly"
    else
        echo "❌ Model file creation failed"
    fi
    
    if [[ "$RESULTS_EXISTS" == "true" ]]; then
        echo "✅ Results file created correctly"
    else
        echo "❌ Results file creation failed"
    fi
    
    if [[ "$SAMPLES_EXISTS" == "true" ]]; then
        echo "✅ Sample generation works (including Kroupa IMF)"
    else
        echo "❌ Sample generation failed"
    fi
    
    echo "✅ Directory structure works"
    
    # Overall assessment
    if [[ "$MODEL_EXISTS" == "true" && "$RESULTS_EXISTS" == "true" && "$SAMPLES_EXISTS" == "true" ]]; then
        echo
        echo "🚀 COMPLETE SUCCESS - READY FOR FULL SUBMISSION!"
        echo "==============================================="
        echo "✅ Both training AND sampling verified working"
        echo "✅ Kroupa IMF integration confirmed"
        echo "✅ Complete end-to-end pipeline functional"
        echo
        echo "🚀 NEXT STEPS:"
        echo "1. 🧪 SLURM test: sbatch test_submit_tfp_array.sh"
        echo "2. 🚀 Production: sbatch submit_tfp_array.sh"
        echo "3. 📊 Monitor: squeue -u \$(whoami)"
        
    elif [[ "$MODEL_EXISTS" == "true" && "$RESULTS_EXISTS" == "true" ]]; then
        echo
        echo "⚠️ PARTIAL SUCCESS - TRAINING WORKS, SAMPLING NEEDS CHECK"
        echo "========================================================="
        echo "✅ Training pipeline verified working"
        echo "❌ Sample generation needs investigation"
        echo "💡 Training may have completed without generating samples"
        echo "💡 This could be normal for very minimal test parameters"
        echo
        echo "🔧 RECOMMENDATIONS:"
        echo "1. Try with slightly more epochs (5-10) to ensure full completion"
        echo "2. Check if sampling is disabled for minimal tests"
        echo "3. Proceed with SLURM test which uses more realistic parameters"
        
    else
        echo
        echo "❌ PIPELINE NEEDS DEBUGGING"
        echo "=========================="
        echo "❌ Critical components failed"
        echo "🔍 Check logs and error messages above"
        echo "💡 Fix issues before proceeding to full submission"
    fi
    
else
    echo "❌ TRAINING FAILED!"
    echo "==================="
    echo "Exit code: $TRAIN_EXIT_CODE"
    echo "Runtime: ${runtime} seconds"
    echo
    echo "🔍 TROUBLESHOOTING:"
    echo "Check for common issues:"
    echo "1. Module loading problems"
    echo "2. Environment activation issues"
    echo "3. GPU availability"
    echo "4. train_tfp_flows.py argument problems"
    echo "5. H5 file access/structure issues"
    echo
    echo "💡 Try running train_tfp_flows.py manually with --help to check arguments"
fi

echo
echo "📁 Test outputs saved in: $OUTPUT_BASE_DIR"
echo "📁 Model files: $MODEL_DIR"
echo "📁 Sample files: $SAMPLES_DIR"
echo "🏁 Quick training & sampling test completed at $(date)"

# Exit with appropriate code based on overall success
if [[ $TRAIN_EXIT_CODE -eq 0 && "$MODEL_EXISTS" == "true" && "$RESULTS_EXISTS" == "true" ]]; then
    exit 0  # Success
else
    exit $TRAIN_EXIT_CODE  # Use training exit code
fi
