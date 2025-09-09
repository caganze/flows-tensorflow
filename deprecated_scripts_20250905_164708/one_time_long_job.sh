# Create a one-time submission bypassing the script's time limit:
sbatch --job-name="tfp_long" \
       --partition=gpu \
       --time=08:00:00 \
       --mem=64GB \
       --gres=gpu:1 \
       --output=logs/long_%A_%a.out \
       --error=logs/long_%A_%a.err \
       --array=23,88,188,268,327,364,415,440,469%3 \
       --wrap="
module purge
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0
export CUDA_VISIBLE_DEVICES=0
source ~/.bashrc
conda activate bosque

H5_FILE=\$(find /oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/ -name 'eden_scaled_Halo*' -type f | head -1)
OUTPUT_BASE=/oak/stanford/orgs/kipac/users/caganze/tfp_flows_output

# Extract data source and halo ID from H5 file
FILENAME=\$(basename \"\$H5_FILE\")
HALO_ID=\$(echo \"\$FILENAME\" | sed 's/.*Halo\\([0-9][0-9]*\\).*/\\1/')

# Determine data source from filename
if [[ \"\$FILENAME\" == *\"eden_scaled\"* ]]; then
    DATA_SOURCE=\"eden\"
elif [[ \"\$FILENAME\" == *\"symphonyHR_scaled\"* ]]; then
    DATA_SOURCE=\"symphony-hr\"
elif [[ \"\$FILENAME\" == *\"symphony_scaled\"* ]]; then
    DATA_SOURCE=\"symphony\"
else
    DATA_SOURCE=\"unknown\"
fi

# Handle fallback file (all_in_one.h5) - use default structure
if [[ \"\$FILENAME\" == \"all_in_one.h5\" ]] || [[ \"\$HALO_ID\" == \"\$FILENAME\" ]]; then
    echo \"⚠️  Using fallback file, setting default halo structure\"
    HALO_ID=\"000\"
    DATA_SOURCE=\"symphony\"
fi

# Create hierarchical output directory
MODEL_DIR=\"\$OUTPUT_BASE/trained_flows/\${DATA_SOURCE}/halo\${HALO_ID}\"
mkdir -p \"\$MODEL_DIR\"

echo \"📁 Data source: \$DATA_SOURCE\"
echo \"📁 Halo ID: \$HALO_ID\"
echo \"📁 Model dir: \$MODEL_DIR\"

# Get particle count for sophisticated parameter selection
echo \"🔍 Determining optimal parameters for PID \$SLURM_ARRAY_TASK_ID...\"
OBJECT_COUNT=\$(python -c \"
import h5py, sys, os
try:
    h5_file = os.environ.get('H5_FILE', '')
    pid = int(os.environ.get('SLURM_ARRAY_TASK_ID', '1'))
    with h5py.File(h5_file, 'r') as f:
        if f'pid{pid}' in f.keys():
            print(f[f'pid{pid}'].shape[0])
        else:
            print(50000)
except: 
    print(50000)
\" 2>/dev/null || echo 50000)

echo \"📊 PID \$SLURM_ARRAY_TASK_ID has \$OBJECT_COUNT particles\"

# SOPHISTICATED parameter selection based on particle count
if [[ \$OBJECT_COUNT -gt 100000 ]]; then
    EPOCHS=60; BATCH_SIZE=1024; N_LAYERS=4; HIDDEN_UNITS=512; LEARNING_RATE=2e-4
    echo \"🐋 Large particle (>100k): epochs=60, layers=4, units=512, lr=2e-4\"
elif [[ \$OBJECT_COUNT -gt 50000 ]]; then
    EPOCHS=45; BATCH_SIZE=1024; N_LAYERS=3; HIDDEN_UNITS=384; LEARNING_RATE=3e-4
    echo \"🐟 Medium-large (50k-100k): epochs=45, layers=3, units=384, lr=3e-4\"
elif [[ \$OBJECT_COUNT -lt 5000 ]]; then
    EPOCHS=35; BATCH_SIZE=512; N_LAYERS=3; HIDDEN_UNITS=256; LEARNING_RATE=5e-4
    echo \"🐭 Small particle (<5k): epochs=35, layers=3, units=256, lr=5e-4\"
else
    EPOCHS=40; BATCH_SIZE=1024; N_LAYERS=3; HIDDEN_UNITS=320; LEARNING_RATE=4e-4
    echo \"🐟 Medium particle (5k-50k): epochs=40, layers=3, units=320, lr=4e-4\"
fi

python train_tfp_flows.py \\
    --data_path \"\$H5_FILE\" \\
    --particle_pid \$SLURM_ARRAY_TASK_ID \\
    --output_dir \"\$MODEL_DIR\" \\
    --epochs \$EPOCHS \\
    --batch_size \$BATCH_SIZE \\
    --learning_rate \$LEARNING_RATE \\
    --n_layers \$N_LAYERS \\
    --hidden_units \$HIDDEN_UNITS \\
    --generate-samples \\
    --use_kroupa_imf
"
