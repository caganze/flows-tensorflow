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

python train_tfp_flows.py \\
    --data_path \"\$H5_FILE\" \\
    --particle_pid \$SLURM_ARRAY_TASK_ID \\
    --output_dir \"\$MODEL_DIR\" \\
    --epochs 50 \\
    --batch_size 1024 \\
    --learning_rate 1e-3 \\
    --n_layers 4 \\
    --hidden_units 64
"
