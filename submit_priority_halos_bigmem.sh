#!/bin/bash
#SBATCH --job-name=priority_halos_training
#SBATCH --time=12:00:00
#SBATCH --mem=1536GB
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/priority_halos_%j.out
#SBATCH --error=logs/priority_halos_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Define arrays
PRIORITY_HALOS=("718" "939" "925")
SUITES=("eden" "symphony")
PARTICLES=(1 2 3 4 5)

# Training parameters
EPOCHS=100
BATCH_SIZE=512
N_LAYERS=4
HIDDEN_UNITS=256
LEARNING_RATE=0.0001

echo "🚀 Starting priority halos training on bigmem partition"
echo "📋 Configuration:"
echo "   Halos: ${PRIORITY_HALOS[*]}"
echo "   Suites: ${SUITES[*]}"
echo "   Particles: ${PARTICLES[*]}"
echo "   Memory: 1536GB per job"
echo "   Time limit: 12 hours"
echo "   Training params: ${EPOCHS} epochs, batch_size=${BATCH_SIZE}, lr=${LEARNING_RATE}"
echo ""

# Function to submit a single training job
submit_training_job() {
    local halo_id=$1
    local particle_pid=$2
    local suite=$3
    
    echo "📤 Submitting job for Halo${halo_id} PID ${particle_pid} (${suite})"
    
    sbatch --job-name="halo${halo_id}_pid${particle_pid}_${suite}" \
           --time=12:00:00 \
           --mem=1536GB \
           --partition=bigmem \
           --nodes=1 \
           --ntasks-per-node=1 \
           --cpus-per-task=8 \
           --output="logs/halo${halo_id}_pid${particle_pid}_${suite}_%j.out" \
           --error="logs/halo${halo_id}_pid${particle_pid}_${suite}_%j.err" \
           --wrap="
           cd /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow
           source ~/.bashrc
           conda activate flows-env
           
           echo \"🚀 Starting training for Halo${halo_id} PID ${particle_pid} (${suite})\"
           echo \"📋 Job started at: \$(date)\"
           echo \"💻 Node: \$(hostname)\"
           echo \"🧠 Memory: 1536GB\"
           echo \"⏱️ Time limit: 12 hours\"
           echo \"\"
           
           python train_tfp_flows.py \\
               --halo_id \"Halo${halo_id}\" \\
               --particle_pid ${particle_pid} \\
               --suite \"${suite}\" \\
               --epochs ${EPOCHS} \\
               --batch_size ${BATCH_SIZE} \\
               --n_layers ${N_LAYERS} \\
               --hidden_units ${HIDDEN_UNITS} \\
               --learning_rate ${LEARNING_RATE}
           
           echo \"\"
           echo \"✅ Training completed for Halo${halo_id} PID ${particle_pid} (${suite})\"
           echo \"📋 Job finished at: \$(date)\"
           "
}

# Submit all combinations
total_jobs=0
for halo in "${PRIORITY_HALOS[@]}"; do
    for suite in "${SUITES[@]}"; do
        for particle in "${PARTICLES[@]}"; do
            submit_training_job "$halo" "$particle" "$suite"
            ((total_jobs++))
        done
    done
done

echo ""
echo "📊 Submission Summary:"
echo "   Total jobs submitted: ${total_jobs}"
echo "   Halos: ${#PRIORITY_HALOS[@]} (${PRIORITY_HALOS[*]})"
echo "   Suites: ${#SUITES[@]} (${SUITES[*]})"
echo "   Particles per halo/suite: ${#PARTICLES[@]} (${PARTICLES[*]})"
echo "   Memory per job: 1536GB"
echo "   Time limit per job: 12 hours"
echo "   Partition: bigmem"
echo ""
echo "🔍 Monitor jobs with: squeue -u \$USER"
echo "📁 Check logs in: logs/"
echo ""

# Wait a moment for jobs to be submitted
sleep 2

# Show current job queue
echo "📋 Current job queue:"
squeue -u $USER --format="%.10i %.20j %.8u %.2t %.10M %.6D %R" | head -20
