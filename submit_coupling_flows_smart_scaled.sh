#!/bin/bash

# Smart SLURM submission script for coupling flows training with particle-count scaling
# Reads from particle_list.txt and scales training parameters based on particle count

set -e

# Configuration
PARTICLE_LIST_FILE="particle_list.txt"
BASE_OUTPUT_DIR="coupling_output"
LOG_DIR="logs"
MAX_CONCURRENT_JOBS=10  # Set to 0 for unlimited submissions (no throttling)
DEFAULT_EPOCHS=100
DEFAULT_PATIENCE=15

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Function to determine training parameters based on particle count
get_training_params() {
    local particle_count=$1
    local size_category=$2
    
    # Base parameters
    local epochs=$DEFAULT_EPOCHS
    local patience=$DEFAULT_PATIENCE
    local batch_size=512
    local learning_rate="1e-3"
    local n_layers=4
    local hidden_units=64
    local time_limit="12:00:00"
    local memory="64GB"
    local cpus=8
    local partition="kipac"
    
    # Scale parameters based on particle count
    if [ $particle_count -ge 200000 ]; then
        # Large datasets (>200k particles)
        epochs=150
        patience=20
        batch_size=1024
        learning_rate="5e-4"
        n_layers=6
        hidden_units=128
        time_limit="24:00:00"
        memory="128GB"
        cpus=16
        partition="kipac"
    elif [ $particle_count -ge 100000 ]; then
        # Medium-Large datasets (100k-200k particles)
        epochs=120
        patience=18
        batch_size=768
        learning_rate="7e-4"
        n_layers=5
        hidden_units=96
        time_limit="18:00:00"
        memory="96GB"
        cpus=12
        partition="kipac"
    elif [ $particle_count -ge 50000 ]; then
        # Medium datasets (50k-100k particles)
        epochs=100
        patience=15
        batch_size=512
        learning_rate="1e-3"
        n_layers=4
        hidden_units=64
        time_limit="12:00:00"
        memory="64GB"
        cpus=8
        partition="kipac"
    elif [ $particle_count -ge 10000 ]; then
        # Small-Medium datasets (10k-50k particles)
        epochs=80
        patience=12
        batch_size=256
        learning_rate="2e-3"
        n_layers=3
        hidden_units=48
        time_limit="8:00:00"
        memory="32GB"
        cpus=4
        partition="kipac"
    else
        # Small datasets (<10k particles)
        epochs=60
        patience=10
        batch_size=128
        learning_rate="3e-3"
        n_layers=3
        hidden_units=32
        time_limit="4:00:00"
        memory="16GB"
        cpus=2
        partition="kipac"
    fi
    
    # Additional scaling based on size category
    case $size_category in
        "Large")
            epochs=$((epochs + 20))
            patience=$((patience + 3))
            ;;
        "Medium-Large")
            epochs=$((epochs + 10))
            patience=$((patience + 2))
            ;;
        "Small")
            epochs=$((epochs - 10))
            patience=$((patience - 2))
            ;;
    esac
    
    # Ensure minimum values
    epochs=$((epochs < 30 ? 30 : epochs))
    patience=$((patience < 5 ? 5 : patience))
    
    echo "$epochs $patience $batch_size $learning_rate $n_layers $hidden_units $time_limit $memory $cpus $partition"
}

# Function to check if training is already completed
is_training_completed() {
    local base_dir="$1"
    local halo_id="$2"
    local particle_pid="$3"
    
    local config_file="${base_dir}/flow_config_${halo_id}_${particle_pid}.pkl"
    local weights_file="${base_dir}/flow_weights_${halo_id}_${particle_pid}.index"
    local preproc_file="${base_dir}/coupling_flow_pid${particle_pid}_preprocessing.npz"
    local results_file="${base_dir}/coupling_flow_pid${particle_pid}_results.json"
    
    if [[ -f "$config_file" && -f "$weights_file" && -f "$preproc_file" && -f "$results_file" ]]; then
        return 0  # Completed
    else
        return 1  # Not completed
    fi
}

# Function to submit a single training job
submit_training_job() {
    local particle_pid="$1"
    local halo_id="$2"
    local suite="$3"
    local particle_count="$4"
    local size_category="$5"
    
    local base_dir="${BASE_OUTPUT_DIR}/${suite}/${halo_id,,}/pid${particle_pid}"
    
    # Check if already completed
    if is_training_completed "$base_dir" "$halo_id" "$particle_pid"; then
        print_warning "Training already completed for PID ${particle_pid}, skipping..."
        return 0
    fi
    
    # Get scaled training parameters
    local params=($(get_training_params $particle_count $size_category))
    local epochs=${params[0]}
    local patience=${params[1]}
    local batch_size=${params[2]}
    local learning_rate=${params[3]}
    local n_layers=${params[4]}
    local hidden_units=${params[5]}
    local time_limit=${params[6]}
    local memory=${params[7]}
    local cpus=${params[8]}
    local partition=${params[9]}
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Create SLURM job script
    local job_script="${LOG_DIR}/job_${halo_id}_${particle_pid}.sh"
    cat > "$job_script" << EOF
#!/bin/bash
#SBATCH --job-name=coupling_${halo_id}_${particle_pid}
#SBATCH --output=${LOG_DIR}/coupling_${halo_id}_${particle_pid}_%j.out
#SBATCH --error=${LOG_DIR}/coupling_${halo_id}_${particle_pid}_%j.err
#SBATCH --partition=${partition}
#SBATCH --time=${time_limit}
#SBATCH --mem=${memory}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --nodes=1

# Load modules (adjust for your system)
module load python/3.9
module load tensorflow/2.10

# Activate conda environment (adjust path as needed)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bosque

# Print job info
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Start time: \$(date)"
echo "Training PID: ${particle_pid} (${particle_count} particles, ${size_category})"
echo "Parameters: epochs=${epochs}, patience=${patience}, batch_size=${batch_size}"
echo "Architecture: layers=${n_layers}, hidden=${hidden_units}, lr=${learning_rate}"

# Run training
cd \$SLURM_SUBMIT_DIR

python train_coupling_flows_conditional.py \\
    --halo_id ${halo_id} \\
    --particle_pid ${particle_pid} \\
    --suite ${suite} \\
    --epochs ${epochs} \\
    --early_stopping_patience ${patience} \\
    --train_val_split 0.8 \\
    --n_layers ${n_layers} \\
    --hidden_units ${hidden_units} \\
    --learning_rate ${learning_rate} \\
    --output_dir ${BASE_OUTPUT_DIR}

echo "End time: \$(date)"
echo "Training completed for PID ${particle_pid}"
EOF
    
    # Submit job
    local job_id=$(sbatch "$job_script" | awk '{print $4}')
    print_success "Submitted job ${job_id} for PID ${particle_pid} (${particle_count} particles, ${size_category})"
    print_status "  Parameters: epochs=${epochs}, patience=${patience}, batch_size=${batch_size}"
    print_status "  Architecture: layers=${n_layers}, hidden=${hidden_units}, lr=${learning_rate}"
    print_status "  Resources: ${cpus} CPUs, ${memory}, ${time_limit}"
    
    echo "$job_id" >> "${LOG_DIR}/submitted_jobs.txt"
}

# Main function
main() {
    print_status "Starting smart coupling flows training submission..."
    print_status "Reading particle list from: ${PARTICLE_LIST_FILE}"
    if [ "$MAX_CONCURRENT_JOBS" -le 0 ]; then
        print_status "Concurrency: unlimited (no throttling)"
    else
        print_status "Concurrency limit: ${MAX_CONCURRENT_JOBS} jobs"
    fi
    
    # Check if particle list file exists
    if [[ ! -f "$PARTICLE_LIST_FILE" ]]; then
        print_error "Particle list file not found: ${PARTICLE_LIST_FILE}"
        exit 1
    fi
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    rm -f "${LOG_DIR}/submitted_jobs.txt"
    
    # Initialize counters
    local total_count=0
    local submitted_count=0
    local skipped_count=0
    
    # Read particle list and submit jobs
    while IFS=',' read -r particle_pid halo_id suite particle_count size_category; do
        # Skip empty lines and comments
        if [[ -z "$particle_pid" || "$particle_pid" =~ ^# ]]; then
            continue
        fi
        
        # Remove any whitespace
        particle_pid=$(echo "$particle_pid" | xargs)
        halo_id=$(echo "$halo_id" | xargs)
        suite=$(echo "$suite" | xargs)
        particle_count=$(echo "$particle_count" | xargs)
        size_category=$(echo "$size_category" | xargs)
        
        total_count=$((total_count + 1))
        
        # Throttle only if a positive limit is set
        if [ "$MAX_CONCURRENT_JOBS" -gt 0 ]; then
            # Count current jobs for this user without header (-h) for accuracy
            local current_jobs=$(squeue -u "$USER" -h -o "%i" | wc -l)
            while [ "$current_jobs" -ge "$MAX_CONCURRENT_JOBS" ]; do
                print_warning "Maximum concurrent jobs (${MAX_CONCURRENT_JOBS}) reached. Waiting... (current: ${current_jobs})"
                sleep 20
                current_jobs=$(squeue -u "$USER" -h -o "%i" | wc -l)
            done
        fi
        
        # Submit training job
        if submit_training_job "$particle_pid" "$halo_id" "$suite" "$particle_count" "$size_category"; then
            submitted_count=$((submitted_count + 1))
        else
            skipped_count=$((skipped_count + 1))
        fi
        
        # Small delay between submissions
        sleep 2
        
    done < "$PARTICLE_LIST_FILE"
    
    # Final summary
    print_status "Submission completed!"
    print_success "Total processed: ${total_count}"
    print_success "Jobs submitted: ${submitted_count}"
    print_warning "Jobs skipped: ${skipped_count}"
    
    if [ $submitted_count -gt 0 ]; then
        print_status "Submitted job IDs saved to: ${LOG_DIR}/submitted_jobs.txt"
        print_status "Monitor jobs with: squeue -u $USER"
        print_status "Check logs in: ${LOG_DIR}/"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -f, --file FILE     Particle list file (default: particle_list.txt)"
    echo "  -o, --output DIR    Base output directory (default: coupling_output)"
    echo "  -j, --jobs NUM      Maximum concurrent jobs (default: 10)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Particle list format:"
    echo "  pid,halo_id,suite,count,size"
    echo "  Example: 1,Halo718,eden,270306,Large"
    echo ""
    echo "Scaling rules:"
    echo "  Large (>200k): epochs=150, patience=20, batch=1024, lr=5e-4, layers=6, hidden=128"
    echo "  Medium-Large (100k-200k): epochs=120, patience=18, batch=768, lr=7e-4, layers=5, hidden=96"
    echo "  Medium (50k-100k): epochs=100, patience=15, batch=512, lr=1e-3, layers=4, hidden=64"
    echo "  Small-Medium (10k-50k): epochs=80, patience=12, batch=256, lr=2e-3, layers=3, hidden=48"
    echo "  Small (<10k): epochs=60, patience=10, batch=128, lr=3e-3, layers=3, hidden=32"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            PARTICLE_LIST_FILE="$2"
            shift 2
            ;;
        -o|--output)
            BASE_OUTPUT_DIR="$2"
            shift 2
            ;;
        -j|--jobs)
            MAX_CONCURRENT_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Run main function
main "$@"
