#!/bin/bash

# 🖥️  CPU Parallel Submission Script
# Submit CPU-optimized TFP flows jobs (one particle per array task)

set -e

echo "🖥️  CPU PARALLEL TFP FLOWS SUBMISSION"
echo "===================================="
echo "🚀 CPU-optimized version of brute force job"
echo "📊 One particle per array task with CPU optimization"
echo ""

# Default parameters
DEFAULT_PARTITION="kipac"
DEFAULT_TIME="12:00:00"
DEFAULT_MEMORY="256GB"
DEFAULT_CPUS=64
DEFAULT_CONCURRENT=50

# Parse command line arguments  
PARTITION=$DEFAULT_PARTITION
TIME_LIMIT=$DEFAULT_TIME
MEMORY=$DEFAULT_MEMORY
CPUS_PER_TASK=$DEFAULT_CPUS
CONCURRENT=$DEFAULT_CONCURRENT
DRY_RUN=false
FORCE_SUBMIT=false

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --concurrent N      Max concurrent tasks (default: $DEFAULT_CONCURRENT)"
    echo "  --partition NAME    SLURM partition (default: $DEFAULT_PARTITION)"
    echo "  --time HOURS        Time limit (default: $DEFAULT_TIME)"
    echo "  --memory SIZE       Memory per task (default: $DEFAULT_MEMORY)"
    echo "  --cpus N            CPUs per task (default: $DEFAULT_CPUS)"
    echo "  --dry-run           Show what would be submitted without submitting"
    echo "  --force             Skip confirmation prompt"
    echo "  --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                                    # Submit with particle list size"
    echo "  $0 --concurrent 5                    # Max 5 concurrent tasks"
    echo "  $0 --partition bigmem --memory 1TB   # Use bigmem partition with 1TB"
    echo "  $0 --dry-run                         # Preview submission"
    echo ""
    echo "NOTE: Array size is automatically determined from particle_list.txt"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --concurrent)
            CONCURRENT="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --cpus)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE_SUBMIT=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate inputs
if ! [[ "$CONCURRENT" =~ ^[0-9]+$ ]] || [[ $CONCURRENT -lt 1 ]]; then
    echo "❌ Invalid concurrent limit: $CONCURRENT"
    exit 1
fi

if ! [[ "$CPUS_PER_TASK" =~ ^[0-9]+$ ]] || [[ $CPUS_PER_TASK -lt 1 ]]; then
    echo "❌ Invalid CPUs per task: $CPUS_PER_TASK"
    exit 1
fi

# Check required files
echo "🔍 Checking prerequisites..."

required_files=(
    "brute_force_cpu_parallel.sh"
    "train_tfp_flows.py"
    "generate_particle_list.sh"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Required file missing: $file"
        exit 1
    fi
done

echo "✅ All required files found"

# Generate particle list if needed
if [[ ! -f "particle_list.txt" ]]; then
    echo "📋 Generating particle list..."
    if ! ./generate_particle_list.sh; then
        echo "❌ Failed to generate particle list"
        exit 1
    fi
else
    echo "✅ Particle list exists"
fi

# Check particle list
if [[ ! -s "particle_list.txt" ]]; then
    echo "❌ Particle list is empty"
    exit 1
fi

TOTAL_PARTICLES=$(wc -l < particle_list.txt)
echo "📊 Found $TOTAL_PARTICLES particles in list"

# Array size equals number of particles (one per task)
ARRAY_SIZE=$TOTAL_PARTICLES

# Handle very large arrays - SLURM often has limits around 1000-5000 array tasks
MAX_ARRAY_SIZE=2000
if [[ $ARRAY_SIZE -gt $MAX_ARRAY_SIZE ]]; then
    echo "⚠️  Large array detected: $ARRAY_SIZE tasks"
    echo "🎯 Consider splitting into multiple submissions or using chunked approach"
    echo "💡 SLURM array limits are typically around 1000-5000 tasks"
    echo ""
    echo "🤔 Options:"
    echo "  1. Proceed with large array (may hit SLURM limits)"
    echo "  2. Reduce to $MAX_ARRAY_SIZE tasks and run multiple times"
    echo "  3. Use chunked brute force approach instead"
    echo ""
    
    if [[ "$FORCE_SUBMIT" != "true" && "$DRY_RUN" != "true" ]]; then
        echo "⚠️  Recommend using chunked approach for >$MAX_ARRAY_SIZE particles"
        echo "🔧 Would you like to:"
        echo "  [1] Proceed with full array (risk SLURM limits)"
        echo "  [2] Submit first $MAX_ARRAY_SIZE particles only"
        echo "  [3] Cancel and use brute_force_gpu_job.sh instead"
        echo ""
        read -p "Enter choice (1/2/3): " choice
        
        case $choice in
            1)
                echo "✅ Proceeding with full array..."
                ;;
            2)
                ARRAY_SIZE=$MAX_ARRAY_SIZE
                echo "✅ Limiting to first $MAX_ARRAY_SIZE particles"
                ;;
            3)
                echo "❌ Cancelled - consider using brute_force_gpu_job.sh for large scale"
                exit 0
                ;;
            *)
                echo "❌ Invalid choice, cancelling"
                exit 1
                ;;
        esac
    fi
fi

# Create necessary directories
mkdir -p logs success_logs failed_jobs

# Display submission summary
echo ""
echo "📋 SUBMISSION SUMMARY"
echo "===================="
echo "🔧 Job script: brute_force_cpu_parallel.sh"
echo "🎯 Array tasks: 1-$ARRAY_SIZE%$CONCURRENT"
echo "🖥️  Partition: $PARTITION"
echo "⏰ Time limit: $TIME_LIMIT"
echo "💾 Memory: $MEMORY"
echo "🧮 CPUs per task: $CPUS_PER_TASK"
echo "📊 Total particles: $TOTAL_PARTICLES (1 per task)"
echo ""

# Calculate resource requirements
echo "📈 RESOURCE ESTIMATION"
echo "====================="
echo "🔢 Total CPU-hours: ~$(( ARRAY_SIZE * CPUS_PER_TASK * 24 / 1000 ))k"
echo "💰 Memory usage: $MEMORY per task"
echo "⏱️  Estimated completion: 24 hours (worst case)"
echo ""

# Dry run check
if [[ "$DRY_RUN" == "true" ]]; then
    echo "🧪 DRY RUN - Command that would be executed:"
    echo "sbatch --array=1-$ARRAY_SIZE%$CONCURRENT --partition=$PARTITION --time=$TIME_LIMIT --mem=$MEMORY --cpus-per-task=$CPUS_PER_TASK brute_force_cpu_parallel.sh"
    echo ""
    echo "✅ Dry run completed - no jobs submitted"
    exit 0
fi

# Confirmation prompt
if [[ "$FORCE_SUBMIT" != "true" ]]; then
    echo "⚠️  This will submit $ARRAY_SIZE array jobs to partition '$PARTITION'"
    echo "🤔 Do you want to proceed? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "❌ Submission cancelled"
        exit 0
    fi
fi

# Submit the job
echo "🚀 Submitting CPU parallel job..."
JOB_ID=$(sbatch \
    --array=1-$ARRAY_SIZE%$CONCURRENT \
    --partition=$PARTITION \
    --time=$TIME_LIMIT \
    --mem=$MEMORY \
    --cpus-per-task=$CPUS_PER_TASK \
    brute_force_cpu_parallel.sh | grep -o '[0-9]\+')

if [[ -n "$JOB_ID" ]]; then
    echo "✅ Job submitted successfully!"
    echo "🆔 Job ID: $JOB_ID"
    echo "📊 Array range: 1-$ARRAY_SIZE%$CONCURRENT"
    echo ""
    echo "📋 MONITORING COMMANDS"
    echo "====================="
    echo "🔍 Check job status:"
    echo "   squeue -u \$USER -j $JOB_ID"
    echo ""
    echo "📊 Monitor progress:"
    echo "   ./monitor_cpu_parallel.sh"
    echo ""
    echo "📁 Check logs:"
    echo "   tail -f logs/cpu_parallel_${JOB_ID}_*.out"
    echo ""
    echo "📈 View progress summary:"
    echo "   cat cpu_parallel_progress/array_progress.log"
    echo ""
    echo "🏁 Happy computing! 🖥️"
else
    echo "❌ Job submission failed"
    exit 1
fi
