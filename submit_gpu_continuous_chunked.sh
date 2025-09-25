#!/bin/bash

# 🎮 GPU Parallel Chunked Submission - Continuous Flows
# For large particle lists - submits in manageable chunks for continuous flow GPU processing

set -e

# Activate bosque conda environment
source ~/.bashrc
conda activate bosque

echo "🎮 GPU CHUNKED SUBMISSION - CONTINUOUS FLOWS"
echo "============================================="
echo "🚀 Submit large particle lists in manageable chunks for continuous flow training"
echo ""

# Default parameters
CHUNK_SIZE=200
PARTITION="owners"
TIME_LIMIT="12:00:00"
MEMORY="128GB"
GPUS=1
CONCURRENT=3
DRY_RUN=false
START_CHUNK=1
PARTICLE_LIST="particle_list.txt"

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --particle-list FILE Particle list file (default: particle_list.txt)"
    echo "  --chunk-size N      Particles per chunk (default: 200)"
    echo "  --start-chunk N     Starting chunk number (default: 1)"
    echo "  --concurrent N      Max concurrent tasks per chunk (default: 3)"
    echo "  --partition NAME    SLURM partition (default: owners)"
    echo "  --time HOURS        Time limit (default: 12:00:00)"
    echo "  --memory SIZE       Memory per task (default: 128GB)"
    echo "  --gpus N            GPUs per task (fixed: 1, specified in job script)"
    echo "  --dry-run           Show what would be submitted"
    echo "  --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                              # Submit in 200-particle chunks"
    echo "  $0 --chunk-size 100             # Smaller chunks"
    echo "  $0 --start-chunk 3              # Start from 3rd chunk"
    echo "  $0 --particle-list particle_list_continuous_incomplete.txt  # Use filtered list"
    echo "  $0 --dry-run                    # Preview all chunks"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --particle-list)
            PARTICLE_LIST="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --start-chunk)
            START_CHUNK="$2"
            shift 2
            ;;
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
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
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

# Check prerequisites
if [[ ! -f "continuous_flow_gpu_job.sh" ]]; then
    echo "❌ continuous_flow_gpu_job.sh not found"
    exit 1
fi

if [[ ! -f "$PARTICLE_LIST" ]]; then
    if [[ "$PARTICLE_LIST" == "particle_list.txt" ]]; then
        echo "❌ Symlib particle list required. Run: ./generate_all_priority_halos.sh"
        exit 1
    else
        echo "❌ Particle list file not found: $PARTICLE_LIST"
        exit 1
    fi
fi

TOTAL_PARTICLES=$(wc -l < "$PARTICLE_LIST")
echo "📊 Found $TOTAL_PARTICLES particles in list"

# Calculate chunks
TOTAL_CHUNKS=$(( (TOTAL_PARTICLES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo "🔢 Will create $TOTAL_CHUNKS chunks of up to $CHUNK_SIZE particles each"

# Validate start chunk
if [[ $START_CHUNK -gt $TOTAL_CHUNKS ]]; then
    echo "❌ Start chunk $START_CHUNK exceeds total chunks ($TOTAL_CHUNKS)"
    exit 1
fi

echo ""
echo "📋 CONTINUOUS FLOW CHUNKED SUBMISSION PLAN"
echo "==========================================="
echo "🎯 Chunk size: $CHUNK_SIZE particles"
echo "🔢 Total chunks: $TOTAL_CHUNKS"
echo "🚀 Starting from chunk: $START_CHUNK"
echo "⏰ Time per chunk: $TIME_LIMIT"
echo "💾 Memory per task: $MEMORY"
echo "🎮 GPUs per task: 1 (specified in job script)"
echo "🎛️  Concurrent per chunk: $CONCURRENT"
echo "📁 Output: /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output_conditional/"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "🧪 DRY RUN - Commands that would be executed:"
    echo ""
fi

# Submit chunks
SUBMITTED_JOBS=()

for (( chunk=START_CHUNK; chunk<=TOTAL_CHUNKS; chunk++ )); do
    # Calculate range for this chunk
    start_particle=$(( (chunk - 1) * CHUNK_SIZE + 1 ))
    end_particle=$(( chunk * CHUNK_SIZE ))
    
    # Don't exceed total particles
    if [[ $end_particle -gt $TOTAL_PARTICLES ]]; then
        end_particle=$TOTAL_PARTICLES
    fi
    
    chunk_size_actual=$((end_particle - start_particle + 1))
    
    echo "📦 Continuous Flow Chunk $chunk: particles $start_particle-$end_particle ($chunk_size_actual particles)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  sbatch --array=$start_particle-$end_particle%$CONCURRENT --partition=$PARTITION --time=$TIME_LIMIT --mem=$MEMORY continuous_flow_gpu_job.sh"
    else
        # Submit the job with custom particle list
        JOB_ID=$(sbatch \
            --array=$start_particle-$end_particle%$CONCURRENT \
            --partition=$PARTITION \
            --time=$TIME_LIMIT \
            --mem=$MEMORY \
            --job-name="continuous_chunk_$chunk" \
            --export=PARTICLE_LIST_FILE="$PARTICLE_LIST" \
            continuous_flow_gpu_job.sh | grep -o '[0-9]\+')
        
        if [[ -n "$JOB_ID" ]]; then
            echo "  ✅ Submitted: Job ID $JOB_ID"
            SUBMITTED_JOBS+=("$JOB_ID")
        else
            echo "  ❌ Failed to submit chunk $chunk"
        fi
        
        # Brief pause between submissions
        sleep 2
    fi
done

echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "✅ Dry run completed - no jobs submitted"
    echo "💡 Run without --dry-run to actually submit"
else
    echo "🎉 CONTINUOUS FLOW SUBMISSION COMPLETED"
    echo "======================================"
    echo "📦 Submitted chunks: $((${#SUBMITTED_JOBS[@]}))"
    echo "🆔 Job IDs: ${SUBMITTED_JOBS[*]}"
    echo ""
    echo "📊 Monitor progress:"
    echo "   ./monitor_continuous_flows.sh --follow"
    echo ""
    echo "🔍 Check job status:"
    echo "   squeue -u \$USER"
    echo ""
    echo "📈 Track specific jobs:"
    for job in "${SUBMITTED_JOBS[@]}"; do
        echo "   squeue -j $job"
    done
fi

echo ""
echo "💡 To submit remaining chunks later:"
echo "   $0 --start-chunk $((chunk + 1))"
