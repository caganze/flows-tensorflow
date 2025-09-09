#!/bin/bash

# 🖥️  CPU Parallel Chunked Submission
# For large particle lists - submits in manageable chunks

set -e

echo "🖥️  CPU CHUNKED SUBMISSION"
echo "========================="
echo "🚀 Submit large particle lists in manageable chunks"
echo ""

# Default parameters
CHUNK_SIZE=1000
PARTITION="kipac"
TIME_LIMIT="24:00:00"
MEMORY="512GB"
CPUS_PER_TASK=64
CONCURRENT=10
DRY_RUN=false
START_CHUNK=1
PARTICLE_LIST="particle_list.txt"

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --particle-list FILE Particle list file (default: particle_list.txt)"
    echo "  --chunk-size N      Particles per chunk (default: 1000)"
    echo "  --start-chunk N     Starting chunk number (default: 1)"
    echo "  --concurrent N      Max concurrent tasks per chunk (default: 10)"
    echo "  --partition NAME    SLURM partition (default: normal)"
    echo "  --time HOURS        Time limit (default: 24:00:00)"
    echo "  --memory SIZE       Memory per task (default: 512GB)"
    echo "  --cpus N            CPUs per task (default: 64)"
    echo "  --dry-run           Show what would be submitted"
    echo "  --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                              # Submit in 1000-particle chunks"
    echo "  $0 --chunk-size 500             # Smaller chunks"
    echo "  $0 --start-chunk 3              # Start from 3rd chunk (skip first 2000)"
    echo "  $0 --particle-list particle_list_incomplete.txt  # Use filtered list"
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
        --cpus)
            CPUS_PER_TASK="$2"
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
if [[ ! -f "brute_force_cpu_parallel.sh" ]]; then
    echo "❌ brute_force_cpu_parallel.sh not found"
    exit 1
fi

if [[ ! -f "$PARTICLE_LIST" ]]; then
    if [[ "$PARTICLE_LIST" == "particle_list.txt" ]]; then
        echo "📋 Generating particle list..."
        if ! ./generate_particle_list.sh; then
            echo "❌ Failed to generate particle list"
            exit 1
        fi
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
echo "📋 CHUNKED SUBMISSION PLAN"
echo "=========================="
echo "🎯 Chunk size: $CHUNK_SIZE particles"
echo "🔢 Total chunks: $TOTAL_CHUNKS"
echo "🚀 Starting from chunk: $START_CHUNK"
echo "⏰ Time per chunk: $TIME_LIMIT"
echo "💾 Memory per task: $MEMORY"
echo "🧮 CPUs per task: $CPUS_PER_TASK"
echo "🎛️  Concurrent per chunk: $CONCURRENT"
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
    
    echo "📦 Chunk $chunk: particles $start_particle-$end_particle ($chunk_size_actual particles)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  sbatch --array=$start_particle-$end_particle%$CONCURRENT --partition=$PARTITION --time=$TIME_LIMIT --mem=$MEMORY --cpus-per-task=$CPUS_PER_TASK brute_force_cpu_parallel.sh"
    else
        # Submit the job with custom particle list
        JOB_ID=$(sbatch \
            --array=$start_particle-$end_particle%$CONCURRENT \
            --partition=$PARTITION \
            --time=$TIME_LIMIT \
            --mem=$MEMORY \
            --cpus-per-task=$CPUS_PER_TASK \
            --job-name="cpu_chunk_$chunk" \
            --export=PARTICLE_LIST_FILE="$PARTICLE_LIST" \
            brute_force_cpu_parallel.sh | grep -o '[0-9]\+')
        
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
    echo "🎉 SUBMISSION COMPLETED"
    echo "======================"
    echo "📦 Submitted chunks: $((${#SUBMITTED_JOBS[@]}))"
    echo "🆔 Job IDs: ${SUBMITTED_JOBS[*]}"
    echo ""
    echo "📊 Monitor progress:"
    echo "   ./monitor_cpu_parallel.sh --follow"
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
