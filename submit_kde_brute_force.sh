#!/bin/bash

# 🚀 Brute Force KDE Submission
# Submit KDE jobs for ALL particles without filtering

set -e

# Activate bosque conda environment
source ~/.bashrc
conda activate bosque

echo "🚀 BRUTE FORCE KDE SUBMISSION"
echo "============================="
echo "🎯 Submit KDE jobs for ALL particles (no filtering)"
echo "📁 Output: /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/kde_output/"
echo ""

# Default parameters
CHUNK_SIZE=50  # Small chunks for CPU
CONCURRENT=10  # Moderate concurrent for stability
PARTITION="kipac"
PARTICLE_LIST_FILE="particle_list.txt"
DRY_RUN=false
VERBOSE=false

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --chunk-size N      Particles per chunk (default: 50)"
    echo "  --concurrent N      Max concurrent tasks per chunk (default: 10)"
    echo "  --partition NAME    SLURM partition (default: kipac)"
    echo "  --particle-list FILE Particle list file (default: particle_list.txt)"
    echo "  --verbose           Show detailed progress"
    echo "  --dry-run           Show what would be submitted without submitting"
    echo "  --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                              # Brute force all particles"
    echo "  $0 --chunk-size 30 --concurrent 15  # Adjust parameters"
    echo "  $0 --verbose --dry-run          # Preview submission"
    echo ""
    echo "WARNING: This submits ALL particles without checking for completion!"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --chunk-size)
            CHUNK_SIZE="$2"
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
        --particle-list)
            PARTICLE_LIST_FILE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
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

# Check required files
required_files=(
    "kde_cpu_job.sh"
    "$PARTICLE_LIST_FILE"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Required file missing: $file"
        exit 1
    fi
done

echo "✅ All required files found"

# Get particle count
TOTAL_PARTICLES=$(wc -l < "$PARTICLE_LIST_FILE")
echo "📊 Found $TOTAL_PARTICLES particles in list"

# Calculate chunks
TOTAL_CHUNKS=$(( (TOTAL_PARTICLES + CHUNK_SIZE - 1) / CHUNK_SIZE ))
echo "🔢 Will create $TOTAL_CHUNKS chunks of up to $CHUNK_SIZE particles each"

echo ""
echo "📋 BRUTE FORCE KDE SUBMISSION PLAN"
echo "=================================="
echo "🎯 Chunk size: $CHUNK_SIZE particles"
echo "🔢 Total chunks: $TOTAL_CHUNKS"
echo "⏰ Time per chunk: 6:00:00"
echo "💾 Memory per task: 32GB"
echo "💻 CPUs per task: 4"
echo "🎛️  Concurrent per chunk: $CONCURRENT"
echo "📁 Output: /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/kde_output/"
echo "⚠️  WARNING: This will submit ALL particles (no completion check)"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "🧪 DRY RUN - Commands that would be executed:"
    echo ""
fi

# Submit chunks
SUBMITTED_JOBS=()

for (( chunk=1; chunk<=TOTAL_CHUNKS; chunk++ )); do
    # Calculate range for this chunk
    start_particle=$(( (chunk - 1) * CHUNK_SIZE + 1 ))
    end_particle=$(( chunk * CHUNK_SIZE ))
    
    # Don't exceed total particles
    if [[ $end_particle -gt $TOTAL_PARTICLES ]]; then
        end_particle=$TOTAL_PARTICLES
    fi
    
    chunk_size_actual=$((end_particle - start_particle + 1))
    
    echo "📦 KDE Brute Force Chunk $chunk: particles $start_particle-$end_particle ($chunk_size_actual particles)"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  sbatch --array=$start_particle-$end_particle%$CONCURRENT --partition=$PARTITION --time=6:00:00 --mem=32GB --cpus-per-task=4 kde_cpu_job.sh"
    else
        # Submit the job
        JOB_ID=$(sbatch \
            --array=$start_particle-$end_particle%$CONCURRENT \
            --partition=$PARTITION \
            --time=6:00:00 \
            --mem=32GB \
            --cpus-per-task=4 \
            --job-name="kde_brute_$chunk" \
            --export=PARTICLE_LIST_FILE="$PARTICLE_LIST_FILE" \
            kde_cpu_job.sh | grep -o '[0-9]\+')
        
        if [[ -n "$JOB_ID" ]]; then
            echo "  ✅ Submitted: Job ID $JOB_ID"
            SUBMITTED_JOBS+=("$JOB_ID")
        else
            echo "  ❌ Failed to submit chunk $chunk"
        fi
        
        # Brief pause between submissions
        sleep 1
    fi
done

echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "✅ Dry run completed - no jobs submitted"
    echo "💡 Run without --dry-run to actually submit"
else
    echo "🎉 BRUTE FORCE KDE SUBMISSION COMPLETED"
    echo "======================================"
    echo "📦 Submitted chunks: $((${#SUBMITTED_JOBS[@]}))"
    echo "🆔 Job IDs: ${SUBMITTED_JOBS[*]}"
    echo ""
    echo "📊 Monitor progress:"
    echo "   ./monitor_kde.sh --follow"
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
echo "⚠️  IMPORTANT: This was a brute force submission!"
echo "   All particles were submitted regardless of completion status"
echo "   Use ./submit_cpu_kde_smart.sh for intelligent filtering next time"


