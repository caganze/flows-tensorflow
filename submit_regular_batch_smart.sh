#!/bin/bash

# 🧠 Smart Regular Flow Batch Submission (50 Array Limit)
# Groups remaining particles into 50 batches and runs them in for loops

set -e

# Activate bosque conda environment
source ~/.bashrc
conda activate bosque

echo "🧠 SMART REGULAR FLOW BATCH SUBMISSION (50 ARRAY LIMIT)"
echo "======================================================"
echo "🔍 Filters completed particles and groups into 50 batches"
echo "📁 Output: /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output/"
echo ""

# Default parameters
MAX_ARRAYS=50
PARTITION="owners"
PARTICLE_LIST_FILE="particle_list.txt"
DRY_RUN=false
VERBOSE=false

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --max-arrays N      Maximum array jobs (default: 50)"
    echo "  --partition NAME    SLURM partition (default: owners)"
    echo "  --particle-list FILE Particle list file (default: particle_list.txt)"
    echo "  --verbose           Show detailed filtering progress"
    echo "  --dry-run           Show what would be submitted without submitting"
    echo "  --help              Show this help"
    echo ""
    echo "PROCESS:"
    echo "  1. Filters completed regular flow particles"
    echo "  2. Groups remaining particles into 50 batches"
    echo "  3. Each batch runs particles in for loop (no array limit hit)"
    echo "  4. 12 hour time limit per batch, GPU processing"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-arrays)
            MAX_ARRAYS="$2"
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
    "regular_batch_job.sh"
    "filter_completed_particles.sh"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Required file missing: $file"
        echo "💡 Creating missing files..."
        exit 1
    fi
done

echo "✅ All required files found"

# Step 1: Filter completed particles
echo ""
echo "🔍 STEP 1: Filtering completed regular flow particles"
echo "=================================================="

FILTER_ARGS=""
if [[ "$VERBOSE" == "true" ]]; then
    FILTER_ARGS="--verbose"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    FILTER_ARGS="$FILTER_ARGS --dry-run"
fi

if ! ./filter_completed_particles.sh --input "$PARTICLE_LIST_FILE" $FILTER_ARGS; then
    echo "❌ Failed to filter particles"
    exit 1
fi

# Check if we have any incomplete particles (only if not dry run)
if [[ "$DRY_RUN" != "true" ]]; then
    if [[ ! -f "particle_list_incomplete.txt" || ! -s "particle_list_incomplete.txt" ]]; then
        echo ""
        echo "🎉 ALL REGULAR FLOW PARTICLES COMPLETED!"
        echo "======================================="
        echo "✅ No incomplete particles found - all regular flow work is done!"
        exit 0
    fi
    
    INCOMPLETE_COUNT=$(wc -l < particle_list_incomplete.txt)
    echo ""
    echo "📊 Found $INCOMPLETE_COUNT incomplete regular flow particles to process"
else
    # For dry run, estimate from filter output
    INCOMPLETE_COUNT=$(grep "Would contain" <<< "$(./filter_completed_particles.sh --dry-run 2>/dev/null)" | grep -o '[0-9]\+' || echo "460")
    echo ""
    echo "📊 Estimated $INCOMPLETE_COUNT incomplete particles (dry run)"
fi

# Step 2: Create batches
echo ""
echo "🚀 STEP 2: Creating batches for submission"
echo "=========================================="

PARTICLES_PER_BATCH=$(( (INCOMPLETE_COUNT + MAX_ARRAYS - 1) / MAX_ARRAYS ))
echo "📊 Grouping $INCOMPLETE_COUNT particles into $MAX_ARRAYS batches"
echo "📦 ~$PARTICLES_PER_BATCH particles per batch"

echo ""
echo "📋 REGULAR FLOW BATCH SUBMISSION PLAN"
echo "===================================="
echo "🎯 Max arrays: $MAX_ARRAYS"
echo "📦 Particles per batch: ~$PARTICLES_PER_BATCH"
echo "⏰ Time per batch: 12:00:00"
echo "💾 Memory per batch: 128GB"
echo "🎮 GPUs per batch: 1"
echo "📁 Output: /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output/"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "🧪 DRY RUN - Commands that would be executed:"
    echo ""
    
    for (( batch=1; batch<=MAX_ARRAYS; batch++ )); do
        start_line=$(( (batch - 1) * PARTICLES_PER_BATCH + 1 ))
        end_line=$(( batch * PARTICLES_PER_BATCH ))
        
        if [[ $start_line -le $INCOMPLETE_COUNT ]]; then
            if [[ $end_line -gt $INCOMPLETE_COUNT ]]; then
                end_line=$INCOMPLETE_COUNT
            fi
            
            actual_particles=$((end_line - start_line + 1))
            echo "📦 Batch $batch: lines $start_line-$end_line ($actual_particles particles)"
            echo "  sbatch --job-name=regular_batch_$batch --time=12:00:00 --mem=128GB --gres=gpu:1 --partition=$PARTITION regular_batch_job.sh $start_line $end_line"
        fi
    done
    
    echo ""
    echo "✅ Dry run completed - no jobs submitted"
    echo "💡 Run without --dry-run to actually submit"
else
    # Step 3: Submit batches
    echo "🚀 STEP 3: Submitting regular flow batches"
    echo "========================================="
    
    SUBMITTED_JOBS=()
    
    for (( batch=1; batch<=MAX_ARRAYS; batch++ )); do
        start_line=$(( (batch - 1) * PARTICLES_PER_BATCH + 1 ))
        end_line=$(( batch * PARTICLES_PER_BATCH ))
        
        if [[ $start_line -le $INCOMPLETE_COUNT ]]; then
            if [[ $end_line -gt $INCOMPLETE_COUNT ]]; then
                end_line=$INCOMPLETE_COUNT
            fi
            
            actual_particles=$((end_line - start_line + 1))
            echo "📦 Submitting Batch $batch: lines $start_line-$end_line ($actual_particles particles)"
            
            JOB_ID=$(sbatch \
                --job-name="regular_batch_$batch" \
                --time=12:00:00 \
                --mem=128GB \
                --gres=gpu:1 \
                --partition=$PARTITION \
                --output="logs/regular_batch_%j.out" \
                --error="logs/regular_batch_%j.err" \
                regular_batch_job.sh $start_line $end_line | grep -o '[0-9]\+')
            
            if [[ -n "$JOB_ID" ]]; then
                echo "  ✅ Submitted: Job ID $JOB_ID"
                SUBMITTED_JOBS+=("$JOB_ID")
            else
                echo "  ❌ Failed to submit batch $batch"
            fi
            
            # Brief pause between submissions
            sleep 1
        fi
    done
    
    echo ""
    echo "🎉 REGULAR FLOW BATCH SUBMISSION COMPLETED"
    echo "========================================="
    echo "📦 Submitted batches: $((${#SUBMITTED_JOBS[@]}))"
    echo "🆔 Job IDs: ${SUBMITTED_JOBS[*]}"
    echo ""
    echo "📊 Monitor progress:"
    echo "   ./monitor_brute_force.sh --follow"
    echo ""
    echo "🔍 Check job status:"
    echo "   squeue -u \$USER"
fi

echo ""
echo "💡 Each batch will process particles sequentially in a for loop"
echo "🎯 No array job limits hit - stays within 50 job limit"


