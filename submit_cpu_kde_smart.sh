#!/bin/bash

# 🧠 Smart CPU Submission - KDE
# Automatically filters completed particles and submits only incomplete ones for KDE training

set -e

# Activate bosque conda environment
source ~/.bashrc
conda activate bosque

echo "🧠 SMART CPU SUBMISSION - KDE"
echo "=============================="
echo "🔍 Filters completed particles before submission"
echo "🚀 Submits only incomplete particles for KDE training"
echo "📁 Output: /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/kde_output/"
echo ""

# Default parameters
CHUNK_SIZE=50  # Smaller chunks for CPU
CONCURRENT=20  # More concurrent for CPU
PARTITION="kipac"  # CPU partition
PARTICLE_LIST_FILE="particle_list.txt"
DRY_RUN=false
VERBOSE=false

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --chunk-size N      Particles per chunk (default: 50)"
    echo "  --concurrent N      Max concurrent tasks per chunk (default: 20)"
    echo "  --partition NAME    SLURM partition (default: kipac)"
    echo "  --particle-list FILE Particle list file (default: particle_list.txt)"
    echo "  --verbose           Show detailed filtering progress"
    echo "  --dry-run           Show what would be submitted without submitting"
    echo "  --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                              # Smart submission with defaults"
    echo "  $0 --chunk-size 100 --concurrent 15  # Larger chunks, fewer concurrent"
    echo "  $0 --particle-list custom_list.txt   # Use custom particle list"
    echo "  $0 --verbose --dry-run          # Preview with detailed info"
    echo ""
    echo "PROCESS:"
    echo "  1. Scans for completed KDE models (.h5 files)"
    echo "  2. Creates filtered list of incomplete particles"
    echo "  3. Submits in manageable chunks to avoid QOS limits"
    echo "  4. Uses CPU partition (kipac) for KDE density estimation"
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
    "filter_completed_kde.sh"
    "submit_cpu_kde_chunked.sh"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Required file missing: $file"
        echo "💡 Run the setup script to create missing files"
        exit 1
    fi
done

echo "✅ All required files found"

# Step 1: Filter completed particles
echo ""
echo "🔍 STEP 1: Filtering completed KDE particles"
echo "============================================"

FILTER_ARGS=""
if [[ "$VERBOSE" == "true" ]]; then
    FILTER_ARGS="--verbose"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    FILTER_ARGS="$FILTER_ARGS --dry-run"
fi

if ! ./filter_completed_kde.sh --input "$PARTICLE_LIST_FILE" $FILTER_ARGS; then
    echo "❌ Failed to filter particles"
    exit 1
fi

# Check if we have any incomplete particles (only if not dry run)
if [[ "$DRY_RUN" != "true" ]]; then
    if [[ ! -f "particle_list_kde_incomplete.txt" || ! -s "particle_list_kde_incomplete.txt" ]]; then
        echo ""
        echo "🎉 ALL KDE MODELS COMPLETED!"
        echo "============================"
        echo "✅ No incomplete particles found - all KDE work is done!"
        exit 0
    fi
    
    INCOMPLETE_COUNT=$(wc -l < particle_list_kde_incomplete.txt)
    echo ""
    echo "📊 Found $INCOMPLETE_COUNT incomplete KDE particles to process"
fi

# Step 2: Submit in chunks
echo ""
echo "🚀 STEP 2: Submitting KDE chunks"
echo "================================"

SUBMIT_ARGS="--chunk-size $CHUNK_SIZE --concurrent $CONCURRENT --partition $PARTITION --particle-list particle_list_kde_incomplete.txt"

if [[ "$DRY_RUN" == "true" ]]; then
    SUBMIT_ARGS="$SUBMIT_ARGS --dry-run"
    echo "🧪 DRY RUN MODE - Preview of KDE submission:"
    echo ""
fi

if ! ./submit_cpu_kde_chunked.sh $SUBMIT_ARGS; then
    echo "❌ Failed to submit KDE chunks"
    exit 1
fi

echo ""
echo "✅ SMART CPU KDE SUBMISSION COMPLETED"
echo "===================================="

if [[ "$DRY_RUN" == "true" ]]; then
    echo "🧪 This was a dry run - no jobs were actually submitted"
    echo "💡 Run without --dry-run to actually submit jobs"
else
    echo "🎯 Only incomplete KDE particles were submitted"
    echo "🔄 Jobs should start running soon (since CPU queue is active)"
    echo "📁 Output directory: /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/kde_output/"
    echo ""
    echo "📊 Monitor progress:"
    echo "   ./monitor_kde.sh --follow"
    echo ""
    echo "🔍 Check job status:"
    echo "   squeue -u \$USER"
fi
