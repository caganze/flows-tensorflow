#!/bin/bash

# 🚀 Maximize 12-Hour Cluster Usage
# Intelligently uses both GPU and CPU resources over 12 hours

echo "🚀 MAXIMIZE 12-HOUR CLUSTER USAGE"
echo "=================================="
echo "🎯 Goal: Process as many particles as possible in 12 hours"
echo "⏰ Start time: $(date)"
echo ""

# Configuration
MAX_RUNTIME_HOURS=12
PARTICLE_LIST="particle_list_incomplete.txt"
LOG_FILE="12hour_run_$(date +%Y%m%d_%H%M%S).log"

# Strategy parameters
GPU_CHUNK_SIZE=100        # Smaller GPU chunks for faster queue
CPU_CHUNK_SIZE=500        # Larger CPU chunks (CPUs are abundant)
GPU_CONCURRENT=2          # Conservative GPU concurrency
CPU_CONCURRENT=20         # Aggressive CPU concurrency

echo "📋 STRATEGY"
echo "==========="
echo "🎮 GPU: ${GPU_CHUNK_SIZE}-particle chunks, ${GPU_CONCURRENT} concurrent"
echo "🖥️  CPU: ${CPU_CHUNK_SIZE}-particle chunks, ${CPU_CONCURRENT} concurrent"
echo "📝 Logging to: $LOG_FILE"
echo ""

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1

echo "🔍 PHASE 1: Current Status Assessment"
echo "======================================"

# Check current jobs
echo "Current queue status:"
squeue -u $USER --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

CURRENT_GPU_JOBS=$(squeue -u $USER -p owners -h | wc -l)
CURRENT_CPU_JOBS=$(squeue -u $USER -p kipac -h | wc -l)

echo ""
echo "📊 Current jobs: GPU=$CURRENT_GPU_JOBS, CPU=$CURRENT_CPU_JOBS"

# Check particle list
if [[ ! -f "$PARTICLE_LIST" ]]; then
    echo "⚠️ Filtered particle list not found, using full list..."
    PARTICLE_LIST="particle_list.txt"
fi

TOTAL_PARTICLES=$(wc -l < "$PARTICLE_LIST" 2>/dev/null || echo "0")
echo "📊 Particles to process: $TOTAL_PARTICLES"

echo ""
echo "🚀 PHASE 2: Multi-Track Submission"
echo "==================================="

# Function to submit GPU chunks
submit_gpu_chunks() {
    echo "🎮 Submitting GPU chunks..."
    ./submit_gpu_chunked.sh \
        --particle-list "$PARTICLE_LIST" \
        --chunk-size $GPU_CHUNK_SIZE \
        --concurrent $GPU_CONCURRENT \
        --time "12:00:00" \
        --dry-run
    
    read -p "Submit GPU chunks? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./submit_gpu_chunked.sh \
            --particle-list "$PARTICLE_LIST" \
            --chunk-size $GPU_CHUNK_SIZE \
            --concurrent $GPU_CONCURRENT \
            --time "12:00:00"
    fi
}

# Function to submit CPU chunks  
submit_cpu_chunks() {
    echo "🖥️ Submitting CPU chunks..."
    ./submit_cpu_chunked.sh \
        --particle-list "$PARTICLE_LIST" \
        --chunk-size $CPU_CHUNK_SIZE \
        --concurrent $CPU_CONCURRENT \
        --time "12:00:00" \
        --dry-run
        
    read -p "Submit CPU chunks? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./submit_cpu_chunked.sh \
            --particle-list "$PARTICLE_LIST" \
            --chunk-size $CPU_CHUNK_SIZE \
            --concurrent $CPU_CONCURRENT \
            --time "12:00:00"
    fi
}

# Submit to both GPU and CPU
echo "🎯 Strategy: Submit to BOTH GPU and CPU partitions"
echo "   GPU will be faster per particle, CPU will have more availability"
echo ""

submit_gpu_chunks
echo ""
submit_cpu_chunks

echo ""
echo "⏰ PHASE 3: Continuous Monitoring (12 hours)"
echo "=============================================="

END_TIME=$(($(date +%s) + MAX_RUNTIME_HOURS * 3600))

while [[ $(date +%s) -lt $END_TIME ]]; do
    CURRENT_TIME=$(date)
    REMAINING_HOURS=$(( (END_TIME - $(date +%s)) / 3600 ))
    
    echo "[$CURRENT_TIME] ⏰ Time remaining: ${REMAINING_HOURS}h"
    
    # Check job status
    GPU_RUNNING=$(squeue -u $USER -p owners -t RUNNING -h | wc -l)
    GPU_PENDING=$(squeue -u $USER -p owners -t PENDING -h | wc -l)
    CPU_RUNNING=$(squeue -u $USER -p kipac -t RUNNING -h | wc -l)
    CPU_PENDING=$(squeue -u $USER -p kipac -t PENDING -h | wc -l)
    
    echo "   📊 GPU: ${GPU_RUNNING} running, ${GPU_PENDING} pending"
    echo "   📊 CPU: ${CPU_RUNNING} running, ${CPU_PENDING} pending"
    
    # Check for completed particles
    if [[ -f "particle_list.txt" ]]; then
        ./filter_completed_particles.sh --dry-run > /dev/null 2>&1
        COMPLETED_COUNT=$(grep -c "✅ Completed" /tmp/filter_output 2>/dev/null || echo "0")
        echo "   ✅ Estimated completed: $COMPLETED_COUNT particles"
    fi
    
    # Adaptive resubmission
    TOTAL_JOBS=$((GPU_RUNNING + GPU_PENDING + CPU_RUNNING + CPU_PENDING))
    
    if [[ $TOTAL_JOBS -lt 5 && $REMAINING_HOURS -gt 1 ]]; then
        echo "   🚀 Low job count, attempting to submit more..."
        
        # Try GPU first (faster)
        if [[ $GPU_PENDING -lt 3 ]]; then
            echo "   🎮 Adding more GPU jobs..."
            ./submit_gpu_chunked.sh \
                --particle-list "$PARTICLE_LIST" \
                --chunk-size $((GPU_CHUNK_SIZE / 2)) \
                --concurrent 1 \
                --time "$((REMAINING_HOURS)):00:00" > /dev/null 2>&1 || true
        fi
        
        # Then CPU (more availability)
        if [[ $CPU_PENDING -lt 5 ]]; then
            echo "   🖥️ Adding more CPU jobs..."
            ./submit_cpu_chunked.sh \
                --particle-list "$PARTICLE_LIST" \
                --chunk-size $CPU_CHUNK_SIZE \
                --concurrent 5 \
                --time "$((REMAINING_HOURS)):00:00" > /dev/null 2>&1 || true
        fi
    fi
    
    # Sleep for 10 minutes before next check
    sleep 600
done

echo ""
echo "🏁 PHASE 4: 12-Hour Run Complete"
echo "================================="
echo "⏰ End time: $(date)"

# Final status
echo "📊 Final job status:"
squeue -u $USER --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

echo ""
echo "✅ 12-hour maximization run complete!"
echo "📝 Full log saved to: $LOG_FILE"
echo ""
echo "🔍 Next steps:"
echo "   1. Check completed particles: ./filter_completed_particles.sh --dry-run"
echo "   2. Download results: ./download_sherlock_logs.sh"
echo "   3. Analyze completion rate"
