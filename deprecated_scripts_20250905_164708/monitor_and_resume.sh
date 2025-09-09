#!/bin/bash

# 📊 Monitor and Resume GPU Jobs
# Monitors current jobs and automatically resumes when queue space is available

echo "📊 MONITOR AND RESUME GPU JOBS"
echo "==============================="

PARTICLE_LIST="particle_list_incomplete.txt"
CHUNK_SIZE=200
START_CHUNK=16

echo "🔍 Current job status:"
squeue -u $USER --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

echo ""
echo "📋 Monitoring configuration:"
echo "  Start chunk: $START_CHUNK"
echo "  Chunk size: $CHUNK_SIZE"
echo "  Particle list: $PARTICLE_LIST"

echo ""
echo "⏳ Waiting for queue space..."
echo "   Press Ctrl+C to stop monitoring"

while true; do
    # Count current jobs
    CURRENT_JOBS=$(squeue -u $USER -h | wc -l)
    echo "$(date): Current jobs: $CURRENT_JOBS"
    
    # If we have fewer than 10 jobs, try to submit more
    if [[ $CURRENT_JOBS -lt 10 ]]; then
        echo "🚀 Queue space available! Resuming submission..."
        ./submit_gpu_chunked.sh --start-chunk $START_CHUNK --chunk-size $CHUNK_SIZE --particle-list $PARTICLE_LIST
        
        # If submission succeeded, we're done
        if [[ $? -eq 0 ]]; then
            echo "✅ All chunks submitted successfully!"
            break
        else
            echo "⚠️ Submission failed, will retry in 5 minutes..."
        fi
    fi
    
    # Wait 5 minutes before checking again
    sleep 300
done

echo "🏁 Monitoring complete!"
