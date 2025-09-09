#!/bin/bash
# Submit flows in smaller batches to work within Sherlock GPU limits
# Based on Sherlock documentation: GPU resources are scarce and have submission limits

echo "🚀 TensorFlow Probability Flows - Batch Submission Strategy"
echo "=========================================================="
echo "Submitting jobs in smaller batches to respect GPU partition limits"
echo

# Configuration
TOTAL_PARTICLES=5000
PARTICLES_PER_TASK=5
TASKS_PER_BATCH=50
MAX_CONCURRENT=10

TOTAL_TASKS=$((TOTAL_PARTICLES / PARTICLES_PER_TASK))
TOTAL_BATCHES=$((TOTAL_TASKS / TASKS_PER_BATCH))

echo "📊 Batch Configuration:"
echo "  Total particles: $TOTAL_PARTICLES"
echo "  Particles per task: $PARTICLES_PER_TASK" 
echo "  Tasks per batch: $TASKS_PER_BATCH"
echo "  Max concurrent: $MAX_CONCURRENT"
echo "  Total tasks needed: $TOTAL_TASKS"
echo "  Total batches: $TOTAL_BATCHES"
echo

# Function to submit a batch
submit_batch() {
    local batch_num=$1
    local start_task=$(( (batch_num - 1) * TASKS_PER_BATCH + 1 ))
    local end_task=$(( batch_num * TASKS_PER_BATCH ))
    
    if [ $end_task -gt $TOTAL_TASKS ]; then
        end_task=$TOTAL_TASKS
    fi
    
    local start_pid=$(( (start_task - 1) * PARTICLES_PER_TASK + 1 ))
    local end_pid=$(( end_task * PARTICLES_PER_TASK ))
    
    echo "📤 Submitting Batch $batch_num:"
    echo "  Tasks: $start_task-$end_task"
    echo "  Particles: $start_pid-$end_pid"
    
    sbatch --job-name="tfp_batch${batch_num}" \
           --array=${start_task}-${end_task}%${MAX_CONCURRENT} \
           submit_flows_array.sh
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "  ✅ Batch $batch_num submitted successfully"
    else
        echo "  ❌ Batch $batch_num failed to submit (exit code: $exit_code)"
        return $exit_code
    fi
    
    echo
    return 0
}

# Submit batches with delay to avoid rate limiting
echo "🎯 Starting batch submissions..."
echo

for batch in $(seq 1 $TOTAL_BATCHES); do
    submit_batch $batch
    
    if [ $? -ne 0 ]; then
        echo "❌ Submission failed at batch $batch"
        echo "💡 Try checking limits with: sacctmgr show user \$USER withassoc"
        exit 1
    fi
    
    # Small delay between submissions
    if [ $batch -lt $TOTAL_BATCHES ]; then
        echo "⏳ Waiting 2 seconds before next batch..."
        sleep 2
    fi
done

echo "🎉 All batches submitted successfully!"
echo
echo "📊 Monitor progress with:"
echo "  squeue -u \$(whoami)"
echo "  watch 'squeue -u \$(whoami)'"
echo
echo "📁 Outputs will be saved to:"
echo "  /oak/stanford/orgs/kipac/users/caganze/tfp_flows_output/"
