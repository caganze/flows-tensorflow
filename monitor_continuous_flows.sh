#!/bin/bash

# 📊 Monitor Continuous Flow Jobs
# Track progress and status of continuous flow training

set -e

echo "📊 CONTINUOUS FLOW MONITOR"
echo "=========================="
echo ""

# Default parameters
FOLLOW=false
SHOW_DETAILS=false
USER=${USER:-$(whoami)}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --follow            Follow mode (update continuously)"
    echo "  --details           Show detailed job information"
    echo "  --user USER         Monitor specific user (default: current user)"
    echo "  --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                  # Quick status check"
    echo "  $0 --follow         # Continuous monitoring"
    echo "  $0 --details        # Detailed job info"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --follow)
            FOLLOW=true
            shift
            ;;
        --details)
            SHOW_DETAILS=true
            shift
            ;;
        --user)
            USER="$2"
            shift 2
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

monitor_once() {
    clear
    echo "📊 CONTINUOUS FLOW MONITOR - $(date)"
    echo "===================================="
    echo "👤 User: $USER"
    echo ""
    
    # Check SLURM queue
    echo "🚀 SLURM QUEUE STATUS"
    echo "--------------------"
    
    # Count jobs by status
    RUNNING=$(squeue -u $USER -t RUNNING -h --name=continuous_chunk_* 2>/dev/null | wc -l || echo "0")
    PENDING=$(squeue -u $USER -t PENDING -h --name=continuous_chunk_* 2>/dev/null | wc -l || echo "0")
    TOTAL=$((RUNNING + PENDING))
    
    echo "🏃 Running: $RUNNING jobs"
    echo "⏳ Pending: $PENDING jobs"
    echo "📊 Total: $TOTAL jobs"
    echo ""
    
    if [[ $TOTAL -gt 0 ]]; then
        echo "📋 JOB DETAILS"
        echo "--------------"
        if [[ "$SHOW_DETAILS" == "true" ]]; then
            squeue -u $USER --name=continuous_chunk_* -o "%.10i %.12j %.8T %.10M %.6D %.20V" 2>/dev/null || echo "No jobs found"
        else
            squeue -u $USER --name=continuous_chunk_* -o "%.10i %.12j %.8T %.10M" 2>/dev/null || echo "No jobs found"
        fi
        echo ""
    fi
    
    # Check output directory
    OUTPUT_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/tfp_output_conditional"
    if [[ -d "$OUTPUT_DIR" ]]; then
        echo "📁 OUTPUT DIRECTORY STATUS"
        echo "-------------------------"
        echo "📂 Location: $OUTPUT_DIR"
        
        # Count completed models
        MODEL_COUNT=$(find "$OUTPUT_DIR/trained_flows" -name "model_pid*.npz" 2>/dev/null | wc -l || echo "0")
        SAMPLE_COUNT=$(find "$OUTPUT_DIR/samples" -name "*_samples.*" 2>/dev/null | wc -l || echo "0")
        
        echo "🧠 Completed models: $MODEL_COUNT"
        echo "📊 Sample files: $SAMPLE_COUNT"
        echo ""
    fi
    
    # Check recent logs
    echo "📝 RECENT LOG ACTIVITY"
    echo "---------------------"
    if [[ -d "logs" ]]; then
        RECENT_LOGS=$(find logs -name "continuous_flow_*.out" -mmin -60 2>/dev/null | wc -l || echo "0")
        echo "📄 Recent log files (last hour): $RECENT_LOGS"
        
        if [[ $RECENT_LOGS -gt 0 ]]; then
            echo "🕐 Latest activity:"
            find logs -name "continuous_flow_*.out" -mmin -60 -exec ls -lt {} \; 2>/dev/null | head -3
        fi
    else
        echo "📁 No logs directory found"
    fi
    echo ""
    
    # Success/failure summary
    if [[ -f "success_logs/continuous_flow_success.log" ]]; then
        SUCCESS_TODAY=$(grep "$(date +%Y-%m-%d)" success_logs/continuous_flow_success.log 2>/dev/null | wc -l || echo "0")
        echo "✅ Successes today: $SUCCESS_TODAY"
    fi
    
    if [[ -f "failed_jobs/continuous_flow_failures.log" ]]; then
        FAILURES_TODAY=$(grep "$(date +%Y-%m-%d)" failed_jobs/continuous_flow_failures.log 2>/dev/null | wc -l || echo "0")
        echo "❌ Failures today: $FAILURES_TODAY"
    fi
    echo ""
    
    echo "🔄 Last updated: $(date)"
    
    if [[ "$FOLLOW" == "true" ]]; then
        echo ""
        echo "💡 Press Ctrl+C to exit follow mode"
        echo "🔄 Updating every 30 seconds..."
    fi
}

if [[ "$FOLLOW" == "true" ]]; then
    while true; do
        monitor_once
        sleep 30
    done
else
    monitor_once
fi
