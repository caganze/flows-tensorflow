#!/bin/bash

# 📊 Monitor KDE Jobs
# Track progress and status of KDE training

set -e

echo "📊 KDE MONITOR"
echo "=============="
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
    echo "📊 KDE MONITOR - $(date)"
    echo "========================"
    echo "👤 User: $USER"
    echo ""
    
    # Check SLURM queue
    echo "🚀 SLURM QUEUE STATUS"
    echo "--------------------"
    
    # Count jobs by status
    RUNNING=$(squeue -u $USER -t RUNNING -h --name=kde_chunk_* 2>/dev/null | wc -l || echo "0")
    PENDING=$(squeue -u $USER -t PENDING -h --name=kde_chunk_* 2>/dev/null | wc -l || echo "0")
    TOTAL=$((RUNNING + PENDING))
    
    echo "🏃 Running: $RUNNING jobs"
    echo "⏳ Pending: $PENDING jobs"
    echo "📊 Total: $TOTAL jobs"
    echo ""
    
    if [[ $TOTAL -gt 0 ]]; then
        echo "📋 JOB DETAILS"
        echo "--------------"
        if [[ "$SHOW_DETAILS" == "true" ]]; then
            squeue -u $USER --name=kde_chunk_* -o "%.10i %.12j %.8T %.10M %.6D %.20V" 2>/dev/null || echo "No jobs found"
        else
            squeue -u $USER --name=kde_chunk_* -o "%.10i %.12j %.8T %.10M" 2>/dev/null || echo "No jobs found"
        fi
        echo ""
    fi
    
    # Check output directory
    OUTPUT_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/kde_output"
    if [[ -d "$OUTPUT_DIR" ]]; then
        echo "📁 OUTPUT DIRECTORY STATUS"
        echo "-------------------------"
        echo "📂 Location: $OUTPUT_DIR"
        
        # Count completed KDE samples
        SAMPLE_COUNT=$(find "$OUTPUT_DIR/kde_samples" -name "kde_samples_*.h5" 2>/dev/null | wc -l || echo "0")
        
        echo "📊 Completed KDE samples: $SAMPLE_COUNT"
        
        # Check disk usage
        if command -v du >/dev/null; then
            SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1 || echo "unknown")
            echo "💾 Output size: $SIZE"
        fi
        echo ""
    fi
    
    # Check recent logs
    echo "📝 RECENT LOG ACTIVITY"
    echo "---------------------"
    if [[ -d "logs" ]]; then
        RECENT_LOGS=$(find logs -name "kde_*.out" -mmin -60 2>/dev/null | wc -l || echo "0")
        echo "📄 Recent log files (last hour): $RECENT_LOGS"
        
        if [[ $RECENT_LOGS -gt 0 ]]; then
            echo "🕐 Latest activity:"
            find logs -name "kde_*.out" -mmin -60 -exec ls -lt {} \; 2>/dev/null | head -3
        fi
    else
        echo "📁 No logs directory found"
    fi
    echo ""
    
    # Success/failure summary
    if [[ -f "success_logs/kde_success.log" ]]; then
        SUCCESS_TODAY=$(grep "$(date +%Y-%m-%d)" success_logs/kde_success.log 2>/dev/null | wc -l || echo "0")
        echo "✅ Successes today: $SUCCESS_TODAY"
    fi
    
    if [[ -f "failed_jobs/kde_failures.log" ]]; then
        FAILURES_TODAY=$(grep "$(date +%Y-%m-%d)" failed_jobs/kde_failures.log 2>/dev/null | wc -l || echo "0")
        echo "❌ Failures today: $FAILURES_TODAY"
    fi
    echo ""
    
    # KDE-specific stats
    if [[ -f "success_logs/kde_success.log" ]]; then
        echo "📈 KDE STATISTICS"
        echo "----------------"
        TOTAL_SUCCESS=$(wc -l < success_logs/kde_success.log 2>/dev/null || echo "0")
        echo "🎯 Total completed: $TOTAL_SUCCESS"
        
        # Show recent completions
        if [[ $TOTAL_SUCCESS -gt 0 ]]; then
            echo "🕐 Recent completions:"
            tail -3 success_logs/kde_success.log 2>/dev/null | while read line; do
                echo "   $line"
            done
        fi
        echo ""
    fi
    
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
