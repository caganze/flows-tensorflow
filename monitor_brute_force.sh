#!/bin/bash

# 📊 Monitor Brute Force GPU Jobs
# Tracks progress of GPU array jobs

set -e

FOLLOW=false
WATCH_INTERVAL=30

show_usage() {
    echo "📊 Monitor Brute Force GPU Jobs"
    echo "=============================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --follow            Continuously monitor (refresh every 30s)"
    echo "  --interval N        Refresh interval in seconds (default: 30)"
    echo "  --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                  # One-time status check"
    echo "  $0 --follow         # Continuous monitoring"
    echo "  $0 --follow --interval 60  # Monitor with 60s refresh"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --follow)
            FOLLOW=true
            shift
            ;;
        --interval)
            WATCH_INTERVAL="$2"
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
    echo "📊 GPU BRUTE FORCE MONITORING"
    echo "============================="
    echo "🕐 Time: $(date)"
    echo ""
    
    # Check job queue
    echo "🔍 ACTIVE JOBS"
    echo "=============="
    if JOBS=$(squeue --me --format="%.10i %.15j %.8T %.10M %.6D %R" 2>/dev/null); then
        if [[ $(echo "$JOBS" | wc -l) -gt 1 ]]; then
            echo "$JOBS"
            echo ""
            
            # Count job states
            RUNNING=$(echo "$JOBS" | grep -c " R " || echo "0")
            PENDING=$(echo "$JOBS" | grep -c " PD " || echo "0")
            COMPLETING=$(echo "$JOBS" | grep -c " CG " || echo "0")
            
            echo "📈 Job Summary:"
            echo "  🏃 Running: $RUNNING"
            echo "  ⏳ Pending: $PENDING"
            echo "  🏁 Completing: $COMPLETING"
        else
            echo "  ✅ No active jobs found"
        fi
    else
        echo "  ❌ Could not query job status"
    fi
    
    echo ""
    
    # Check recent completions
    echo "📈 RECENT COMPLETIONS"
    echo "===================="
    
    # Look for recent success logs
    if [[ -f "success_logs/brute_force_success.log" ]]; then
        RECENT_SUCCESS=$(tail -10 success_logs/brute_force_success.log 2>/dev/null | wc -l)
        if [[ $RECENT_SUCCESS -gt 0 ]]; then
            echo "✅ Recent successes (last 10):"
            tail -10 success_logs/brute_force_success.log 2>/dev/null | while read -r line; do
                echo "  $line"
            done
        else
            echo "  No recent successes logged"
        fi
    else
        echo "  No success log found"
    fi
    
    echo ""
    
    # Check recent failures
    echo "❌ RECENT FAILURES"
    echo "=================="
    
    if [[ -f "failed_jobs/brute_force_failures.log" ]]; then
        RECENT_FAILURES=$(tail -5 failed_jobs/brute_force_failures.log 2>/dev/null | wc -l)
        if [[ $RECENT_FAILURES -gt 0 ]]; then
            echo "❌ Recent failures (last 5):"
            tail -5 failed_jobs/brute_force_failures.log 2>/dev/null | while read -r line; do
                echo "  $line"
            done
        else
            echo "  No recent failures logged"
        fi
    else
        echo "  No failure log found"
    fi
    
    echo ""
    
    # Check log directory
    echo "📁 LOG FILES"
    echo "============"
    
    if [[ -d "logs" ]]; then
        RECENT_LOGS=$(find logs -name "brute_force_*.out" -mtime -1 2>/dev/null | wc -l)
        echo "📄 Recent log files (last 24h): $RECENT_LOGS"
        
        if [[ $RECENT_LOGS -gt 0 ]]; then
            echo "🔍 Most recent logs:"
            find logs -name "brute_force_*.out" -mtime -1 2>/dev/null | head -5 | while read -r logfile; do
                SIZE=$(du -h "$logfile" 2>/dev/null | cut -f1)
                MODIFIED=$(stat -f "%Sm" -t "%H:%M" "$logfile" 2>/dev/null || echo "??:??")
                echo "  $(basename $logfile) ($SIZE, $MODIFIED)"
            done
        fi
    else
        echo "  No logs directory found"
    fi
    
    echo ""
    
    # System resources
    echo "🖥️  SYSTEM STATUS"
    echo "================"
    
    # Check GPU queue
    if GPU_QUEUE=$(squeue -p owners --format="%.10i %.8T" 2>/dev/null | grep -v JOBID); then
        GPU_JOBS=$(echo "$GPU_QUEUE" | wc -l)
        echo "🎮 Total GPU jobs in owners partition: $GPU_JOBS"
    fi
    
    # Quick disk usage check
    if [[ -d "/oak/stanford/orgs/kipac/users/caganze" ]]; then
        DISK_USAGE=$(du -sh /oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/tfp_output 2>/dev/null | cut -f1 || echo "N/A")
        echo "💾 Output directory size: $DISK_USAGE"
    fi
    
    echo ""
    echo "💡 QUICK COMMANDS"
    echo "================="
    echo "📊 Check detailed job status:     squeue --me -l"
    echo "🔍 Download latest logs:          ./download_sherlock_logs.sh"
    echo "❌ Cancel all jobs:               scancel -u \$USER"
    echo "🚀 Submit smart GPU jobs:         ./submit_gpu_smart.sh"
    echo ""
}

if [[ "$FOLLOW" == "true" ]]; then
    echo "🔄 CONTINUOUS MONITORING MODE"
    echo "============================="
    echo "Refreshing every ${WATCH_INTERVAL}s (Press Ctrl+C to stop)"
    echo ""
    
    while true; do
        clear
        monitor_once
        echo "⏰ Next refresh in ${WATCH_INTERVAL}s..."
        sleep "$WATCH_INTERVAL"
    done
else
    monitor_once
fi