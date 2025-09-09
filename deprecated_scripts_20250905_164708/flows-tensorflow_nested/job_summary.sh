#!/bin/bash

# SLURM Job Analysis Script
echo "🔍 SLURM Job Analysis Summary"
echo "============================="

# Get user's username
USER=$(whoami)

# Get all jobs for user (last 24 hours)
echo "📊 Fetching job data..."
JOBS=$(sacct -u $USER -S $(date -d '1 day ago' '+%Y-%m-%d') --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,NodeList -P)

# Parse jobs
echo "$JOBS" | tail -n +2 | while IFS='|' read -r jobid jobname state exitcode start end elapsed nodelist; do
    # Skip if empty or header
    [[ -z "$jobid" || "$jobid" == "JobID" ]] && continue
    
    # Skip sub-jobs (contain dots)
    [[ "$jobid" == *"."* ]] && continue
    
    echo "Job: $jobid | $jobname | $state | Runtime: $elapsed"
done

echo ""
echo "📈 Job Statistics:"

# Count by status
COMPLETED=$(echo "$JOBS" | grep -c "COMPLETED")
FAILED=$(echo "$JOBS" | grep -c "FAILED")
RUNNING=$(echo "$JOBS" | grep -c "RUNNING")
PENDING=$(echo "$JOBS" | grep -c "PENDING")

echo "✅ Completed: $COMPLETED"
echo "❌ Failed: $FAILED" 
echo "🏃 Running: $RUNNING"
echo "⏳ Pending: $PENDING"

echo ""
echo "🕐 Runtime Analysis:"

# Get completed job runtimes
echo "$JOBS" | grep "COMPLETED" | while IFS='|' read -r jobid jobname state exitcode start end elapsed nodelist; do
    [[ "$jobid" == *"."* ]] && continue
    echo "  $jobid: $elapsed"
done

echo ""
echo "❌ Failure Analysis:"

# Check for common failure patterns in recent .err files
echo "Checking recent error logs..."
find logs/ -name "*.err" -mtime -1 2>/dev/null | head -10 | while read errfile; do
    if [[ -s "$errfile" ]]; then
        echo "📄 $errfile:"
        # Look for common error patterns
        if grep -q "OutOfMemoryError\|OOM\|out of memory" "$errfile" 2>/dev/null; then
            echo "  🧠 Memory issue detected"
        fi
        if grep -q "ModuleNotFoundError\|ImportError" "$errfile" 2>/dev/null; then
            echo "  📦 Import/module issue detected"
        fi
        if grep -q "CUDA\|GPU" "$errfile" 2>/dev/null; then
            echo "  🖥️  GPU/CUDA issue detected"
        fi
        if grep -q "Permission denied" "$errfile" 2>/dev/null; then
            echo "  🔒 Permission issue detected"
        fi
        if grep -q "No such file" "$errfile" 2>/dev/null; then
            echo "  📁 File not found issue detected"
        fi
        # Show last few lines of error
        echo "  Last error lines:"
        tail -3 "$errfile" 2>/dev/null | sed 's/^/    /'
    fi
done

echo ""
echo "🎯 Quick Summary:"
TOTAL=$((COMPLETED + FAILED + RUNNING + PENDING))
if [[ $TOTAL -gt 0 ]]; then
    SUCCESS_RATE=$((COMPLETED * 100 / (COMPLETED + FAILED)))
    echo "Success rate: $COMPLETED/$((COMPLETED + FAILED)) ($SUCCESS_RATE%)"
fi

echo "Current queue status:"
squeue --me 2>/dev/null | tail -n +2 | wc -l | xargs echo "Active jobs:"
