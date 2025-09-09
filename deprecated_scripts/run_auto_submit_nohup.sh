#!/bin/bash

echo "🚀 STARTING AUTO-SUBMISSION WITH NOHUP"
echo "======================================"
echo "This will run in the background and continue even if you disconnect"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Get timestamp for unique log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/auto_submit_${TIMESTAMP}.log"

echo "📝 Auto-submission will log to: $LOG_FILE"
echo "📊 Monitor progress with: tail -f $LOG_FILE"
echo "🛑 Stop with: pkill -f auto_submit_flows.sh"
echo ""

# Start the auto-submission in background with nohup
echo "Starting auto-submission in background..."
nohup ./auto_submit_flows.sh > "$LOG_FILE" 2>&1 &

# Get the process ID
PID=$!
echo "✅ Auto-submission started!"
echo "   Process ID: $PID"
echo "   Log file: $LOG_FILE"
echo ""

# Show initial status
echo "📋 Initial status:"
echo "   Time: $(date)"
echo "   Working directory: $(pwd)"
echo "   Process running: $(ps -p $PID -o pid,cmd --no-headers 2>/dev/null || echo 'Process not found')"
echo ""

echo "🔍 MONITORING COMMANDS:"
echo "======================"
echo "# Watch live progress:"
echo "tail -f $LOG_FILE"
echo ""
echo "# Check if still running:"
echo "ps aux | grep auto_submit_flows"
echo ""
echo "# Stop the process:"
echo "pkill -f auto_submit_flows.sh"
echo "# OR kill specific PID:"
echo "kill $PID"
echo ""
echo "# Check queue status:"
echo "squeue --me"
echo ""

echo "🎯 The auto-submission is now running in the background!"
echo "   You can safely disconnect from Sherlock"
echo "   Check back later with: tail -f $LOG_FILE"
