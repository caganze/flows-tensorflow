#!/bin/bash

# 📥 Download Relevant Logs from Sherlock
# SCP script to download diagnostic logs for local analysis

set -e

echo "📥 DOWNLOAD SHERLOCK LOGS"
echo "========================="

# Configuration
SHERLOCK_HOST="sherlock.stanford.edu"
SHERLOCK_USER="caganze"
SHERLOCK_PATH="/oak/stanford/orgs/kipac/users/caganze/old/flows-tensorflow"
LOCAL_LOGS_DIR="sherlock_logs"

# Create local directory
echo "📁 Creating local logs directory..."
mkdir -p "$LOCAL_LOGS_DIR"
cd "$LOCAL_LOGS_DIR"

echo "📋 Configuration:"
echo "   Remote: ${SHERLOCK_USER}@${SHERLOCK_HOST}:${SHERLOCK_PATH}"
echo "   Local: $(pwd)"
echo ""

# Setup SSH connection multiplexing (single password entry)
SSH_CONTROL_PATH="/tmp/ssh_sherlock_%r@%h:%p"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$SSH_CONTROL_PATH -o ControlPersist=300"

# Test connection and establish persistent connection
echo "🔍 Establishing connection (password required once)..."
if ! ssh $SSH_OPTS -o ConnectTimeout=10 "$SHERLOCK_USER@$SHERLOCK_HOST" "echo 'Connection successful'" >/dev/null 2>&1; then
    echo "❌ Cannot connect to Sherlock"
    echo "💡 Make sure you have SSH access and VPN is connected"
    exit 1
fi
echo "✅ Connection established and will persist for 5 minutes"
echo ""

# Get list of recent log files from Sherlock (using existing connection)
echo "🔍 Finding recent log files on Sherlock..."
ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "
cd $SHERLOCK_PATH

echo '=== RECENT LOG FILES ==='
echo 'GPU Brute Force Logs:'
ls -la logs/brute_force_*.out logs/brute_force_*.err 2>/dev/null | tail -5

echo ''
echo 'CPU Parallel Logs:'
ls -la logs/cpu_flows_*.out logs/cpu_flows_*.err 2>/dev/null | tail -5

echo ''
echo 'SLURM System Logs:'
ls -la slurm-*.out slurm-*.err 2>/dev/null | tail -5

echo ''
echo 'Other Logs:'
ls -la logs/*.log logs/*smart* logs/*filter* 2>/dev/null | head -5
"

echo ""
echo "📥 DOWNLOADING PRIORITY LOGS"
echo "============================="

# Download specific known logs first
echo "1️⃣ GPU Brute Force Logs (Priority):"
scp "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/logs/brute_force_5019746.out" . 2>/dev/null && echo "   ✅ brute_force_5019746.out" || echo "   ❌ brute_force_5019746.out (not found)"
scp "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/logs/brute_force_5019746.err" . 2>/dev/null && echo "   ✅ brute_force_5019746.err" || echo "   ❌ brute_force_5019746.err (not found)"

echo ""
echo "2️⃣ CPU Parallel Logs (Sample):"
scp "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/logs/cpu_flows_5019747_24.out" . 2>/dev/null && echo "   ✅ cpu_flows_5019747_24.out" || echo "   ❌ cpu_flows_5019747_24.out (not found)"
scp "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/logs/cpu_flows_5019747_95.err" . 2>/dev/null && echo "   ✅ cpu_flows_5019747_95.err" || echo "   ❌ cpu_flows_5019747_95.err (not found)"

echo ""
echo "3️⃣ Recent GPU Brute Force Logs:"
# Download 3 most recent GPU brute force logs (using existing connection)
ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && ls -t logs/brute_force_*.out 2>/dev/null | head -3" | while read -r logfile; do
    if [[ -n "$logfile" ]]; then
        scp -o ControlPath="$SSH_CONTROL_PATH" "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/$logfile" . && echo "   ✅ $(basename $logfile)"
    fi
done

echo ""
echo "4️⃣ Recent GPU Brute Force Error Logs:"
ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && ls -t logs/brute_force_*.err 2>/dev/null | head -3" | while read -r logfile; do
    if [[ -n "$logfile" ]]; then
        scp -o ControlPath="$SSH_CONTROL_PATH" "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/$logfile" . && echo "   ✅ $(basename $logfile)"
    fi
done

echo ""
echo "5️⃣ Recent CPU Parallel Logs (Sample):"
# Download a few CPU parallel logs to see the pattern (using existing connection)
ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && ls -t logs/cpu_flows_*.out 2>/dev/null | head -3" | while read -r logfile; do
    if [[ -n "$logfile" ]]; then
        scp -o ControlPath="$SSH_CONTROL_PATH" "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/$logfile" . && echo "   ✅ $(basename $logfile)"
    fi
done

ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && ls -t logs/cpu_flows_*.err 2>/dev/null | head -3" | while read -r logfile; do
    if [[ -n "$logfile" ]]; then
        scp -o ControlPath="$SSH_CONTROL_PATH" "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/$logfile" . && echo "   ✅ $(basename $logfile)"
    fi
done

echo ""
echo "6️⃣ SLURM System Logs:"
ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && ls -t slurm-*.out 2>/dev/null | head -2" | while read -r logfile; do
    if [[ -n "$logfile" ]]; then
        scp -o ControlPath="$SSH_CONTROL_PATH" "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/$logfile" . && echo "   ✅ $(basename $logfile)"
    fi
done

ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && ls -t slurm-*.err 2>/dev/null | head -2" | while read -r logfile; do
    if [[ -n "$logfile" ]]; then
        scp -o ControlPath="$SSH_CONTROL_PATH" "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/$logfile" . && echo "   ✅ $(basename $logfile)"
    fi
done

echo ""
echo "7️⃣ Other Important Files:"
# Download job information (using existing connection)
ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && sacct -u $USER --starttime=today --format=JobID,JobName,State,ExitCode,Elapsed,Partition" > job_history.txt && echo "   ✅ job_history.txt"

# Download particle list info (using existing connection)
scp -o ControlPath="$SSH_CONTROL_PATH" "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/particle_list.txt" . 2>/dev/null && echo "   ✅ particle_list.txt" || echo "   ❌ particle_list.txt (not found)"
scp -o ControlPath="$SSH_CONTROL_PATH" "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/particle_list_incomplete.txt" . 2>/dev/null && echo "   ✅ particle_list_incomplete.txt" || echo "   ❌ particle_list_incomplete.txt (not found)"

# Close SSH connection
echo ""
echo "🔌 Closing SSH connection..."
ssh $SSH_OPTS -O exit "$SHERLOCK_USER@$SHERLOCK_HOST" 2>/dev/null || true

echo ""
echo "📊 DOWNLOAD SUMMARY"
echo "==================="
echo "📁 Downloaded to: $(pwd)"
echo "📄 Files downloaded:"
ls -la *.out *.err *.txt 2>/dev/null | wc -l
echo ""

echo "📋 Log files by type:"
echo "   GPU Brute Force: $(ls -1 brute_force_*.out brute_force_*.err 2>/dev/null | wc -l) files"
echo "   CPU Parallel: $(ls -1 cpu_flows_*.out cpu_flows_*.err 2>/dev/null | wc -l) files"
echo "   SLURM System: $(ls -1 slurm-*.out slurm-*.err 2>/dev/null | wc -l) files"
echo "   Other: $(ls -1 *.txt 2>/dev/null | wc -l) files"

echo ""
echo "🔍 QUICK ANALYSIS COMMANDS"
echo "=========================="
echo "📊 Check GPU job progress:"
echo "   grep -n 'Processing.*PID\\|SUCCESS\\|FAILED' brute_force_*.out"
echo ""
echo "📊 Check for errors:"
echo "   grep -i 'error\\|exception\\|fail\\|traceback' *.out *.err"
echo ""
echo "📊 Check job timing:"
echo "   grep -E 'Started:|Completed:|Time:' brute_force_*.out"
echo ""
echo "📊 Check where jobs stopped:"
echo "   tail -50 brute_force_*.out"
echo ""
echo "📊 Check CPU job status:"
echo "   head -20 cpu_flows_*.out | grep -E 'Array Task|Processing|SUCCESS|FAILED'"
echo ""

echo "✅ Download complete! Check the files above for analysis."
