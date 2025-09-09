#!/bin/bash

# 🔄 Combined Upload/Download Script for Sherlock
# Single password entry for both operations

set -e

echo "🔄 SHERLOCK SYNC UTILITY"
echo "========================"

# Configuration
SHERLOCK_HOST="sherlock.stanford.edu"
SHERLOCK_USER="caganze"
SHERLOCK_PATH="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow"

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --upload            Upload local files to Sherlock"
    echo "  --download          Download logs from Sherlock" 
    echo "  --both              Upload files then download logs (default)"
    echo "  --logs-only         Download logs only (same as --download)"
    echo "  --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                  # Upload files, then download logs"
    echo "  $0 --upload         # Upload files only"
    echo "  $0 --download       # Download logs only"
    echo "  $0 --both           # Upload then download (explicit)"
    echo ""
}

# Parse arguments
OPERATION="both"

while [[ $# -gt 0 ]]; do
    case $1 in
        --upload)
            OPERATION="upload"
            shift
            ;;
        --download|--logs-only)
            OPERATION="download"
            shift
            ;;
        --both)
            OPERATION="both"
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

echo "🎯 Operation: $OPERATION"
echo "📍 Remote: ${SHERLOCK_USER}@${SHERLOCK_HOST}:${SHERLOCK_PATH}"
echo ""

# Setup SSH connection multiplexing (single password entry)
SSH_CONTROL_PATH="/tmp/ssh_sherlock_%r@%h:%p"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$SSH_CONTROL_PATH -o ControlPersist=600"

# Establish persistent connection (only password prompt)
echo "🔍 Establishing connection (password required once)..."
if ! ssh $SSH_OPTS -o ConnectTimeout=10 "$SHERLOCK_USER@$SHERLOCK_HOST" "echo 'Connection successful'" >/dev/null 2>&1; then
    echo "❌ Cannot connect to Sherlock"
    echo "💡 Make sure you have SSH access and VPN is connected"
    exit 1
fi
echo "✅ Connection established and will persist for 10 minutes"
echo ""

# Upload files if requested
if [[ "$OPERATION" == "upload" || "$OPERATION" == "both" ]]; then
    echo "📤 UPLOADING FILES TO SHERLOCK"
    echo "==============================="
    
    # Create remote directory
    ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "mkdir -p $SHERLOCK_PATH"
    
    # Upload files
    rsync -avz \
        --progress \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude '.git/' \
        --exclude '.DS_Store' \
        --exclude 'logs/' \
        --exclude 'sherlock_logs/' \
        --exclude 'test_*output*/' \
        --exclude '*.log' \
        --exclude 'slurm-*.out' \
        --exclude 'comprehensive_test_*.out' \
        --exclude 'comprehensive_test_*.err' \
        --exclude 'redundant_scripts_backup_*/' \
        ./ \
        "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/"
    
    if [ $? -eq 0 ]; then
        echo "✅ Upload successful!"
        
        # Make scripts executable
        ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && chmod +x *.sh *.py"
        echo "🔧 Scripts made executable"
    else
        echo "❌ Upload failed!"
        ssh $SSH_OPTS -O exit "$SHERLOCK_USER@$SHERLOCK_HOST" 2>/dev/null || true
        exit 1
    fi
    
    echo ""
fi

# Download logs if requested
if [[ "$OPERATION" == "download" || "$OPERATION" == "both" ]]; then
    echo "📥 DOWNLOADING LOGS FROM SHERLOCK"
    echo "=================================="
    
    # Create local logs directory
    mkdir -p sherlock_logs_new
    cd sherlock_logs_new
    
    # Download recent logs
    echo "🔍 Downloading recent GPU brute force logs..."
    ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && ls -t logs/brute_force_*.out logs/brute_force_*.err 2>/dev/null | head -5" | while read -r logfile; do
        if [[ -n "$logfile" ]]; then
            scp -o ControlPath="$SSH_CONTROL_PATH" "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/$logfile" . && echo "   ✅ $(basename $logfile)"
        fi
    done
    
    echo ""
    echo "🔍 Downloading recent CPU parallel logs..."
    ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && ls -t logs/cpu_flows_*.out logs/cpu_flows_*.err 2>/dev/null | head -5" | while read -r logfile; do
        if [[ -n "$logfile" ]]; then
            scp -o ControlPath="$SSH_CONTROL_PATH" "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/$logfile" . && echo "   ✅ $(basename $logfile)"
        fi
    done
    
    echo ""
    echo "🔍 Downloading submit_tfp_array logs..."
    ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && ls -t logs/tfp_*.out logs/tfp_*.err 2>/dev/null | head -5" | while read -r logfile; do
        if [[ -n "$logfile" ]]; then
            scp -o ControlPath="$SSH_CONTROL_PATH" "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/$logfile" . && echo "   ✅ $(basename $logfile)"
        fi
    done
    
    echo ""
    echo "🔍 Downloading job history..."
    ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && sacct -u $USER --starttime=today --format=JobID,JobName,State,ExitCode,Elapsed,Partition" > job_history.txt && echo "   ✅ job_history.txt"
    
    cd ..
    echo "✅ Logs downloaded to: sherlock_logs_new/"
    
    echo ""
fi

# Close SSH connection
echo "🔌 Closing SSH connection..."
ssh $SSH_OPTS -O exit "$SHERLOCK_USER@$SHERLOCK_HOST" 2>/dev/null || true

echo ""
echo "🎉 SYNC COMPLETE!"
echo "================="

if [[ "$OPERATION" == "upload" || "$OPERATION" == "both" ]]; then
    echo "✅ Files uploaded to Sherlock"
fi

if [[ "$OPERATION" == "download" || "$OPERATION" == "both" ]]; then
    echo "✅ Logs downloaded from Sherlock"
    echo "📁 Check: sherlock_logs_new/"
fi

echo ""
echo "🔗 To access Sherlock manually:"
echo "   ssh $SHERLOCK_USER@$SHERLOCK_HOST"
echo "   cd $SHERLOCK_PATH"

