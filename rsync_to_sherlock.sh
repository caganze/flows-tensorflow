#!/bin/bash

# 🚀 Rsync to Sherlock
# Upload local changes to Sherlock efficiently

set -e

echo "🚀 RSYNC TO SHERLOCK"
echo "===================="

# Configuration
SHERLOCK_HOST="sherlock.stanford.edu"
SHERLOCK_USER="caganze"  # Use current username
SHERLOCK_PATH="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow"
LOCAL_PATH="."

echo "📋 Configuration:"
echo "   Local path: $(pwd)"
echo "   Remote: ${SHERLOCK_USER}@${SHERLOCK_HOST}:${SHERLOCK_PATH}"
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

# Create remote directory (using existing connection)
echo "📁 Ensuring remote directory exists..."
ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "mkdir -p $SHERLOCK_PATH"

# Main rsync command
echo "📤 Syncing files..."
rsync -avz \
    --progress \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude '.DS_Store' \
    --exclude 'logs/' \
    --exclude 'test_*output*/' \
    --exclude '*.log' \
    --exclude 'slurm-*.out' \
    --exclude 'comprehensive_test_*.out' \
    --exclude 'comprehensive_test_*.err' \
    --exclude 'redundant_scripts_backup_*/' \
    "$LOCAL_PATH/" \
    "$SHERLOCK_USER@$SHERLOCK_HOST:$SHERLOCK_PATH/"

RSYNC_EXIT=$?

if [ $RSYNC_EXIT -eq 0 ]; then
    echo ""
    echo "✅ RSYNC COMPLETED SUCCESSFULLY!"
    echo "================================="
    
    # Make scripts executable (using existing connection)
    echo "🔧 Making scripts executable..."
    ssh $SSH_OPTS "$SHERLOCK_USER@$SHERLOCK_HOST" "cd $SHERLOCK_PATH && chmod +x *.sh *.py"
    
    # Close SSH connection
    echo "🔌 Closing SSH connection..."
    ssh $SSH_OPTS -O exit "$SHERLOCK_USER@$SHERLOCK_HOST" 2>/dev/null || true
    
    echo "📊 Files uploaded to: $SHERLOCK_HOST:$SHERLOCK_PATH"
    echo ""
    echo "🔗 To access Sherlock:"
    echo "   ssh $SHERLOCK_USER@$SHERLOCK_HOST"
    echo "   cd $SHERLOCK_PATH"
    echo ""
    echo "🧪 To test on Sherlock:"
    echo "   module purge"
    echo "   module load math devel python/3.9.0"
    echo "   source ~/.bashrc"
    echo "   conda activate bosque"
    echo "   python final_verification.py"
    echo ""
    echo "🚀 Ready for production testing!"
    
else
    echo "❌ RSYNC FAILED with exit code $RSYNC_EXIT"
    exit 1
fi
