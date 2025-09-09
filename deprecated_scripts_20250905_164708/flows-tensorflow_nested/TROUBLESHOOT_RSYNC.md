# 🔧 Troubleshooting rsync to Sherlock

## Common Issues and Solutions

### 1. **Test SSH Connection First**
```bash
# Test if you can SSH to Sherlock
ssh caganze@login.sherlock.stanford.edu

# If this fails, you need to:
# - Be on Stanford network or VPN
# - Have valid Sherlock account
# - Check your username
```

### 2. **Test Basic rsync**
```bash
# Test with a single small file first
rsync -av README.md caganze@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/

# If this works, then try all files
rsync -av * caganze@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/
```

### 3. **Check Directory Permissions**
```bash
# SSH to Sherlock and check if directory exists
ssh caganze@login.sherlock.stanford.edu "ls -la /oak/stanford/orgs/kipac/users/caganze/"

# Create directory if it doesn't exist
ssh caganze@login.sherlock.stanford.edu "mkdir -p /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow"
```

### 4. **Alternative Methods**

#### Method A: Using scp for individual files
```bash
scp brute_force_gpu_job.sh caganze@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/
scp monitor_brute_force.sh caganze@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/
scp deploy_to_sherlock.sh caganze@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/
scp BRUTE_FORCE_USAGE_GUIDE.md caganze@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/
```

#### Method B: Create a tar file and upload
```bash
# Create tar file locally
tar -czf brute_force_system.tar.gz brute_force_gpu_job.sh monitor_brute_force.sh deploy_to_sherlock.sh BRUTE_FORCE_USAGE_GUIDE.md QUICK_DEPLOY.md

# Upload tar file
scp brute_force_system.tar.gz caganze@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/

# SSH and extract
ssh caganze@login.sherlock.stanford.edu
cd /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow
tar -xzf brute_force_system.tar.gz
chmod +x *.sh
```

#### Method C: Direct rsync with specific files
```bash
rsync -av brute_force_gpu_job.sh monitor_brute_force.sh deploy_to_sherlock.sh BRUTE_FORCE_USAGE_GUIDE.md QUICK_DEPLOY.md caganze@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/
```

### 5. **Debug the rsync Command**
```bash
# Add verbose and progress flags to see what's happening
rsync -avP * caganze@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/

# Or use dry-run to see what would be transferred
rsync -avn * caganze@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/
```

### 6. **Common Error Messages**

**"Permission denied"** → Check SSH keys or try with password
**"No such file or directory"** → Create the target directory first  
**"Connection refused"** → Check VPN/network connection
**"Host key verification failed"** → Remove old host key: `ssh-keygen -R login.sherlock.stanford.edu`

### 7. **Quick Test Script**
```bash
# Test all connection methods
echo "Testing SSH..."
ssh -o ConnectTimeout=10 caganze@login.sherlock.stanford.edu "echo 'SSH works'"

echo "Testing directory creation..."
ssh caganze@login.sherlock.stanford.edu "mkdir -p /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow && echo 'Directory ready'"

echo "Testing single file transfer..."
echo "test" > test_file.txt
rsync -av test_file.txt caganze@login.sherlock.stanford.edu:/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/
rm test_file.txt

echo "If all tests pass, run the full rsync command"
```

## What specific error are you seeing?

Please run one of the test commands above and let me know the exact error message you get!
