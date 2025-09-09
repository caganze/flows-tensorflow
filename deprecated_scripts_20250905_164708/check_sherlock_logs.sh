#!/bin/bash

# 🔍 Comprehensive Log Check for Sherlock Jobs
# Run this on Sherlock to diagnose what happened

echo "🔍 SHERLOCK LOG DIAGNOSTIC"
echo "=========================="
echo "📍 Current directory: $(pwd)"
echo "📅 Checking logs from today..."
echo ""

echo "1️⃣ SLURM JOB HISTORY"
echo "===================="
echo "📋 Recent jobs:"
sacct -u $USER --starttime=today --format=JobID,JobName,State,ExitCode,Elapsed,Partition | head -20

echo ""
echo "📊 Job summary:"
sacct -u $USER --starttime=today --state=COMPLETED,FAILED,CANCELLED,TIMEOUT | wc -l
echo ""

echo "2️⃣ GPU BRUTE FORCE LOGS"
echo "======================="
echo "📄 GPU brute force output files:"
ls -la logs/brute_force_*.out 2>/dev/null | tail -10

echo ""
echo "📄 GPU brute force error files:"
ls -la logs/brute_force_*.err 2>/dev/null | tail -10

echo ""
echo "🔍 Latest GPU brute force logs to check:"
LATEST_GPU_OUT=$(ls -t logs/brute_force_*.out 2>/dev/null | head -1)
LATEST_GPU_ERR=$(ls -t logs/brute_force_*.err 2>/dev/null | head -1)

if [[ -n "$LATEST_GPU_OUT" ]]; then
    echo "   📤 Latest output: $LATEST_GPU_OUT"
    echo "      Command: head -50 $LATEST_GPU_OUT"
    echo "      Command: tail -50 $LATEST_GPU_OUT"
    echo "      Command: grep -i 'error\|fail\|exception' $LATEST_GPU_OUT"
fi

if [[ -n "$LATEST_GPU_ERR" ]]; then
    echo "   📤 Latest error: $LATEST_GPU_ERR"
    echo "      Command: cat $LATEST_GPU_ERR"
fi

echo ""
echo "3️⃣ CPU PARALLEL LOGS"
echo "===================="
echo "📄 CPU parallel output files:"
ls -la logs/cpu_flows_*.out 2>/dev/null | tail -10

echo ""
echo "📄 CPU parallel error files:"
ls -la logs/cpu_flows_*.err 2>/dev/null | tail -10

echo ""
echo "🔍 Latest CPU parallel logs to check:"
LATEST_CPU_OUT=$(ls -t logs/cpu_flows_*.out 2>/dev/null | head -1)
LATEST_CPU_ERR=$(ls -t logs/cpu_flows_*.err 2>/dev/null | head -1)

if [[ -n "$LATEST_CPU_OUT" ]]; then
    echo "   📤 Latest output: $LATEST_CPU_OUT"
    echo "      Command: head -30 $LATEST_CPU_OUT"
    echo "      Command: tail -30 $LATEST_CPU_OUT"
fi

if [[ -n "$LATEST_CPU_ERR" ]]; then
    echo "   📤 Latest error: $LATEST_CPU_ERR"
    echo "      Command: cat $LATEST_CPU_ERR"
fi

echo ""
echo "4️⃣ SLURM SYSTEM LOGS"
echo "===================="
echo "📄 SLURM output files:"
ls -la slurm-*.out 2>/dev/null | tail -5

echo ""
echo "📄 SLURM error files:"
ls -la slurm-*.err 2>/dev/null | tail -5

echo ""
echo "🔍 Latest SLURM logs to check:"
LATEST_SLURM_OUT=$(ls -t slurm-*.out 2>/dev/null | head -1)
LATEST_SLURM_ERR=$(ls -t slurm-*.err 2>/dev/null | head -1)

if [[ -n "$LATEST_SLURM_OUT" ]]; then
    echo "   📤 Latest SLURM output: $LATEST_SLURM_OUT"
    echo "      Command: cat $LATEST_SLURM_OUT"
fi

if [[ -n "$LATEST_SLURM_ERR" ]]; then
    echo "   📤 Latest SLURM error: $LATEST_SLURM_ERR"
    echo "      Command: cat $LATEST_SLURM_ERR"
fi

echo ""
echo "5️⃣ SMART CPU SUBMISSION LOGS"
echo "============================="
echo "📄 Looking for smart CPU logs..."
ls -la logs/*smart* logs/*filter* 2>/dev/null

echo ""
echo "6️⃣ OUTPUT VERIFICATION"
echo "======================"
echo "📁 Checking output directories:"
echo "   Trained flows:"
find /oak/stanford/orgs/kipac/users/$USER/milkyway-*/tfp_output/trained_flows/ -name "*.npz" -mtime -1 2>/dev/null | head -10

echo ""
echo "   Samples:"
find /oak/stanford/orgs/kipac/users/$USER/milkyway-*/tfp_output/samples/ -name "*.npz" -mtime -1 2>/dev/null | head -10

echo ""
echo "7️⃣ ERROR PATTERN SEARCH"
echo "======================="
echo "🔍 Searching for common errors in recent logs:"
echo ""

echo "📊 Memory errors:"
grep -i "out of memory\|memory error\|OOM" logs/*.out logs/*.err 2>/dev/null | head -5

echo ""
echo "📊 CUDA/GPU errors:"
grep -i "cuda\|gpu error\|device" logs/*.out logs/*.err 2>/dev/null | head -5

echo ""
echo "📊 File access errors:"
grep -i "no such file\|permission denied\|cannot open" logs/*.out logs/*.err 2>/dev/null | head -5

echo ""
echo "📊 Python/import errors:"
grep -i "importerror\|modulenotfounderror\|traceback" logs/*.out logs/*.err 2>/dev/null | head -5

echo ""
echo "📊 SLURM time/resource errors:"
grep -i "time limit\|exceeded\|killed\|cancelled" logs/*.out logs/*.err 2>/dev/null | head -5

echo ""
echo "8️⃣ PARTICLE LIST STATUS"
echo "======================="
echo "📋 Particle list info:"
if [[ -f "particle_list.txt" ]]; then
    echo "   Total particles: $(wc -l < particle_list.txt)"
    echo "   First 5 particles:"
    head -5 particle_list.txt
    echo "   Halo breakdown:"
    cut -d',' -f2 particle_list.txt | grep -o 'Halo[0-9]*' | sort | uniq -c | head -10
fi

echo ""
if [[ -f "particle_list_incomplete.txt" ]]; then
    echo "📋 Incomplete particle list:"
    echo "   Remaining particles: $(wc -l < particle_list_incomplete.txt)"
    echo "   First 5 remaining:"
    head -5 particle_list_incomplete.txt
else
    echo "📋 No incomplete particle list found"
fi

echo ""
echo "9️⃣ COMMANDS TO RUN FOR DETAILED ANALYSIS"
echo "========================================"
echo ""
echo "🔍 Check specific logs:"
echo "   cat logs/brute_force_5019746.out    # GPU job output"
echo "   cat logs/brute_force_5019746.err    # GPU job errors"
echo "   cat logs/cpu_flows_5019747_24.out   # CPU job sample output"
echo "   cat logs/cpu_flows_5019747_95.err   # CPU job sample error"
echo ""
echo "🔍 Search for the last successful particle:"
echo "   grep -n 'SUCCESS\|COMPLETED\|PID.*completed' logs/brute_force_*.out"
echo ""
echo "🔍 Find where jobs failed:"
echo "   grep -n 'FAILED\|ERROR\|Exception\|Traceback' logs/brute_force_*.out logs/brute_force_*.err"
echo ""
echo "🔍 Check job timing:"
echo "   grep -E 'Started:|Completed:|Time:' logs/brute_force_*.out | tail -10"
echo ""
echo "🔍 Check particle processing order:"
echo "   grep 'Processing.*PID' logs/brute_force_*.out | tail -20"

echo ""
echo "✅ LOG DIAGNOSTIC COMPLETE"
echo "=========================="
echo "📝 Next steps:"
echo "   1. Run the specific 'cat' commands above for detailed logs"
echo "   2. Look for error patterns in the Error Pattern Search section"
echo "   3. Check which particles completed successfully"
echo "   4. Identify where the jobs stopped/failed"

