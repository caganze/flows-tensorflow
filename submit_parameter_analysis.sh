#!/bin/bash

echo "🚀 Parameter Analysis Job Submission"
echo "===================================="
echo ""

# Detect login node more robustly (sh**-ln** or login hostnames)
if [[ $HOSTNAME == *"-ln"* ]] || [[ $HOSTNAME == *"login"* ]]; then
    echo "✅ Running on login node: $HOSTNAME"
else
    echo "⚠️  Warning: Not on a login node ($HOSTNAME)."
fi

echo ""
echo "Available job types:"
echo "1. Quick test (4 hours, 25 tests)"
echo "2. Comprehensive test (12 hours, 100+ tests)"
echo ""

read -p "Select job type (1 or 2): " job_type

case $job_type in
    1)
        echo "📋 Submitting quick parameter test job..."
        echo "   - Duration: 4 hours"
        echo "   - Tests: ~25 parameter combinations"
        echo "   - Output: quick_param_test_<job_id>.out"
        echo ""
        
        SUBMIT_OUTPUT=$(sbatch quick_parameter_job.sh 2>&1)
        STATUS=$?
        if [ $STATUS -ne 0 ]; then
            echo "❌ sbatch failed: $SUBMIT_OUTPUT"
            exit $STATUS
        fi
        JOB_ID=$(echo "$SUBMIT_OUTPUT" | awk '{print $4}')
        if [ -z "$JOB_ID" ]; then
            echo "❌ Could not parse Job ID from sbatch output: $SUBMIT_OUTPUT"
            exit 1
        fi
        echo "✅ Job submitted successfully!"
        echo "   Job ID: $JOB_ID"
        echo "   Monitor with: squeue -j $JOB_ID"
        echo "   Cancel with: scancel $JOB_ID"
        echo ""
        echo "📁 Results will be saved to:"
        echo "   - quick_param_test_<timestamp>/"
        echo "   - /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/quick_param_test_results/"
        ;;
    2)
        echo "📋 Submitting comprehensive parameter analysis job..."
        echo "   - Duration: 12 hours"
        echo "   - Tests: 100+ parameter combinations"
        echo "   - Output: parameter_analysis_<job_id>.out"
        echo ""
        
        SUBMIT_OUTPUT=$(sbatch parameter_analysis_job.sh 2>&1)
        STATUS=$?
        if [ $STATUS -ne 0 ]; then
            echo "❌ sbatch failed: $SUBMIT_OUTPUT"
            exit $STATUS
        fi
        JOB_ID=$(echo "$SUBMIT_OUTPUT" | awk '{print $4}')
        if [ -z "$JOB_ID" ]; then
            echo "❌ Could not parse Job ID from sbatch output: $SUBMIT_OUTPUT"
            exit 1
        fi
        echo "✅ Job submitted successfully!"
        echo "   Job ID: $JOB_ID"
        echo "   Monitor with: squeue -j $JOB_ID"
        echo "   Cancel with: scancel $JOB_ID"
        echo ""
        echo "📁 Results will be saved to:"
        echo "   - parameter_analysis_results_<timestamp>/"
        echo "   - /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/parameter_analysis_results/"
        ;;
    *)
        echo "❌ Invalid selection. Please run the script again and select 1 or 2."
        exit 1
        ;;
esac

echo ""
echo "🔍 To monitor your job:"
echo "   squeue -u $USER"
echo "   squeue -j $JOB_ID"
echo ""
echo "📊 To check job output:"
echo "   tail -f quick_param_test_${JOB_ID}.out    # (for quick test)"
echo "   tail -f parameter_analysis_${JOB_ID}.out  # (for comprehensive test)"
echo ""
echo "❌ To cancel the job:"
echo "   scancel $JOB_ID"
echo ""
echo "✅ Job submission complete!"
