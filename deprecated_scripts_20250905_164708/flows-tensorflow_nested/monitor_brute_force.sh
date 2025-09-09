#!/bin/bash

#=============================================================================
# BRUTE FORCE MONITOR SCRIPT
# 
# This script monitors the progress of the brute force GPU job and provides
# comprehensive status reports, success/failure analysis, and progress tracking.
#
# Usage:
#   ./monitor_brute_force.sh               # Show current status
#   ./monitor_brute_force.sh --detailed    # Show detailed analysis
#   ./monitor_brute_force.sh --live        # Live monitoring (updates every 30s)
#=============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default mode
MODE="status"
if [[ "$1" == "--detailed" ]]; then
    MODE="detailed"
elif [[ "$1" == "--live" ]]; then
    MODE="live"
fi

show_header() {
    echo -e "${CYAN}=============================================================================="
    echo -e "🚀 BRUTE FORCE GPU JOB MONITOR"
    echo -e "==============================================================================${NC}"
    echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

check_job_status() {
    echo -e "${BLUE}📊 SLURM JOB STATUS${NC}"
    echo "----------------------------------------"
    
    # Check if any brute force jobs are running
    RUNNING_JOBS=$(squeue -u $(whoami) --name="brute_force_all_halos_pids" --noheader 2>/dev/null | wc -l)
    
    if [[ $RUNNING_JOBS -gt 0 ]]; then
        echo -e "${GREEN}✅ Active jobs: $RUNNING_JOBS${NC}"
        echo ""
        squeue -u $(whoami) --name="brute_force_all_halos_pids" --format="%.10i %.12j %.8T %.10M %.6D %R" 2>/dev/null
    else
        echo -e "${YELLOW}⚠️  No active brute force jobs found${NC}"
    fi
    echo ""
}

analyze_logs() {
    echo -e "${BLUE}📋 LOG ANALYSIS${NC}"
    echo "----------------------------------------"
    
    # Count log files
    OUT_LOGS=$(find logs/ -name "brute_force_*.out" 2>/dev/null | wc -l)
    ERR_LOGS=$(find logs/ -name "brute_force_*.err" 2>/dev/null | wc -l)
    
    echo "Output logs: $OUT_LOGS"
    echo "Error logs: $ERR_LOGS"
    echo ""
    
    # Check for recent activity
    if [[ $OUT_LOGS -gt 0 ]]; then
        echo "Most recent activity:"
        find logs/ -name "brute_force_*.out" -exec ls -lt {} \; 2>/dev/null | head -3 | while read line; do
            echo "  $line"
        done
        echo ""
    fi
}

analyze_success_failures() {
    echo -e "${BLUE}✅ SUCCESS/FAILURE ANALYSIS${NC}"
    echo "----------------------------------------"
    
    # Success log analysis
    if [[ -f "success_logs/brute_force_success.log" ]]; then
        SUCCESS_COUNT=$(wc -l < "success_logs/brute_force_success.log" 2>/dev/null || echo "0")
        echo -e "${GREEN}✅ Successful completions: $SUCCESS_COUNT${NC}"
        
        if [[ $SUCCESS_COUNT -gt 0 ]]; then
            echo "Recent successes:"
            tail -5 "success_logs/brute_force_success.log" | while read line; do
                timestamp=$(echo "$line" | awk '{print $1, $2}')
                halo=$(echo "$line" | grep -o 'halo:[0-9]*' | cut -d: -f2)
                pid=$(echo "$line" | grep -o 'pid:[0-9]*' | cut -d: -f2)
                echo -e "  ${GREEN}$timestamp${NC} - Halo $halo, PID $pid"
            done
        fi
    else
        echo -e "${YELLOW}⚠️  No success log found yet${NC}"
    fi
    echo ""
    
    # Failure log analysis
    if [[ -f "failed_jobs/brute_force_failures.log" ]]; then
        FAILURE_COUNT=$(wc -l < "failed_jobs/brute_force_failures.log" 2>/dev/null || echo "0")
        echo -e "${RED}❌ Failed attempts: $FAILURE_COUNT${NC}"
        
        if [[ $FAILURE_COUNT -gt 0 ]]; then
            echo "Recent failures:"
            tail -5 "failed_jobs/brute_force_failures.log" | while read line; do
                timestamp=$(echo "$line" | awk '{print $1, $2}')
                halo=$(echo "$line" | grep -o 'halo:[0-9]*' | cut -d: -f2)
                pid=$(echo "$line" | grep -o 'pid:[0-9]*' | cut -d: -f2)
                error_type=$(echo "$line" | grep -o 'error_type:[^ ]*' | cut -d: -f2)
                echo -e "  ${RED}$timestamp${NC} - Halo $halo, PID $pid ($error_type)"
            done
            
            echo ""
            echo "Failure types summary:"
            grep -o 'error_type:[^ ]*' "failed_jobs/brute_force_failures.log" | sort | uniq -c | sort -nr | while read count type; do
                error_type=$(echo "$type" | cut -d: -f2)
                echo "  $error_type: $count times"
            done
        fi
    else
        echo -e "${GREEN}✅ No failure log found (good!)${NC}"
    fi
    echo ""
}

analyze_outputs() {
    echo -e "${BLUE}📁 OUTPUT ANALYSIS${NC}"
    echo "----------------------------------------"
    
    # Define Sherlock paths
    BASE_DIR="/oak/stanford/orgs/kipac/users/caganze"
    OUTPUT_BASE="$BASE_DIR/tfp_flows_output"
    
    # Count trained models (new hierarchical structure)
    if [[ -d "$OUTPUT_BASE/trained_flows" ]]; then
        MODEL_FILES=$(find "$OUTPUT_BASE/trained_flows" -name "model_pid*.npz" 2>/dev/null | wc -l)
        echo "Model files (.npz): $MODEL_FILES"
        
        # Show breakdown by data source
        if [[ $MODEL_FILES -gt 0 ]]; then
            echo "By data source:"
            for source in eden symphony symphony-hr unknown; do
                if [[ -d "$OUTPUT_BASE/trained_flows/$source" ]]; then
                    count=$(find "$OUTPUT_BASE/trained_flows/$source" -name "model_pid*.npz" 2>/dev/null | wc -l)
                    if [[ $count -gt 0 ]]; then
                        echo "  $source: $count models"
                    fi
                fi
            done
            
            echo "Recent models:"
            find "$OUTPUT_BASE/trained_flows" -name "model_pid*.npz" -exec ls -lt {} \; 2>/dev/null | head -3 | while read line; do
                model_path=$(echo $line | awk '{print $9}')
                rel_path=${model_path#$OUTPUT_BASE/trained_flows/}
                echo "  $rel_path"
            done
        fi
    else
        echo "Model output directory not found: $OUTPUT_BASE/trained_flows"
    fi
    
    # Count sample files (new hierarchical structure)
    if [[ -d "$OUTPUT_BASE/samples" ]]; then
        SAMPLE_NPZ=$(find "$OUTPUT_BASE/samples" -name "model_pid*_samples.npz" 2>/dev/null | wc -l)
        SAMPLE_H5=$(find "$OUTPUT_BASE/samples" -name "model_pid*_samples.h5" 2>/dev/null | wc -l)
        echo "Sample files (.npz): $SAMPLE_NPZ"
        echo "Sample files (.h5): $SAMPLE_H5"
        
        # Show breakdown by data source
        if [[ $((SAMPLE_NPZ + SAMPLE_H5)) -gt 0 ]]; then
            echo "By data source:"
            for source in eden symphony symphony-hr unknown; do
                if [[ -d "$OUTPUT_BASE/samples/$source" ]]; then
                    npz_count=$(find "$OUTPUT_BASE/samples/$source" -name "*.npz" 2>/dev/null | wc -l)
                    h5_count=$(find "$OUTPUT_BASE/samples/$source" -name "*.h5" 2>/dev/null | wc -l)
                    total_count=$((npz_count + h5_count))
                    if [[ $total_count -gt 0 ]]; then
                        echo "  $source: $total_count samples ($npz_count npz, $h5_count h5)"
                    fi
                fi
            done
            
            TOTAL_SIZE=$(find "$OUTPUT_BASE/samples" -name "*.npz" -o -name "*.h5" 2>/dev/null | xargs du -ch 2>/dev/null | tail -1 | awk '{print $1}')
            echo "Total sample data: $TOTAL_SIZE"
        fi
    else
        echo "Sample output directory not found: $OUTPUT_BASE/samples"
    fi
    
    echo ""
}

show_progress_estimate() {
    echo -e "${BLUE}📈 PROGRESS ESTIMATION${NC}"
    echo "----------------------------------------"
    
    # Try to estimate total work and progress
    if [[ -f "success_logs/brute_force_success.log" ]]; then
        SUCCESS_COUNT=$(wc -l < "success_logs/brute_force_success.log" 2>/dev/null || echo "0")
        
        # Estimate total combinations (this is approximate)
        # We'll try to extract this from the logs if possible
        TOTAL_ESTIMATE=$(grep -h "Total halo-PID combinations:" logs/brute_force_*.out 2>/dev/null | tail -1 | grep -o '[0-9]*' | tail -1)
        
        if [[ -n "$TOTAL_ESTIMATE" && "$TOTAL_ESTIMATE" -gt 0 ]]; then
            PROGRESS_PERCENT=$(( 100 * SUCCESS_COUNT / TOTAL_ESTIMATE ))
            echo "Estimated total combinations: $TOTAL_ESTIMATE"
            echo "Completed successfully: $SUCCESS_COUNT"
            echo -e "Progress: ${GREEN}$PROGRESS_PERCENT%${NC}"
            
            # Estimate completion time if there's a pattern
            if [[ $SUCCESS_COUNT -gt 5 ]]; then
                # Calculate average time per success (rough estimate)
                FIRST_SUCCESS=$(head -1 "success_logs/brute_force_success.log" | awk '{print $1, $2}')
                LAST_SUCCESS=$(tail -1 "success_logs/brute_force_success.log" | awk '{print $1, $2}')
                
                if [[ -n "$FIRST_SUCCESS" && -n "$LAST_SUCCESS" ]]; then
                    echo "First success: $FIRST_SUCCESS"
                    echo "Latest success: $LAST_SUCCESS"
                fi
            fi
        else
            echo "Total work estimate not available yet"
            echo "Completed successfully: $SUCCESS_COUNT"
        fi
    else
        echo "No progress data available yet"
    fi
    echo ""
}

show_detailed_analysis() {
    echo -e "${BLUE}🔍 DETAILED ANALYSIS${NC}"
    echo "----------------------------------------"
    
    # Show unique halos being processed
    if [[ -f "success_logs/brute_force_success.log" ]]; then
        echo "Halos with successful completions:"
        grep -o 'halo:[0-9]*' "success_logs/brute_force_success.log" | cut -d: -f2 | sort -nu | head -10 | tr '\n' ' '
        echo ""
        echo ""
        
        echo "PIDs with successful completions:"
        grep -o 'pid:[0-9]*' "success_logs/brute_force_success.log" | cut -d: -f2 | sort -nu | head -20 | tr '\n' ' '
        echo ""
        echo ""
    fi
    
    # Show resource usage from recent logs
    echo "Recent resource usage (from latest log):"
    LATEST_LOG=$(find logs/ -name "brute_force_*.out" -exec ls -t {} \; 2>/dev/null | head -1)
    if [[ -n "$LATEST_LOG" ]]; then
        echo "Latest log: $(basename $LATEST_LOG)"
        grep -E "(Training Duration|Total Duration|Model size|Sample size)" "$LATEST_LOG" 2>/dev/null | tail -5
    else
        echo "No recent logs found"
    fi
    echo ""
}

# Main execution
show_header

if [[ "$MODE" == "live" ]]; then
    echo -e "${YELLOW}🔴 LIVE MONITORING MODE (Ctrl+C to exit)${NC}"
    echo "Updates every 30 seconds..."
    echo ""
    
    while true; do
        clear
        show_header
        check_job_status
        analyze_success_failures
        show_progress_estimate
        echo -e "${YELLOW}Next update in 30 seconds... (Ctrl+C to exit)${NC}"
        sleep 30
    done
    
elif [[ "$MODE" == "detailed" ]]; then
    check_job_status
    analyze_logs
    analyze_success_failures
    analyze_outputs
    show_progress_estimate
    show_detailed_analysis
    
else
    # Standard status mode
    check_job_status
    analyze_success_failures
    show_progress_estimate
    analyze_outputs
fi

echo -e "${CYAN}==============================================================================${NC}"
echo -e "${CYAN}Monitor completed at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${CYAN}==============================================================================${NC}"
