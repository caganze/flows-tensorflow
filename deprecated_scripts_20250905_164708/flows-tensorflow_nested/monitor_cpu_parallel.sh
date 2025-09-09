#!/bin/bash

# 🖥️  CPU Parallel Job Monitor
# Real-time monitoring and progress tracking for CPU parallel jobs

set -e

# Default parameters
REFRESH_INTERVAL=30
SHOW_DETAILS=false
FOLLOW_MODE=false
AUTO_REFRESH=false

show_usage() {
    echo "🖥️  CPU Parallel Job Monitor"
    echo "=========================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --details           Show detailed progress breakdown"
    echo "  --follow            Follow mode (auto-refresh every 30s)"
    echo "  --interval N        Refresh interval for follow mode (default: 30s)"
    echo "  --once              Run once and exit (default)"
    echo "  --help              Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                  # Quick status check"
    echo "  $0 --details        # Detailed breakdown"
    echo "  $0 --follow         # Auto-refresh mode"
    echo "  $0 --follow --interval 10  # Auto-refresh every 10s"
    echo ""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --details)
            SHOW_DETAILS=true
            shift
            ;;
        --follow)
            FOLLOW_MODE=true
            AUTO_REFRESH=true
            shift
            ;;
        --interval)
            REFRESH_INTERVAL="$2"
            shift 2
            ;;
        --once)
            FOLLOW_MODE=false
            AUTO_REFRESH=false
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

# Color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}🖥️  CPU PARALLEL TFP FLOWS MONITOR${NC}"
    echo -e "${CYAN}===================================${NC}"
    echo -e "📅 $(date)"
    echo -e "🔄 Refresh: $([[ $AUTO_REFRESH == true ]] && echo "Auto (${REFRESH_INTERVAL}s)" || echo "Manual")"
    echo ""
}

get_slurm_jobs() {
    # Get all TFP CPU parallel jobs for current user
    squeue -u "$USER" -h -o "%A %T %j %M %N %C %m" | grep -E "(cpu_flows_parallel|cpu_parallel)" || true
}

count_jobs_by_status() {
    local status="$1"
    echo "$SLURM_JOBS" | grep -c "$status" || echo "0"
}

analyze_progress_logs() {
    local success_count=0
    local failed_count=0
    local total_processed=0
    
    # Count from success logs
    if [[ -f "success_logs/cpu_flows_success.log" ]]; then
        success_count=$(wc -l < success_logs/cpu_flows_success.log 2>/dev/null || echo "0")
    fi
    
    # Count from failure logs
    if [[ -f "failed_jobs/cpu_flows_failures.log" ]]; then
        failed_count=$(wc -l < failed_jobs/cpu_flows_failures.log 2>/dev/null || echo "0")
    fi
    
    total_processed=$((success_count + failed_count))
    
    echo "$success_count $failed_count $total_processed"
}

get_particle_stats() {
    local total_particles=0
    
    if [[ -f "particle_list.txt" ]]; then
        total_particles=$(wc -l < particle_list.txt 2>/dev/null || echo "0")
    fi
    
    echo "$total_particles"
}

analyze_array_progress() {
    local completed_arrays=0
    local total_arrays=0
    
    if [[ -f "cpu_parallel_progress/array_progress.log" ]]; then
        completed_arrays=$(wc -l < cpu_parallel_progress/array_progress.log 2>/dev/null || echo "0")
        
        # Get max array ID from the log to estimate total
        if [[ $completed_arrays -gt 0 ]]; then
            total_arrays=$(grep -o 'array_id:[0-9]*' cpu_parallel_progress/array_progress.log | cut -d':' -f2 | sort -n | tail -1)
        fi
    fi
    
    echo "$completed_arrays $total_arrays"
}

show_quick_status() {
    print_header
    
    # Get SLURM job data
    SLURM_JOBS=$(get_slurm_jobs)
    
    if [[ -z "$SLURM_JOBS" ]]; then
        echo -e "${YELLOW}📭 No CPU parallel jobs found${NC}"
        echo ""
        echo -e "${BLUE}💡 To submit a new job:${NC}"
        echo -e "   ./submit_cpu_parallel.sh"
        return
    fi
    
    # Job status counts
    local running=$(count_jobs_by_status "RUNNING")
    local pending=$(count_jobs_by_status "PENDING")
    local total_jobs=$(echo "$SLURM_JOBS" | wc -l)
    
    echo -e "${GREEN}🎯 SLURM JOB STATUS${NC}"
    echo -e "${GREEN}==================${NC}"
    echo -e "🟢 Running: $running"
    echo -e "🟡 Pending: $pending"
    echo -e "📊 Total jobs: $total_jobs"
    echo ""
    
    # Progress analysis
    read success_count failed_count total_processed <<< $(analyze_progress_logs)
    local total_particles=$(get_particle_stats)
    
    echo -e "${BLUE}📈 PARTICLE PROCESSING PROGRESS${NC}"
    echo -e "${BLUE}===============================${NC}"
    echo -e "✅ Successful: $success_count"
    echo -e "❌ Failed: $failed_count"
    echo -e "📊 Total processed: $total_processed"
    
    if [[ $total_particles -gt 0 ]]; then
        local completion_pct=$((total_processed * 100 / total_particles))
        echo -e "🎯 Total particles: $total_particles"
        echo -e "📈 Completion: ${completion_pct}%"
        
        # Progress bar
        local bar_width=40
        local filled=$((completion_pct * bar_width / 100))
        local empty=$((bar_width - filled))
        
        printf "📊 Progress: ["
        printf "%*s" "$filled" | tr ' ' '█'
        printf "%*s" "$empty" | tr ' ' '░'
        printf "] %d%%\n" "$completion_pct"
    fi
    echo ""
    
    # Array task progress
    read completed_arrays total_arrays <<< $(analyze_array_progress)
    if [[ $completed_arrays -gt 0 ]]; then
        echo -e "${PURPLE}🔢 ARRAY TASK PROGRESS${NC}"
        echo -e "${PURPLE}=====================${NC}"
        echo -e "✅ Completed arrays: $completed_arrays"
        if [[ $total_arrays -gt 0 ]]; then
            echo -e "🎯 Max array ID seen: $total_arrays"
        fi
        echo ""
    fi
    
    # Success rate
    if [[ $total_processed -gt 0 ]]; then
        local success_rate=$((success_count * 100 / total_processed))
        echo -e "${GREEN}📊 Success rate: ${success_rate}%${NC}"
        echo ""
    fi
}

show_detailed_status() {
    show_quick_status
    
    # Recent successes
    echo -e "${GREEN}🎉 RECENT SUCCESSES (last 10)${NC}"
    echo -e "${GREEN}=============================${NC}"
    if [[ -f "success_logs/cpu_flows_success.log" ]]; then
        tail -10 success_logs/cpu_flows_success.log | while read -r line; do
            if [[ -n "$line" ]]; then
                # Parse log line for key info
                local timestamp=$(echo "$line" | cut -d' ' -f1-5)
                local info=$(echo "$line" | grep -o 'halo:[^ ]*\|pid:[^ ]*' | tr '\n' ' ')
                echo -e "  ${GREEN}✓${NC} $timestamp - $info"
            fi
        done
    else
        echo -e "  ${YELLOW}No success log found${NC}"
    fi
    echo ""
    
    # Recent failures
    echo -e "${RED}⚠️  RECENT FAILURES (last 5)${NC}"
    echo -e "${RED}===========================${NC}"
    if [[ -f "failed_jobs/cpu_flows_failures.log" ]]; then
        tail -5 failed_jobs/cpu_flows_failures.log | while read -r line; do
            if [[ -n "$line" ]]; then
                # Parse log line for key info
                local timestamp=$(echo "$line" | cut -d' ' -f1-5)
                local info=$(echo "$line" | grep -o 'halo:[^ ]*\|pid:[^ ]*' | tr '\n' ' ')
                echo -e "  ${RED}✗${NC} $timestamp - $info"
            fi
        done
    else
        echo -e "  ${GREEN}No failure log found${NC}"
    fi
    echo ""
    
    # Running jobs details
    SLURM_JOBS=$(get_slurm_jobs)
    if [[ -n "$SLURM_JOBS" ]]; then
        echo -e "${BLUE}🔄 RUNNING JOBS DETAILS${NC}"
        echo -e "${BLUE}=======================${NC}"
        echo -e "JobID    Status   Runtime  Node        CPUs Memory"
        echo -e "-------- -------- -------- ----------- ---- ------"
        echo "$SLURM_JOBS" | while read -r jobid status name runtime node cpus memory; do
            if [[ "$status" == "RUNNING" ]]; then
                printf "%-8s %-8s %-8s %-11s %-4s %s\n" "$jobid" "$status" "$runtime" "$node" "$cpus" "$memory"
            fi
        done
        echo ""
    fi
    
    # File system usage
    echo -e "${CYAN}💾 STORAGE USAGE${NC}"
    echo -e "${CYAN}===============${NC}"
    
    # Count output files
    local model_files=0
    local sample_files=0
    
    if command -v find >/dev/null 2>&1; then
        model_files=$(find . -name "model_pid*.npz" 2>/dev/null | wc -l || echo "0")
        sample_files=$(find . -name "*_samples.*" 2>/dev/null | wc -l || echo "0")
    fi
    
    echo -e "📁 Model files: $model_files"
    echo -e "📁 Sample files: $sample_files"
    
    # Log directory sizes
    if [[ -d "logs" ]]; then
        local log_size=$(du -sh logs 2>/dev/null | cut -f1 || echo "Unknown")
        echo -e "📋 Log directory: $log_size"
    fi
    echo ""
}

# Main execution
main() {
    if [[ "$AUTO_REFRESH" == "true" ]]; then
        echo -e "${CYAN}🔄 Follow mode active - Press Ctrl+C to exit${NC}"
        echo ""
        
        while true; do
            clear
            if [[ "$SHOW_DETAILS" == "true" ]]; then
                show_detailed_status
            else
                show_quick_status
            fi
            
            echo -e "${YELLOW}⏰ Auto-refreshing in ${REFRESH_INTERVAL}s... (Ctrl+C to exit)${NC}"
            sleep "$REFRESH_INTERVAL"
        done
    else
        # Single run
        if [[ "$SHOW_DETAILS" == "true" ]]; then
            show_detailed_status
        else
            show_quick_status
        fi
        
        echo -e "${YELLOW}💡 Tip: Use --follow for auto-refresh or --details for more info${NC}"
    fi
}

# Handle Ctrl+C gracefully in follow mode
trap 'echo -e "\n${GREEN}✅ Monitoring stopped${NC}"; exit 0' INT

# Run main function
main
