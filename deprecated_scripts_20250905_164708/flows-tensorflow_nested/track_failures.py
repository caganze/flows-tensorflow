#!/usr/bin/env python3
"""
Failure tracking system for parallel training pipeline
Logs failed particles and generates summary reports
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

def log_failure(halo_id: str, particle_pid: int, error_type: str, error_message: str, 
                log_dir: str = "failed_particles"):
    """Log a failed particle training"""
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.time()
    date_str = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Individual failure log
    failure_log = {
        'halo_id': halo_id,
        'particle_pid': particle_pid,
        'error_type': error_type,
        'error_message': error_message,
        'timestamp': timestamp,
        'date': date_str,
        'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
    }
    
    # Save individual log
    log_file = log_path / f"halo{halo_id}_pid{particle_pid:03d}_error.json"
    with open(log_file, 'w') as f:
        json.dump(failure_log, f, indent=2)
    
    # Update summary
    update_failure_summary(log_dir)
    
    print(f"❌ Logged failure: Halo {halo_id}, PID {particle_pid}")
    print(f"📁 Log saved: {log_file}")

def update_failure_summary(log_dir: str = "failed_particles"):
    """Update the failure summary file"""
    
    log_path = Path(log_dir)
    if not log_path.exists():
        return
    
    # Collect all failure logs
    failures = []
    for log_file in log_path.glob("halo*_pid*_error.json"):
        try:
            with open(log_file, 'r') as f:
                failure = json.load(f)
                failures.append(failure)
        except Exception as e:
            print(f"⚠️ Could not read {log_file}: {e}")
    
    if not failures:
        return
    
    # Generate summary statistics
    summary = {
        'total_failures': len(failures),
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'error_types': {},
        'halos_with_failures': {},
        'pids_with_failures': {},
        'failure_timeline': []
    }
    
    for failure in failures:
        # Count error types
        error_type = failure.get('error_type', 'unknown')
        summary['error_types'][error_type] = summary['error_types'].get(error_type, 0) + 1
        
        # Count failures per halo
        halo_id = failure.get('halo_id', 'unknown')
        summary['halos_with_failures'][halo_id] = summary['halos_with_failures'].get(halo_id, 0) + 1
        
        # Count failures per PID
        pid = failure.get('particle_pid', 'unknown')
        summary['pids_with_failures'][str(pid)] = summary['pids_with_failures'].get(str(pid), 0) + 1
        
        # Timeline entry
        summary['failure_timeline'].append({
            'halo_id': halo_id,
            'particle_pid': pid,
            'error_type': error_type,
            'date': failure.get('date', 'unknown'),
            'timestamp': failure.get('timestamp', 0)
        })
    
    # Sort timeline by timestamp
    summary['failure_timeline'].sort(key=lambda x: x['timestamp'])
    
    # Save summary
    summary_file = log_path / "failure_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create human-readable report
    report_file = log_path / "failure_report.txt"
    with open(report_file, 'w') as f:
        f.write("PARALLEL TRAINING FAILURE REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Last Updated: {summary['last_updated']}\n")
        f.write(f"Total Failures: {summary['total_failures']}\n\n")
        
        f.write("ERROR TYPES:\n")
        f.write("-" * 20 + "\n")
        for error_type, count in sorted(summary['error_types'].items()):
            f.write(f"  {error_type}: {count}\n")
        
        f.write(f"\nHALOS WITH FAILURES:\n")
        f.write("-" * 30 + "\n")
        for halo_id, count in sorted(summary['halos_with_failures'].items()):
            f.write(f"  Halo {halo_id}: {count} failures\n")
            
        f.write(f"\nPIDs WITH FAILURES:\n")
        f.write("-" * 25 + "\n")
        for pid, count in sorted(summary['pids_with_failures'].items(), key=lambda x: int(x[0])):
            f.write(f"  PID {pid}: {count} failures\n")
        
        f.write(f"\nRECENT FAILURES (last 10):\n")
        f.write("-" * 35 + "\n")
        for failure in summary['failure_timeline'][-10:]:
            f.write(f"  {failure['date']}: Halo {failure['halo_id']} PID {failure['particle_pid']} - {failure['error_type']}\n")
    
    print(f"📊 Updated failure summary: {summary_file}")
    print(f"📄 Human-readable report: {report_file}")

def get_failure_stats(log_dir: str = "failed_particles") -> Dict[str, Any]:
    """Get failure statistics"""
    
    summary_file = Path(log_dir) / "failure_summary.json"
    if not summary_file.exists():
        return {'total_failures': 0}
    
    with open(summary_file, 'r') as f:
        return json.load(f)

def list_failed_particles(log_dir: str = "failed_particles") -> List[Dict[str, Any]]:
    """Get list of all failed particles"""
    
    stats = get_failure_stats(log_dir)
    return stats.get('failure_timeline', [])

def main():
    """Command line interface for failure tracking"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python track_failures.py log <halo_id> <particle_pid> <error_type> <error_message>")
        print("  python track_failures.py summary")
        print("  python track_failures.py list")
        return
    
    command = sys.argv[1]
    
    if command == "log" and len(sys.argv) >= 6:
        halo_id = sys.argv[2]
        particle_pid = int(sys.argv[3])
        error_type = sys.argv[4]
        error_message = " ".join(sys.argv[5:])
        log_failure(halo_id, particle_pid, error_type, error_message)
        
    elif command == "summary":
        stats = get_failure_stats()
        print(f"📊 FAILURE SUMMARY")
        print(f"Total failures: {stats.get('total_failures', 0)}")
        print(f"Error types: {list(stats.get('error_types', {}).keys())}")
        print(f"Affected halos: {len(stats.get('halos_with_failures', {}))}")
        print(f"Affected PIDs: {len(stats.get('pids_with_failures', {}))}")
        
    elif command == "list":
        failures = list_failed_particles()
        print(f"📋 FAILED PARTICLES ({len(failures)} total):")
        for failure in failures[-20:]:  # Show last 20
            print(f"  Halo {failure['halo_id']} PID {failure['particle_pid']}: {failure['error_type']}")
    
    else:
        print("❌ Invalid command")

if __name__ == "__main__":
    main()
