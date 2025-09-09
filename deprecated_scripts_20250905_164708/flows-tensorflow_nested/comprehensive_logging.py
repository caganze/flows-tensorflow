#!/usr/bin/env python3
"""
Comprehensive logging and job management for TensorFlow Probability flows
Ensures we never lose results and can track all progress
"""

import os
import sys
import time
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available, system monitoring will be limited")

import subprocess

class ComprehensiveLogger:
    """
    Comprehensive logger that tracks everything during training
    """
    
    def __init__(self, log_dir: str, job_name: str, particle_pid: int = None):
        """
        Initialize comprehensive logging
        
        Args:
            log_dir: Base directory for all logs
            job_name: Name of the job (e.g., 'training', 'sampling')
            particle_pid: Particle ID being processed
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.job_name = job_name
        self.particle_pid = particle_pid
        self.start_time = time.time()
        
        # Create unique log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if particle_pid is not None:
            log_filename = f"{job_name}_pid{particle_pid}_{timestamp}.log"
            self.job_id = f"{job_name}_pid{particle_pid}_{timestamp}"
        else:
            log_filename = f"{job_name}_{timestamp}.log"
            self.job_id = f"{job_name}_{timestamp}"
        
        self.log_file = self.log_dir / log_filename
        
        # Set up logging
        self.logger = logging.getLogger(self.job_id)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler with detailed formatting
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | PID:%(process)d | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Job tracking
        self.job_info = {
            'job_id': self.job_id,
            'job_name': job_name,
            'particle_pid': particle_pid,
            'start_time': self.start_time,
            'start_time_str': datetime.fromtimestamp(self.start_time).isoformat(),
            'log_file': str(self.log_file),
            'pid': os.getpid(),
            'hostname': os.uname().nodename,
            'working_dir': str(Path.cwd()),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
            'status': 'RUNNING'
        }
        
        # Save initial job info
        self.save_job_info()
        
        # Log startup
        self.info("="*80)
        self.info(f"🚀 STARTING JOB: {self.job_id}")
        self.info("="*80)
        self.log_system_info()
        
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical error"""
        self.logger.critical(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_system_info(self):
        """Log comprehensive system information"""
        self.info("💻 SYSTEM INFORMATION:")
        self.info(f"  Hostname: {self.job_info['hostname']}")
        self.info(f"  PID: {self.job_info['pid']}")
        self.info(f"  Working directory: {self.job_info['working_dir']}")
        
        if self.job_info['slurm_job_id']:
            self.info(f"  SLURM Job ID: {self.job_info['slurm_job_id']}")
        if self.job_info['slurm_array_task_id']:
            self.info(f"  SLURM Array Task: {self.job_info['slurm_array_task_id']}")
        
        # Memory info
        if HAS_PSUTIL:
            memory = psutil.virtual_memory()
            self.info(f"  Memory: {memory.total/1e9:.1f}GB total, {memory.available/1e9:.1f}GB available")
        else:
            self.info("  Memory: psutil not available, cannot get memory info")
        
        # GPU info
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                self.info(f"  GPU: {gpu_info}")
            else:
                self.warning("  No GPU information available")
        except:
            self.warning("  Could not query GPU information")
    
    def log_progress(self, stage: str, progress: float, details: str = ""):
        """Log training/processing progress"""
        elapsed = time.time() - self.start_time
        message = f"📊 {stage}: {progress:.1%} complete ({elapsed:.1f}s elapsed)"
        if details:
            message += f" - {details}"
        self.info(message)
        
        # Update job info
        self.job_info['current_stage'] = stage
        self.job_info['progress'] = progress
        self.job_info['elapsed_time'] = elapsed
        self.save_job_info()
    
    def log_metric(self, name: str, value: float, step: int = None):
        """Log training metrics"""
        message = f"📈 {name}: {value:.6f}"
        if step is not None:
            message += f" (step {step})"
        self.info(message)
    
    def log_error_with_traceback(self, error: Exception, context: str = ""):
        """Log error with full traceback"""
        self.error(f"❌ ERROR {context}: {str(error)}")
        self.error("Full traceback:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                self.error(f"  {line}")
    
    def log_file_operation(self, operation: str, filepath: str, success: bool, 
                          size_mb: float = None, details: str = ""):
        """Log file operations (save/load)"""
        status = "✅" if success else "❌"
        message = f"{status} {operation}: {filepath}"
        if size_mb is not None:
            message += f" ({size_mb:.1f}MB)"
        if details:
            message += f" - {details}"
        self.info(message)
    
    def save_job_info(self):
        """Save job information to JSON file"""
        job_info_file = self.log_dir / f"{self.job_id}_info.json"
        try:
            with open(job_info_file, 'w') as f:
                json.dump(self.job_info, f, indent=2)
        except Exception as e:
            self.warning(f"Could not save job info: {e}")
    
    def mark_completed(self, success: bool = True, final_message: str = ""):
        """Mark job as completed"""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        self.job_info['end_time'] = end_time
        self.job_info['end_time_str'] = datetime.fromtimestamp(end_time).isoformat()
        self.job_info['total_time_seconds'] = total_time
        self.job_info['status'] = 'COMPLETED' if success else 'FAILED'
        
        if final_message:
            self.job_info['final_message'] = final_message
        
        self.save_job_info()
        
        # Log completion
        status = "✅ COMPLETED" if success else "❌ FAILED"
        self.info("="*80)
        self.info(f"{status}: {self.job_id}")
        self.info(f"⏱️  Total runtime: {total_time:.1f} seconds ({total_time/3600:.2f} hours)")
        if final_message:
            self.info(f"📝 Final message: {final_message}")
        self.info("="*80)

class JobQueueManager:
    """
    Manages job submissions and tracks results to prevent loss
    """
    
    def __init__(self, queue_dir: str):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        self.submitted_jobs_file = self.queue_dir / "submitted_jobs.json"
        self.completed_jobs_file = self.queue_dir / "completed_jobs.json"
        
        # Load existing job lists
        self.submitted_jobs = self.load_job_list(self.submitted_jobs_file)
        self.completed_jobs = self.load_job_list(self.completed_jobs_file)
    
    def load_job_list(self, filepath: Path) -> Dict[str, Any]:
        """Load job list from JSON file"""
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def save_job_list(self, job_list: Dict[str, Any], filepath: Path):
        """Save job list to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(job_list, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"Warning: Could not save job list to {filepath}: {e}")
    
    def register_job_submission(self, job_id: str, job_info: Dict[str, Any]):
        """Register a submitted job"""
        self.submitted_jobs[job_id] = {
            **job_info,
            'submission_time': time.time(),
            'submission_time_str': datetime.now().isoformat(),
            'status': 'SUBMITTED'
        }
        self.save_job_list(self.submitted_jobs, self.submitted_jobs_file)
        print(f"📝 Registered job submission: {job_id}")
    
    def mark_job_completed(self, job_id: str, success: bool, results: Dict[str, Any] = None):
        """Mark a job as completed and save results"""
        if job_id in self.submitted_jobs:
            job_info = self.submitted_jobs[job_id].copy()
            job_info.update({
                'completion_time': time.time(),
                'completion_time_str': datetime.now().isoformat(),
                'status': 'COMPLETED' if success else 'FAILED',
                'success': success
            })
            
            if results:
                job_info['results'] = results
            
            self.completed_jobs[job_id] = job_info
            self.save_job_list(self.completed_jobs, self.completed_jobs_file)
            
            # Remove from submitted list
            del self.submitted_jobs[job_id]
            self.save_job_list(self.submitted_jobs, self.submitted_jobs_file)
            
            status = "✅ COMPLETED" if success else "❌ FAILED"
            print(f"📋 Marked job {status}: {job_id}")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a specific job"""
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        elif job_id in self.submitted_jobs:
            return self.submitted_jobs[job_id]
        else:
            return {'status': 'NOT_FOUND'}
    
    def list_running_jobs(self) -> Dict[str, Any]:
        """List all currently running jobs"""
        return self.submitted_jobs
    
    def list_completed_jobs(self) -> Dict[str, Any]:
        """List all completed jobs"""
        return self.completed_jobs
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all jobs"""
        running_count = len(self.submitted_jobs)
        completed_count = len(self.completed_jobs)
        success_count = sum(1 for job in self.completed_jobs.values() 
                           if job.get('success', False))
        
        return {
            'running_jobs': running_count,
            'completed_jobs': completed_count,
            'successful_jobs': success_count,
            'failed_jobs': completed_count - success_count,
            'total_jobs': running_count + completed_count
        }

def test_comprehensive_logging():
    """Test the comprehensive logging system"""
    print("🧪 Testing comprehensive logging...")
    
    logger = ComprehensiveLogger("test_logs", "test_job", particle_pid=123)
    
    # Test various log types
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.debug("This is a debug message")
    
    # Test progress logging
    for i in range(5):
        logger.log_progress("Training", i/4, f"Epoch {i+1}/5")
        time.sleep(0.1)
    
    # Test metric logging
    logger.log_metric("loss", 2.456, step=100)
    logger.log_metric("accuracy", 0.892)
    
    # Test file operation logging
    logger.log_file_operation("SAVE", "/path/to/model.h5", True, 15.6, "Model weights")
    
    # Test completion
    logger.mark_completed(True, "Test completed successfully")
    
    print("✅ Comprehensive logging test completed")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_logs")

if __name__ == "__main__":
    test_comprehensive_logging()
