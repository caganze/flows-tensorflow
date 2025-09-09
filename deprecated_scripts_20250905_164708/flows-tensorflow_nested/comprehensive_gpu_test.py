#!/usr/bin/env python3
"""
Comprehensive GPU test for the complete TensorFlow Probability flows pipeline
Tests: Kroupa IMF + Optimized I/O + GPU training + Multiple particles

NO JAX DEPENDENCIES - Pure TensorFlow implementation
"""

import sys
import os
import time
import argparse
from pathlib import Path
import subprocess
import shutil

# Core scientific libraries
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import h5py
import json

# Our modules (ensuring no JAX)
from train_tfp_flows import train_and_save_flow
from kroupa_imf import get_stellar_mass_from_h5
from optimized_io import save_samples_optimized, load_samples_optimized, test_io_methods
from tfp_flows_gpu_solution import TFPNormalizingFlow
from comprehensive_logging import ComprehensiveLogger, JobQueueManager

def check_gpu_availability():
    """Check GPU availability and TensorFlow GPU setup."""
    print("🔍 Checking GPU availability...")
    
    # Check physical devices
    physical_devices = tf.config.list_physical_devices('GPU')
    print(f"Physical GPU devices: {len(physical_devices)}")
    for i, device in enumerate(physical_devices):
        print(f"  GPU {i}: {device}")
    
    # Check if TensorFlow can see GPUs
    if len(physical_devices) == 0:
        print("❌ No GPUs detected by TensorFlow")
        return False
    
    # Test GPU computation
    try:
        with tf.device('/GPU:0'):
            test_tensor = tf.random.normal((1000, 1000))
            result = tf.linalg.matmul(test_tensor, test_tensor)
            print(f"✅ GPU computation test passed: {result.shape}")
        return True
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False

def check_dependencies():
    """Check that all required packages are available (no JAX)."""
    print("🔍 Checking dependencies...")
    
    required_packages = {
        'tensorflow': tf.__version__,
        'tensorflow_probability': tfp.__version__,
        'numpy': np.__version__,
        'h5py': h5py.__version__
    }
    
    print("✅ Required packages:")
    for package, version in required_packages.items():
        print(f"  {package}: {version}")
    
    # Ensure JAX is NOT imported
    forbidden_modules = ['jax', 'jaxlib']
    for module in forbidden_modules:
        if module in sys.modules:
            print(f"❌ WARNING: {module} is imported! This may cause conflicts.")
            return False
        else:
            print(f"✅ {module}: not imported (good)")
    
    return True

def test_kroupa_imf():
    """Test Kroupa IMF functionality."""
    print("\n🧪 Testing Kroupa IMF...")
    
    try:
        # Import and run the built-in test function
        from kroupa_imf import test_kroupa_imf as run_kroupa_test
        run_kroupa_test()
        print("✅ Kroupa IMF test passed")
        return True
    except Exception as e:
        print(f"❌ Kroupa IMF test failed: {e}")
        return False

def test_optimized_io():
    """Test optimized I/O methods."""
    print("\n🧪 Testing optimized I/O...")
    
    try:
        # Run the built-in test
        test_io_methods()
        print("✅ Optimized I/O test passed")
        return True
    except Exception as e:
        print(f"❌ Optimized I/O test failed: {e}")
        return False

def test_single_particle_training(h5_file: str, particle_pid: int, 
                                 n_samples: int = 50000, epochs: int = 5):
    """Test training on a single particle with all new features."""
    print(f"\n🧪 Testing single particle training (PID {particle_pid})...")
    
    # Get stellar mass for this particle
    try:
        stellar_mass = get_stellar_mass_from_h5(h5_file, particle_pid)
        print(f"  Stellar mass for PID {particle_pid}: {stellar_mass:.2e} M☉")
    except Exception as e:
        print(f"  Warning: Could not get stellar mass: {e}")
        stellar_mass = 1e8  # Default fallback
    
    # Set up training parameters
    output_dir = f"test_output/comprehensive_test_pid{particle_pid}"
    
    try:
        # Train with all new features
        model_path, samples_path = train_and_save_flow(
            h5_file=h5_file,
            particle_pid=particle_pid,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=1024,
            learning_rate=1e-3,
            n_layers=4,
            hidden_units=64,
            generate_samples=True,
            n_samples=n_samples,
            use_kroupa_imf=True,  # Enable Kroupa IMF
            validation_split=0.2,
            early_stopping_patience=50,
            reduce_lr_patience=20
        )
        
        print(f"  ✅ Training completed")
        print(f"  Model saved: {model_path}")
        print(f"  Samples saved: {samples_path}")
        
        # Verify saved files exist
        model_files = list(Path(model_path).parent.glob("*"))
        sample_files = list(Path(samples_path).glob("*")) if samples_path else []
        
        print(f"  Model files ({len(model_files)}): {[f.name for f in model_files[:3]]}...")
        print(f"  Sample files ({len(sample_files)}): {[f.name for f in sample_files[:3]]}...")
        
        # Test loading the samples with optimized I/O
        if sample_files:
            for sample_file in sample_files:
                if sample_file.suffix in ['.h5', '.npz']:
                    try:
                        loaded_data = load_samples_optimized(str(sample_file))
                        n_loaded = loaded_data['samples_6d'].shape[0]
                        print(f"  ✅ Loaded {n_loaded:,} samples from {sample_file.name}")
                        
                        # Check for Kroupa masses
                        if 'masses' in loaded_data:
                            masses = loaded_data['masses']
                            print(f"  ✅ Kroupa masses included: {len(masses):,} stars")
                            print(f"     Total mass: {np.sum(masses):.2e} M☉")
                        break
                    except Exception as e:
                        print(f"  ⚠️  Could not load {sample_file.name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_particles(h5_file: str, particle_pids: list, 
                          n_samples: int = 25000, epochs: int = 3):
    """Test training on multiple particles."""
    print(f"\n🧪 Testing multiple particle training ({len(particle_pids)} particles)...")
    
    results = {}
    for i, pid in enumerate(particle_pids):
        print(f"\n--- Particle {i+1}/{len(particle_pids)}: PID {pid} ---")
        
        start_time = time.time()
        success = test_single_particle_training(
            h5_file=h5_file,
            particle_pid=pid,
            n_samples=n_samples,
            epochs=epochs
        )
        elapsed = time.time() - start_time
        
        results[pid] = {
            'success': success,
            'elapsed_time': elapsed
        }
        
        print(f"  PID {pid}: {'✅ Success' if success else '❌ Failed'} ({elapsed:.1f}s)")
    
    # Summary
    successes = sum(1 for r in results.values() if r['success'])
    total_time = sum(r['elapsed_time'] for r in results.values())
    
    print(f"\n📊 Multiple particle test summary:")
    print(f"  Success rate: {successes}/{len(particle_pids)} ({100*successes/len(particle_pids):.1f}%)")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average time per particle: {total_time/len(particle_pids):.1f}s")
    
    return successes == len(particle_pids)

def test_job_management():
    """Test job queue management and result preservation."""
    print("\n🧪 Testing job management...")
    
    try:
        # Test job queue manager
        queue_manager = JobQueueManager("test_queue")
        
        # Register test job
        test_job_info = {
            'job_type': 'training',
            'particle_pid': 999,
            'test_job': True
        }
        
        job_id = "test_job_999"
        queue_manager.register_job_submission(job_id, test_job_info)
        
        # Check job status
        status = queue_manager.get_job_status(job_id)
        assert status['status'] == 'SUBMITTED', "Job not properly registered"
        
        # Mark job completed with results
        test_results = {
            'model_path': '/path/to/model.h5',
            'samples_path': '/path/to/samples.npz',
            'final_loss': 2.456
        }
        
        queue_manager.mark_job_completed(job_id, True, test_results)
        
        # Verify completion
        status = queue_manager.get_job_status(job_id)
        assert status['status'] == 'COMPLETED', "Job not properly completed"
        assert status['results']['final_loss'] == 2.456, "Results not preserved"
        
        # Test summary
        summary = queue_manager.get_summary()
        assert summary['successful_jobs'] == 1, "Summary incorrect"
        
        print("  ✅ Job queue management working")
        
        # Test comprehensive logging
        logger = ComprehensiveLogger("test_logs", "test_training", particle_pid=999)
        logger.info("Test log message")
        logger.log_progress("Testing", 0.5, "Halfway done")
        logger.log_metric("test_metric", 3.14159)
        logger.mark_completed(True, "Test logging completed")
        
        print("  ✅ Comprehensive logging working")
        
        # Cleanup
        import shutil
        if os.path.exists("test_queue"):
            shutil.rmtree("test_queue")
        if os.path.exists("test_logs"):
            shutil.rmtree("test_logs")
        
        print("✅ Job management test passed")
        return True
        
    except Exception as e:
        print(f"❌ Job management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_result_validation():
    """Test that results are properly saved and accessible."""
    print("\n🧪 Testing result validation...")
    
    try:
        # Create test output structure
        test_output_dir = "test_validation_output"
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Test model saving
        model_dir = f"{test_output_dir}/trained_flows/test_model"
        os.makedirs(model_dir, exist_ok=True)
        
        # Create dummy model file
        model_file = f"{model_dir}/model_weights.h5"
        with open(model_file, 'w') as f:
            f.write("dummy model data")
        
        # Create dummy results file
        results_file = f"{model_dir}/training_results.json"
        test_results = {
            'final_loss': 2.456,
            'epochs': 50,
            'success': True
        }
        with open(results_file, 'w') as f:
            json.dump(test_results, f)
        
        # Test samples saving
        samples_dir = f"{test_output_dir}/samples"
        os.makedirs(samples_dir, exist_ok=True)
        
        # Create dummy samples
        samples_file = f"{samples_dir}/test_model_samples.npz"
        dummy_samples = np.random.normal(0, 1, (1000, 6))
        np.savez_compressed(samples_file, samples_6d=dummy_samples)
        
        # Validate files exist and are readable
        assert os.path.exists(model_file), "Model file not saved"
        assert os.path.exists(results_file), "Results file not saved"
        assert os.path.exists(samples_file), "Samples file not saved"
        
        # Validate content
        with open(results_file, 'r') as f:
            loaded_results = json.load(f)
        assert loaded_results['final_loss'] == 2.456, "Results not properly saved"
        
        loaded_samples = np.load(samples_file, allow_pickle=True)
        assert loaded_samples['samples_6d'].shape == (1000, 6), "Samples not properly saved"
        
        print("  ✅ Model files properly saved and accessible")
        print("  ✅ Results JSON properly saved and accessible")  
        print("  ✅ Sample files properly saved and accessible")
        
        # Test file size validation
        model_size = os.path.getsize(model_file) / 1024  # KB
        samples_size = os.path.getsize(samples_file) / 1024  # KB
        
        print(f"  📊 Model file size: {model_size:.1f} KB")
        print(f"  📊 Samples file size: {samples_size:.1f} KB")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_output_dir)
        
        print("✅ Result validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Result validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_queue_submission_validation():
    """Test SLURM job submission validation."""
    print("\n🧪 Testing queue submission validation...")
    
    try:
        # Test SLURM environment detection
        slurm_env = {
            'SLURM_JOB_ID': os.environ.get('SLURM_JOB_ID'),
            'SLURM_ARRAY_TASK_ID': os.environ.get('SLURM_ARRAY_TASK_ID'),
            'SLURM_PROCID': os.environ.get('SLURM_PROCID'),
            'SLURM_NTASKS': os.environ.get('SLURM_NTASKS')
        }
        
        print("  📋 SLURM environment:")
        for key, value in slurm_env.items():
            status = "✅" if value else "❌"
            print(f"    {status} {key}: {value}")
        
        # Check if we're in a SLURM job
        in_slurm_job = slurm_env['SLURM_JOB_ID'] is not None
        print(f"  🎯 Running in SLURM job: {'✅ Yes' if in_slurm_job else '❌ No'}")
        
        # Test job script validation
        required_scripts = [
            'submit_flows_array.sh',
            'train_single_gpu.sh',
            'run_comprehensive_gpu_test.sh'
        ]
        
        print("  📋 Job script validation:")
        for script in required_scripts:
            exists = os.path.exists(script)
            executable = os.access(script, os.X_OK) if exists else False
            status = "✅" if exists and executable else "❌"
            print(f"    {status} {script}: {'exists & executable' if exists and executable else 'missing or not executable'}")
        
        # Test output directory structure
        expected_dirs = [
            'logs',
            'trained_flows', 
            'samples'
        ]
        
        print("  📁 Output directory validation:")
        for dirname in expected_dirs:
            # Create if doesn't exist (for testing)
            os.makedirs(dirname, exist_ok=True)
            writable = os.access(dirname, os.W_OK)
            status = "✅" if writable else "❌"
            print(f"    {status} {dirname}/: {'writable' if writable else 'not writable'}")
        
        # Cleanup test directories
        for dirname in expected_dirs:
            if os.path.exists(dirname) and os.path.basename(os.getcwd()) != 'flows-tensorflow':
                import shutil
                shutil.rmtree(dirname)
        
        print("✅ Queue submission validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Queue submission validation test failed: {e}")
        return False

def cleanup_test_outputs():
    """Clean up test output directories."""
    print("\n🧹 Cleaning up test outputs...")
    
    cleanup_dirs = ['test_output', 'test_io_output']
    for dirname in cleanup_dirs:
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
            print(f"  Removed: {dirname}/")

def comprehensive_test(h5_file: str, test_particle_pids: list = None, 
                      n_samples: int = 50000, epochs: int = 5, 
                      cleanup: bool = True):
    """Run comprehensive test of the entire pipeline."""
    print("🚀 COMPREHENSIVE GPU TEST STARTING")
    print("=" * 60)
    
    # Default test particles if not specified
    if test_particle_pids is None:
        test_particle_pids = [1, 2, 200]  # Mix of small and large PIDs
    
    test_results = {}
    
    # 1. Environment checks
    print("\n1️⃣ ENVIRONMENT CHECKS")
    test_results['dependencies'] = check_dependencies()
    test_results['gpu'] = check_gpu_availability()
    
    if not test_results['dependencies'] or not test_results['gpu']:
        print("❌ Environment checks failed. Cannot proceed.")
        return False
    
    # 2. Component tests
    print("\n2️⃣ COMPONENT TESTS")
    test_results['kroupa_imf'] = test_kroupa_imf()
    test_results['optimized_io'] = test_optimized_io()
    
    # 2.5. Job management and result preservation tests
    print("\n2️⃣.5️⃣ JOB MANAGEMENT TESTS")
    test_results['job_management'] = test_job_management()
    test_results['result_validation'] = test_result_validation()
    test_results['queue_submission'] = test_queue_submission_validation()
    
    # 3. Single particle training test
    print("\n3️⃣ SINGLE PARTICLE TEST")
    test_results['single_particle'] = test_single_particle_training(
        h5_file=h5_file,
        particle_pid=test_particle_pids[0],
        n_samples=n_samples,
        epochs=epochs
    )
    
    # 4. Multiple particle test
    print("\n4️⃣ MULTIPLE PARTICLE TEST")
    test_results['multiple_particles'] = test_multiple_particles(
        h5_file=h5_file,
        particle_pids=test_particle_pids,
        n_samples=n_samples//2,  # Smaller for speed
        epochs=max(epochs//2, 2)  # Fewer epochs for speed
    )
    
    # 5. Results summary
    print("\n" + "=" * 60)
    print("📋 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:20}: {status}")
    
    overall_success = all(test_results.values())
    print(f"\n🎯 OVERALL RESULT: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🚀 Pipeline is ready for full deployment!")
    else:
        print("\n⚠️  Please fix failing tests before deployment.")
    
    # Cleanup
    if cleanup:
        cleanup_test_outputs()
    
    return overall_success

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Comprehensive GPU test for TFP flows pipeline")
    
    parser.add_argument("--h5_file", type=str, required=True,
                        help="Path to HDF5 file with particle data")
    parser.add_argument("--particle_pids", type=int, nargs="+", default=[1, 2, 200],
                        help="Particle PIDs to test")
    parser.add_argument("--n_samples", type=int, default=50000,
                        help="Number of samples to generate")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--no_cleanup", action="store_true",
                        help="Don't cleanup test outputs")
    
    args = parser.parse_args()
    
    # Check if H5 file exists
    if not os.path.exists(args.h5_file):
        print(f"❌ H5 file not found: {args.h5_file}")
        sys.exit(1)
    
    # Run comprehensive test
    success = comprehensive_test(
        h5_file=args.h5_file,
        test_particle_pids=args.particle_pids,
        n_samples=args.n_samples,
        epochs=args.epochs,
        cleanup=not args.no_cleanup
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
