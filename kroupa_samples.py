#!/usr/bin/env python3
"""
Kroupa IMF Sampling Script
Loads trained flows and generates realistic stellar populations using Kroupa IMF
Creates kroupa-samples directory with same structure as samples
"""

import sys
import argparse
import time
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

import numpy as np
import h5py

# Configure TensorFlow before importing to avoid threading issues
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import tensorflow as tf
import tensorflow_probability as tfp

# Configure TensorFlow for CPU-only mode with limited threading
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.set_visible_devices([], 'GPU')  # Disable GPU

# Import our modules
from tfp_flows_gpu_solution import TFPNormalizingFlow, load_trained_flow
from kroupa_imf import sample_with_kroupa_imf, get_stellar_mass_from_h5
from optimized_io import save_samples_optimized

def setup_logging(log_dir: str, particle_pid: int) -> logging.Logger:
    """Setup logging for the sampling process"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(f'kroupa_sampling_{particle_pid}')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    log_file = Path(log_dir) / f"kroupa_sampling_pid{particle_pid}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def find_trained_models(base_dir: str, particle_pid: int = None) -> List[Dict[str, Any]]:
    """
    Find all trained model files and their metadata
    
    Args:
        base_dir: Base directory to search for models
        particle_pid: Specific particle ID to find (None for all)
    
    Returns:
        List of model info dictionaries
    """
    models = []
    
    # Search for model files
    search_pattern = f"model_pid{particle_pid}.npz" if particle_pid else "model_pid*.npz"
    
    for model_file in Path(base_dir).rglob(search_pattern):
        # Extract info from path
        model_dir = model_file.parent
        model_name = model_file.stem
        
        # Extract PID from filename
        try:
            pid = int(model_name.split('_pid')[1])
        except (IndexError, ValueError):
            continue
        
        # Find corresponding preprocessing and results files
        preprocessing_file = model_dir / f"{model_name}_preprocessing.npz"
        results_file = model_dir / f"{model_name}_results.json"
        
        # Extract data source and halo info from path
        path_parts = str(model_dir).split('/')
        data_source = "unknown"
        halo_id = "unknown"
        
        for i, part in enumerate(path_parts):
            if part == "trained_flows" and i + 1 < len(path_parts):
                data_source = path_parts[i + 1]
            if part.startswith("halo"):
                halo_id = part
        
        model_info = {
            'pid': pid,
            'model_file': str(model_file),
            'preprocessing_file': str(preprocessing_file) if preprocessing_file.exists() else None,
            'results_file': str(results_file) if results_file.exists() else None,
            'data_source': data_source,
            'halo_id': halo_id,
            'model_dir': str(model_dir)
        }
        
        models.append(model_info)
    
    return sorted(models, key=lambda x: x['pid'])

def load_model_and_preprocessing(model_info: Dict[str, Any], logger: logging.Logger) -> Tuple[TFPNormalizingFlow, Dict, Dict]:
    """
    Load trained model, preprocessing stats, and metadata
    
    Returns:
        Tuple of (flow, preprocessing_stats, metadata)
    """
    logger.info(f"Loading model for PID {model_info['pid']}")
    
    # Load the trained flow
    flow = load_trained_flow(model_info['model_file'])
    logger.info(f"✅ Loaded flow model: {flow.input_dim}D, {flow.n_layers} layers")
    
    # Load preprocessing stats
    preprocessing_stats = {}
    if model_info['preprocessing_file'] and os.path.exists(model_info['preprocessing_file']):
        preprocessing_data = np.load(model_info['preprocessing_file'], allow_pickle=True)
        preprocessing_stats = {key: preprocessing_data[key] for key in preprocessing_data.files}
        logger.info("✅ Loaded preprocessing statistics")
    else:
        logger.warning("⚠️ No preprocessing file found - using defaults")
        preprocessing_stats = {
            'mean': np.zeros(flow.input_dim, dtype=np.float32),
            'std': np.ones(flow.input_dim, dtype=np.float32),
            'standardize': True,
            'clip_outliers': 5.0
        }
    
    # Load metadata from results file
    metadata = {}
    if model_info['results_file'] and os.path.exists(model_info['results_file']):
        with open(model_info['results_file'], 'r') as f:
            results_data = json.load(f)
            metadata = results_data.get('metadata', {})
        logger.info("✅ Loaded model metadata")
    else:
        logger.warning("⚠️ No results file found - using minimal metadata")
        metadata = {
            'particle_pid': model_info['pid'],
            'stellar_mass': 1e8  # Default fallback
        }
    
    return flow, preprocessing_stats, metadata

def find_original_h5_file(model_info: Dict[str, Any], logger: logging.Logger) -> Optional[str]:
    """
    Find the original H5 file that was used to train this model
    """
    # Common search paths based on data source
    search_paths = [
        "/oak/stanford/orgs/kipac/users/caganze/milkyway-eden-mocks/",
        "/oak/stanford/orgs/kipac/users/caganze/symphony_mocks/",
        "/oak/stanford/orgs/kipac/users/caganze/",
    ]
    
    # Construct likely filenames based on data source and halo
    halo_num = model_info['halo_id'].replace('halo', '')
    
    possible_filenames = []
    if model_info['data_source'] == 'eden':
        possible_filenames.extend([
            f"eden_scaled_Halo{halo_num}_sunrot0_0kpc200kpcoriginal_particles.h5",
            f"eden_scaled_Halo{halo_num}.h5"
        ])
    elif model_info['data_source'] == 'symphony':
        possible_filenames.extend([
            f"symphony_scaled_Halo{halo_num}.h5",
            "all_in_one.h5"
        ])
    elif model_info['data_source'] == 'symphony-hr':
        possible_filenames.extend([
            f"symphonyHR_scaled_Halo{halo_num}.h5"
        ])
    
    # Search for files
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        for filename in possible_filenames:
            h5_file = os.path.join(search_path, filename)
            if os.path.exists(h5_file):
                logger.info(f"✅ Found original H5 file: {h5_file}")
                return h5_file
    
    logger.warning(f"⚠️ Could not find original H5 file for {model_info}")
    return None

def get_stellar_mass_for_particle(h5_file: str, particle_pid: int, logger: logging.Logger) -> float:
    """
    Get stellar mass for specific particle from H5 file
    """
    if not h5_file or not os.path.exists(h5_file):
        logger.warning(f"⚠️ H5 file not available, using default stellar mass")
        return 1e8
    
    try:
        from test_h5_read_single_particle import read_h5_to_dict
        
        # Read H5 file
        data_dict = read_h5_to_dict(h5_file)
        
        # Get stellar mass using existing function
        stellar_mass = get_stellar_mass_from_h5(data_dict, particle_pid)
        logger.info(f"✅ Retrieved stellar mass: {stellar_mass:.2e} M☉")
        
        return stellar_mass
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to get stellar mass from H5: {e}")
        return 1e8  # Default fallback

def generate_kroupa_samples(model_info: Dict[str, Any], output_base_dir: str, logger: logging.Logger) -> bool:
    """
    Generate Kroupa IMF samples for a single particle
    
    Returns:
        True if successful, False if failed
    """
    try:
        logger.info(f"🌟 Starting Kroupa sampling for PID {model_info['pid']}")
        
        # Load model and preprocessing
        flow, preprocessing_stats, metadata = load_model_and_preprocessing(model_info, logger)
        
        # Find original H5 file to get stellar mass
        h5_file = find_original_h5_file(model_info, logger)
        
        # Get stellar mass
        if h5_file:
            stellar_mass = get_stellar_mass_for_particle(h5_file, model_info['pid'], logger)
        else:
            # Try to get from metadata
            stellar_mass = metadata.get('stellar_mass', 1e8)
            logger.info(f"Using stellar mass from metadata: {stellar_mass:.2e} M☉")
        
        # Create output directory
        output_dir = Path(output_base_dir) / "kroupa-samples" / model_info['data_source'] / model_info['halo_id']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Output directory: {output_dir}")
        
        # Generate samples using Kroupa IMF
        logger.info(f"🎲 Generating Kroupa IMF samples...")
        logger.info(f"   Target stellar mass: {stellar_mass:.2e} M☉")
        
        samples, masses = sample_with_kroupa_imf(
            flow=flow,
            n_target_mass=stellar_mass,
            preprocessing_stats=preprocessing_stats,
            seed=42
        )
        
        n_samples = len(samples)
        total_mass = np.sum(masses)
        
        logger.info(f"✅ Generated {n_samples:,} stellar samples")
        logger.info(f"   Total mass: {total_mass:.2e} M☉")
        logger.info(f"   Mean stellar mass: {np.mean(masses):.2e} M☉")
        
        # Prepare comprehensive metadata
        kroupa_metadata = {
            **metadata,
            'pid': model_info['pid'],
            'data_source': model_info['data_source'],
            'halo_id': model_info['halo_id'],
            'kroupa_sampling': True,
            'n_samples': n_samples,
            'target_stellar_mass': stellar_mass,
            'actual_total_mass': float(total_mass),
            'mean_stellar_mass': float(np.mean(masses)),
            'min_stellar_mass': float(np.min(masses)),
            'max_stellar_mass': float(np.max(masses)),
            'sampling_timestamp': time.time(),
            'sampling_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'original_h5_file': h5_file,
            'source_model': model_info['model_file']
        }
        
        # Save samples using optimized I/O
        model_name = f"kroupa_pid{model_info['pid']}"
        
        saved_files = save_samples_optimized(
            samples=samples,
            masses=masses,
            output_dir=str(output_dir),
            model_name=model_name,
            metadata=kroupa_metadata
        )
        
        logger.info(f"✅ Saved Kroupa samples:")
        for file_type, file_path in saved_files.items():
            logger.info(f"   {file_type.upper()}: {file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to generate Kroupa samples for PID {model_info['pid']}: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate Kroupa IMF samples from trained flows")
    
    parser.add_argument("--base-dir", required=True, help="Base directory containing trained_flows")
    parser.add_argument("--particle-pid", type=int, help="Specific particle ID to process (default: all)")
    parser.add_argument("--output-base", help="Base output directory (default: same as base-dir)")
    parser.add_argument("--failed-log", default="kroupa_failed_pids.txt", help="File to log failed PIDs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without doing it")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup
    output_base = args.output_base or args.base_dir
    
    print("🌟 KROUPA IMF SAMPLING")
    print("=" * 50)
    print(f"Base directory: {args.base_dir}")
    print(f"Output base: {output_base}")
    if args.particle_pid:
        print(f"Target PID: {args.particle_pid}")
    else:
        print("Processing: All found models")
    print()
    
    # Find trained models
    print("🔍 Finding trained models...")
    models = find_trained_models(args.base_dir, args.particle_pid)
    
    if not models:
        print("❌ No trained models found!")
        return 1
    
    print(f"✅ Found {len(models)} trained model(s)")
    
    if args.dry_run:
        print("\n🧪 DRY RUN - Models that would be processed:")
        for model in models:
            print(f"   PID {model['pid']}: {model['data_source']}/{model['halo_id']}")
        return 0
    
    # Process models
    print(f"\n🚀 Processing {len(models)} model(s)...")
    
    successful = 0
    failed = 0
    failed_pids = []
    
    for i, model_info in enumerate(models):
        print(f"\n📦 Processing {i+1}/{len(models)}: PID {model_info['pid']}")
        
        # Setup logging
        log_dir = Path(output_base) / "kroupa-samples" / "logs"
        logger = setup_logging(str(log_dir), model_info['pid'])
        
        # Generate samples
        if generate_kroupa_samples(model_info, output_base, logger):
            successful += 1
            print(f"✅ Success: PID {model_info['pid']}")
        else:
            failed += 1
            failed_pids.append(model_info['pid'])
            print(f"❌ Failed: PID {model_info['pid']}")
    
    # Save failed PIDs
    if failed_pids:
        with open(args.failed_log, 'w') as f:
            for pid in failed_pids:
                f.write(f"{pid}\n")
        print(f"\n📝 Failed PIDs saved to: {args.failed_log}")
    
    # Summary
    print(f"\n📊 KROUPA SAMPLING SUMMARY")
    print("=" * 30)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success rate: {successful * 100 / len(models):.1f}%")
    
    if successful > 0:
        print(f"\n🎉 Kroupa samples saved to: {output_base}/kroupa-samples/")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())

