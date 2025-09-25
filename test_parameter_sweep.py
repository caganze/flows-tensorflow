#!/usr/bin/env python3
"""
Comprehensive parameter sweep testing script for conditional CNF flows
Tests different parameter combinations across multiple particle IDs to find optimal settings
"""

import os
import sys
import argparse
import time
import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import pandas as pd

# Configure TensorFlow environment before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@dataclass
class TestConfig:
    """Configuration for a single test run"""
    particle_pid: int
    hidden_units: List[int]
    epochs: int
    learning_rate: float
    batch_size: int
    integration_time: float
    num_integration_steps: int
    validation_freq: int
    clip_outliers: float

@dataclass
class TestResult:
    """Results from a single test run"""
    config: TestConfig
    success: bool
    training_time: float
    final_train_loss: float
    final_val_loss: float
    n_particles: int
    model_size_mb: float
    error_message: str = ""

def get_particle_info(halo_id: str, particle_pid: int, suite: str = 'eden') -> Dict[str, Any]:
    """Get information about a particle ID including number of particles"""
    try:
        # Import symlib utilities
        from symlib_utils import load_particle_data, validate_symlib_environment
        
        if not validate_symlib_environment():
            return {"n_particles": 0, "error": "Symlib environment not available"}
        
        # Load particle data to get count
        data, metadata = load_particle_data(halo_id, particle_pid, suite)
        
        return {
            "n_particles": len(data),
            "stellar_mass": metadata.get('stellar_mass', 0),
            "mass_range": [float(np.min(data[:, 6])), float(np.max(data[:, 6]))] if data.shape[1] > 6 else [0, 0]
        }
    except Exception as e:
        return {"n_particles": 0, "error": str(e)}

def run_single_test(halo_id: str, suite: str, config: TestConfig, output_dir: str) -> TestResult:
    """Run a single test with given configuration"""
    
    print(f"\n🧪 Testing PID {config.particle_pid}: {config.hidden_units} units, {config.epochs} epochs, LR={config.learning_rate}")
    
    # Get particle info
    particle_info = get_particle_info(halo_id, config.particle_pid, suite)
    n_particles = particle_info.get("n_particles", 0)
    
    if n_particles == 0:
        return TestResult(
            config=config,
            success=False,
            training_time=0.0,
            final_train_loss=float('inf'),
            final_val_loss=float('inf'),
            n_particles=0,
            model_size_mb=0.0,
            error_message=particle_info.get("error", "No particles found")
        )
    
    # Build command
    hidden_units_str = " ".join(map(str, config.hidden_units))
    cmd = [
        "python", "train_cnf_flows_conditional.py",
        "--halo_id", halo_id,
        "--particle_pid", str(config.particle_pid),
        "--suite", suite,
        "--hidden_units", hidden_units_str,
        "--epochs", str(config.epochs),
        "--learning_rate", str(config.learning_rate),
        "--batch_size", str(config.batch_size),
        "--integration_time", str(config.integration_time),
        "--num_integration_steps", str(config.num_integration_steps),
        "--validation_freq", str(config.validation_freq),
        "--clip_outliers", str(config.clip_outliers),
        "--output_dir", output_dir
    ]
    
    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        training_time = time.time() - start_time
        
        if result.returncode == 0:
            # Parse results from stdout
            stdout_lines = result.stdout.split('\n')
            
            # Extract final losses
            final_train_loss = float('inf')
            final_val_loss = float('inf')
            
            for line in stdout_lines:
                if "final_training_loss:" in line:
                    try:
                        final_train_loss = float(line.split(":")[-1].strip())
                    except:
                        pass
                elif "final_validation_loss:" in line:
                    try:
                        final_val_loss = float(line.split(":")[-1].strip())
                    except:
                        pass
            
            # Get model size
            model_path = f"{output_dir}/conditional_model_pid{config.particle_pid}.npz"
            model_size_mb = 0.0
            if os.path.exists(model_path):
                model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            return TestResult(
                config=config,
                success=True,
                training_time=training_time,
                final_train_loss=final_train_loss,
                final_val_loss=final_val_loss,
                n_particles=n_particles,
                model_size_mb=model_size_mb
            )
        else:
            return TestResult(
                config=config,
                success=False,
                training_time=training_time,
                final_train_loss=float('inf'),
                final_val_loss=float('inf'),
                n_particles=n_particles,
                model_size_mb=0.0,
                error_message=result.stderr
            )
            
    except subprocess.TimeoutExpired:
        return TestResult(
            config=config,
            success=False,
            training_time=3600.0,
            final_train_loss=float('inf'),
            final_val_loss=float('inf'),
            n_particles=n_particles,
            model_size_mb=0.0,
            error_message="Training timed out after 1 hour"
        )
    except Exception as e:
        return TestResult(
            config=config,
            success=False,
            training_time=time.time() - start_time,
            final_train_loss=float('inf'),
            final_val_loss=float('inf'),
            n_particles=n_particles,
            model_size_mb=0.0,
            error_message=str(e)
        )

def create_parameter_sweep_configs() -> List[TestConfig]:
    """Create a comprehensive set of test configurations"""
    
    configs = []
    
    # Define parameter ranges
    particle_pids = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]  # Test multiple PIDs
    
    # Model complexity levels
    model_complexities = [
        ([16, 16], "small"),      # 2 layers, 16 units each
        ([32, 32], "medium"),     # 2 layers, 32 units each  
        ([64, 64], "large"),      # 2 layers, 64 units each
        ([32, 32, 32], "deep"),   # 3 layers, 32 units each
        ([64, 64, 64], "very_large"), # 3 layers, 64 units each
    ]
    
    # Training configurations
    training_configs = [
        (50, 1e-3, 256, "fast"),      # Fast training
        (100, 1e-4, 512, "standard"), # Standard training
        (200, 5e-5, 1024, "thorough"), # Thorough training
    ]
    
    # Integration settings
    integration_configs = [
        (1.0, 10, "standard"),    # Standard ODE integration
        (2.0, 20, "precise"),     # More precise integration
    ]
    
    # Generate all combinations
    for pid in particle_pids:
        for hidden_units, complexity_name in model_complexities:
            for epochs, lr, batch_size, training_name in training_configs:
                for integration_time, integration_steps, integration_name in integration_configs:
                    
                    config = TestConfig(
                        particle_pid=pid,
                        hidden_units=hidden_units,
                        epochs=epochs,
                        learning_rate=lr,
                        batch_size=batch_size,
                        integration_time=integration_time,
                        num_integration_steps=integration_steps,
                        validation_freq=5,
                        clip_outliers=5.0
                    )
                    
                    configs.append(config)
    
    return configs

def analyze_results(results: List[TestResult]) -> pd.DataFrame:
    """Analyze test results and create summary DataFrame"""
    
    data = []
    for result in results:
        if result.success:
            # Calculate model complexity metrics
            total_params = sum(result.config.hidden_units) * 6 * 2  # Rough estimate
            complexity_score = len(result.config.hidden_units) * sum(result.config.hidden_units)
            
            # Calculate efficiency metrics
            particles_per_second = result.n_particles / result.training_time if result.training_time > 0 else 0
            loss_per_particle = result.final_val_loss / result.n_particles if result.n_particles > 0 else float('inf')
            
            data.append({
                'particle_pid': result.config.particle_pid,
                'n_particles': result.n_particles,
                'hidden_units': str(result.config.hidden_units),
                'epochs': result.config.epochs,
                'learning_rate': result.config.learning_rate,
                'batch_size': result.config.batch_size,
                'integration_time': result.config.integration_time,
                'integration_steps': result.config.num_integration_steps,
                'training_time': result.training_time,
                'final_train_loss': result.final_train_loss,
                'final_val_loss': result.final_val_loss,
                'model_size_mb': result.model_size_mb,
                'total_params_est': total_params,
                'complexity_score': complexity_score,
                'particles_per_second': particles_per_second,
                'loss_per_particle': loss_per_particle,
                'success': True
            })
        else:
            data.append({
                'particle_pid': result.config.particle_pid,
                'n_particles': result.n_particles,
                'hidden_units': str(result.config.hidden_units),
                'epochs': result.config.epochs,
                'learning_rate': result.config.learning_rate,
                'batch_size': result.config.batch_size,
                'integration_time': result.config.integration_time,
                'integration_steps': result.config.num_integration_steps,
                'training_time': result.training_time,
                'final_train_loss': float('inf'),
                'final_val_loss': float('inf'),
                'model_size_mb': 0.0,
                'total_params_est': 0,
                'complexity_score': 0,
                'particles_per_second': 0,
                'loss_per_particle': float('inf'),
                'success': False,
                'error_message': result.error_message
            })
    
    return pd.DataFrame(data)

def create_particle_size_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Create more granular particle size categories"""
    
    # Define particle size categories based on actual data
    df['particle_size_category'] = pd.cut(
        df['n_particles'], 
        bins=[0, 1000, 5000, 10000, 20000, float('inf')],
        labels=['tiny', 'small', 'medium', 'large', 'huge']
    )
    
    return df

def generate_recommendations(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate parameter recommendations based on results"""
    
    # Filter successful results
    successful_df = df[df['success'] == True].copy()
    
    if len(successful_df) == 0:
        return {"error": "No successful training runs found"}
    
    recommendations = {}
    
    # Recommendations by particle size
    for size_category in ['tiny', 'small', 'medium', 'large', 'huge']:
        size_data = successful_df[successful_df['particle_size_category'] == size_category]
        
        if len(size_data) > 0:
            # Find best configuration for this size category
            # Weight by validation loss (lower is better) and training time (lower is better)
            size_data['score'] = 1.0 / (size_data['final_val_loss'] + 1e-6) / (size_data['training_time'] + 1e-6)
            
            best_config = size_data.loc[size_data['score'].idxmax()]
            
            recommendations[size_category] = {
                'optimal_hidden_units': best_config['hidden_units'],
                'optimal_epochs': int(best_config['epochs']),
                'optimal_learning_rate': best_config['learning_rate'],
                'optimal_batch_size': int(best_config['batch_size']),
                'optimal_integration_time': best_config['integration_time'],
                'optimal_integration_steps': int(best_config['integration_steps']),
                'expected_training_time': best_config['training_time'],
                'expected_val_loss': best_config['final_val_loss'],
                'n_particles_range': (size_data['n_particles'].min(), size_data['n_particles'].max())
            }
    
    # Overall recommendations
    recommendations['overall'] = {
        'fastest_training': successful_df.loc[successful_df['training_time'].idxmin()].to_dict(),
        'best_validation_loss': successful_df.loc[successful_df['final_val_loss'].idxmin()].to_dict(),
        'most_efficient': successful_df.loc[successful_df['particles_per_second'].idxmax()].to_dict()
    }
    
    return recommendations

def main():
    parser = argparse.ArgumentParser(description="Comprehensive parameter sweep for conditional CNF flows")
    parser.add_argument("--halo_id", required=True, help="Halo ID (e.g., Halo268)")
    parser.add_argument("--suite", default="eden", help="Simulation suite name")
    parser.add_argument("--output_dir", default="./parameter_sweep_results", help="Output directory for results")
    parser.add_argument("--max_tests", type=int, default=50, help="Maximum number of tests to run")
    parser.add_argument("--particle_pids", nargs='+', type=int, help="Specific particle PIDs to test (default: auto-select)")
    
    args = parser.parse_args()
    
    print("🚀 Starting Comprehensive Parameter Sweep")
    print("=" * 60)
    print(f"Halo ID: {args.halo_id}")
    print(f"Suite: {args.suite}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get test configurations
    if args.particle_pids:
        # Use specified PIDs
        configs = []
        for pid in args.particle_pids:
            # Get particle info first
            particle_info = get_particle_info(args.halo_id, pid, args.suite)
            if particle_info.get("n_particles", 0) > 0:
                # Add a few test configurations for this PID
                configs.extend([
                    TestConfig(pid, [32, 32], 100, 1e-4, 512, 1.0, 10, 5, 5.0),
                    TestConfig(pid, [64, 64], 100, 1e-4, 512, 1.0, 10, 5, 5.0),
                    TestConfig(pid, [32, 32], 200, 5e-5, 1024, 1.0, 10, 5, 5.0),
                ])
    else:
        # Use predefined configurations
        configs = create_parameter_sweep_configs()
    
    # Limit number of tests
    configs = configs[:args.max_tests]
    
    print(f"📊 Running {len(configs)} test configurations...")
    print()
    
    # Run tests
    results = []
    for i, config in enumerate(configs):
        print(f"Progress: {i+1}/{len(configs)}")
        result = run_single_test(args.halo_id, args.suite, config, args.output_dir)
        results.append(result)
        
        # Save intermediate results
        if (i + 1) % 10 == 0:
            df = analyze_results(results)
            df.to_csv(f"{args.output_dir}/intermediate_results_{i+1}.csv", index=False)
    
    # Analyze results
    print("\n📈 Analyzing results...")
    df = analyze_results(results)
    df = create_particle_size_categories(df)
    
    # Generate recommendations
    recommendations = generate_recommendations(df)
    
    # Save results
    df.to_csv(f"{args.output_dir}/parameter_sweep_results.csv", index=False)
    
    with open(f"{args.output_dir}/recommendations.json", 'w') as f:
        json.dump(recommendations, f, indent=2, default=str)
    
    # Print summary
    print("\n📊 RESULTS SUMMARY")
    print("=" * 60)
    
    successful_tests = len(df[df['success'] == True])
    total_tests = len(df)
    
    print(f"Successful tests: {successful_tests}/{total_tests}")
    print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests > 0:
        successful_df = df[df['success'] == True]
        print(f"Average training time: {successful_df['training_time'].mean():.1f} seconds")
        print(f"Average validation loss: {successful_df['final_val_loss'].mean():.3f}")
        print(f"Best validation loss: {successful_df['final_val_loss'].min():.3f}")
        
        print("\n🎯 RECOMMENDATIONS BY PARTICLE SIZE:")
        for size_category in ['tiny', 'small', 'medium', 'large', 'huge']:
            if size_category in recommendations:
                rec = recommendations[size_category]
                print(f"\n{size_category.upper()} particles ({rec['n_particles_range'][0]}-{rec['n_particles_range'][1]} particles):")
                print(f"  Hidden units: {rec['optimal_hidden_units']}")
                print(f"  Epochs: {rec['optimal_epochs']}")
                print(f"  Learning rate: {rec['optimal_learning_rate']}")
                print(f"  Batch size: {rec['optimal_batch_size']}")
                print(f"  Expected training time: {rec['expected_training_time']:.1f}s")
                print(f"  Expected validation loss: {rec['expected_val_loss']:.3f}")
    
    print(f"\n✅ Results saved to: {args.output_dir}/")
    print("📁 Files created:")
    print("  - parameter_sweep_results.csv (detailed results)")
    print("  - recommendations.json (parameter recommendations)")

if __name__ == "__main__":
    main()
