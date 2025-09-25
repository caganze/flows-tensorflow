#!/usr/bin/env python3
"""
Quick parameter test script for conditional CNF flows
Tests a focused set of parameters across multiple particle IDs
"""

import os
import sys
import subprocess
import time
import json
import pandas as pd
from pathlib import Path

def run_test(halo_id, particle_pid, hidden_units, epochs, learning_rate, batch_size, output_dir):
    """Run a single test and return results"""
    
    print(f"🧪 Testing PID {particle_pid}: {hidden_units} units, {epochs} epochs, LR={learning_rate}")
    
    hidden_units_str = " ".join(map(str, hidden_units))
    cmd = [
        "python", "train_cnf_flows_conditional.py",
        "--halo_id", halo_id,
        "--particle_pid", str(particle_pid),
        "--suite", "eden",
        "--hidden_units", hidden_units_str,
        "--epochs", str(epochs),
        "--learning_rate", str(learning_rate),
        "--batch_size", str(batch_size),
        "--validation_freq", "5",
        "--clip_outliers", "5.0",
        "--output_dir", output_dir
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        training_time = time.time() - start_time
        
        if result.returncode == 0:
            # Parse results
            stdout_lines = result.stdout.split('\n')
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
            
            return {
                'success': True,
                'training_time': training_time,
                'final_train_loss': final_train_loss,
                'final_val_loss': final_val_loss,
                'error': None
            }
        else:
            return {
                'success': False,
                'training_time': training_time,
                'final_train_loss': float('inf'),
                'final_val_loss': float('inf'),
                'error': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'training_time': 1800.0,
            'final_train_loss': float('inf'),
            'final_val_loss': float('inf'),
            'error': "Training timed out after 30 minutes"
        }
    except Exception as e:
        return {
            'success': False,
            'training_time': time.time() - start_time,
            'final_train_loss': float('inf'),
            'final_val_loss': float('inf'),
            'error': str(e)
        }

def main():
    halo_id = "Halo268"
    output_dir = "./quick_test_results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Test configurations: (hidden_units, epochs, learning_rate, batch_size)
    test_configs = [
        ([16, 16], 50, 1e-3, 256),    # Fast, small model
        ([32, 32], 100, 1e-4, 512),   # Standard
        ([64, 64], 100, 1e-4, 512),   # Larger model
        ([32, 32], 200, 5e-5, 1024),  # Thorough training
        ([64, 64], 200, 5e-5, 1024),  # Large model, thorough training
    ]
    
    # Test particle IDs
    particle_pids = [20, 21, 22, 23, 24]  # Test 5 different PIDs
    
    print("🚀 Quick Parameter Test for Conditional CNF Flows")
    print("=" * 60)
    print(f"Halo ID: {halo_id}")
    print(f"Testing {len(test_configs)} configurations on {len(particle_pids)} particle IDs")
    print(f"Total tests: {len(test_configs) * len(particle_pids)}")
    print()
    
    results = []
    test_count = 0
    total_tests = len(test_configs) * len(particle_pids)
    
    for pid in particle_pids:
        for hidden_units, epochs, lr, batch_size in test_configs:
            test_count += 1
            print(f"\nProgress: {test_count}/{total_tests}")
            
            result = run_test(halo_id, pid, hidden_units, epochs, lr, batch_size, output_dir)
            
            results.append({
                'particle_pid': pid,
                'hidden_units': str(hidden_units),
                'epochs': epochs,
                'learning_rate': lr,
                'batch_size': batch_size,
                'success': result['success'],
                'training_time': result['training_time'],
                'final_train_loss': result['final_train_loss'],
                'final_val_loss': result['final_val_loss'],
                'error': result['error']
            })
            
            # Save intermediate results
            if test_count % 5 == 0:
                df = pd.DataFrame(results)
                df.to_csv(f"{output_dir}/intermediate_results.csv", index=False)
    
    # Save final results
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/quick_test_results.csv", index=False)
    
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
        
        # Find best configurations
        best_config = successful_df.loc[successful_df['final_val_loss'].idxmin()]
        fastest_config = successful_df.loc[successful_df['training_time'].idxmin()]
        
        print(f"\n🏆 BEST CONFIGURATION (lowest validation loss):")
        print(f"  PID: {best_config['particle_pid']}")
        print(f"  Hidden units: {best_config['hidden_units']}")
        print(f"  Epochs: {best_config['epochs']}")
        print(f"  Learning rate: {best_config['learning_rate']}")
        print(f"  Batch size: {best_config['batch_size']}")
        print(f"  Training time: {best_config['training_time']:.1f}s")
        print(f"  Validation loss: {best_config['final_val_loss']:.3f}")
        
        print(f"\n⚡ FASTEST CONFIGURATION:")
        print(f"  PID: {fastest_config['particle_pid']}")
        print(f"  Hidden units: {fastest_config['hidden_units']}")
        print(f"  Epochs: {fastest_config['epochs']}")
        print(f"  Learning rate: {fastest_config['learning_rate']}")
        print(f"  Batch size: {fastest_config['batch_size']}")
        print(f"  Training time: {fastest_config['training_time']:.1f}s")
        print(f"  Validation loss: {fastest_config['final_val_loss']:.3f}")
    
    print(f"\n✅ Results saved to: {output_dir}/quick_test_results.csv")

if __name__ == "__main__":
    main()
