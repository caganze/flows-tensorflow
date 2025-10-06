#!/usr/bin/env python3
"""
Test script for KDE-informed coupling flows training
"""

import subprocess
import sys
import os

def test_kde_coupling_flows():
    """Test KDE-informed coupling flows training"""
    
    print("🧪 Testing KDE-informed coupling flows training...")
    
    # Test command with KDE loss enabled
    cmd = [
        "python", "train_coupling_flows_conditional.py",
        "--halo_id", "Halo939",
        "--particle_pid", "20", 
        "--suite", "eden",
        "--n_layers", "2",  # Smaller for testing
        "--hidden_units", "32",
        "--epochs", "3",  # Few epochs for testing
        "--learning_rate", "1e-3",
        "--n_mass_bins", "4",  # Fewer bins for testing
        "--embedding_dim", "2",
        "--use_kde_loss",  # Enable KDE loss
        "--lambda_kde", "0.1",
        "--output_dir", "test_kde_output"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ KDE-informed training completed successfully!")
            print("Output:")
            print(result.stdout)
        else:
            print("❌ KDE-informed training failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False
    
    # Test standard training for comparison
    print("\n🧪 Testing standard training (no KDE)...")
    
    cmd_standard = [
        "python", "train_coupling_flows_conditional.py",
        "--halo_id", "Halo939",
        "--particle_pid", "20",
        "--suite", "eden", 
        "--n_layers", "2",
        "--hidden_units", "32",
        "--epochs", "3",
        "--learning_rate", "1e-3",
        "--n_mass_bins", "4",
        "--embedding_dim", "2",
        "--output_dir", "test_standard_output"
    ]
    
    print(f"Running command: {' '.join(cmd_standard)}")
    
    try:
        result = subprocess.run(cmd_standard, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Standard training completed successfully!")
            print("Output:")
            print(result.stdout)
        else:
            print("❌ Standard training failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False
    
    print("\n🎉 Both KDE-informed and standard training completed successfully!")
    print("You can now compare the results in:")
    print("  - test_kde_output/ (with KDE regularization)")
    print("  - test_standard_output/ (standard training)")
    
    return True

if __name__ == "__main__":
    success = test_kde_coupling_flows()
    sys.exit(0 if success else 1)







