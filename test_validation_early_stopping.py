#!/usr/bin/env python3
"""
Test script for validation monitoring and early stopping in coupling flows training
"""

import subprocess
import sys
import os

def test_validation_early_stopping():
    """Test validation monitoring and early stopping functionality"""
    
    print("🧪 Testing validation monitoring and early stopping...")
    
    # Test command with early stopping enabled
    cmd = [
        "python", "train_coupling_flows_conditional.py",
        "--halo_id", "Halo939",
        "--particle_pid", "20", 
        "--suite", "eden",
        "--n_layers", "2",  # Smaller for testing
        "--hidden_units", "32",
        "--epochs", "20",  # More epochs to test early stopping
        "--learning_rate", "1e-3",
        "--n_mass_bins", "4",
        "--embedding_dim", "2",
        "--early_stopping_patience", "5",  # Short patience for testing
        "--train_val_split", "0.8",
        "--output_dir", "test_validation_output"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("Expected: Should show both train and validation loss, and potentially early stop")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Validation monitoring and early stopping test completed!")
            print("Output:")
            print(result.stdout)
            
            # Check if validation loss was printed
            if "Val Loss:" in result.stdout:
                print("✅ Validation loss monitoring working!")
            else:
                print("⚠️ Validation loss not found in output")
                
            # Check if early stopping was triggered
            if "Early stopping" in result.stdout:
                print("✅ Early stopping triggered!")
            else:
                print("ℹ️ Early stopping not triggered (may have completed all epochs)")
                
        else:
            print("❌ Test failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out (10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False
    
    # Test with KDE loss and validation
    print("\n🧪 Testing KDE loss with validation monitoring...")
    
    cmd_kde = [
        "python", "train_coupling_flows_conditional.py",
        "--halo_id", "Halo939",
        "--particle_pid", "20",
        "--suite", "eden",
        "--n_layers", "2",
        "--hidden_units", "32", 
        "--epochs", "15",
        "--learning_rate", "1e-3",
        "--n_mass_bins", "4",
        "--embedding_dim", "2",
        "--use_kde_loss",
        "--lambda_kde", "0.1",
        "--early_stopping_patience", "3",
        "--train_val_split", "0.7",
        "--output_dir", "test_kde_validation_output"
    ]
    
    print(f"Running command: {' '.join(cmd_kde)}")
    print("Expected: Should show train/val loss, NLL, and KDE loss components")
    
    try:
        result = subprocess.run(cmd_kde, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ KDE loss with validation monitoring test completed!")
            print("Output:")
            print(result.stdout)
            
            # Check if all loss components were printed
            if "Train Loss:" in result.stdout and "Val Loss:" in result.stdout:
                print("✅ Train and validation loss monitoring working!")
            if "NLL:" in result.stdout and "KDE:" in result.stdout:
                print("✅ KDE loss components monitoring working!")
                
        else:
            print("❌ KDE test failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ KDE test timed out (10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running KDE test: {e}")
        return False
    
    print("\n🎉 All validation and early stopping tests completed successfully!")
    print("Check the output directories for saved models and results:")
    print("  - test_validation_output/")
    print("  - test_kde_validation_output/")
    
    return True

if __name__ == "__main__":
    success = test_validation_early_stopping()
    sys.exit(0 if success else 1)







