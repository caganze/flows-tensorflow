#!/usr/bin/env python3
"""
Example script showing how to use KDE-informed coupling flows training
"""

import argparse
import sys

def main():
    """Example usage of KDE-informed coupling flows training"""
    
    parser = argparse.ArgumentParser(description="Example KDE-informed coupling flows training")
    parser.add_argument("--halo_id", type=str, default="Halo939", help="Halo ID to train on")
    parser.add_argument("--particle_pid", type=int, default=20, help="Particle type ID")
    parser.add_argument("--suite", type=str, default="eden", help="Simulation suite")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lambda_kde", type=float, default=0.1, help="KDE regularization weight")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--train_split", type=float, default=0.8, help="Training data fraction")
    
    args = parser.parse_args()
    
    print("🎯 KDE-Informed Coupling Flows Training Example")
    print("=" * 50)
    print(f"Halo: {args.halo_id}")
    print(f"Particle PID: {args.particle_pid}")
    print(f"Suite: {args.suite}")
    print(f"Epochs: {args.epochs}")
    print(f"KDE weight (λ): {args.lambda_kde}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Train/val split: {args.train_split:.1%}")
    print("=" * 50)
    
    # Example 1: Standard training (no KDE)
    print("\n📊 Example 1: Standard Training")
    print("Command:")
    cmd_standard = [
        "python", "train_coupling_flows_conditional.py",
        "--halo_id", args.halo_id,
        "--particle_pid", str(args.particle_pid),
        "--suite", args.suite,
        "--epochs", str(args.epochs),
        "--early_stopping_patience", str(args.patience),
        "--train_val_split", str(args.train_split),
        "--output_dir", f"standard_output_{args.halo_id.lower()}"
    ]
    print(" ".join(cmd_standard))
    
    # Example 2: KDE-informed training
    print("\n🎯 Example 2: KDE-Informed Training")
    print("Command:")
    cmd_kde = [
        "python", "train_coupling_flows_conditional.py",
        "--halo_id", args.halo_id,
        "--particle_pid", str(args.particle_pid),
        "--suite", args.suite,
        "--epochs", str(args.epochs),
        "--use_kde_loss",  # Enable KDE regularization
        "--lambda_kde", str(args.lambda_kde),
        "--early_stopping_patience", str(args.patience),
        "--train_val_split", str(args.train_split),
        "--output_dir", f"kde_output_{args.halo_id.lower()}"
    ]
    print(" ".join(cmd_kde))
    
    # Example 3: High KDE weight (strong regularization)
    print("\n🔒 Example 3: Strong KDE Regularization")
    print("Command:")
    cmd_strong_kde = [
        "python", "train_coupling_flows_conditional.py",
        "--halo_id", args.halo_id,
        "--particle_pid", str(args.particle_pid),
        "--suite", args.suite,
        "--epochs", str(args.epochs),
        "--use_kde_loss",
        "--lambda_kde", "0.5",  # Strong regularization
        "--early_stopping_patience", str(args.patience),
        "--train_val_split", str(args.train_split),
        "--output_dir", f"strong_kde_output_{args.halo_id.lower()}"
    ]
    print(" ".join(cmd_strong_kde))
    
    print("\n💡 Key Features:")
    print("  - Standard training: Only uses negative log-likelihood loss")
    print("  - KDE-informed: Combines NLL + λ × MSE(flow_log_prob, kde_log_prob)")
    print("  - Higher λ_kde: Stronger regularization toward KDE density estimates")
    print("  - KDE models: Created separately for each mass bin")
    print("  - Validation monitoring: Tracks both train and validation loss")
    print("  - Early stopping: Prevents overfitting with configurable patience")
    print("  - Train/val split: Configurable data split (default 80/20)")
    
    print("\n📈 Expected Benefits of KDE Loss:")
    print("  - Better density estimation in low-density regions")
    print("  - More stable training with regularization")
    print("  - Improved sample quality matching observed data")
    print("  - Reduced mode collapse")
    
    print("\n🔧 Tuning Tips:")
    print("  - Start with λ_kde = 0.1")
    print("  - Increase λ_kde if flow doesn't match data well")
    print("  - Decrease λ_kde if training becomes unstable")
    print("  - Monitor both NLL and KDE loss components")
    print("  - Watch for train/val loss divergence (overfitting)")
    print("  - Adjust patience based on convergence speed")
    print("  - Use smaller train/val split if you have limited data")
    
    return True

if __name__ == "__main__":
    main()
