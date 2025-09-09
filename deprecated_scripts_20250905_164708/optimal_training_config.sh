#!/bin/bash

# 🎯 OPTIMAL TRAINING CONFIGURATION
# Fixes overfitting and spline artifacts in normalizing flows
# Based on analysis of KDE vs Flow comparison plots

echo "🎯 OPTIMAL NORMALIZING FLOW TRAINING CONFIGURATION"
echo "=================================================="
echo ""

# 🔧 OVERFITTING SOLUTION: Key Parameters
echo "🔧 ANTI-OVERFITTING HYPERPARAMETERS:"
echo "====================================="

echo "1️⃣ EPOCHS: 25-40 (was 100)"
echo "   • Reason: 100 epochs = severe overfitting"
echo "   • Solution: Early stopping around 25-40 epochs"
echo ""

echo "2️⃣ LEARNING RATE: 5e-4 (was 1e-3)" 
echo "   • Reason: Lower LR = smoother convergence"
echo "   • Solution: Halve the learning rate for stability"
echo ""

echo "3️⃣ LAYERS: 3 (was 4)"
echo "   • Reason: Fewer layers = less overfitting capacity"
echo "   • Solution: Reduce model complexity"
echo ""

echo "4️⃣ HIDDEN UNITS: 256-384 (was 512)"
echo "   • Reason: Smaller networks generalize better"
echo "   • Solution: Reduce parameter count"
echo ""

echo "5️⃣ REGULARIZATION: Add L2 weight decay"
echo "   • Reason: Explicit regularization prevents overfitting"
echo "   • Solution: Add weight decay to optimizer"
echo ""

echo "6️⃣ EARLY STOPPING: Stricter validation monitoring"
echo "   • Reason: Stop before overfitting begins"
echo "   • Solution: Monitor validation loss closely"
echo ""

# Recommended configurations for different use cases
echo "📋 RECOMMENDED CONFIGURATIONS:"
echo "==============================="

echo ""
echo "🚀 QUICK TEST (2-5 min per particle):"
echo "  --epochs 15"
echo "  --learning_rate 5e-4" 
echo "  --n_layers 2"
echo "  --hidden_units 256"
echo "  --batch_size 1024"
echo ""

echo "⚖️ BALANCED (5-10 min per particle):"
echo "  --epochs 25"
echo "  --learning_rate 5e-4"
echo "  --n_layers 3" 
echo "  --hidden_units 320"
echo "  --batch_size 512"
echo ""

echo "🎯 HIGH QUALITY (10-15 min per particle):"
echo "  --epochs 35"
echo "  --learning_rate 3e-4"
echo "  --n_layers 3"
echo "  --hidden_units 384"
echo "  --batch_size 512"
echo ""

echo "🎨 EXAMPLE UPDATED COMMAND:"
echo "==========================="
echo "python train_tfp_flows.py \\"
echo "    --data_path \"\$H5_FILE\" \\"
echo "    --particle_pid \"\$PID\" \\"
echo "    --output_dir \"\$MODEL_DIR\" \\"
echo "    --epochs 25 \\"
echo "    --learning_rate 5e-4 \\"
echo "    --n_layers 3 \\"
echo "    --hidden_units 320 \\"
echo "    --batch_size 512 \\"
echo "    --generate-samples \\"
echo "    --use_kroupa_imf"
echo ""

echo "💡 VALIDATION STRATEGY:"
echo "======================="
echo "• Monitor validation loss every 5 epochs"
echo "• Stop if validation loss increases for 3 consecutive checks"
echo "• Compare flow samples to KDE visually every 10 particles"
echo ""

echo "🎉 EXPECTED IMPROVEMENTS:"
echo "========================"
echo "✅ Smoother, more natural distributions"
echo "✅ Better match to KDE reference"
echo "✅ Elimination of sharp spline artifacts"
echo "✅ Faster training (less epochs needed)"
echo "✅ More robust generalization"
