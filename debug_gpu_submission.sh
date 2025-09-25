#!/bin/bash

echo "🔍 GPU SUBMISSION DEBUG"
echo "======================="
echo ""

echo "Testing different GPU specification formats..."
echo ""

echo "1. Testing --gres=gpu:1 format:"
sbatch --test-only --partition=owners --time=1:00:00 --mem=16GB --gres=gpu:1 --job-name=test_gres /bin/echo "test" && echo "✅ --gres=gpu:1 works" || echo "❌ --gres=gpu:1 failed"

echo ""
echo "2. Testing --gpus=1 format:"  
sbatch --test-only --partition=owners --time=1:00:00 --mem=16GB --gpus=1 --job-name=test_gpus /bin/echo "test" && echo "✅ --gpus=1 works" || echo "❌ --gpus=1 failed"

echo ""
echo "3. Testing conflict (both --gres and --gpus):"
sbatch --test-only --partition=owners --time=1:00:00 --mem=16GB --gres=gpu:1 --gpus=1 --job-name=test_conflict /bin/echo "test" && echo "⚠️ Both formats work together" || echo "❌ Conflict detected (expected)"

echo ""
echo "Recommendations will be provided based on results..."
