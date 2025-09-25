#!/bin/bash
#SBATCH --job-name=quick_param_test
#SBATCH --partition=owners
#SBATCH --qos=owners
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=quick_param_test_%j.out
#SBATCH --error=quick_param_test_%j.err

# Load modules and set environment (fallback if modules unavailable)
if command -v module >/dev/null 2>&1; then
    module load python/3.11 || true
    module load cuda/12.2.0 || true
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_FLAGS='--xla_gpu_cuda_data_dir=/share/software/user/open/cuda/12.2.0'

# Change to working directory
cd /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow

# Activate conda environment
if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi
conda activate bosque 2>/dev/null || source activate bosque 2>/dev/null || true

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="quick_param_test_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "🚀 Quick Parameter Test for Conditional CNF Flows"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Output directory: $OUTPUT_DIR"
echo "Start time: $(date)"
echo ""

# Step 1: Analyze particle sizes (quick scan)
echo "📊 Step 1: Quick particle size analysis..."
python analyze_particle_sizes.py --halo_id Halo268 --suite eden --max_pid 30 > $OUTPUT_DIR/particle_analysis.log 2>&1

echo ""
echo "⏳ Waiting 5 seconds before starting parameter tests..."
sleep 5

# Step 2: Run quick parameter tests
echo "🧪 Step 2: Running quick parameter tests..."
python quick_parameter_test.py > $OUTPUT_DIR/parameter_tests.log 2>&1

# Step 3: Generate summary
echo "📈 Step 3: Generating summary..."
python -c "
import pandas as pd
import json
import os

output_dir = '$OUTPUT_DIR'

try:
    # Load particle analysis
    particle_df = pd.read_csv('particle_size_analysis_Halo268.csv')
    
    # Load parameter test results
    param_df = pd.read_csv('quick_test_results/quick_test_results.csv')
    
    print(f'📊 QUICK PARAMETER TEST SUMMARY')
    print(f'=' * 50)
    print(f'Particle PIDs analyzed: {len(particle_df)}')
    print(f'Parameter tests run: {len(param_df)}')
    
    successful = param_df[param_df['success'] == True]
    print(f'Successful tests: {len(successful)}/{len(param_df)}')
    print(f'Success rate: {len(successful)/len(param_df)*100:.1f}%')
    
    if len(successful) > 0:
        print(f'Average training time: {successful[\"training_time\"].mean():.1f} seconds')
        print(f'Average validation loss: {successful[\"final_val_loss\"].mean():.3f}')
        print(f'Best validation loss: {successful[\"final_val_loss\"].min():.3f}')
        
        # Best configuration
        best = successful.loc[successful['final_val_loss'].idxmin()]
        print(f'\\n🏆 BEST CONFIGURATION:')
        print(f'  PID: {best[\"particle_pid\"]}')
        print(f'  Hidden units: {best[\"hidden_units\"]}')
        print(f'  Epochs: {best[\"epochs\"]}')
        print(f'  Learning rate: {best[\"learning_rate\"]}')
        print(f'  Batch size: {best[\"batch_size\"]}')
        print(f'  Training time: {best[\"training_time\"]:.1f}s')
        print(f'  Validation loss: {best[\"final_val_loss\"]:.3f}')
        
        # Fastest configuration
        fastest = successful.loc[successful['training_time'].idxmin()]
        print(f'\\n⚡ FASTEST CONFIGURATION:')
        print(f'  PID: {fastest[\"particle_pid\"]}')
        print(f'  Hidden units: {fastest[\"hidden_units\"]}')
        print(f'  Epochs: {fastest[\"epochs\"]}')
        print(f'  Learning rate: {fastest[\"learning_rate\"]}')
        print(f'  Batch size: {fastest[\"batch_size\"]}')
        print(f'  Training time: {fastest[\"training_time\"]:.1f}s')
        print(f'  Validation loss: {fastest[\"final_val_loss\"]:.3f}')
    
    # Save summary
    summary = {
        'particle_pids_analyzed': int(len(particle_df)),
        'parameter_tests_run': int(len(param_df)),
        'successful_tests': int(len(successful)),
        'success_rate': float(len(successful)/len(param_df)*100) if len(param_df) > 0 else 0,
        'average_training_time': float(successful['training_time'].mean()) if len(successful) > 0 else 0,
        'average_validation_loss': float(successful['final_val_loss'].mean()) if len(successful) > 0 else 0,
        'best_validation_loss': float(successful['final_val_loss'].min()) if len(successful) > 0 else float('inf')
    }
    
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f'\\n✅ Summary saved to {output_dir}/summary.json')
    
except Exception as e:
    print(f'❌ Error generating summary: {e}')
    import traceback
    traceback.print_exc()
" > $OUTPUT_DIR/summary_report.log 2>&1

# Copy results to accessible location
echo "📁 Copying results..."
cp -r $OUTPUT_DIR /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/quick_param_test_results/
cp particle_size_analysis_Halo268.csv /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/quick_param_test_results/
cp -r quick_test_results /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/quick_param_test_results/

echo ""
echo "✅ Quick parameter test completed!"
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds"
echo ""
echo "📁 Results saved to:"
echo "  - $OUTPUT_DIR/"
echo "  - /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/quick_param_test_results/"
echo ""
echo "📄 Key files:"
echo "  - particle_size_analysis_Halo268.csv (particle size analysis)"
echo "  - quick_test_results.csv (parameter test results)"
echo "  - summary.json (executive summary)"
echo ""
echo "💡 Next steps:"
echo "  1. Review the summary.json for optimal parameters"
echo "  2. Use the particle size categories to group similar training runs"
echo "  3. Apply the best configurations to your production training"
