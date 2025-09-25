#!/bin/bash
#SBATCH --job-name=param_analysis
#SBATCH --partition=owners
#SBATCH --qos=owners
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=parameter_analysis_%j.out
#SBATCH --error=parameter_analysis_%j.err

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
OUTPUT_DIR="parameter_analysis_results_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "🚀 Starting Parameter Analysis for Conditional CNF Flows"
echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Output directory: $OUTPUT_DIR"
echo "Start time: $(date)"
echo ""

# Step 1: Analyze particle sizes
echo "📊 Step 1: Analyzing particle size distributions..."
python analyze_particle_sizes.py --halo_id Halo268 --suite eden --max_pid 50 > $OUTPUT_DIR/particle_analysis.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Particle size analysis completed successfully"
else
    echo "❌ Particle size analysis failed"
fi

echo ""
echo "⏳ Waiting 10 seconds before starting parameter tests..."
sleep 10

# Step 2: Run comprehensive parameter tests
echo "🧪 Step 2: Running comprehensive parameter tests..."
python test_parameter_sweep.py \
    --halo_id Halo268 \
    --suite eden \
    --output_dir $OUTPUT_DIR \
    --max_tests 100 \
    > $OUTPUT_DIR/parameter_tests.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Parameter tests completed successfully"
else
    echo "❌ Parameter tests failed"
fi

# Step 3: Generate summary report
echo "📈 Step 3: Generating summary report..."
python -c "
import pandas as pd
import json
import os

output_dir = '$OUTPUT_DIR'

# Load results
try:
    df = pd.read_csv(f'{output_dir}/parameter_sweep_results.csv')
    
    # Generate summary
    successful = df[df['success'] == True]
    total = len(df)
    
    print(f'📊 PARAMETER ANALYSIS SUMMARY')
    print(f'=' * 50)
    print(f'Total tests run: {total}')
    print(f'Successful tests: {len(successful)}')
    print(f'Success rate: {len(successful)/total*100:.1f}%')
    
    if len(successful) > 0:
        print(f'Average training time: {successful[\"training_time\"].mean():.1f} seconds')
        print(f'Average validation loss: {successful[\"final_val_loss\"].mean():.3f}')
        print(f'Best validation loss: {successful[\"final_val_loss\"].min():.3f}')
        
        # Best configurations by particle size
        print(f'\\n🏆 BEST CONFIGURATIONS BY PARTICLE SIZE:')
        for size in successful['particle_size_category'].unique():
            if pd.notna(size):
                size_data = successful[successful['particle_size_category'] == size]
                best = size_data.loc[size_data['final_val_loss'].idxmin()]
                print(f'{size:12s}: PID {best[\"particle_pid\"]}, {best[\"hidden_units\"]}, {best[\"epochs\"]} epochs, {best[\"training_time\"]:.1f}s')
    
    # Save summary
    summary = {
        'total_tests': int(total),
        'successful_tests': int(len(successful)),
        'success_rate': float(len(successful)/total*100) if total > 0 else 0,
        'average_training_time': float(successful['training_time'].mean()) if len(successful) > 0 else 0,
        'average_validation_loss': float(successful['final_val_loss'].mean()) if len(successful) > 0 else 0,
        'best_validation_loss': float(successful['final_val_loss'].min()) if len(successful) > 0 else float('inf')
    }
    
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f'\\n✅ Summary saved to {output_dir}/summary.json')
    
except Exception as e:
    print(f'❌ Error generating summary: {e}')
" > $OUTPUT_DIR/summary_report.log 2>&1

# Step 4: Copy results to accessible location
echo "📁 Step 4: Organizing results..."
cp -r $OUTPUT_DIR /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/parameter_analysis_results/

echo ""
echo "✅ Parameter analysis completed!"
echo "End time: $(date)"
echo "Total runtime: $SECONDS seconds"
echo ""
echo "📁 Results saved to:"
echo "  - $OUTPUT_DIR/"
echo "  - /oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/parameter_analysis_results/"
echo ""
echo "📄 Key files:"
echo "  - parameter_sweep_results.csv (detailed results)"
echo "  - recommendations.json (parameter recommendations)"
echo "  - summary.json (executive summary)"
echo "  - particle_analysis.log (particle size analysis log)"
echo "  - parameter_tests.log (parameter tests log)"
echo ""
echo "💡 Next steps:"
echo "  1. Review the recommendations.json for optimal parameters"
echo "  2. Use the particle size categories to group similar training runs"
echo "  3. Apply the best configurations to your production training"
