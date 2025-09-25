#!/bin/bash

# Parameter analysis and testing script for conditional CNF flows

HALO_ID="Halo268"
SUITE="eden"

echo "🚀 Starting Parameter Analysis for Conditional CNF Flows"
echo "=========================================================="
echo "Halo ID: $HALO_ID"
echo "Suite: $SUITE"
echo ""

# Step 1: Analyze particle sizes
echo "📊 Step 1: Analyzing particle size distributions..."
python analyze_particle_sizes.py --halo_id $HALO_ID --suite $SUITE --max_pid 30

echo ""
echo "⏳ Waiting 5 seconds before starting parameter tests..."
sleep 5

# Step 2: Run quick parameter tests
echo "🧪 Step 2: Running quick parameter tests..."
python quick_parameter_test.py

echo ""
echo "✅ Analysis complete!"
echo "📁 Check the following files:"
echo "  - particle_size_analysis_${HALO_ID}.csv (particle size analysis)"
echo "  - quick_test_results/quick_test_results.csv (parameter test results)"
echo ""
echo "💡 Use the results to:"
echo "  1. Identify optimal parameters for different particle sizes"
echo "  2. Group similar PIDs for batch training"
echo "  3. Estimate training times for different configurations"
