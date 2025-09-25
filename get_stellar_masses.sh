#!/bin/bash

# 🌟 Get Stellar Masses for Incomplete KDE Particles
# Calculates total stellar mass for each incomplete particle after KDE filtering

set -e

show_usage() {
    echo "Usage: $0 [INCOMPLETE_FILE]"
    echo ""
    echo "Arguments:"
    echo "  INCOMPLETE_FILE    Path to incomplete particle list (default: particle_list_kde_incomplete.txt)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default file"
    echo "  $0 my_incomplete_particles.txt       # Use custom file"
    echo "  $0 particle_list_flow_incomplete.txt # Use flow incomplete list"
    echo ""
}

# Parse arguments
INCOMPLETE_FILE="particle_list_kde_incomplete.txt"  # Default

if [[ $# -eq 1 ]]; then
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        show_usage
        exit 0
    else
        INCOMPLETE_FILE="$1"
    fi
elif [[ $# -gt 1 ]]; then
    echo "❌ Too many arguments"
    show_usage
    exit 1
fi

echo "🌟 STELLAR MASS CALCULATOR"
echo "========================="
echo "📁 Working directory: $(pwd)"
echo "📋 Incomplete file: $INCOMPLETE_FILE"
echo ""

# Check if incomplete particle list exists
if [[ ! -f "$INCOMPLETE_FILE" ]]; then
    echo "❌ File not found: $INCOMPLETE_FILE"
    if [[ "$INCOMPLETE_FILE" == "particle_list_kde_incomplete.txt" ]]; then
        echo "   Please run ./filter_completed_kde.sh first"
    else
        echo "   Please check the file path and try again"
    fi
    echo ""
    show_usage
    exit 1
fi

# Check particle count
PARTICLE_COUNT=$(wc -l < "$INCOMPLETE_FILE")
echo "📊 Found $PARTICLE_COUNT incomplete particles"
echo ""

# Load symlib environment if needed
if ! python3 -c "import symlib" 2>/dev/null; then
    echo "🔧 Loading symlib environment..."
    # Try common symlib environment setups on Sherlock
    if [[ -f "/home/users/caganze/.bashrc" ]]; then
        source /home/users/caganze/.bashrc
    fi
    if [[ -f "/home/users/caganze/symlib_env/bin/activate" ]]; then
        source /home/users/caganze/symlib_env/bin/activate
    fi
fi

# Check if symlib is available
if ! python3 -c "import symlib" 2>/dev/null; then
    echo "❌ Symlib not available"
    echo "   Please make sure symlib environment is properly loaded"
    exit 1
fi

echo "✅ Symlib environment ready"
echo ""

# Make script executable
chmod +x get_stellar_masses_incomplete.py

# Run the stellar mass calculation
echo "🚀 Starting stellar mass calculation..."
echo "   This may take a while for large particle lists..."
echo ""

python3 get_stellar_masses_incomplete.py "$INCOMPLETE_FILE"

echo ""
echo "✅ Stellar mass calculation complete!"
echo ""
echo "💡 Next steps:"
echo "   - Review the stellar mass rankings above"
echo "   - Use the highest mass particles for priority processing"
echo "   - Consider using ./submit_cpu_kde_smart.sh for remaining particles"
