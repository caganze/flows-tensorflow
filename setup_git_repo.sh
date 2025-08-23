#!/bin/bash
# Git repository setup script for TensorFlow Probability flows pipeline

echo "🚀 Setting up Git repository for TensorFlow Probability flows pipeline..."

# Initialize git repo
git init

# Create a comprehensive .gitignore file
cat > .gitignore << 'EOF'
# Logs and outputs
logs/
*.log
*.out
*.err
auto_submit.log

# Data and model outputs
/oak/
*.h5
*.npz
*_output/
*_test_output/

# Python cache
__pycache__/
*.pyc
*.pyo

# Temporary files
*.tmp
slurm-*
nohup.out

# Editor files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db
EOF

# Add all source files
git add .

# Create initial commit
git commit -m "Initial commit: TensorFlow Probability flows training pipeline

- Complete TFP normalizing flows implementation
- SLURM array job deployment scripts  
- Intelligent auto-submission system with beautiful logging
- Kroupa IMF integration and optimized I/O
- Comprehensive testing and validation suite
- Ready for production deployment on 478 particles"

# Create GitHub repo and push (replace with your actual repo name)
echo "📚 Creating GitHub repository..."
gh repo create caganze/flows-tensorflow --private --description "TensorFlow Probability normalizing flows for astrophysical data"

# Add remote and push
git remote add origin git@github.com:caganze/flows-tensorflow.git
git branch -M main
git push -u origin main

echo "✅ Repository successfully created and pushed to GitHub!"
echo "📖 Repository URL: https://github.com/caganze/flows-tensorflow"
