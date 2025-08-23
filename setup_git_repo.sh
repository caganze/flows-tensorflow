#!/bin/bash
# Git repository setup script for TensorFlow Probability flows pipeline

echo "🚀 Setting up Git repository for TensorFlow Probability flows pipeline..."

# Initialize git repo
git init

# Create a comprehensive .gitignore file
cat > .gitignore << 'GITIGNORE'
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
GITIGNORE

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

echo "📚 Local Git repository created successfully!"
echo ""
echo "🔗 Next steps:"
echo "1. Go to https://github.com/new"
echo "2. Create a private repository named 'flows-tensorflow'"
echo "3. Then run these commands:"
echo ""
echo "git remote add origin git@github.com:caganze/flows-tensorflow.git"
echo "git branch -M main"
echo "git push -u origin main"
echo ""
echo "✅ All your code is committed and ready to push!"
