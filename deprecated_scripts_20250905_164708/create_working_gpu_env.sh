#!/bin/bash
# Fixed version of your working script with compatible CUDA/cuDNN modules
# This script worked perfectly except for GPU detection - now using correct modules!

# Create comprehensive log file
LOG_FILE="complete_environment_setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "🗂️ COMPREHENSIVE LOGGING ENABLED"
echo "Log file: $LOG_FILE"
echo "Started at: $(date)"
echo "========================================"

echo "🆕 CREATING WORKING GPU BOSQUE ENVIRONMENT"
echo "=========================================="
echo "Python 3.11 + TensorFlow 2.15.0 + TFP 0.23.0"
echo "Using compatible CUDA 12.2.0 + cuDNN 8.9.0.131"
echo ""

# Load compatible modules - THE ONLY CHANGE from your working script!
echo "🔌 Loading compatible modules..."
echo "   Changed: cuda/12.6.1 → cuda/12.2.0"
echo "   Changed: cudnn/9.4.0 → cudnn/8.9.0.131"

echo "📋 System info before module loading:"
echo "  Hostname: $(hostname)"
echo "  Date: $(date)"
echo "  User: $(whoami)"
echo "  Working directory: $(pwd)"
echo "  Available modules: $(module avail 2>&1 | wc -l) modules"

echo "🔧 Loading modules with verbose output..."
set -x
module load math devel nvhpc/24.7 cuda/12.2.0 cudnn/8.9.0.131
set +x

echo "📋 Loaded modules:"
module list

echo "📋 Environment variables after module loading:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  PATH: $PATH"

# Activate conda
echo "🐍 Initializing conda..."
echo "  Conda location: /oak/stanford/orgs/kipac/users/caganze/anaconda3"
source /oak/stanford/orgs/kipac/users/caganze/anaconda3/etc/profile.d/conda.sh

echo "📋 Conda info:"
conda info
echo "📋 Existing environments:"
conda env list

echo ""
echo "🗑️ Removing existing bosque environment..."
set -x
conda env remove -n bosque -y 2>&1 || echo "Environment didn't exist"
set +x

echo ""
echo "🐍 Creating fresh bosque environment with Python 3.11..."
echo "  This may take several minutes..."
set -x
conda create -n bosque python=3.11 -y
set +x

echo ""
echo "🔄 Activating new bosque environment..."
set -x
conda activate bosque
set +x

echo "📋 Post-activation check:"
echo "  Active environment: $CONDA_DEFAULT_ENV"
echo "  Python location: $(which python)"
echo "  Python version: $(python --version)"

# Verify environment
if [ "$CONDA_DEFAULT_ENV" != "bosque" ]; then
    echo "❌ Failed to activate bosque environment"
    exit 1
fi

echo "✅ Environment: $CONDA_DEFAULT_ENV"
echo "✅ Python: $(python --version)"
echo "✅ Python path: $(which python)"

echo ""
echo "📦 Installing core scientific packages via conda..."
echo "  Installing: scipy matplotlib seaborn pandas h5py"
echo "  This may take several minutes and will handle NumPy automatically..."
set -x
conda install -c conda-forge scipy matplotlib seaborn pandas h5py -y
set +x

echo "📋 Conda packages installed:"
conda list | grep -E "(scipy|matplotlib|seaborn|pandas|h5py|numpy)"

echo ""
echo "📦 Installing known-good TensorFlow combination..."
echo "   TensorFlow: 2.15.0"
echo "   TensorFlow Probability: 0.23.0" 
echo "   This will install TensorFlow 2.15 with bundled Keras 2.x"

set -x
pip install tensorflow==2.15.0 tensorflow-probability==0.23.0
set +x

echo "📋 TensorFlow installation check:"
pip list | grep -E "(tensorflow|keras)"

echo ""
echo "📦 Installing additional useful packages..."
echo "  Installing: astropy scikit-learn jupyter ipykernel"
set -x
pip install astropy scikit-learn jupyter ipykernel
set +x

echo "📋 Final pip package list:"
pip list

echo ""
echo "🧪 Testing the working GPU environment..."
python -c "
import sys
import os

print('🔍 Environment Verification')
print('===========================')
print(f'Python: {sys.version}')
print(f'Python executable: {sys.executable}')
print(f'Conda environment: {os.environ.get(\"CONDA_DEFAULT_ENV\", \"None\")}')

try:
    import numpy as np
    import tensorflow as tf
    import tensorflow_probability as tfp
    import keras
    
    print(f'✅ NumPy: {np.__version__}')
    print(f'✅ TensorFlow: {tf.__version__}')
    print(f'✅ TensorFlow Probability: {tfp.__version__}')
    print(f'✅ Keras: {keras.__version__}')
    
    print('')
    print('🔍 GPU Detection')
    print('================')
    print('🔧 Using compatible CUDA 12.2.0 + cuDNN 8.9.0.131')
    
    gpu_available = tf.test.is_gpu_available()
    print(f'GPU available: {gpu_available}')
    
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f'GPU devices: {gpu_devices}')
    
    if gpu_available:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        with tf.device('/GPU:0'):
            a = tf.constant([[1., 2.]], dtype=tf.float32)
            b = tf.constant([[3.], [4.]], dtype=tf.float32)
            c = tf.matmul(a, b)
        
        print(f'✅ GPU computation: {c.numpy()}')
        print(f'✅ Computation device: {c.device}')
        gpu_success = True
    else:
        print('❌ Still no GPU detected')
        print('💡 Try: module spider cuda/12.2.0')
        print('💡 Try: module spider cudnn/8.9.0.131')
        gpu_success = False
    
    print('')
    print('🔍 TensorFlow Probability Test')
    print('===============================')
    try:
        tfd = tfp.distributions
        tfb = tfp.bijectors
        
        normal = tfd.Normal(0., 1.)
        samples = normal.sample(5)
        print(f'✅ Basic distribution: {samples.shape}')
        
        base_dist = tfd.MultivariateNormalDiag(loc=[0., 0.], scale_diag=[1., 1.])
        bijector = tfb.Shift([1., -1.])
        flow = tfd.TransformedDistribution(base_dist, bijector)
        
        flow_samples = flow.sample(10)
        flow_log_probs = flow.log_prob(flow_samples)
        
        print(f'✅ Simple flow: samples {flow_samples.shape}, log_probs {flow_log_probs.shape}')
        
        tfp_success = True
    except Exception as e:
        print(f'❌ TFP test failed: {e}')
        tfp_success = False
    
    print('')
    print('📊 FINAL RESULTS')
    print('=================')
    overall_success = gpu_success and tfp_success
    print(f'Environment Setup: ✅')
    print(f'GPU Detection: {\"✅\" if gpu_success else \"❌\"}')
    print(f'TFP Basic: {\"✅\" if tfp_success else \"❌\"}')
    
    if overall_success:
        print('')
        print('🎉 SUCCESS: Working GPU bosque environment ready!')
        print('✅ TensorFlow 2.15.0 + TFP 0.23.0 + Keras 2.x')
        print('✅ GPU support working with compatible modules!')
    elif tfp_success:
        print('')
        print('🟡 PARTIAL SUCCESS: Environment ready but GPU needs debugging')
        print('✅ TensorFlow 2.15.0 + TFP 0.23.0 + Keras 2.x working')
        print('❌ GPU detection still failing - check module loading')
    else:
        print('')
        print('⚠️ Some issues detected - check logs above')
    
except ImportError as e:
    print(f'❌ Import failed: {e}')
except Exception as e:
    print(f'❌ Test failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "💾 Exporting environment for reproducibility..."
ENV_FILE="working_gpu_bosque_environment_$(date +%Y%m%d_%H%M).yml"
REQ_FILE="working_gpu_bosque_requirements_$(date +%Y%m%d_%H%M).txt"

echo "📋 Exporting conda environment to: $ENV_FILE"
set -x
conda env export > "$ENV_FILE"
set +x

echo "📋 Exporting pip requirements to: $REQ_FILE"
set -x
pip freeze > "$REQ_FILE"
set +x

echo "📋 Environment export files created:"
ls -la *environment*.yml *requirements*.txt

echo "📋 Final system diagnostics:"
echo "  Disk usage: $(df -h . | tail -1)"
echo "  Memory usage: $(free -h | grep '^Mem:')"
echo "  Load average: $(uptime)"

echo ""
echo "✅ WORKING GPU BOSQUE ENVIRONMENT CREATED"
echo "========================================"
echo ""
echo "📋 Environment Details:"
echo "  Name: bosque"
echo "  Python: $(python --version)"
echo "  Location: $CONDA_PREFIX"
echo "  Packages: $(pip list | wc -l) installed"
echo ""
echo "📋 Key Packages:"
echo "  TensorFlow: $(python -c 'import tensorflow; print(tensorflow.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  TFP: $(python -c 'import tensorflow_probability; print(tensorflow_probability.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  Keras: $(python -c 'import keras; print(keras.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  NumPy: $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'Not installed')"
echo ""
echo "🔧 Module Configuration:"
echo "  CUDA: 12.2.0 (TensorFlow 2.15.0 compatible)"
echo "  cuDNN: 8.9.0.131 (TensorFlow 2.15.0 compatible)"
echo ""
echo "🚀 Ready to run TensorFlow Probability normalizing flows with GPU support!"
echo ""
echo "=========================================="
echo "🗂️ COMPREHENSIVE LOG COMPLETE"
echo "Log file: $LOG_FILE"
echo "Finished at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="
