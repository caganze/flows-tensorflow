# Flow Architectures in flows-tensorflow

This document provides a comprehensive overview of all normalizing flow architectures implemented in this codebase, along with their graphical representations.

## Overview

The codebase contains **6 main types** of normalizing flow architectures, each designed for specific use cases in astrophysical data modeling:

1. **TFP Normalizing Flow (MAF)** - Masked Autoregressive Flow
2. **CNF (Continuous Normalizing Flow)** - Neural ODE-based flows
3. **Conditional TFP Flow** - Mass-conditioned MAF flows
4. **Conditional CNF Flow** - Mass-conditioned Neural ODE flows
5. **Coupling Flow** - Efficient coupling layer-based flows
6. **KDE-Informed Flow** - Hybrid KDE + Flow approach

## Generated Diagrams

All flow architecture diagrams have been generated and saved in the `flow_architecture_diagrams/` directory:

### Enhanced Architecture Diagrams (with Layer Details & Data Shapes)
- `tfp_maf_architecture.png` - **TFP MAF architecture** with detailed MLP layers, data shapes, and transformation equations
- `cnf_architecture.png` - **CNF architecture** with Neural ODE details, ODE solver parameters, and integration steps
- `conditional_tfp_architecture.png` - **Conditional TFP flow** with mass embedding details and conditional MLP architecture
- `conditional_cnf_architecture.png` - Conditional CNF flow diagram
- `coupling_flow_architecture.png` - Coupling flow diagram
- `kde_informed_architecture.png` - KDE-informed flow diagram

### Overview & Comparison Diagrams
- `flow_architectures_overview.png` - Comprehensive overview of all flow types
- `flow_comparison.png` - Comparison table of flow characteristics

### Detailed Technical Diagrams
- `mlp_architecture_details.png` - **Detailed MLP architectures** showing layer-by-layer breakdown for MAF, Neural ODE, and Conditional networks
- `transform_equations.png` - **Comprehensive transform equations** showing mathematical formulas for all flow types

### Combined Diagram
- `all_flow_architectures.png` - All diagrams combined in one figure

## Flow Architecture Details

### 1. TFP Normalizing Flow (MAF)

**File**: `tfp_flows_gpu_solution.py`
**Class**: `TFPNormalizingFlow`

**Architecture**:
- **Base Distribution**: Gaussian (or Gaussian Mixture)
- **Bijectors**: Chain of Masked Autoregressive Flow (MAF) layers
- **Permutation**: Random permutations between MAF layers
- **Batch Normalization**: Optional invertible batch normalization

**Key Features**:
- Discrete transformations via MAF layers
- Autoregressive networks for shift and scale functions
- Good for high-dimensional data (6D phase space)
- Fast training and sampling

**Use Case**: General-purpose normalizing flows for 6D phase space data

### 2. CNF (Continuous Normalizing Flow)

**File**: `cnf_flows_solution.py`
**Classes**: `CNFNormalizingFlow`, `NeuralODE`

**Architecture**:
- **Base Distribution**: Gaussian
- **Neural ODE**: Neural network modeling continuous dynamics
- **ODE Solver**: BDF/RK45 solver for integration
- **Jacobian**: Computed via divergence (trace of Jacobian)

**Key Features**:
- Continuous transformations via Neural ODEs
- Time integration from t=0 to t=T
- More expressive than discrete flows
- Slower training due to ODE solving

**Use Case**: Continuous dynamics modeling, more expressive transformations

### 3. Conditional TFP Flow

**File**: `train_tfp_flows_conditional.py`
**Class**: `ConditionalTFPNormalizingFlow`

**Architecture**:
- **Mass Conditioning**: Mass bin embedding for conditioning
- **Conditional MAF**: MAF layers that take mass conditions as input
- **Embedding Layer**: Maps mass bins to embedding vectors
- **Base Distribution**: Gaussian

**Key Features**:
- Learns p(x|m) - phase space conditioned on mass
- Mass bin embedding for hierarchical conditioning
- Conditional autoregressive networks
- Fast training with conditioning

**Use Case**: Mass-conditioned learning for astrophysical data

### 4. Conditional CNF Flow

**File**: `cnf_flows_solution.py`
**Class**: `ConditionalCNFNormalizingFlow`

**Architecture**:
- **Conditional Neural ODE**: Neural ODE that takes mass conditions
- **Mass Conditioning**: Mass information fed to Neural ODE
- **ODE Solver**: BDF/RK45 solver with conditioning
- **Jacobian**: Computed via divergence with conditioning

**Key Features**:
- Continuous transformations conditioned on mass
- Learns p(x|m) via Neural ODE
- More expressive than conditional MAF
- Slower due to ODE solving

**Use Case**: Continuous mass-conditioned dynamics

### 5. Coupling Flow

**File**: `train_coupling_flows_conditional.py`
**Class**: `ConditionalCouplingFlow`

**Architecture**:
- **Coupling Layers**: Split-transform-merge operations
- **Split**: Divide input into two parts
- **Transform**: Apply neural network to one part conditioned on the other
- **Merge**: Combine transformed and unchanged parts

**Key Features**:
- Most efficient computation
- Split-transform-merge paradigm
- Fast training and sampling
- Less expressive than MAF

**Use Case**: Efficient alternative to MAF for 6D data

### 6. KDE-Informed Flow

**Files**: `kde_informed_maf_*.py`, `kde_informed_continuous_flow_*.py`
**Classes**: `KDEInformedMAFTrainer`, `KDEInformedCNFTrainer`

**Architecture**:
- **KDE Models**: Gaussian KDE for different mass bins
- **Flow Model**: MAF or CNF flow
- **Combined Loss**: L = L_NLL + λ·L_KDE
- **KDE Regularization**: MSE between flow and KDE log probabilities

**Key Features**:
- Hybrid approach combining KDE and flows
- KDE provides density estimates for regularization
- λ parameter controls KDE influence
- Improved density estimation

**Use Case**: Hybrid approach for better density modeling

## Performance Comparison

| Architecture | Complexity | Speed | Conditioning | Best Use Case |
|--------------|------------|-------|--------------|---------------|
| TFP MAF | Medium | Fast | No | General 6D flows |
| CNF | High | Slow | No | Continuous dynamics |
| Conditional TFP | Medium | Fast | Yes | Mass-conditioned |
| Conditional CNF | High | Slow | Yes | Continuous + conditional |
| Coupling Flow | Low | Very Fast | No | Efficient alternative |
| KDE-Informed | Medium | Medium | Yes | Hybrid approach |

## Implementation Files

### Core Flow Implementations
- `tfp_flows_gpu_solution.py` - TFP MAF implementation
- `cnf_flows_solution.py` - CNF and Conditional CNF implementations
- `train_tfp_flows_conditional.py` - Conditional TFP implementation
- `train_coupling_flows_conditional.py` - Coupling flow implementation

### KDE-Informed Implementations
- `kde_informed_maf_unconditional_training.py` - KDE-informed MAF
- `kde_informed_maf_conditional_training.py` - KDE-informed conditional MAF
- `kde_informed_continuous_flow_unconditional_training.py` - KDE-informed CNF
- `kde_informed_continuous_flow_conditional_training.py` - KDE-informed conditional CNF

### Training Scripts
- `train_tfp_flows.py` - TFP MAF training
- `train_cnf_flows.py` - CNF training
- `train_cnf_flows_conditional.py` - Conditional CNF training

### Sampling Scripts
- `sample_cnf_flow.py` - CNF sampling
- `sample_conditional_flow.py` - Conditional flow sampling
- `sample_conditional_cnf_flow.py` - Conditional CNF sampling

## Key Components

### Base Distributions
- **Gaussian**: Standard multivariate normal distribution
- **Gaussian Mixture**: Mixture of Gaussians for more complex latent space

### Bijectors
- **Masked Autoregressive Flow (MAF)**: Autoregressive transformations
- **Permutation**: Random permutations for better mixing
- **Batch Normalization**: Invertible normalization layers
- **Coupling Layers**: Split-transform-merge operations

### Neural Networks
- **Autoregressive Networks**: For MAF shift and scale functions
- **Neural ODE**: For continuous transformations
- **Conditional Networks**: Networks that take conditioning variables

### Training Components
- **Trainers**: Specialized training classes for each flow type
- **Loss Functions**: Negative log-likelihood with optional regularization
- **Optimizers**: Adam optimizer with gradient clipping
- **Early Stopping**: Validation-based early stopping

## Usage Recommendations

### For 6D Phase Space Data:
1. **Conditional TFP MAF** - Best balance of speed and conditioning
2. **TFP MAF** - If no conditioning needed
3. **Coupling Flow** - If speed is critical

### For Continuous Dynamics:
1. **CNF** - If no conditioning needed
2. **Conditional CNF** - If mass conditioning needed

### For Hybrid Approaches:
1. **KDE-Informed Flows** - When combining KDE and flow benefits

## Enhanced Visualization Features

### 🎯 **Detailed Architecture Diagrams**
- **Data Shapes**: All tensors show exact dimensions (batch_size, features)
- **Layer Details**: MLP architectures with input/output dimensions and activations
- **Transformation Equations**: Mathematical formulas for forward/inverse transforms
- **Component Breakdown**: Detailed view of each architectural component

### 🧠 **MLP Architecture Details**
- **MAF Networks**: Autoregressive networks with masking details
- **Neural ODE Networks**: Time-dependent networks with ODE solver parameters
- **Conditional Networks**: Mass embedding and conditional MLP architectures
- **Layer Specifications**: Input/output dimensions, activations, and connections

### 📐 **Mathematical Transform Equations**
- **MAF Transforms**: Complete autoregressive forward/inverse equations
- **CNF Transforms**: Neural ODE integration and divergence computation
- **Conditional Transforms**: Mass-conditioned autoregressive equations
- **Coupling Transforms**: Split-transform-merge mathematical formulas
- **Base Distributions**: Gaussian and GMM probability density functions
- **Log-determinants**: Complete Jacobian determinant calculations

### 📊 **Technical Specifications**
- **Data Flow**: Complete tensor shapes throughout the pipeline
- **Hyperparameters**: ODE solver tolerances, integration steps, embedding dimensions
- **Mathematical Details**: Log-determinant computation, Jacobian traces, divergence
- **Implementation Details**: Batch normalization, permutation layers, base distributions

### 🎨 **Visual Quality**
- **High-Resolution**: 300 DPI PNG files suitable for presentations/papers
- **Professional Layout**: Clean, organized diagrams with clear annotations
- **Color Coding**: Consistent color scheme for different component types
- **Comprehensive Coverage**: All flow types with detailed technical information

## Generation Script

The diagrams were generated using `generate_flow_architectures.py`, which:
1. Analyzes all Python files for flow architectures
2. Extracts architectural components and parameters
3. Generates individual diagrams for each flow type with detailed layer information
4. Creates comprehensive overview and comparison diagrams
5. Produces detailed MLP architecture breakdowns

To regenerate the diagrams:
```bash
python generate_flow_architectures.py
```

To view the diagrams:
```bash
python view_diagrams.py list    # List all diagrams
python view_diagrams.py view    # View in matplotlib window
python view_diagrams.py save    # Save combined diagram
```

This will create all diagrams in the `flow_architecture_diagrams/` directory.