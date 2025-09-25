# KDE-Informed Flow Training Implementations

This directory contains four implementations of KDE-informed flow training, combining Kernel Density Estimation (KDE) with different types of normalizing flows through regularization in the loss function.

## Overview

All implementations use the approach suggested by Google Gemini:
- **Loss Function**: `L = L_NLL + λ * L_KDE`
- **KDE Loss**: `L_KDE = MSE(log P_Flow, log P_KDE)`
- **Goal**: Use KDE as a "teacher" to guide flow training

## The Four Implementations

### 1. **Unconditional MAF** 
**File**: `kde_informed_maf_unconditional_training.py`

**Architecture**: 
- Learns `p(x,y,z,vx,vy,vz,m)` - joint distribution of phase space and mass
- Uses Masked Autoregressive Flow (MAF) without conditioning
- Single KDE model for the entire dataset

**Usage**:
```bash
python kde_informed_maf_unconditional_training.py --toy_dataset --epochs 50 --lambda_kde 0.1
```

**Key Parameters**:
- `--lambda_kde`: KDE regularization weight (0.01-0.5)
- `--n_layers`: Number of MAF layers (3-5)
- `--hidden_units`: Hidden units per layer (32-128)

---

### 2. **Conditional MAF**
**File**: `kde_informed_maf_conditional_training.py`

**Architecture**:
- Learns `p(x,y,z,vx,vy,vz|m)` - phase space conditioned on mass
- Uses Masked Autoregressive Flow (MAF) with mass conditioning
- KDE models created for different mass bins

**Usage**:
```bash
python kde_informed_maf_conditional_training.py --toy_dataset --epochs 50 --lambda_kde 0.1 --mass_bins 8
```

**Key Parameters**:
- `--lambda_kde`: KDE regularization weight (0.01-0.5)
- `--mass_bins`: Number of mass bins for KDE models (5-15)
- `--n_layers`: Number of MAF layers (3-5)
- `--hidden_units`: Hidden units per layer (32-128)

---

### 3. **Unconditional Continuous Flow**
**File**: `kde_informed_continuous_flow_unconditional_training.py`

**Architecture**:
- Learns `p(x,y,z,vx,vy,vz,m)` - joint distribution of phase space and mass
- Uses Neural ODE-based continuous normalizing flow without conditioning
- Single KDE model for the entire dataset

**Usage**:
```bash
python kde_informed_continuous_flow_unconditional_training.py --toy_dataset --epochs 50 --lambda_kde 0.1
```

**Key Parameters**:
- `--lambda_kde`: KDE regularization weight (0.01-0.5)
- `--hidden_units`: Hidden units per layer (e.g., `--hidden_units 64 64`)
- `--integration_time`: ODE integration time (0.5-2.0)
- `--num_integration_steps`: Number of ODE solver steps (5-20)

---

### 4. **Conditional Continuous Flow**
**File**: `kde_informed_continuous_flow_conditional_training.py`

**Architecture**:
- Learns `p(x,y,z,vx,vy,vz|m)` - phase space conditioned on mass
- Uses Neural ODE-based continuous normalizing flow with mass conditioning
- KDE models created for different mass bins

**Usage**:
```bash
python kde_informed_continuous_flow_conditional_training.py --toy_dataset --epochs 50 --lambda_kde 0.1 --mass_bins 8
```

**Key Parameters**:
- `--lambda_kde`: KDE regularization weight (0.01-0.5)
- `--mass_bins`: Number of mass bins for KDE models (5-15)
- `--hidden_units`: Hidden units per layer (e.g., `--hidden_units 64 64`)
- `--integration_time`: ODE integration time (0.5-2.0)
- `--num_integration_steps`: Number of ODE solver steps (5-20)

---

## Key Differences

| Implementation | Architecture | Conditioning | KDE Models | Use Case |
|----------------|--------------|--------------|------------|----------|
| **Unconditional MAF** | MAF | None | Single | Learn joint distribution |
| **Conditional MAF** | MAF | Mass | Multiple (mass bins) | Learn conditional distribution |
| **Unconditional Continuous** | Neural ODE | None | Single | Learn joint distribution (continuous) |
| **Conditional Continuous** | Neural ODE | Mass | Multiple (mass bins) | Learn conditional distribution (continuous) |

## Hyperparameter Tuning Guide

### **λ (lambda_kde)** - Most Important Parameter
- **0.01-0.05**: Light KDE guidance, flow learns mostly from raw data
- **0.1-0.2**: Balanced approach (recommended starting point)
- **0.3-0.5**: Strong KDE guidance, flow closely follows KDE

### **Mass Bins** (Conditional only)
- **5-8 bins**: Coarse but stable conditioning
- **10-15 bins**: Fine-grained conditioning
- **>15 bins**: May be unstable with limited data

### **Architecture Parameters**
- **MAF layers**: 3-5 layers work well
- **Hidden units**: 32-128 per layer
- **Continuous flow**: 2-3 hidden layers with 64-128 units each

### **ODE Parameters** (Continuous flows only)
- **Integration time**: 0.5-2.0 (higher = more complex transformations)
- **Integration steps**: 5-20 (more steps = more accurate but slower)

## Output Files

Each implementation generates:
- **Model file**: `*.npz` with trained flow weights
- **Training plots**: `*_training_results.png` showing loss curves
- **Comparison plots**: `*_vs_kde_comparison.png` comparing flow vs KDE samples
- **Training history**: `*_training_history.npz` with loss data

## Example Commands

### Quick Test (2D, 10 epochs)
```bash
python kde_informed_maf_unconditional_training.py --epochs 10 --n_samples 1000
```

### Full Training (6D astrophysical, 50 epochs)
```bash
python kde_informed_maf_conditional_training.py --toy_dataset --epochs 50 --lambda_kde 0.1 --mass_bins 8
```

### Continuous Flow with Custom ODE Settings
```bash
python kde_informed_continuous_flow_conditional_training.py --toy_dataset --epochs 50 --lambda_kde 0.2 --integration_time 1.5 --num_integration_steps 15
```

## Tips for Success

1. **Start with λ=0.1** and adjust based on results
2. **Use validation data** to monitor overfitting
3. **For conditional flows**, ensure sufficient data per mass bin
4. **For continuous flows**, start with simple ODE settings
5. **Monitor KDE contribution** in final loss statistics
6. **Compare samples** visually to verify KDE guidance is working

## Troubleshooting

- **NaN losses**: Reduce λ or increase batch size
- **Poor KDE guidance**: Increase λ or check KDE model quality
- **Slow training**: Reduce integration steps or hidden units
- **Memory issues**: Reduce batch size or number of mass bins


