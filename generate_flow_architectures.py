#!/usr/bin/env python3
"""
Generate Graphical Representations of All Flow Architectures

This script analyzes all Python files in the codebase and generates graphical
representations of the different normalizing flow architectures implemented.

Flow Types Identified:
1. TFP Normalizing Flow (MAF-based)
2. CNF (Continuous Normalizing Flow) 
3. Conditional TFP Flow
4. Conditional CNF Flow
5. Coupling Flow
6. KDE-Informed Flows

The script creates both individual architecture diagrams and a comprehensive
overview diagram showing all flow types and their relationships.
"""

import os
import sys
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from pathlib import Path
import networkx as nx
from typing import Dict, List, Set, Tuple, Any
import re

# Set up matplotlib for better quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

class FlowArchitectureAnalyzer:
    """Analyzes Python files to extract flow architecture information"""
    
    def __init__(self, codebase_path: str):
        self.codebase_path = Path(codebase_path)
        self.flow_classes = {}
        self.flow_files = []
        self.architecture_info = {}
        
    def analyze_codebase(self):
        """Analyze all Python files for flow architectures"""
        print("🔍 Analyzing codebase for flow architectures...")
        
        # Find all Python files
        py_files = list(self.codebase_path.rglob("*.py"))
        print(f"📁 Found {len(py_files)} Python files")
        
        for py_file in py_files:
            if self._is_flow_file(py_file):
                self._analyze_file(py_file)
        
        print(f"✅ Found {len(self.flow_classes)} flow architectures")
        return self.architecture_info
    
    def _is_flow_file(self, file_path: Path) -> bool:
        """Check if file likely contains flow architectures"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Look for flow-related keywords
            flow_keywords = [
                'NormalizingFlow', 'CNF', 'MAF', 'CouplingFlow', 
                'tfp.bijectors', 'TransformedDistribution',
                'MaskedAutoregressiveFlow', 'NeuralODE'
            ]
            
            return any(keyword in content for keyword in flow_keywords)
        except:
            return False
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file for flow architectures"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if self._is_flow_class(node, content):
                        self._extract_flow_info(node, file_path, content)
                        
        except Exception as e:
            print(f"⚠️ Error analyzing {file_path}: {e}")
    
    def _is_flow_class(self, class_node: ast.ClassDef, content: str) -> bool:
        """Check if a class is a flow architecture"""
        class_name = class_node.name
        
        # Check class name patterns
        flow_patterns = [
            r'.*Flow.*', r'.*CNF.*', r'.*MAF.*', 
            r'.*Coupling.*', r'.*NeuralODE.*'
        ]
        
        for pattern in flow_patterns:
            if re.match(pattern, class_name, re.IGNORECASE):
                return True
        
        # Check for flow-related methods
        flow_methods = ['log_prob', 'sample', 'forward_transform', 'inverse_transform']
        methods = [n.name for n in class_node.body if isinstance(n, ast.FunctionDef)]
        
        return any(method in methods for method in flow_methods)
    
    def _extract_flow_info(self, class_node: ast.ClassDef, file_path: Path, content: str):
        """Extract architecture information from a flow class"""
        class_name = class_node.name
        file_name = file_path.name
        
        # Determine flow type
        flow_type = self._classify_flow_type(class_name, content)
        
        # Extract key components
        components = self._extract_components(class_node, content)
        
        # Extract parameters
        parameters = self._extract_parameters(class_node, content)
        
        self.flow_classes[class_name] = {
            'file': file_name,
            'type': flow_type,
            'components': components,
            'parameters': parameters,
            'class_node': class_node
        }
        
        print(f"📊 Found {flow_type}: {class_name} in {file_name}")
    
    def _classify_flow_type(self, class_name: str, content: str) -> str:
        """Classify the type of flow architecture"""
        class_name_lower = class_name.lower()
        
        if 'cnf' in class_name_lower or 'continuous' in class_name_lower:
            return 'CNF (Continuous Normalizing Flow)'
        elif 'conditional' in class_name_lower:
            if 'cnf' in class_name_lower:
                return 'Conditional CNF'
            else:
                return 'Conditional TFP Flow'
        elif 'coupling' in class_name_lower:
            return 'Coupling Flow'
        elif 'kde' in class_name_lower:
            return 'KDE-Informed Flow'
        elif 'neuralode' in class_name_lower:
            return 'Neural ODE'
        else:
            return 'TFP Normalizing Flow (MAF)'
    
    def _extract_components(self, class_node: ast.ClassDef, content: str) -> List[str]:
        """Extract key architectural components"""
        components = []
        
        # Look for specific patterns in the class
        if 'base_dist' in content:
            components.append('Base Distribution')
        if 'bijector' in content:
            components.append('Bijector Chain')
        if 'MAF' in content or 'MaskedAutoregressiveFlow' in content:
            components.append('Masked Autoregressive Flow')
        if 'NeuralODE' in content:
            components.append('Neural ODE')
        if 'embedding' in content:
            components.append('Conditional Embedding')
        if 'permutation' in content:
            components.append('Permutation Layer')
        if 'batch' in content.lower():
            components.append('Batch Normalization')
        
        return components
    
    def _extract_parameters(self, class_node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Extract key parameters from the class"""
        params = {}
        
        # Look for common parameters in __init__ method
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                for arg in node.args.args:
                    if arg.arg not in ['self']:
                        params[arg.arg] = 'Parameter'
        
        return params

class FlowArchitectureVisualizer:
    """Creates visual representations of flow architectures"""
    
    def __init__(self, analyzer: FlowArchitectureAnalyzer):
        self.analyzer = analyzer
        self.output_dir = Path("flow_architecture_diagrams")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_all_diagrams(self):
        """Generate all flow architecture diagrams"""
        print("🎨 Generating flow architecture diagrams...")
        
        # Generate individual architecture diagrams
        self._generate_individual_diagrams()
        
        # Generate comprehensive overview
        self._generate_overview_diagram()
        
        # Generate flow comparison diagram
        self._generate_comparison_diagram()
        
        print(f"✅ All diagrams saved to {self.output_dir}")
    
    def _generate_individual_diagrams(self):
        """Generate individual diagrams for each flow type"""
        
        # TFP Normalizing Flow (MAF)
        self._create_tfp_maf_diagram()
        
        # CNF Flow
        self._create_cnf_diagram()
        
        # Conditional TFP Flow
        self._create_conditional_tfp_diagram()
        
        # Conditional CNF Flow
        self._create_conditional_cnf_diagram()
        
        # Coupling Flow
        self._create_coupling_flow_diagram()
        
        # KDE-Informed Flow
        self._create_kde_informed_diagram()
        
        # MLP Architecture Details
        self._create_mlp_architecture_diagram()
        
        # Transform Equations Details
        self._create_transform_equations_diagram()
    
    def _create_tfp_maf_diagram(self):
        """Create diagram for TFP MAF architecture with detailed layers and data shapes"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Title
        ax.text(8, 11.5, 'TFP Normalizing Flow (MAF) - Detailed Architecture', 
                fontsize=18, fontweight='bold', ha='center')
        
        # Data flow with shapes
        y_positions = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        
        # Input data with shape
        self._draw_box(ax, 1, y_positions[0], 2, 0.8, 'Input Data\n(batch_size, 6)\n[x, y, z, vx, vy, vz]', 'lightblue')
        
        # MAF Layer 1 with MLP details
        self._draw_box(ax, 5, y_positions[1], 3, 1.2, 'MAF Layer 1\nAutoregressive Network\nInput: (batch_size, 6)\nHidden: [64, 64] → ReLU\nOutput: (batch_size, 12)\n[shift(6), log_scale(6)]', 'lightgreen')
        
        # Permutation with details
        self._draw_box(ax, 5, y_positions[2], 3, 0.8, 'Permutation Layer\nRandom permutation\nInput: (batch_size, 6)\nOutput: (batch_size, 6)', 'lightyellow')
        
        # MAF Layer 2 with MLP details
        self._draw_box(ax, 5, y_positions[3], 3, 1.2, 'MAF Layer 2\nAutoregressive Network\nInput: (batch_size, 6)\nHidden: [64, 64] → ReLU\nOutput: (batch_size, 12)\n[shift(6), log_scale(6)]', 'lightgreen')
        
        # Batch Norm with details
        self._draw_box(ax, 5, y_positions[4], 3, 0.8, 'Batch Normalization\n(Optional)\nInput: (batch_size, 6)\nOutput: (batch_size, 6)\nLearnable scale/shift', 'lightcoral')
        
        # MAF Layer N with MLP details
        self._draw_box(ax, 5, y_positions[5], 3, 1.2, 'MAF Layer N\nAutoregressive Network\nInput: (batch_size, 6)\nHidden: [64, 64] → ReLU\nOutput: (batch_size, 12)\n[shift(6), log_scale(6)]', 'lightgreen')
        
        # Base Distribution with details
        self._draw_box(ax, 10, y_positions[6], 2.5, 1.0, 'Base Distribution\nMultivariate Normal\nμ = 0, σ = 1\nShape: (batch_size, 6)\nlog_prob: (batch_size,)', 'lightpink')
        
        # Output with shape
        self._draw_box(ax, 1, y_positions[7], 2, 0.8, 'Latent Space\n(batch_size, 6)\n[z₁, z₂, ..., z₆]', 'lightblue')
        
        # Arrows
        arrows = [
            (3, y_positions[0], 5, y_positions[1]),
            (6.5, y_positions[1], 6.5, y_positions[2]),
            (6.5, y_positions[2], 6.5, y_positions[3]),
            (6.5, y_positions[3], 6.5, y_positions[4]),
            (6.5, y_positions[4], 6.5, y_positions[5]),
            (8, y_positions[5], 10, y_positions[6]),
            (10, y_positions[6], 3, y_positions[7])
        ]
        
        for x1, y1, x2, y2 in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Add MLP architecture details
        ax.text(13, 8, 'Autoregressive Network Architecture:', fontsize=12, fontweight='bold')
        ax.text(13, 7.5, '• Input Layer: 6 → 64', fontsize=10)
        ax.text(13, 7, '• Hidden Layer: 64 → 64 (ReLU)', fontsize=10)
        ax.text(13, 6.5, '• Output Layer: 64 → 12', fontsize=10)
        ax.text(13, 6, '• Split: 6 for shift, 6 for log_scale', fontsize=10)
        ax.text(13, 5.5, '• Masking: Ensures autoregressive property', fontsize=10)
        
        # Add detailed transformation equations
        ax.text(13, 4.5, 'MAF Transform Equations:', fontsize=12, fontweight='bold')
        ax.text(13, 4, 'Forward Transform:', fontsize=10, fontweight='bold')
        ax.text(13, 3.7, 'y₁ = x₁', fontsize=9)
        ax.text(13, 3.4, 'y₂ = x₂ ⊙ exp(s₂(x₁)) + t₂(x₁)', fontsize=9)
        ax.text(13, 3.1, 'y₃ = x₃ ⊙ exp(s₃(x₁,x₂)) + t₃(x₁,x₂)', fontsize=9)
        ax.text(13, 2.8, '... (autoregressive)', fontsize=9)
        ax.text(13, 2.5, 'Inverse Transform:', fontsize=10, fontweight='bold')
        ax.text(13, 2.2, 'x₁ = y₁', fontsize=9)
        ax.text(13, 1.9, 'x₂ = (y₂ - t₂(x₁)) ⊙ exp(-s₂(x₁))', fontsize=9)
        ax.text(13, 1.6, 'x₃ = (y₃ - t₃(x₁,x₂)) ⊙ exp(-s₃(x₁,x₂))', fontsize=9)
        ax.text(13, 1.3, 'Log-det: Σᵢ sᵢ(x₁,...,xᵢ₋₁)', fontsize=9)
        
        # Add details
        ax.text(8, 0.5, 'Forward: Data → Latent Space (log_prob computation)\nInverse: Latent Space → Data (sampling)', 
                fontsize=11, ha='center', style='italic')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tfp_maf_architecture.png', bbox_inches='tight')
        plt.close()
    
    def _create_cnf_diagram(self):
        """Create diagram for CNF architecture with detailed layers and data shapes"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Title
        ax.text(8, 11.5, 'Continuous Normalizing Flow (CNF) - Detailed Architecture', 
                fontsize=18, fontweight='bold', ha='center')
        
        # Data flow with shapes
        y_positions = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        
        # Input data with shape
        self._draw_box(ax, 1, y_positions[0], 2, 0.8, 'Input Data\n(batch_size, 6)\n[x, y, z, vx, vy, vz]', 'lightblue')
        
        # Neural ODE with MLP details
        self._draw_box(ax, 5, y_positions[1], 3.5, 1.4, 'Neural ODE\nf(x,t) = dx/dt\nInput: (batch_size, 6+1)\n[x, t] → [64, 64] → 6\nActivation: tanh', 'lightgreen')
        
        # ODE Solver with details
        self._draw_box(ax, 5, y_positions[2], 3.5, 0.8, 'ODE Solver\nBDF/RK45\nInput: f(x,t), x₀, t∈[0,T]\nOutput: x(t) trajectory\nSteps: 10-50', 'lightyellow')
        
        # Time Integration with details
        self._draw_box(ax, 5, y_positions[3], 3.5, 0.8, 'Time Integration\nt ∈ [0, T], T=1.0\nTime points: [0, 0.1, ..., 1.0]\nAugmented state: [x, log_det]', 'lightcoral')
        
        # Jacobian Computation with details
        self._draw_box(ax, 5, y_positions[4], 3.5, 0.8, 'Jacobian Trace\n∇·f(x,t) = Σᵢ ∂fᵢ/∂xᵢ\nComputed via autodiff\nLog-det: ∫₀ᵀ -∇·f(x,t) dt', 'lightpink')
        
        # Base Distribution with details
        self._draw_box(ax, 10, y_positions[5], 2.5, 1.0, 'Base Distribution\nMultivariate Normal\nμ = 0, σ = 1\nShape: (batch_size, 6)\nlog_prob: (batch_size,)', 'lightgray')
        
        # Output with shape
        self._draw_box(ax, 1, y_positions[6], 2, 0.8, 'Latent Space\n(batch_size, 6)\n[z₁, z₂, ..., z₆]', 'lightblue')
        
        # Arrows
        arrows = [
            (3, y_positions[0], 5, y_positions[1]),
            (6.75, y_positions[1], 6.75, y_positions[2]),
            (6.75, y_positions[2], 6.75, y_positions[3]),
            (6.75, y_positions[3], 6.75, y_positions[4]),
            (8.5, y_positions[4], 10, y_positions[5]),
            (10, y_positions[5], 3, y_positions[6])
        ]
        
        for x1, y1, x2, y2 in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Add Neural ODE architecture details
        ax.text(13, 8.5, 'Neural ODE Architecture:', fontsize=12, fontweight='bold')
        ax.text(13, 8, '• Input Layer: 7 → 64 (x + t)', fontsize=10)
        ax.text(13, 7.5, '• Hidden Layer: 64 → 64 (tanh)', fontsize=10)
        ax.text(13, 7, '• Output Layer: 64 → 6 (linear)', fontsize=10)
        ax.text(13, 6.5, '• Time concatenation: [x, t]', fontsize=10)
        ax.text(13, 6, '• Activation: tanh (smooth)', fontsize=10)
        
        # Add ODE details
        ax.text(13, 5.5, 'ODE Integration Details:', fontsize=12, fontweight='bold')
        ax.text(13, 5, '• Solver: BDF (stiff ODEs)', fontsize=10)
        ax.text(13, 4.5, '• Tolerance: rtol=1e-5, atol=1e-8', fontsize=10)
        ax.text(13, 4, '• Time span: [0, T], T=1.0', fontsize=10)
        ax.text(13, 3.5, '• Steps: 10-50 (adaptive)', fontsize=10)
        
        # Add detailed CNF transform equations
        ax.text(13, 3, 'CNF Transform Equations:', fontsize=12, fontweight='bold')
        ax.text(13, 2.7, 'ODE: dx/dt = f(x,t)', fontsize=10, fontweight='bold')
        ax.text(13, 2.4, 'Forward: x(0) → x(T)', fontsize=9)
        ax.text(13, 2.1, 'x(T) = x(0) + ∫₀ᵀ f(x(s),s) ds', fontsize=9)
        ax.text(13, 1.8, 'Inverse: x(T) → x(0)', fontsize=9)
        ax.text(13, 1.5, 'x(0) = x(T) - ∫₀ᵀ f(x(s),s) ds', fontsize=9)
        ax.text(13, 1.2, 'Log-det: ∫₀ᵀ -∇·f(x,t) dt', fontsize=9)
        ax.text(13, 0.9, 'Divergence: ∇·f = Σᵢ ∂fᵢ/∂xᵢ', fontsize=9)
        
        # Add details
        ax.text(8, 0.5, 'Continuous transformation via Neural ODE\nLog-det computed via divergence (trace of Jacobian)', 
                fontsize=11, ha='center', style='italic')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cnf_architecture.png', bbox_inches='tight')
        plt.close()
    
    def _create_conditional_tfp_diagram(self):
        """Create diagram for Conditional TFP Flow with detailed layers and data shapes"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Title
        ax.text(8, 11.5, 'Conditional TFP Normalizing Flow - Detailed Architecture', 
                fontsize=18, fontweight='bold', ha='center')
        
        # Data flow with shapes
        y_positions = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        
        # Input data with shape
        self._draw_box(ax, 1, y_positions[0], 2, 0.8, 'Phase Space\n(batch_size, 6)\n[x, y, z, vx, vy, vz]', 'lightblue')
        
        # Mass conditioning with shape
        self._draw_box(ax, 4, y_positions[0], 2, 0.8, 'Mass Condition\n(batch_size, 1)\n[mass_bin_index]', 'lightcoral')
        
        # Mass embedding with details
        self._draw_box(ax, 4, y_positions[1], 2, 1.0, 'Mass Embedding\nEmbedding Layer\nInput: (batch_size, 1)\nOutput: (batch_size, 4)\n8 bins → 4D embedding', 'lightyellow')
        
        # Conditional MAF Layer 1 with MLP details
        self._draw_box(ax, 7.5, y_positions[2], 3.5, 1.4, 'Conditional MAF Layer 1\nAutoregressive Network\nInput: (batch_size, 6+4)\n[x, mass_embed] → [64, 64] → 12\n[shift(6), log_scale(6)]', 'lightgreen')
        
        # Permutation with details
        self._draw_box(ax, 7.5, y_positions[3], 3.5, 0.8, 'Permutation Layer\nRandom permutation\nInput: (batch_size, 6)\nOutput: (batch_size, 6)', 'lightyellow')
        
        # Conditional MAF Layer N with MLP details
        self._draw_box(ax, 7.5, y_positions[4], 3.5, 1.4, 'Conditional MAF Layer N\nAutoregressive Network\nInput: (batch_size, 6+4)\n[x, mass_embed] → [64, 64] → 12\n[shift(6), log_scale(6)]', 'lightgreen')
        
        # Base Distribution with details
        self._draw_box(ax, 1, y_positions[5], 2, 1.0, 'Base Distribution\nMultivariate Normal\nμ = 0, σ = 1\nShape: (batch_size, 6)\nlog_prob: (batch_size,)', 'lightpink')
        
        # Output with shape
        self._draw_box(ax, 7.5, y_positions[6], 2, 0.8, 'Latent Space\n(batch_size, 6)\n[z₁, z₂, ..., z₆]', 'lightblue')
        
        # Arrows
        arrows = [
            (3, y_positions[0], 7.5, y_positions[2]),  # Phase space to MAF
            (5, y_positions[0], 5, y_positions[1]),  # Mass to embedding
            (5, y_positions[1], 7.5, y_positions[2]),  # Embedding to MAF
            (9.25, y_positions[2], 9.25, y_positions[3]),  # MAF to permutation
            (9.25, y_positions[3], 9.25, y_positions[4]),  # Permutation to MAF
            (9.25, y_positions[4], 3, y_positions[5]),  # MAF to base
            (3, y_positions[5], 7.5, y_positions[6])  # Base to output
        ]
        
        for x1, y1, x2, y2 in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Add conditional architecture details
        ax.text(13, 8.5, 'Conditional Architecture:', fontsize=12, fontweight='bold')
        ax.text(13, 8, '• Mass Binning: 8 bins (log-uniform)', fontsize=10)
        ax.text(13, 7.5, '• Embedding: 8 → 4 dimensions', fontsize=10)
        ax.text(13, 7, '• Concatenation: [x, mass_embed]', fontsize=10)
        ax.text(13, 6.5, '• Input: (batch_size, 10)', fontsize=10)
        
        # Add MLP details
        ax.text(13, 6, 'Conditional MLP Architecture:', fontsize=12, fontweight='bold')
        ax.text(13, 5.5, '• Input Layer: 10 → 64', fontsize=10)
        ax.text(13, 5, '• Hidden Layer: 64 → 64 (ReLU)', fontsize=10)
        ax.text(13, 4.5, '• Output Layer: 64 → 12', fontsize=10)
        ax.text(13, 4, '• Split: 6 for shift, 6 for log_scale', fontsize=10)
        
        # Add detailed conditional transform equations
        ax.text(13, 3.5, 'Conditional MAF Transforms:', fontsize=12, fontweight='bold')
        ax.text(13, 3.2, 'Mass Embedding: m → e(m)', fontsize=10, fontweight='bold')
        ax.text(13, 2.9, 'Forward Transform:', fontsize=10, fontweight='bold')
        ax.text(13, 2.6, 'y₁ = x₁', fontsize=9)
        ax.text(13, 2.3, 'y₂ = x₂ ⊙ exp(s₂(x₁,e(m))) + t₂(x₁,e(m))', fontsize=9)
        ax.text(13, 2.0, 'y₃ = x₃ ⊙ exp(s₃(x₁,x₂,e(m))) + t₃(x₁,x₂,e(m))', fontsize=9)
        ax.text(13, 1.7, '... (conditional autoregressive)', fontsize=9)
        ax.text(13, 1.4, 'Inverse Transform:', fontsize=10, fontweight='bold')
        ax.text(13, 1.1, 'x₁ = y₁', fontsize=9)
        ax.text(13, 0.8, 'x₂ = (y₂ - t₂(x₁,e(m))) ⊙ exp(-s₂(x₁,e(m)))', fontsize=9)
        ax.text(13, 0.5, 'Log-det: Σᵢ sᵢ(x₁,...,xᵢ₋₁,e(m))', fontsize=9)
        
        # Add details
        ax.text(8, 0.5, 'Learns p(x|m) - phase space conditioned on mass\nMass embedding provides hierarchical conditioning', 
                fontsize=11, ha='center', style='italic')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'conditional_tfp_architecture.png', bbox_inches='tight')
        plt.close()
    
    def _create_conditional_cnf_diagram(self):
        """Create diagram for Conditional CNF"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, 'Conditional Continuous Normalizing Flow', 
                fontsize=16, fontweight='bold', ha='center')
        
        # Data flow
        y_positions = [8, 7, 6, 5, 4, 3, 2, 1]
        
        # Input data
        self._draw_box(ax, 1, y_positions[0], 1.5, 0.6, 'Phase Space\n(x₁, x₂, ..., x₆)', 'lightblue')
        
        # Mass conditioning
        self._draw_box(ax, 3.5, y_positions[0], 1.5, 0.6, 'Mass\nCondition', 'lightcoral')
        
        # Conditional Neural ODE
        self._draw_box(ax, 6, y_positions[1], 2.5, 0.8, 'Conditional Neural ODE\nf(x,m,t) = dx/dt', 'lightgreen')
        
        # ODE Solver
        self._draw_box(ax, 6, y_positions[2], 2.5, 0.6, 'ODE Solver\n(BDF/RK45)', 'lightyellow')
        
        # Time Integration
        self._draw_box(ax, 6, y_positions[3], 2.5, 0.6, 'Time Integration\nt ∈ [0, T]', 'lightcoral')
        
        # Jacobian Computation
        self._draw_box(ax, 6, y_positions[4], 2.5, 0.6, 'Jacobian Trace\n∇·f(x,m,t)', 'lightpink')
        
        # Base Distribution
        self._draw_box(ax, 1, y_positions[5], 1.5, 0.6, 'Base Distribution\n(Gaussian)', 'lightgray')
        
        # Output
        self._draw_box(ax, 6, y_positions[6], 1.5, 0.6, 'Latent Space\n(z₁, z₂, ..., z₆)', 'lightblue')
        
        # Arrows
        arrows = [
            (2.5, y_positions[0], 6, y_positions[1]),  # Phase space to ODE
            (4.25, y_positions[0], 6, y_positions[1]),  # Mass to ODE
            (7.25, y_positions[1], 7.25, y_positions[2]),  # ODE to solver
            (7.25, y_positions[2], 7.25, y_positions[3]),  # Solver to integration
            (7.25, y_positions[3], 7.25, y_positions[4]),  # Integration to Jacobian
            (7.25, y_positions[4], 2.5, y_positions[5]),  # Jacobian to base
            (2.5, y_positions[5], 6, y_positions[6])  # Base to output
        ]
        
        for x1, y1, x2, y2 in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Add details
        ax.text(5, 0.5, 'Continuous transformation conditioned on mass\nLearns p(x|m) via Neural ODE', 
                fontsize=10, ha='center', style='italic')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'conditional_cnf_architecture.png', bbox_inches='tight')
        plt.close()
    
    def _create_coupling_flow_diagram(self):
        """Create diagram for Coupling Flow"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, 'Coupling Flow Architecture', 
                fontsize=16, fontweight='bold', ha='center')
        
        # Data flow
        y_positions = [8, 7, 6, 5, 4, 3, 2, 1]
        
        # Input data
        self._draw_box(ax, 1, y_positions[0], 1.5, 0.6, 'Input Data\n(x₁, x₂, ..., x₆)', 'lightblue')
        
        # Coupling Layer 1
        self._draw_box(ax, 4, y_positions[1], 2, 0.6, 'Coupling Layer 1\n(Split & Transform)', 'lightgreen')
        
        # Coupling Layer 2
        self._draw_box(ax, 4, y_positions[2], 2, 0.6, 'Coupling Layer 2\n(Split & Transform)', 'lightgreen')
        
        # Coupling Layer N
        self._draw_box(ax, 4, y_positions[3], 2, 0.6, 'Coupling Layer N\n(Split & Transform)', 'lightgreen')
        
        # Base Distribution
        self._draw_box(ax, 7.5, y_positions[4], 2, 0.6, 'Base Distribution\n(Gaussian)', 'lightpink')
        
        # Output
        self._draw_box(ax, 1, y_positions[5], 1.5, 0.6, 'Latent Space\n(z₁, z₂, ..., z₆)', 'lightblue')
        
        # Arrows
        arrows = [
            (2.5, y_positions[0], 4, y_positions[1]),
            (5, y_positions[1], 5, y_positions[2]),
            (5, y_positions[2], 5, y_positions[3]),
            (6, y_positions[3], 7.5, y_positions[4]),
            (7.5, y_positions[4], 2.5, y_positions[5])
        ]
        
        for x1, y1, x2, y2 in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Add coupling details
        ax.text(4, 2.5, 'Split: x₁, x₂, x₃ | x₄, x₅, x₆\nTransform: y₁, y₂, y₃ = f(x₁, x₂, x₃, x₄, x₅, x₆)', 
                fontsize=9, ha='center', style='italic',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'coupling_flow_architecture.png', bbox_inches='tight')
        plt.close()
    
    def _create_kde_informed_diagram(self):
        """Create diagram for KDE-Informed Flow"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, 'KDE-Informed Flow Architecture', 
                fontsize=16, fontweight='bold', ha='center')
        
        # Data flow
        y_positions = [8, 7, 6, 5, 4, 3, 2, 1]
        
        # Training Data
        self._draw_box(ax, 1, y_positions[0], 1.5, 0.6, 'Training Data\n(x₁, x₂, ..., x₆)', 'lightblue')
        
        # KDE Models
        self._draw_box(ax, 3.5, y_positions[0], 1.5, 0.6, 'KDE Models\n(Mass Bins)', 'lightcoral')
        
        # Flow Model
        self._draw_box(ax, 6, y_positions[1], 2, 0.6, 'Flow Model\n(MAF/CNF)', 'lightgreen')
        
        # Combined Loss
        self._draw_box(ax, 6, y_positions[2], 2, 0.6, 'Combined Loss\nL = L_NLL + λ·L_KDE', 'lightyellow')
        
        # KDE Regularization
        self._draw_box(ax, 3.5, y_positions[3], 1.5, 0.6, 'KDE Regularization\nMSE(log p)', 'lightpink')
        
        # Trained Model
        self._draw_box(ax, 6, y_positions[4], 2, 0.6, 'Trained Model\n(KDE-Informed)', 'lightgreen')
        
        # Output
        self._draw_box(ax, 1, y_positions[5], 1.5, 0.6, 'Generated Samples', 'lightblue')
        
        # Arrows
        arrows = [
            (2.5, y_positions[0], 6, y_positions[1]),  # Data to flow
            (4.25, y_positions[0], 4.25, y_positions[3]),  # Data to KDE
            (7, y_positions[1], 7, y_positions[2]),  # Flow to loss
            (4.25, y_positions[3], 6, y_positions[2]),  # KDE to loss
            (7, y_positions[2], 7, y_positions[4]),  # Loss to trained model
            (7, y_positions[4], 2.5, y_positions[5])  # Model to output
        ]
        
        for x1, y1, x2, y2 in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Add details
        ax.text(5, 0.5, 'KDE provides density estimates for regularization\nλ controls KDE influence in training', 
                fontsize=10, ha='center', style='italic')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'kde_informed_architecture.png', bbox_inches='tight')
        plt.close()
    
    def _generate_overview_diagram(self):
        """Generate comprehensive overview of all flow types"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Title
        ax.text(8, 11.5, 'Flow Architectures Overview', 
                fontsize=20, fontweight='bold', ha='center')
        
        # Flow types with descriptions
        flow_types = [
            ('TFP MAF', 'Masked Autoregressive Flow\nDiscrete transformations\nGood for high-dim data', 2, 9),
            ('CNF', 'Continuous Normalizing Flow\nNeural ODE-based\nContinuous transformations', 6, 9),
            ('Conditional TFP', 'Conditional MAF\nMass-conditioned flows\np(x|m) learning', 10, 9),
            ('Conditional CNF', 'Conditional CNF\nMass-conditioned Neural ODE\nContinuous p(x|m)', 14, 9),
            ('Coupling Flow', 'Coupling Layers\nSplit-transform-merge\nEfficient computation', 2, 6),
            ('KDE-Informed', 'KDE Regularized Flow\nCombines KDE + Flow\nImproved density estimation', 6, 6),
            ('Neural ODE', 'Neural ODE Component\nUsed in CNF flows\nContinuous dynamics', 10, 6),
            ('Base Distributions', 'Gaussian/GMM Base\nSimple latent distributions\nStandard normal', 14, 6)
        ]
        
        # Draw flow type boxes
        for name, description, x, y in flow_types:
            self._draw_box(ax, x, y, 3, 2, f'{name}\n\n{description}', 'lightblue', alpha=0.8)
        
        # Add connections
        connections = [
            (3.5, 8, 4.5, 7),  # TFP MAF to Coupling
            (7.5, 8, 7.5, 7),  # CNF to KDE-Informed
            (11.5, 8, 11.5, 7),  # Conditional TFP to Neural ODE
            (15.5, 8, 15.5, 7),  # Conditional CNF to Base Distributions
        ]
        
        for x1, y1, x2, y2 in connections:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.6))
        
        # Add usage information
        ax.text(8, 4, 'Usage Patterns:', fontsize=14, fontweight='bold', ha='center')
        
        usage_text = """
• TFP MAF: General-purpose normalizing flows for 6D phase space
• CNF: Continuous transformations with Neural ODEs
• Conditional Flows: Mass-conditioned learning for astrophysical data
• Coupling Flows: Efficient alternative to MAF
• KDE-Informed: Hybrid approach combining KDE and flows
• Neural ODE: Continuous-time dynamics modeling
• Base Distributions: Simple latent space representations
        """
        
        ax.text(8, 2, usage_text, fontsize=10, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'flow_architectures_overview.png', bbox_inches='tight')
        plt.close()
    
    def _generate_comparison_diagram(self):
        """Generate comparison diagram of flow characteristics"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(7, 9.5, 'Flow Architecture Comparison', 
                fontsize=18, fontweight='bold', ha='center')
        
        # Comparison table
        headers = ['Architecture', 'Complexity', 'Speed', 'Conditioning', 'Use Case']
        rows = [
            ['TFP MAF', 'Medium', 'Fast', 'No', 'General 6D flows'],
            ['CNF', 'High', 'Slow', 'No', 'Continuous dynamics'],
            ['Conditional TFP', 'Medium', 'Fast', 'Yes', 'Mass-conditioned'],
            ['Conditional CNF', 'High', 'Slow', 'Yes', 'Continuous + conditional'],
            ['Coupling Flow', 'Low', 'Very Fast', 'No', 'Efficient alternative'],
            ['KDE-Informed', 'Medium', 'Medium', 'Yes', 'Hybrid approach']
        ]
        
        # Draw table
        cell_width = 2.5
        cell_height = 0.8
        start_x = 1
        start_y = 8
        
        # Headers
        for i, header in enumerate(headers):
            x = start_x + i * cell_width
            self._draw_box(ax, x, start_y, cell_width, cell_height, header, 'lightgray', fontweight='bold')
        
        # Rows
        for row_idx, row in enumerate(rows):
            y = start_y - (row_idx + 1) * cell_height
            for col_idx, cell in enumerate(row):
                x = start_x + col_idx * cell_width
                color = 'lightblue' if row_idx % 2 == 0 else 'lightgreen'
                self._draw_box(ax, x, y, cell_width, cell_height, cell, color, alpha=0.7)
        
        # Add performance metrics
        ax.text(7, 2, 'Performance Characteristics:', fontsize=14, fontweight='bold', ha='center')
        
        metrics_text = """
Speed Ranking: Coupling > TFP MAF > KDE-Informed > CNF
Complexity: Coupling < TFP MAF < KDE-Informed < CNF
Conditioning: TFP MAF/CNF (none) vs Conditional variants (mass-based)
Best for 6D Phase Space: Conditional TFP MAF (fast + conditioned)
Best for Continuous Dynamics: CNF (but slower)
Most Efficient: Coupling Flow (but less expressive)
        """
        
        ax.text(7, 0.5, metrics_text, fontsize=10, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'flow_comparison.png', bbox_inches='tight')
        plt.close()
    
    def _create_mlp_architecture_diagram(self):
        """Create detailed MLP architecture diagram showing layer details"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(8, 9.5, 'MLP Architecture Details for Flow Networks', 
                fontsize=18, fontweight='bold', ha='center')
        
        # MAF Autoregressive Network
        ax.text(2, 8.5, 'MAF Autoregressive Network', fontsize=14, fontweight='bold', ha='center')
        
        # Input layer
        self._draw_box(ax, 2, 7.5, 1.5, 0.6, 'Input\n(batch_size, 6)\n[x₁, x₂, ..., x₆]', 'lightblue')
        
        # Hidden layers
        self._draw_box(ax, 2, 6.5, 1.5, 0.6, 'Dense 1\n6 → 64\nReLU', 'lightgreen')
        self._draw_box(ax, 2, 5.5, 1.5, 0.6, 'Dense 2\n64 → 64\nReLU', 'lightgreen')
        
        # Output layer
        self._draw_box(ax, 2, 4.5, 1.5, 0.6, 'Output\n64 → 12\nLinear', 'lightcoral')
        
        # Split
        self._draw_box(ax, 2, 3.5, 1.5, 0.6, 'Split\n12 → [6, 6]\n[shift, log_scale]', 'lightyellow')
        
        # Arrows for MAF
        arrows_maf = [(2, 7.2, 2, 6.8), (2, 6.2, 2, 5.8), (2, 5.2, 2, 4.8), (2, 4.2, 2, 3.8)]
        for x1, y1, x2, y2 in arrows_maf:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Neural ODE Network
        ax.text(6, 8.5, 'Neural ODE Network', fontsize=14, fontweight='bold', ha='center')
        
        # Input layer (with time)
        self._draw_box(ax, 6, 7.5, 1.5, 0.6, 'Input\n(batch_size, 7)\n[x₁, x₂, ..., x₆, t]', 'lightblue')
        
        # Hidden layers
        self._draw_box(ax, 6, 6.5, 1.5, 0.6, 'Dense 1\n7 → 64\ntanh', 'lightgreen')
        self._draw_box(ax, 6, 5.5, 1.5, 0.6, 'Dense 2\n64 → 64\ntanh', 'lightgreen')
        
        # Output layer
        self._draw_box(ax, 6, 4.5, 1.5, 0.6, 'Output\n64 → 6\nLinear', 'lightcoral')
        
        # ODE function
        self._draw_box(ax, 6, 3.5, 1.5, 0.6, 'ODE Function\nf(x,t) = dx/dt', 'lightpink')
        
        # Arrows for Neural ODE
        arrows_ode = [(6, 7.2, 6, 6.8), (6, 6.2, 6, 5.8), (6, 5.2, 6, 4.8), (6, 4.2, 6, 3.8)]
        for x1, y1, x2, y2 in arrows_ode:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Conditional Network
        ax.text(10, 8.5, 'Conditional Network', fontsize=14, fontweight='bold', ha='center')
        
        # Input layer (with conditioning)
        self._draw_box(ax, 10, 7.5, 1.5, 0.6, 'Input\n(batch_size, 10)\n[x₁, x₂, ..., x₆, m₁, m₂, m₃, m₄]', 'lightblue')
        
        # Hidden layers
        self._draw_box(ax, 10, 6.5, 1.5, 0.6, 'Dense 1\n10 → 64\nReLU', 'lightgreen')
        self._draw_box(ax, 10, 5.5, 1.5, 0.6, 'Dense 2\n64 → 64\nReLU', 'lightgreen')
        
        # Output layer
        self._draw_box(ax, 10, 4.5, 1.5, 0.6, 'Output\n64 → 12\nLinear', 'lightcoral')
        
        # Split
        self._draw_box(ax, 10, 3.5, 1.5, 0.6, 'Split\n12 → [6, 6]\n[shift, log_scale]', 'lightyellow')
        
        # Arrows for Conditional
        arrows_cond = [(10, 7.2, 10, 6.8), (10, 6.2, 10, 5.8), (10, 5.2, 10, 4.8), (10, 4.2, 10, 3.8)]
        for x1, y1, x2, y2 in arrows_cond:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Add details
        ax.text(8, 2.5, 'Key Differences:', fontsize=12, fontweight='bold', ha='center')
        ax.text(8, 2, '• MAF: Autoregressive masking ensures causal dependencies', fontsize=10, ha='center')
        ax.text(8, 1.5, '• Neural ODE: Time-dependent, smooth dynamics', fontsize=10, ha='center')
        ax.text(8, 1, '• Conditional: Mass embedding provides hierarchical conditioning', fontsize=10, ha='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mlp_architecture_details.png', bbox_inches='tight')
        plt.close()
    
    def _create_transform_equations_diagram(self):
        """Create comprehensive transform equations diagram"""
        fig, ax = plt.subplots(1, 1, figsize=(18, 14))
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 14)
        ax.axis('off')
        
        # Title
        ax.text(9, 13.5, 'Normalizing Flow Transform Equations', 
                fontsize=20, fontweight='bold', ha='center')
        
        # MAF Transform Equations
        ax.text(3, 12.5, 'MAF (Masked Autoregressive Flow)', 
                fontsize=16, fontweight='bold', ha='center')
        
        # MAF Forward
        ax.text(1, 11.5, 'Forward Transform:', fontsize=12, fontweight='bold')
        ax.text(1, 11, 'y₁ = x₁', fontsize=11)
        ax.text(1, 10.7, 'y₂ = x₂ ⊙ exp(s₂(x₁)) + t₂(x₁)', fontsize=11)
        ax.text(1, 10.4, 'y₃ = x₃ ⊙ exp(s₃(x₁,x₂)) + t₃(x₁,x₂)', fontsize=11)
        ax.text(1, 10.1, 'y₄ = x₄ ⊙ exp(s₄(x₁,x₂,x₃)) + t₄(x₁,x₂,x₃)', fontsize=11)
        ax.text(1, 9.8, 'y₅ = x₅ ⊙ exp(s₅(x₁,x₂,x₃,x₄)) + t₅(x₁,x₂,x₃,x₄)', fontsize=11)
        ax.text(1, 9.5, 'y₆ = x₆ ⊙ exp(s₆(x₁,x₂,x₃,x₄,x₅)) + t₆(x₁,x₂,x₃,x₄,x₅)', fontsize=11)
        
        # MAF Inverse
        ax.text(1, 9, 'Inverse Transform:', fontsize=12, fontweight='bold')
        ax.text(1, 8.7, 'x₁ = y₁', fontsize=11)
        ax.text(1, 8.4, 'x₂ = (y₂ - t₂(x₁)) ⊙ exp(-s₂(x₁))', fontsize=11)
        ax.text(1, 8.1, 'x₃ = (y₃ - t₃(x₁,x₂)) ⊙ exp(-s₃(x₁,x₂))', fontsize=11)
        ax.text(1, 7.8, 'x₄ = (y₄ - t₄(x₁,x₂,x₃)) ⊙ exp(-s₄(x₁,x₂,x₃))', fontsize=11)
        ax.text(1, 7.5, 'x₅ = (y₅ - t₅(x₁,x₂,x₃,x₄)) ⊙ exp(-s₅(x₁,x₂,x₃,x₄))', fontsize=11)
        ax.text(1, 7.2, 'x₆ = (y₆ - t₆(x₁,x₂,x₃,x₄,x₅)) ⊙ exp(-s₆(x₁,x₂,x₃,x₄,x₅))', fontsize=11)
        
        # MAF Log-det
        ax.text(1, 6.8, 'Log-determinant:', fontsize=12, fontweight='bold')
        ax.text(1, 6.5, 'log|det(∂y/∂x)| = Σᵢ₌₁⁶ sᵢ(x₁,...,xᵢ₋₁)', fontsize=11)
        
        # CNF Transform Equations
        ax.text(9, 12.5, 'CNF (Continuous Normalizing Flow)', 
                fontsize=16, fontweight='bold', ha='center')
        
        # CNF ODE
        ax.text(7, 11.5, 'Neural ODE:', fontsize=12, fontweight='bold')
        ax.text(7, 11.2, 'dx/dt = f(x,t)', fontsize=11)
        ax.text(7, 10.9, 'where f(x,t) = NeuralNet([x,t])', fontsize=11)
        
        # CNF Forward
        ax.text(7, 10.5, 'Forward Transform:', fontsize=12, fontweight='bold')
        ax.text(7, 10.2, 'x(T) = x(0) + ∫₀ᵀ f(x(s),s) ds', fontsize=11)
        ax.text(7, 9.9, 'Solved numerically with ODE solver', fontsize=11)
        
        # CNF Inverse
        ax.text(7, 9.5, 'Inverse Transform:', fontsize=12, fontweight='bold')
        ax.text(7, 9.2, 'x(0) = x(T) - ∫₀ᵀ f(x(s),s) ds', fontsize=11)
        ax.text(7, 8.9, 'Or: dx/dt = -f(x,T-t)', fontsize=11)
        
        # CNF Log-det
        ax.text(7, 8.5, 'Log-determinant:', fontsize=12, fontweight='bold')
        ax.text(7, 8.2, 'log|det(∂x(T)/∂x(0))| = ∫₀ᵀ -∇·f(x,t) dt', fontsize=11)
        ax.text(7, 7.9, 'where ∇·f = Σᵢ ∂fᵢ/∂xᵢ (divergence)', fontsize=11)
        
        # Conditional Transform Equations
        ax.text(15, 12.5, 'Conditional MAF', 
                fontsize=16, fontweight='bold', ha='center')
        
        # Conditional Forward
        ax.text(13, 11.5, 'Forward Transform:', fontsize=12, fontweight='bold')
        ax.text(13, 11.2, 'y₁ = x₁', fontsize=11)
        ax.text(13, 10.9, 'y₂ = x₂ ⊙ exp(s₂(x₁,m)) + t₂(x₁,m)', fontsize=11)
        ax.text(13, 10.6, 'y₃ = x₃ ⊙ exp(s₃(x₁,x₂,m)) + t₃(x₁,x₂,m)', fontsize=11)
        ax.text(13, 10.3, '... (conditioned on mass m)', fontsize=11)
        
        # Conditional Inverse
        ax.text(13, 9.9, 'Inverse Transform:', fontsize=12, fontweight='bold')
        ax.text(13, 9.6, 'x₁ = y₁', fontsize=11)
        ax.text(13, 9.3, 'x₂ = (y₂ - t₂(x₁,m)) ⊙ exp(-s₂(x₁,m))', fontsize=11)
        ax.text(13, 9.0, 'x₃ = (y₃ - t₃(x₁,x₂,m)) ⊙ exp(-s₃(x₁,x₂,m))', fontsize=11)
        ax.text(13, 8.7, '... (conditioned on mass m)', fontsize=11)
        
        # Conditional Log-det
        ax.text(13, 8.3, 'Log-determinant:', fontsize=12, fontweight='bold')
        ax.text(13, 8.0, 'log|det(∂y/∂x)| = Σᵢ sᵢ(x₁,...,xᵢ₋₁,m)', fontsize=11)
        
        # Coupling Transform Equations
        ax.text(3, 6, 'Coupling Layer', 
                fontsize=16, fontweight='bold', ha='center')
        
        # Coupling Forward
        ax.text(1, 5.5, 'Forward Transform:', fontsize=12, fontweight='bold')
        ax.text(1, 5.2, 'Split: x = [x₁,x₂] where x₁,x₂ ∈ ℝ³', fontsize=11)
        ax.text(1, 4.9, 'y₁ = x₁', fontsize=11)
        ax.text(1, 4.6, 'y₂ = x₂ ⊙ exp(s(x₁)) + t(x₁)', fontsize=11)
        ax.text(1, 4.3, 'Output: y = [y₁,y₂]', fontsize=11)
        
        # Coupling Inverse
        ax.text(1, 3.9, 'Inverse Transform:', fontsize=12, fontweight='bold')
        ax.text(1, 3.6, 'x₁ = y₁', fontsize=11)
        ax.text(1, 3.3, 'x₂ = (y₂ - t(y₁)) ⊙ exp(-s(y₁))', fontsize=11)
        
        # Coupling Log-det
        ax.text(1, 2.9, 'Log-determinant:', fontsize=12, fontweight='bold')
        ax.text(1, 2.6, 'log|det(∂y/∂x)| = Σᵢ sᵢ(x₁)', fontsize=11)
        
        # Base Distribution
        ax.text(9, 6, 'Base Distribution', 
                fontsize=16, fontweight='bold', ha='center')
        
        ax.text(7, 5.5, 'Gaussian Base:', fontsize=12, fontweight='bold')
        ax.text(7, 5.2, 'p(z) = N(z; 0, I)', fontsize=11)
        ax.text(7, 4.9, 'log p(z) = -½||z||² - ½d log(2π)', fontsize=11)
        
        ax.text(7, 4.5, 'GMM Base:', fontsize=12, fontweight='bold')
        ax.text(7, 4.2, 'p(z) = Σᵢ πᵢ N(z; μᵢ, Σᵢ)', fontsize=11)
        ax.text(7, 3.9, 'log p(z) = log Σᵢ πᵢ exp(log N(z; μᵢ, Σᵢ))', fontsize=11)
        
        # Overall Flow
        ax.text(15, 6, 'Overall Flow', 
                fontsize=16, fontweight='bold', ha='center')
        
        ax.text(13, 5.5, 'Forward (Data → Latent):', fontsize=12, fontweight='bold')
        ax.text(13, 5.2, 'x → z = Tₙ(...T₂(T₁(x)))', fontsize=11)
        ax.text(13, 4.9, 'log p(x) = log p(z) + Σᵢ log|det(∂Tᵢ/∂x)|', fontsize=11)
        
        ax.text(13, 4.5, 'Inverse (Latent → Data):', fontsize=12, fontweight='bold')
        ax.text(13, 4.2, 'z → x = T₁⁻¹(T₂⁻¹(...Tₙ⁻¹(z)))', fontsize=11)
        ax.text(13, 3.9, 'Sampling: z ~ p(z), x = T⁻¹(z)', fontsize=11)
        
        # Add mathematical notation legend
        ax.text(9, 2.5, 'Notation:', fontsize=12, fontweight='bold', ha='center')
        ax.text(9, 2.2, '⊙ = element-wise multiplication, ⊙ = Hadamard product', fontsize=10, ha='center')
        ax.text(9, 1.9, 'sᵢ = log-scale function, tᵢ = shift function', fontsize=10, ha='center')
        ax.text(9, 1.6, '∇·f = divergence of vector field f', fontsize=10, ha='center')
        ax.text(9, 1.3, 'Tᵢ = i-th transformation layer', fontsize=10, ha='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'transform_equations.png', bbox_inches='tight')
        plt.close()
    
    def _draw_box(self, ax, x, y, width, height, text, color='lightblue', alpha=1.0, fontweight='normal'):
        """Draw a box with text"""
        # Create rounded rectangle
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                            boxstyle="round,pad=0.1", 
                            facecolor=color, 
                            edgecolor='black',
                            linewidth=1.5,
                            alpha=alpha)
        ax.add_patch(box)
        
        # Add text
        ax.text(x, y, text, ha='center', va='center', 
               fontsize=9, fontweight=fontweight, wrap=True)

def main():
    """Main function to generate all flow architecture diagrams"""
    print("🚀 Flow Architecture Diagram Generator")
    print("=" * 50)
    
    # Get the current directory (flows-tensorflow)
    codebase_path = Path.cwd()
    
    # Analyze the codebase
    analyzer = FlowArchitectureAnalyzer(codebase_path)
    architecture_info = analyzer.analyze_codebase()
    
    # Generate visualizations
    visualizer = FlowArchitectureVisualizer(analyzer)
    visualizer.generate_all_diagrams()
    
    print("\n✅ Flow architecture diagrams generated successfully!")
    print(f"📁 Diagrams saved in: {visualizer.output_dir}")
    print("\nGenerated files:")
    for file in visualizer.output_dir.glob("*.png"):
        print(f"  • {file.name}")
    
    print("\n📊 Summary of Flow Architectures Found:")
    for class_name, info in analyzer.flow_classes.items():
        print(f"  • {info['type']}: {class_name} ({info['file']})")

if __name__ == "__main__":
    main()