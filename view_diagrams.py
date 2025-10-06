#!/usr/bin/env python3
"""
View Flow Architecture Diagrams

This script provides a simple way to view all generated flow architecture diagrams.
It can display them in a matplotlib window or save them as a combined figure.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

def view_all_diagrams():
    """Display all flow architecture diagrams"""
    diagram_dir = Path("flow_architecture_diagrams")
    
    if not diagram_dir.exists():
        print("❌ Diagram directory not found. Run generate_flow_architectures.py first.")
        return
    
    # Get all PNG files
    diagram_files = list(diagram_dir.glob("*.png"))
    
    if not diagram_files:
        print("❌ No diagrams found in the directory.")
        return
    
    print(f"📊 Found {len(diagram_files)} diagrams:")
    for i, file in enumerate(diagram_files):
        print(f"  {i+1}. {file.name}")
    
    # Create a grid layout
    n_diagrams = len(diagram_files)
    n_cols = 2
    n_rows = (n_diagrams + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, file in enumerate(diagram_files):
        row = i // n_cols
        col = i % n_cols
        
        # Load and display image
        img = mpimg.imread(file)
        axes[row, col].imshow(img)
        axes[row, col].set_title(file.stem.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_diagrams, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Flow Architecture Diagrams', fontsize=16, fontweight='bold', y=0.98)
    plt.show()

def save_combined_diagram():
    """Save all diagrams as a combined figure"""
    diagram_dir = Path("flow_architecture_diagrams")
    
    if not diagram_dir.exists():
        print("❌ Diagram directory not found. Run generate_flow_architectures.py first.")
        return
    
    # Get all PNG files
    diagram_files = list(diagram_dir.glob("*.png"))
    
    if not diagram_files:
        print("❌ No diagrams found in the directory.")
        return
    
    # Create a grid layout
    n_diagrams = len(diagram_files)
    n_cols = 2
    n_rows = (n_diagrams + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, file in enumerate(diagram_files):
        row = i // n_cols
        col = i % n_cols
        
        # Load and display image
        img = mpimg.imread(file)
        axes[row, col].imshow(img)
        axes[row, col].set_title(file.stem.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_diagrams, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Flow Architecture Diagrams', fontsize=16, fontweight='bold', y=0.98)
    
    # Save combined diagram
    output_file = "all_flow_architectures.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Combined diagram saved as {output_file}")
    plt.close()

def list_diagrams():
    """List all available diagrams"""
    diagram_dir = Path("flow_architecture_diagrams")
    
    if not diagram_dir.exists():
        print("❌ Diagram directory not found. Run generate_flow_architectures.py first.")
        return
    
    diagram_files = list(diagram_dir.glob("*.png"))
    
    if not diagram_files:
        print("❌ No diagrams found in the directory.")
        return
    
    print("📊 Available Flow Architecture Diagrams:")
    print("=" * 50)
    
    for i, file in enumerate(diagram_files, 1):
        print(f"{i:2d}. {file.name}")
        print(f"    Path: {file}")
        print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "list":
            list_diagrams()
        elif command == "save":
            save_combined_diagram()
        elif command == "view":
            view_all_diagrams()
        else:
            print("Usage: python view_diagrams.py [list|view|save]")
            print("  list - List all available diagrams")
            print("  view - Display all diagrams in matplotlib window")
            print("  save - Save all diagrams as combined figure")
    else:
        print("🎨 Flow Architecture Diagram Viewer")
        print("=" * 40)
        print("Available commands:")
        print("  python view_diagrams.py list  - List all diagrams")
        print("  python view_diagrams.py view - View all diagrams")
        print("  python view_diagrams.py save  - Save combined diagram")
        print()
        list_diagrams()