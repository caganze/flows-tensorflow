#!/usr/bin/env python3
"""
Quick fix for f-string syntax error in training loop
"""

import re

def fix_fstring_error():
    """Fix the f-string conditional format specifier error"""
    
    # Read the file
    with open('tfp_flows_gpu_solution.py', 'r') as f:
        content = f.read()
    
    # Find and fix the problematic f-string
    # Look for the pattern: {val_loss:.4f if val_loss is not None else 'N/A'}
    pattern = r'\{val_loss:\.4f if val_loss is not None else \'N/A\'\}'
    
    if pattern in content:
        print("Found the problematic f-string pattern")
        
        # Replace the problematic print statement with a corrected version
        old_print = '''            if verbose and epoch % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:4d}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f if val_loss is not None else 'N/A'} | "
                      f"Time: {elapsed:.1f}s")'''
        
        new_print = '''            if verbose and epoch % 10 == 0:
                elapsed = time.time() - start_time
                val_loss_str = f"{val_loss:.4f}" if val_loss is not None else 'N/A'
                print(f"Epoch {epoch:4d}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_loss_str} | "
                      f"Time: {elapsed:.1f}s")'''
        
        content = content.replace(old_print, new_print)
        
        # Write the fixed content back
        with open('tfp_flows_gpu_solution.py', 'w') as f:
            f.write(content)
        
        print("✅ Fixed f-string error!")
        return True
    else:
        print("❌ Pattern not found")
        return False

if __name__ == "__main__":
    fix_fstring_error()

