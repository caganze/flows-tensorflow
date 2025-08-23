#!/usr/bin/env python3
"""
Manual fix for the f-string syntax error in training loop
"""

def fix_training_print():
    """Fix the problematic print statement in tfp_flows_gpu_solution.py"""
    
    try:
        # Read the file
        with open('tfp_flows_gpu_solution.py', 'r') as f:
            content = f.read()
        
        print("✅ File read successfully")
        
        # Look for the problematic line (line 227 based on the error)
        lines = content.split('\n')
        
        # Find the problematic print statement
        for i, line in enumerate(lines):
            if 'Val Loss:' in line and '.4f if val_loss is not None else' in line:
                print(f"Found problematic line {i+1}: {line.strip()}")
                
                # Find the complete print statement (might span multiple lines)
                start_idx = i
                while start_idx > 0 and 'print(f"' not in lines[start_idx]:
                    start_idx -= 1
                
                end_idx = i
                while end_idx < len(lines) and ')' not in lines[end_idx]:
                    end_idx += 1
                
                print(f"Print statement spans lines {start_idx+1} to {end_idx+1}")
                
                # Replace the problematic section
                # Find the validation loss formatting part
                for j in range(start_idx, end_idx + 1):
                    if '.4f if val_loss is not None else' in lines[j]:
                        # Replace this line
                        old_line = lines[j]
                        # Extract the indentation
                        indent = old_line[:len(old_line) - len(old_line.lstrip())]
                        
                        # Create new lines with proper formatting
                        new_lines = [
                            f'{indent}val_loss_str = f"{{val_loss:.4f}}" if val_loss is not None else "N/A"',
                            old_line.replace('{val_loss:.4f if val_loss is not None else \'N/A\'}', '{val_loss_str}')
                        ]
                        
                        # Replace the line
                        lines[j:j+1] = new_lines
                        print(f"✅ Fixed line {j+1}")
                        break
                break
        
        # Write the fixed content back
        with open('tfp_flows_gpu_solution.py', 'w') as f:
            f.write('\n'.join(lines))
        
        print("✅ File fixed and saved!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    fix_training_print()

