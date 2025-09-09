#!/usr/bin/env python3
"""
Final fix for the f-string syntax error - handle both the conditional logic AND syntax
"""

def fix_training_print_final():
    """Fix the f-string issues in tfp_flows_gpu_solution.py"""
    
    try:
        # Read the file
        with open('tfp_flows_gpu_solution.py', 'r') as f:
            content = f.read()
        
        print("✅ File read successfully")
        
        # Look for the problematic print statement around line 227
        lines = content.split('\n')
        
        # Find the print statement that spans multiple lines
        for i, line in enumerate(lines):
            if 'print(f"Epoch' in line and '{epoch:4d}' in line:
                print(f"Found problematic print statement starting at line {i+1}")
                
                # Find the complete print statement (spans multiple lines)
                start_idx = i
                end_idx = i
                
                # Find the end of this print statement
                paren_count = line.count('(') - line.count(')')
                while end_idx < len(lines) - 1 and paren_count > 0:
                    end_idx += 1
                    paren_count += lines[end_idx].count('(') - lines[end_idx].count(')')
                
                print(f"Print statement spans lines {start_idx+1} to {end_idx+1}")
                
                # Get the indentation from the first line
                indent = lines[start_idx][:len(lines[start_idx]) - len(lines[start_idx].lstrip())]
                
                # Create the corrected print statement
                new_lines = [
                    f'{indent}# Fix: Handle val_loss formatting properly',
                    f'{indent}if val_loss is not None:',
                    f'{indent}    val_loss_str = f"{{val_loss:.4f}}"',
                    f'{indent}else:',
                    f'{indent}    val_loss_str = "N/A"',
                    f'{indent}',
                    f'{indent}print((',
                    f'{indent}    f"Epoch {{epoch:4d}}/{{epochs}} | "',
                    f'{indent}    f"Train Loss: {{avg_train_loss:.4f}} | "',
                    f'{indent}    f"Val Loss: {{val_loss_str}} | "',
                    f'{indent}    f"Time: {{elapsed:.1f}}s"',
                    f'{indent}))'
                ]
                
                # Replace the problematic lines
                lines[start_idx:end_idx+1] = new_lines
                print(f"✅ Fixed lines {start_idx+1} to {end_idx+1}")
                break
        
        # Write the fixed content back
        with open('tfp_flows_gpu_solution.py', 'w') as f:
            f.write('\n'.join(lines))
        
        print("✅ File fixed and saved!")
        print("🎯 The print statement now uses proper conditional logic and syntax!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    fix_training_print_final()

