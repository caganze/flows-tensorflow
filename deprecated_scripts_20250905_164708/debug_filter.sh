#!/bin/bash

# 🔍 Debug Filter Script
# Simple debugging version to understand what's failing

INPUT_FILE="particle_list.txt"

echo "🔍 DEBUG FILTER SCRIPT"
echo "====================="
echo ""

# Check if input file exists
echo "1️⃣ Checking input file..."
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "❌ Input file not found: $INPUT_FILE"
    echo "📁 Current directory: $(pwd)"
    echo "📋 Files in current directory:"
    ls -la *.txt 2>/dev/null || echo "No .txt files found"
    exit 1
else
    echo "✅ Found input file: $INPUT_FILE"
    echo "📊 File size: $(wc -l < "$INPUT_FILE") lines"
fi

echo ""
echo "2️⃣ Checking first few lines of particle list..."
head -3 "$INPUT_FILE"

echo ""
echo "3️⃣ Testing path construction for first particle..."
# Read first line
first_line=$(head -1 "$INPUT_FILE")
echo "📝 First particle: $first_line"

# Parse it
IFS=',' read -r pid h5_file object_count size_category <<< "$first_line"
echo "🔍 Parsed:"
echo "   PID: $pid"
echo "   H5 file: $h5_file"
echo "   Object count: $object_count"
echo "   Size: $size_category"

# Check if H5 file exists
echo ""
echo "4️⃣ Checking if H5 file exists..."
if [[ -f "$h5_file" ]]; then
    echo "✅ H5 file exists: $h5_file"
else
    echo "❌ H5 file not found: $h5_file"
    echo "🔍 Let's check what's in the parent directory:"
    parent_dir=$(dirname "$h5_file")
    echo "📁 Parent directory: $parent_dir"
    if [[ -d "$parent_dir" ]]; then
        echo "📋 Contents of parent directory:"
        ls -la "$parent_dir" | head -10
    else
        echo "❌ Parent directory doesn't exist"
    fi
fi

# Test path construction
filename=$(basename "$h5_file")
halo_id=$(echo "$filename" | sed 's/.*Halo\([0-9]\+\).*/\1/')

data_source="unknown"
if [[ "$filename" == *"eden_scaled"* ]]; then
    data_source="eden"
elif [[ "$filename" == *"symphonyHR_scaled"* ]]; then
    data_source="symphony-hr"
elif [[ "$filename" == *"symphony_scaled"* ]]; then
    data_source="symphony"
fi

echo ""
echo "5️⃣ Path construction test..."
echo "🏷️ Filename: $filename"
echo "🆔 Halo ID: $halo_id"
echo "📊 Data source: $data_source"

# Construct expected paths
h5_parent_dir=$(dirname "$h5_file")
output_base_dir="$h5_parent_dir/tfp_output"
model_dir="$output_base_dir/trained_flows/${data_source}/halo${halo_id}"
samples_dir="$output_base_dir/samples/${data_source}/halo${halo_id}"

echo ""
echo "6️⃣ Expected output paths..."
echo "📁 Output base: $output_base_dir"
echo "🤖 Model dir: $model_dir"
echo "📊 Samples dir: $samples_dir"

# Check if these directories exist
echo ""
echo "7️⃣ Directory existence check..."
if [[ -d "$output_base_dir" ]]; then
    echo "✅ Output base exists"
    if [[ -d "$model_dir" ]]; then
        echo "✅ Model dir exists"
        echo "📋 Model files:"
        ls -la "$model_dir"/ 2>/dev/null | head -5
    else
        echo "❌ Model dir doesn't exist: $model_dir"
    fi
    
    if [[ -d "$samples_dir" ]]; then
        echo "✅ Samples dir exists"
        echo "📋 Sample files:"
        ls -la "$samples_dir"/ 2>/dev/null | head -5
    else
        echo "❌ Samples dir doesn't exist: $samples_dir"
    fi
else
    echo "❌ Output base doesn't exist: $output_base_dir"
    echo "🔍 Let's see what's in the H5 parent directory:"
    ls -la "$h5_parent_dir"/ 2>/dev/null || echo "Can't access parent directory"
fi

echo ""
echo "🏁 Debug complete!"
