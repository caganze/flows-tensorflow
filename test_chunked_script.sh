#!/bin/bash

# Test script to verify argument parsing
set -e

# Default configuration
CHUNK_SIZE=50
CONCURRENT=5
PARTITION="kipac"
TIME_LIMIT="12:00:00"
MEMORY="64GB"
CPUS_PER_TASK=8
PARTICLE_LIST="particle_list_coupling_incomplete.txt"
OUTPUT_DIR="/oak/stanford/orgs/kipac/users/caganze/flows-tensorflow/coupling_output"
EPOCHS=150
LEARNING_RATE=5e-3
BATCH_SIZE=256
N_MASS_BINS=100
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --concurrent)
            CONCURRENT="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --time-limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --cpus)
            CPUS_PER_TASK="$2"
            shift 2
            ;;
        --particle-list)
            PARTICLE_LIST="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --n-mass-bins)
            N_MASS_BINS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "❌ Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "✅ All arguments parsed successfully!"
echo "   BATCH_SIZE: $BATCH_SIZE"
echo "   N_MASS_BINS: $N_MASS_BINS"
echo "   EPOCHS: $EPOCHS"
echo "   LEARNING_RATE: $LEARNING_RATE"






