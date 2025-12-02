#!/bin/bash
# HypoBench HypotheSAEs Hypothesis Generation
# Generate 20 hypotheses for supported datasets using HypotheSAEs
# Usage: ./run_all_hypothesaes.sh [model_name] [num_hypotheses]

MODEL_NAME="${1:-gpt-4o-mini}"
NUM_HYP="${2:-20}"

echo "=========================================="
echo "HypoBench - HypotheSAEs Hypothesis Generator"
echo "Model: $MODEL_NAME"
echo "Hypotheses: $NUM_HYP"
echo "=========================================="

# Base path
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$BASE_DIR/hypogenic/data"

# ==========================================
# SUPPORTED REAL DATASETS
# ==========================================
echo ""
echo "=== REAL DATASETS (HypotheSAEs) ==="

# Deceptive Reviews
echo "Running: deceptive_reviews"
python "$BASE_DIR/comparison/hypothesaes_runner.py" \
    --dataset_name deceptive_reviews \
    --data_dir "$DATA_DIR/real/deceptive_reviews" \
    --output_dir "./outputs/hypothesaes/deceptive_reviews" \
    --num_train 200 --num_test 300 \
    --max_num_hypotheses $NUM_HYP \
    --interpreter_model "$MODEL_NAME" \
    --annotator_model "$MODEL_NAME" \
    --seed 42

# Retweet
echo "Running: retweet"
python "$BASE_DIR/comparison/hypothesaes_runner.py" \
    --dataset_name retweet \
    --data_dir "$DATA_DIR/real/retweet" \
    --output_dir "./outputs/hypothesaes/retweet" \
    --num_train 200 --num_test 300 \
    --max_num_hypotheses $NUM_HYP \
    --interpreter_model "$MODEL_NAME" \
    --annotator_model "$MODEL_NAME" \
    --seed 42

# Dreaddit
echo "Running: dreaddit"
python "$BASE_DIR/comparison/hypothesaes_runner.py" \
    --dataset_name dreaddit \
    --data_dir "$DATA_DIR/real/dreaddit" \
    --output_dir "./outputs/hypothesaes/dreaddit" \
    --num_train 200 --num_test 300 \
    --max_num_hypotheses $NUM_HYP \
    --interpreter_model "$MODEL_NAME" \
    --annotator_model "$MODEL_NAME" \
    --seed 42

# Headline Binary
echo "Running: headline_binary"
python "$BASE_DIR/comparison/hypothesaes_runner.py" \
    --dataset_name headline_binary \
    --data_dir "$DATA_DIR/real/headline_binary" \
    --output_dir "./outputs/hypothesaes/headline_binary" \
    --num_train 200 --num_test 300 \
    --max_num_hypotheses $NUM_HYP \
    --interpreter_model "$MODEL_NAME" \
    --annotator_model "$MODEL_NAME" \
    --seed 42

# GPT-GC Detect
echo "Running: gptgc_detect"
python "$BASE_DIR/comparison/hypothesaes_runner.py" \
    --dataset_name gptgc_detect \
    --data_dir "$DATA_DIR/real/gptgc_detect" \
    --output_dir "./outputs/hypothesaes/gptgc_detect" \
    --num_train 200 --num_test 300 \
    --max_num_hypotheses $NUM_HYP \
    --interpreter_model "$MODEL_NAME" \
    --annotator_model "$MODEL_NAME" \
    --seed 42

# LLaMA-GC Detect
echo "Running: llamagc_detect"
python "$BASE_DIR/comparison/hypothesaes_runner.py" \
    --dataset_name llamagc_detect \
    --data_dir "$DATA_DIR/real/llamagc_detect" \
    --output_dir "./outputs/hypothesaes/llamagc_detect" \
    --num_train 200 --num_test 300 \
    --max_num_hypotheses $NUM_HYP \
    --interpreter_model "$MODEL_NAME" \
    --annotator_model "$MODEL_NAME" \
    --seed 42

# Persuasive Pairs
echo "Running: persuasive_pairs"
python "$BASE_DIR/comparison/hypothesaes_runner.py" \
    --dataset_name persuasive_pairs \
    --data_dir "$DATA_DIR/real/persuasive_pairs" \
    --output_dir "./outputs/hypothesaes/persuasive_pairs" \
    --num_train 200 --num_test 300 \
    --max_num_hypotheses $NUM_HYP \
    --interpreter_model "$MODEL_NAME" \
    --annotator_model "$MODEL_NAME" \
    --seed 42

echo ""
echo "=========================================="
echo "HypotheSAEs generation complete!"
echo "=========================================="

