#!/bin/bash
# HypoBench HypoGenic Hypothesis Generation
# Generate 20 hypotheses for all datasets using HypoGenic
# Usage: ./run_all_hypogenic.sh [model_name] [num_hypotheses]

MODEL_NAME="${1:-gpt-4o-mini}"
NUM_HYP="${2:-20}"

echo "=========================================="
echo "HypoBench - HypoGenic Hypothesis Generator"
echo "Model: $MODEL_NAME"
echo "Hypotheses: $NUM_HYP"
echo "=========================================="

# Base path
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$BASE_DIR/hypogenic/data"

# ==========================================
# REAL DATASETS
# ==========================================
echo ""
echo "=== REAL DATASETS ==="

# Deceptive Reviews
echo "Running: deceptive_reviews"
hypogenic_generation \
    --task_config_path "$DATA_DIR/real/deceptive_reviews/config.yaml" \
    --model_name "$MODEL_NAME" --model_type gpt \
    --max_num_hypotheses $NUM_HYP \
    --output_folder "./outputs/hypogenic/deceptive_reviews" \
    --num_train 200 --num_test 100 --num_val 100 --seed 42 \
    --num_init 10 --k 5 --alpha 0.5 \
    4096 1e-5

# Dreaddit
echo "Running: dreaddit"
hypogenic_generation \
    --task_config_path "$DATA_DIR/real/dreaddit/config.yaml" \
    --model_name "$MODEL_NAME" --model_type gpt \
    --max_num_hypotheses $NUM_HYP \
    --output_folder "./outputs/hypogenic/dreaddit" \
    --num_train 200 --num_test 100 --num_val 100 --seed 42 \
    --num_init 10 --k 5 --alpha 0.5 \
    4096 1e-5

# GPT-GC Detect
echo "Running: gptgc_detect"
hypogenic_generation \
    --task_config_path "$DATA_DIR/real/gptgc_detect/config.yaml" \
    --model_name "$MODEL_NAME" --model_type gpt \
    --max_num_hypotheses $NUM_HYP \
    --output_folder "./outputs/hypogenic/gptgc_detect" \
    --num_train 200 --num_test 100 --num_val 100 --seed 42 \
    --num_init 10 --k 5 --alpha 0.5 \
    4096 1e-5

# Headline Binary
echo "Running: headline_binary"
hypogenic_generation \
    --task_config_path "$DATA_DIR/real/headline_binary/config.yaml" \
    --model_name "$MODEL_NAME" --model_type gpt \
    --max_num_hypotheses $NUM_HYP \
    --output_folder "./outputs/hypogenic/headline_binary" \
    --num_train 200 --num_test 100 --num_val 100 --seed 42 \
    --num_init 10 --k 5 --alpha 0.5 \
    4096 1e-5

# LLaMA-GC Detect
echo "Running: llamagc_detect"
hypogenic_generation \
    --task_config_path "$DATA_DIR/real/llamagc_detect/config.yaml" \
    --model_name "$MODEL_NAME" --model_type gpt \
    --max_num_hypotheses $NUM_HYP \
    --output_folder "./outputs/hypogenic/llamagc_detect" \
    --num_train 200 --num_test 100 --num_val 100 --seed 42 \
    --num_init 10 --k 5 --alpha 0.5 \
    4096 1e-5

# Persuasive Pairs
echo "Running: persuasive_pairs"
hypogenic_generation \
    --task_config_path "$DATA_DIR/real/persuasive_pairs/config.yaml" \
    --model_name "$MODEL_NAME" --model_type gpt \
    --max_num_hypotheses $NUM_HYP \
    --output_folder "./outputs/hypogenic/persuasive_pairs" \
    --num_train 200 --num_test 100 --num_val 100 --seed 42 \
    --num_init 10 --k 5 --alpha 0.5 \
    4096 1e-5

# Retweet
echo "Running: retweet"
hypogenic_generation \
    --task_config_path "$DATA_DIR/real/retweet/config.yaml" \
    --model_name "$MODEL_NAME" --model_type gpt \
    --max_num_hypotheses $NUM_HYP \
    --output_folder "./outputs/hypogenic/retweet" \
    --num_train 200 --num_test 100 --num_val 100 --seed 42 \
    --num_init 10 --k 5 --alpha 0.5 \
    4096 1e-5

# Journal Cross datasets
for variant in cross_journal_health_nips cross_journal_health_radiology cross_journal_nips_health cross_journal_nips_radiolody cross_journal_radiology_health cross_journal_radiology_nips; do
    echo "Running: journal_cross/$variant"
    hypogenic_generation \
        --task_config_path "$DATA_DIR/real/journal_cross/$variant/config.yaml" \
        --model_name "$MODEL_NAME" --model_type gpt \
        --max_num_hypotheses $NUM_HYP \
        --output_folder "./outputs/hypogenic/journal_cross_$variant" \
        --num_train 200 --num_test 100 --num_val 100 --seed 42 \
        --num_init 10 --k 5 --alpha 0.5 \
        4096 1e-5
done

# Journal Same datasets
for variant in same_journal_health same_journal_nips same_journal_radiology; do
    echo "Running: journal_same/$variant"
    hypogenic_generation \
        --task_config_path "$DATA_DIR/real/journal_same/$variant/config.yaml" \
        --model_name "$MODEL_NAME" --model_type gpt \
        --max_num_hypotheses $NUM_HYP \
        --output_folder "./outputs/hypogenic/journal_same_$variant" \
        --num_train 200 --num_test 100 --num_val 100 --seed 42 \
        --num_init 10 --k 5 --alpha 0.5 \
        4096 1e-5
done

# ==========================================
# SYNTHETIC DATASETS
# ==========================================
echo ""
echo "=== SYNTHETIC DATASETS ==="

# Shoe
echo "Running: synthetic/shoe"
hypogenic_generation \
    --task_config_path "$DATA_DIR/synthetic/shoe/config.yaml" \
    --model_name "$MODEL_NAME" --model_type gpt \
    --max_num_hypotheses $NUM_HYP \
    --output_folder "./outputs/hypogenic/synthetic_shoe" \
    --num_train 200 --num_test 100 --num_val 100 --seed 42 \
    --num_init 10 --k 5 --alpha 0.5 \
    4096 1e-5

# Shoe Two Level
for variant in hard simple; do
    echo "Running: synthetic/shoe_two_level/$variant"
    hypogenic_generation \
        --task_config_path "$DATA_DIR/synthetic/shoe_two_level/$variant/config.yaml" \
        --model_name "$MODEL_NAME" --model_type gpt \
        --max_num_hypotheses $NUM_HYP \
        --output_folder "./outputs/hypogenic/synthetic_shoe_two_level_$variant" \
        --num_train 200 --num_test 100 --num_val 100 --seed 42 \
        --num_init 10 --k 5 --alpha 0.5 \
        4096 1e-5
done

# Admission
for level in level_1 level_2 level_3 level_4; do
    for variant in $(ls "$DATA_DIR/synthetic/admission/$level/" 2>/dev/null); do
        if [ -f "$DATA_DIR/synthetic/admission/$level/$variant/config.yaml" ]; then
            echo "Running: synthetic/admission/$level/$variant"
            hypogenic_generation \
                --task_config_path "$DATA_DIR/synthetic/admission/$level/$variant/config.yaml" \
                --model_name "$MODEL_NAME" --model_type gpt \
                --max_num_hypotheses $NUM_HYP \
                --output_folder "./outputs/hypogenic/synthetic_admission_${level}_${variant}" \
                --num_train 200 --num_test 100 --num_val 100 --seed 42 \
                --num_init 10 --k 5 --alpha 0.5 \
                4096 1e-5
        fi
    done
done

# Admission Advanced
for level in level_1 level_2 level_3 level_4; do
    for variant in $(ls "$DATA_DIR/synthetic/admission_adv/$level/" 2>/dev/null); do
        if [ -f "$DATA_DIR/synthetic/admission_adv/$level/$variant/config.yaml" ]; then
            echo "Running: synthetic/admission_adv/$level/$variant"
            hypogenic_generation \
                --task_config_path "$DATA_DIR/synthetic/admission_adv/$level/$variant/config.yaml" \
                --model_name "$MODEL_NAME" --model_type gpt \
                --max_num_hypotheses $NUM_HYP \
                --output_folder "./outputs/hypogenic/synthetic_admission_adv_${level}_${variant}" \
                --num_train 200 --num_test 100 --num_val 100 --seed 42 \
                --num_init 10 --k 5 --alpha 0.5 \
                4096 1e-5
        fi
    done
done

# Election
for variant in $(ls "$DATA_DIR/synthetic/election/" 2>/dev/null); do
    if [ -f "$DATA_DIR/synthetic/election/$variant/config.yaml" ]; then
        echo "Running: synthetic/election/$variant"
        hypogenic_generation \
            --task_config_path "$DATA_DIR/synthetic/election/$variant/config.yaml" \
            --model_name "$MODEL_NAME" --model_type gpt \
            --max_num_hypotheses $NUM_HYP \
            --output_folder "./outputs/hypogenic/synthetic_election_$variant" \
            --num_train 200 --num_test 100 --num_val 100 --seed 42 \
            --num_init 10 --k 5 --alpha 0.5 \
            4096 1e-5
    fi
done

# Preference
for variant in $(ls "$DATA_DIR/synthetic/preference/" 2>/dev/null); do
    if [ -f "$DATA_DIR/synthetic/preference/$variant/config.yaml" ]; then
        echo "Running: synthetic/preference/$variant"
        hypogenic_generation \
            --task_config_path "$DATA_DIR/synthetic/preference/$variant/config.yaml" \
            --model_name "$MODEL_NAME" --model_type gpt \
            --max_num_hypotheses $NUM_HYP \
            --output_folder "./outputs/hypogenic/synthetic_preference_$variant" \
            --num_train 200 --num_test 100 --num_val 100 --seed 42 \
            --num_init 10 --k 5 --alpha 0.5 \
            4096 1e-5
    fi
done

echo ""
echo "=========================================="
echo "HypoGenic generation complete!"
echo "=========================================="

