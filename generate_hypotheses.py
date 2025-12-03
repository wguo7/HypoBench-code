#!/usr/bin/env python3
"""
HypoBench Hypothesis Generator
Generate hypotheses using HypoGenic and HypotheSAEs for all HypoBench datasets.

Usage:
    python generate_hypotheses.py --method hypogenic --dataset deceptive_reviews --num_hypotheses 20
    python generate_hypotheses.py --method hypothesaes --dataset retweet --num_hypotheses 20
    python generate_hypotheses.py --list
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


# Dataset configurations for HypoBench
REAL_DATASETS = [
    "deceptive_reviews",
    "dreaddit", 
    "gptgc_detect",
    "headline_binary",
    "llamagc_detect",
    "persuasive_pairs",
    "retweet",
]

SYNTHETIC_DATASETS = [
    "shoe",
    "admission",
    "election", 
    "preference",
]

# HypotheSAEs supported datasets with their configurations
HYPOTHESAES_CONFIGS = {
    'deceptive_reviews': {
        'train_file': 'hotel_reviews_train.json',
        'test_file': 'hotel_reviews_test.json',
        'text_field': 'review_sentence',
        'label_field': 'label',
        'label_map': {'deceptive': 1, 'truthful': 0},
    },
    'retweet': {
        'train_file': 'retweet_train.json',
        'test_file': 'retweet_test.json',
        'text_fields': ['first_tweet', 'second_tweet'],
        'label_field': 'label',
        'label_map': {'first': 0, 'second': 1},
    },
    'dreaddit': {
        'train_file': 'dreaddit_ind_train.json',
        'test_file': 'dreaddit_ind_test.json',
        'text_field': 'text',
        'label_field': 'label',
        'label_map': {'has stress': 1, 'no stress': 0},
    },
    'headline_binary': {
        'train_file': 'headline_binary_train.json',
        'test_file': 'headline_binary_test.json',
        'text_field': 'headline',
        'label_field': 'label',
        'label_map': {'sarcastic': 1, 'not_sarcastic': 0},
    },
    'gptgc_detect': {
        'train_file': 'WP_aigc_detect_train.json',
        'test_file': 'WP_aigc_detect_test.json',
        'text_field': 'text',
        'label_field': 'label',
        'label_map': {'AI-generated': 1, 'human-written': 0},
    },
    'llamagc_detect': {
        'train_file': 'WP_aigc_detect_train.json',
        'test_file': 'WP_aigc_detect_test.json',
        'text_field': 'text',
        'label_field': 'label',
        'label_map': {'AI-generated': 1, 'human-written': 0},
    },
    'persuasive_pairs': {
        'train_file': 'persuasive_pairs_human_gt_ind_train.json',
        'test_file': 'persuasive_pairs_human_gt_ind_test.json',
        'text_fields': ['more_persuasive', 'less_persuasive'],
        'label_field': 'label',
        'label_map': {'first': 0, 'second': 1},
    },
}


def get_hypogenic_command(dataset: str, num_hypotheses: int = 20, model: str = "gpt-4o-mini") -> str:
    """Generate HypoGenic command for a dataset."""
    base_dir = Path(__file__).parent
    
    # Check real_datasets first, then synthetic_datasets
    config_path = None
    for folder in ["real_datasets", "synthetic_datasets"]:
        potential_path = base_dir / folder / dataset / "config.yaml"
        if potential_path.exists():
            config_path = potential_path
            break
    
    if config_path is None:
        return f"# Config not found for {dataset}"
    
    output_dir = f"./outputs/hypogenic/{dataset}"
    
    cmd = f"""hypogenic_generation \\
    --task_config_path {config_path} \\
    --model_name {model} --model_type gpt \\
    --max_num_hypotheses {num_hypotheses} \\
    --output_folder {output_dir} \\
    --num_train 200 --num_test 100 --num_val 100 --seed 42 \\
    --num_init 10 --k 5 --alpha 0.5 \\
    4096 1e-5"""
    
    return cmd


def get_hypothesaes_command(dataset: str, num_hypotheses: int = 20, model: str = "gpt-4o-mini") -> str:
    """Generate HypotheSAEs command for a dataset."""
    if dataset not in HYPOTHESAES_CONFIGS:
        return f"# {dataset} not supported by HypotheSAEs"
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / "real_datasets" / dataset
    output_dir = f"./outputs/hypothesaes/{dataset}"
    
    cmd = f"""python hypothesaes_runner.py \\
    --dataset_name {dataset} \\
    --data_dir {data_dir} \\
    --output_dir {output_dir} \\
    --num_train 200 --num_test 300 \\
    --max_num_hypotheses {num_hypotheses} \\
    --interpreter_model {model} \\
    --annotator_model {model} \\
    --seed 42"""
    
    return cmd


def list_datasets():
    """List all available datasets."""
    print("\n" + "=" * 60)
    print("HypoBench Datasets")
    print("=" * 60)
    
    print("\nReal-World Datasets:")
    for ds in REAL_DATASETS:
        sae_support = "✓ HypotheSAEs" if ds in HYPOTHESAES_CONFIGS else ""
        print(f"  - {ds} {sae_support}")
    
    print("\nSynthetic Datasets:")
    for ds in SYNTHETIC_DATASETS:
        print(f"  - {ds}")
    
    print("\n" + "=" * 60)


def print_all_commands(method: str, num_hypotheses: int, model: str):
    """Print all commands for a method."""
    print(f"\n# {method.upper()} Commands for All Datasets")
    print("# " + "=" * 58)
    
    datasets = REAL_DATASETS + SYNTHETIC_DATASETS
    
    for dataset in datasets:
        print(f"\n# Dataset: {dataset}")
        if method == "hypogenic":
            print(get_hypogenic_command(dataset, num_hypotheses, model))
        elif method == "hypothesaes":
            print(get_hypothesaes_command(dataset, num_hypotheses, model))


def main():
    parser = argparse.ArgumentParser(description="Generate hypotheses for HypoBench datasets")
    parser.add_argument('--method', choices=['hypogenic', 'hypothesaes'], 
                        help='Hypothesis generation method')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--num_hypotheses', type=int, default=20)
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--list', action='store_true', help='List all datasets')
    parser.add_argument('--all', action='store_true', help='Print commands for all datasets')
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    if args.all and args.method:
        print_all_commands(args.method, args.num_hypotheses, args.model)
        return
    
    if args.method and args.dataset:
        if args.method == "hypogenic":
            print(get_hypogenic_command(args.dataset, args.num_hypotheses, args.model))
        elif args.method == "hypothesaes":
            print(get_hypothesaes_command(args.dataset, args.num_hypotheses, args.model))
        return
    
    parser.print_help()


if __name__ == '__main__':
    main()

