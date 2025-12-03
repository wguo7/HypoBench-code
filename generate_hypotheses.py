#!/usr/bin/env python3
"""
HypoBench Hypothesis Generator
==============================

A unified command-line tool for generating hypotheses using HypoGenic and 
HypotheSAEs methods on all HypoBench datasets.

This script provides:
- One-liner commands for hypothesis generation
- Support for all real-world and synthetic datasets
- Configurable parameters (model, num_hypotheses, etc.)

Requirements:
    - pip install hypogenic
    - pip install hypothesaes
    - Set OPENAI_API_KEY environment variable

Usage Examples:
    # List all available datasets
    python generate_hypotheses.py --list
    
    # Generate HypoGenic command for a specific dataset
    python generate_hypotheses.py --method hypogenic --dataset deceptive_reviews --num_hypotheses 20
    
    # Generate HypotheSAEs command for a specific dataset
    python generate_hypotheses.py --method hypothesaes --dataset retweet --num_hypotheses 20
    
    # Print commands for all datasets
    python generate_hypotheses.py --method hypogenic --all

Author: HypoBench Team
License: MIT
"""

import os
import argparse
from pathlib import Path
from typing import Optional


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

# List of real-world datasets available in HypoBench
REAL_DATASETS = [
    "deceptive_reviews",  # Hotel review authenticity detection
    "dreaddit",           # Reddit stress detection
    "gptgc_detect",       # GPT-generated content detection
    "headline_binary",    # Sarcasm detection in headlines
    "llamagc_detect",     # LLaMA-generated content detection
    "persuasive_pairs",   # Persuasiveness comparison
    "retweet",            # Retweet prediction
]

# List of synthetic dataset families (each has multiple variants)
SYNTHETIC_DATASETS = [
    "shoe",        # Shoe color prediction (basic)
    "admission",   # University admission prediction
    "election",    # Election voting preference prediction
    "preference",  # User preference prediction
]

# HypotheSAEs requires specific field configurations for each dataset
# Only datasets with text-based features are supported
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
        'text_fields': ['first_tweet', 'second_tweet'],  # Pairwise comparison
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
        'text_fields': ['more_persuasive', 'less_persuasive'],  # Pairwise comparison
        'label_field': 'label',
        'label_map': {'first': 0, 'second': 1},
    },
}


# =============================================================================
# COMMAND GENERATION FUNCTIONS
# =============================================================================

def get_hypogenic_command(
    dataset: str,
    num_hypotheses: int = 20,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Generate a HypoGenic command for hypothesis generation on a dataset.
    
    HypoGenic uses an iterative refinement approach where hypotheses are
    generated, evaluated, and updated based on training data.
    
    Args:
        dataset: Name of the dataset (e.g., 'deceptive_reviews', 'shoe')
        num_hypotheses: Maximum number of hypotheses to generate (default: 20)
        model: LLM model name to use (default: 'gpt-4o-mini')
    
    Returns:
        A shell command string that can be executed to run HypoGenic,
        or a comment if the dataset config is not found.
    
    Example:
        >>> cmd = get_hypogenic_command('deceptive_reviews', 20)
        >>> print(cmd)
        hypogenic_generation --task_config_path ./real_datasets/deceptive_reviews/config.yaml ...
    """
    base_dir = Path(__file__).parent
    
    # Search for config.yaml in both real and synthetic dataset directories
    config_path = None
    for folder in ["real_datasets", "synthetic_datasets"]:
        potential_path = base_dir / folder / dataset / "config.yaml"
        if potential_path.exists():
            config_path = potential_path
            break
    
    if config_path is None:
        return f"# Config not found for {dataset}"
    
    output_dir = f"./outputs/hypogenic/{dataset}"
    
    # Build the HypoGenic generation command with recommended parameters
    cmd = f"""hypogenic_generation \\
    --task_config_path {config_path} \\
    --model_name {model} --model_type gpt \\
    --max_num_hypotheses {num_hypotheses} \\
    --output_folder {output_dir} \\
    --num_train 200 --num_test 100 --num_val 100 --seed 42 \\
    --num_init 10 --k 5 --alpha 0.5 \\
    4096 1e-5"""
    
    return cmd


def get_hypothesaes_command(
    dataset: str,
    num_hypotheses: int = 20,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Generate a HypotheSAEs command for hypothesis generation on a dataset.
    
    HypotheSAEs uses Sparse Autoencoders to identify interpretable features
    in text embeddings, then generates hypotheses by interpreting these features.
    
    Args:
        dataset: Name of the dataset (must be in HYPOTHESAES_CONFIGS)
        num_hypotheses: Maximum number of hypotheses to generate (default: 20)
        model: LLM model name for interpretation/annotation (default: 'gpt-4o-mini')
    
    Returns:
        A shell command string that can be executed to run HypotheSAEs,
        or a comment if the dataset is not supported.
    
    Note:
        HypotheSAEs only supports datasets with text-based features.
        Synthetic datasets with structured features (e.g., shoe) are not supported.
    
    Example:
        >>> cmd = get_hypothesaes_command('retweet', 20)
        >>> print(cmd)
        python hypothesaes_runner.py --dataset_name retweet ...
    """
    # Check if dataset is supported by HypotheSAEs
    if dataset not in HYPOTHESAES_CONFIGS:
        return f"# {dataset} not supported by HypotheSAEs (requires text-based features)"
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / "real_datasets" / dataset
    output_dir = f"./outputs/hypothesaes/{dataset}"
    
    # Build the HypotheSAEs runner command
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


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def list_datasets() -> None:
    """
    Print a formatted list of all available datasets in HypoBench.
    
    Displays:
        - Real-world datasets with HypotheSAEs support indicator
        - Synthetic dataset families
    """
    print("\n" + "=" * 60)
    print("HypoBench Datasets")
    print("=" * 60)
    
    print("\nReal-World Datasets:")
    print("-" * 40)
    for ds in REAL_DATASETS:
        # Indicate which datasets support HypotheSAEs
        sae_support = "✓ HypotheSAEs" if ds in HYPOTHESAES_CONFIGS else ""
        print(f"  - {ds} {sae_support}")
    
    print("\nSynthetic Datasets:")
    print("-" * 40)
    for ds in SYNTHETIC_DATASETS:
        print(f"  - {ds}/* (multiple variants)")
    
    print("\n" + "=" * 60)
    print("Note: HypotheSAEs requires text-based features.")
    print("      Synthetic datasets only support HypoGenic.")
    print("=" * 60 + "\n")


def print_all_commands(
    method: str,
    num_hypotheses: int,
    model: str
) -> None:
    """
    Print hypothesis generation commands for all datasets.
    
    Args:
        method: Either 'hypogenic' or 'hypothesaes'
        num_hypotheses: Number of hypotheses to generate
        model: LLM model name to use
    """
    print(f"\n# {method.upper()} Commands for All Datasets")
    print("# " + "=" * 58)
    print(f"# Model: {model}")
    print(f"# Hypotheses: {num_hypotheses}")
    print("# " + "=" * 58)
    
    # Combine real and synthetic datasets
    datasets = REAL_DATASETS + SYNTHETIC_DATASETS
    
    for dataset in datasets:
        print(f"\n# Dataset: {dataset}")
        if method == "hypogenic":
            print(get_hypogenic_command(dataset, num_hypotheses, model))
        elif method == "hypothesaes":
            print(get_hypothesaes_command(dataset, num_hypotheses, model))


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """
    Main entry point for the HypoBench hypothesis generator CLI.
    
    Parses command-line arguments and executes the appropriate action:
        - List datasets
        - Generate command for a specific dataset
        - Print commands for all datasets
    """
    parser = argparse.ArgumentParser(
        description="Generate hypotheses for HypoBench datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python generate_hypotheses.py --list
  
  # Generate HypoGenic command for deceptive_reviews
  python generate_hypotheses.py --method hypogenic --dataset deceptive_reviews
  
  # Generate HypotheSAEs command for retweet
  python generate_hypotheses.py --method hypothesaes --dataset retweet
  
  # Print all HypoGenic commands
  python generate_hypotheses.py --method hypogenic --all
  
  # Use a different model
  python generate_hypotheses.py --method hypogenic --dataset dreaddit --model gpt-4o
        """
    )
    
    parser.add_argument(
        '--method',
        choices=['hypogenic', 'hypothesaes'],
        help='Hypothesis generation method to use'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Name of the dataset to generate hypotheses for'
    )
    parser.add_argument(
        '--num_hypotheses',
        type=int,
        default=20,
        help='Maximum number of hypotheses to generate (default: 20)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='LLM model name to use (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available datasets'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Print commands for all datasets (requires --method)'
    )
    
    args = parser.parse_args()
    
    # Handle --list flag
    if args.list:
        list_datasets()
        return
    
    # Handle --all flag (requires --method)
    if args.all and args.method:
        print_all_commands(args.method, args.num_hypotheses, args.model)
        return
    
    # Handle single dataset command generation
    if args.method and args.dataset:
        if args.method == "hypogenic":
            print(get_hypogenic_command(args.dataset, args.num_hypotheses, args.model))
        elif args.method == "hypothesaes":
            print(get_hypothesaes_command(args.dataset, args.num_hypotheses, args.model))
        return
    
    # No valid action specified, show help
    parser.print_help()


if __name__ == '__main__':
    main()
