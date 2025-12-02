#!/usr/bin/env python3
"""
HypoBench Hypothesis Generator
Unified script for generating hypotheses using HypoGenic and HypotheSAEs on all HypoBench datasets.

Usage:
    # Generate hypotheses for all datasets using HypoGenic
    python hypobench_generator.py --method hypogenic --num_hypotheses 20

    # Generate hypotheses for a specific dataset using HypotheSAEs
    python hypobench_generator.py --method hypothesaes --dataset deceptive_reviews --num_hypotheses 20

    # List all available datasets
    python hypobench_generator.py --list-datasets
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DatasetInfo:
    """Information about a HypoBench dataset."""
    name: str
    config_path: str
    data_dir: str
    category: str  # 'real' or 'synthetic'
    subcategory: Optional[str] = None  # e.g., 'admission', 'election', etc.


class HypoBenchDatasetDiscovery:
    """Discovers all datasets in HypoBench."""
    
    def __init__(self, base_path: str = None):
        """Initialize dataset discovery.
        
        Args:
            base_path: Base path to the hypothesis_comparison directory.
        """
        if base_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "hypogenic" / "data"
        
    def discover_all_datasets(self) -> List[DatasetInfo]:
        """Discover all datasets in HypoBench.
        
        Returns:
            List of DatasetInfo objects for all discovered datasets.
        """
        datasets = []
        
        # Discover real datasets
        real_path = self.data_path / "real"
        if real_path.exists():
            datasets.extend(self._discover_real_datasets(real_path))
        
        # Discover synthetic datasets
        synthetic_path = self.data_path / "synthetic"
        if synthetic_path.exists():
            datasets.extend(self._discover_synthetic_datasets(synthetic_path))
            
        return datasets
    
    def _discover_real_datasets(self, real_path: Path) -> List[DatasetInfo]:
        """Discover real-world datasets."""
        datasets = []
        
        for item in real_path.iterdir():
            if item.is_dir():
                config_file = item / "config.yaml"
                if config_file.exists():
                    datasets.append(DatasetInfo(
                        name=item.name,
                        config_path=str(config_file),
                        data_dir=str(item),
                        category='real'
                    ))
                else:
                    # Check for nested directories (journal_cross, journal_same)
                    for sub_item in item.iterdir():
                        if sub_item.is_dir():
                            sub_config = sub_item / "config.yaml"
                            if sub_config.exists():
                                datasets.append(DatasetInfo(
                                    name=f"{item.name}/{sub_item.name}",
                                    config_path=str(sub_config),
                                    data_dir=str(sub_item),
                                    category='real',
                                    subcategory=item.name
                                ))
        return datasets
    
    def _discover_synthetic_datasets(self, synthetic_path: Path) -> List[DatasetInfo]:
        """Discover synthetic datasets."""
        datasets = []
        
        for item in synthetic_path.iterdir():
            if item.is_dir():
                config_file = item / "config.yaml"
                if config_file.exists():
                    # Simple synthetic dataset (e.g., shoe)
                    datasets.append(DatasetInfo(
                        name=f"synthetic/{item.name}",
                        config_path=str(config_file),
                        data_dir=str(item),
                        category='synthetic',
                        subcategory=item.name
                    ))
                else:
                    # Nested synthetic dataset (e.g., admission, election, preference)
                    datasets.extend(self._discover_nested_synthetic(item))
        
        return datasets
    
    def _discover_nested_synthetic(self, parent: Path) -> List[DatasetInfo]:
        """Discover nested synthetic datasets."""
        datasets = []
        
        for item in parent.iterdir():
            if item.is_dir():
                config_file = item / "config.yaml"
                if config_file.exists():
                    datasets.append(DatasetInfo(
                        name=f"synthetic/{parent.name}/{item.name}",
                        config_path=str(config_file),
                        data_dir=str(item),
                        category='synthetic',
                        subcategory=parent.name
                    ))
                else:
                    # Even deeper nesting (e.g., admission/level_1/base)
                    for sub_item in item.iterdir():
                        if sub_item.is_dir():
                            sub_config = sub_item / "config.yaml"
                            if sub_config.exists():
                                datasets.append(DatasetInfo(
                                    name=f"synthetic/{parent.name}/{item.name}/{sub_item.name}",
                                    config_path=str(sub_config),
                                    data_dir=str(sub_item),
                                    category='synthetic',
                                    subcategory=parent.name
                                ))
        
        return datasets


class HypoGenicRunner:
    """Runner for HypoGenic hypothesis generation."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        model_type: str = "gpt",
        num_train: int = 200,
        num_test: int = 100,
        num_val: int = 100,
        seed: int = 42
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.num_train = num_train
        self.num_test = num_test
        self.num_val = num_val
        self.seed = seed
        
    def generate_hypotheses(
        self,
        dataset: DatasetInfo,
        num_hypotheses: int = 20,
        output_dir: str = None
    ) -> str:
        """Generate hypotheses using HypoGenic.
        
        Args:
            dataset: Dataset to generate hypotheses for.
            num_hypotheses: Number of hypotheses to generate.
            output_dir: Output directory for results.
            
        Returns:
            Path to the output file.
        """
        if output_dir is None:
            output_dir = f"./outputs/hypogenic/{dataset.name.replace('/', '_')}"
        
        # Build command for hypogenic_generation
        cmd = [
            "hypogenic_generation",
            "--task_config_path", dataset.config_path,
            "--model_name", self.model_name,
            "--model_type", self.model_type,
            "--max_num_hypotheses", str(num_hypotheses),
            "--output_folder", output_dir,
            "--num_train", str(self.num_train),
            "--num_test", str(self.num_test),
            "--num_val", str(self.num_val),
            "--seed", str(self.seed),
            "--num_init", "10",
            "--k", "5",
            "--alpha", "0.5",
            "--update_batch_size", "10",
            "--init_batch_size", "10",
            "--init_hypotheses_per_batch", "10",
            "4096",  # max_tokens (positional)
            "1e-5",  # temperature (positional)
        ]
        
        return cmd, output_dir


class HypotheSAEsRunner:
    """Runner for HypotheSAEs hypothesis generation."""
    
    # Dataset configurations for HypotheSAEs
    DATASET_CONFIGS = {
        'deceptive_reviews': {
            'train_file': 'hotel_reviews_train.json',
            'test_file': 'hotel_reviews_test.json',
            'val_file': 'hotel_reviews_val.json',
            'text_field': 'review_sentence',
            'label_field': 'label',
            'label_map': {'deceptive': 1, 'truthful': 0},
            'M': 512, 'K': 8,
            'task_instructions': """All texts are hotel reviews.
Features should describe specific aspects of review authenticity."""
        },
        'retweet': {
            'train_file': 'retweet_train.json',
            'test_file': 'retweet_test.json',
            'val_file': 'retweet_val.json',
            'text_fields': ['first_tweet', 'second_tweet'],
            'label_field': 'label',
            'label_map': {'first': 0, 'second': 1},
            'M': 512, 'K': 8,
            'task_instructions': """All texts are pairs of tweets about the same content.
Features should describe linguistic patterns that predict retweet likelihood."""
        },
        'dreaddit': {
            'train_file': 'dreaddit_ind_train.json',
            'test_file': 'dreaddit_ind_test.json',
            'val_file': 'dreaddit_ind_val.json',
            'text_field': 'text',
            'label_field': 'label',
            'label_map': {'has stress': 1, 'no stress': 0},
            'M': 512, 'K': 8,
            'task_instructions': """All texts are Reddit posts.
Features should describe patterns indicating stress levels."""
        },
        'headline_binary': {
            'train_file': 'headline_binary_train.json',
            'test_file': 'headline_binary_test.json',
            'val_file': None,
            'text_field': 'headline',
            'label_field': 'label',
            'label_map': {'sarcastic': 1, 'not sarcastic': 0},
            'M': 512, 'K': 8,
            'task_instructions': """All texts are news headlines.
Features should describe patterns indicating sarcasm."""
        },
        'gptgc_detect': {
            'train_file': 'WP_aigc_detect_train.json',
            'test_file': 'WP_aigc_detect_test.json',
            'val_file': 'WP_aigc_detect_val.json',
            'text_field': 'text',
            'label_field': 'label',
            'label_map': {'AI-generated': 1, 'human-written': 0},
            'M': 512, 'K': 8,
            'task_instructions': """All texts are writing samples.
Features should describe patterns distinguishing AI-generated from human-written text."""
        },
        'llamagc_detect': {
            'train_file': 'WP_aigc_detect_train.json',
            'test_file': 'WP_aigc_detect_test.json',
            'val_file': 'WP_aigc_detect_val.json',
            'text_field': 'text',
            'label_field': 'label',
            'label_map': {'AI-generated': 1, 'human-written': 0},
            'M': 512, 'K': 8,
            'task_instructions': """All texts are writing samples.
Features should describe patterns distinguishing AI-generated from human-written text."""
        },
        'persuasive_pairs': {
            'train_file': 'persuasive_pairs_human_gt_ind_train.json',
            'test_file': 'persuasive_pairs_human_gt_ind_test.json',
            'val_file': 'persuasive_pairs_human_gt_ind_val.json',
            'text_fields': ['more_persuasive', 'less_persuasive'],
            'label_field': 'label',
            'label_map': {'first': 0, 'second': 1},
            'M': 512, 'K': 8,
            'task_instructions': """All texts are pairs of persuasive texts.
Features should describe patterns that make text more persuasive."""
        },
    }
    
    def __init__(
        self,
        embedder: str = "text-embedding-3-small",
        interpreter_model: str = "gpt-4o-mini",
        annotator_model: str = "gpt-4o-mini",
        num_train: int = 200,
        num_test: int = 300,
        seed: int = 42
    ):
        self.embedder = embedder
        self.interpreter_model = interpreter_model
        self.annotator_model = annotator_model
        self.num_train = num_train
        self.num_test = num_test
        self.seed = seed
    
    def generate_command(
        self,
        dataset: DatasetInfo,
        num_hypotheses: int = 20,
        output_dir: str = None
    ) -> Tuple[List[str], str]:
        """Generate command for HypotheSAEs.
        
        Args:
            dataset: Dataset to generate hypotheses for.
            num_hypotheses: Number of hypotheses to generate.
            output_dir: Output directory for results.
            
        Returns:
            Tuple of (command list, output directory).
        """
        # Get base dataset name (remove synthetic/ prefix if present)
        base_name = dataset.name.split('/')[-1] if '/' in dataset.name else dataset.name
        
        if output_dir is None:
            output_dir = f"./outputs/hypothesaes/{dataset.name.replace('/', '_')}"
        
        cmd = [
            "python", "comparison/hypothesaes_runner.py",
            "--dataset_name", base_name,
            "--data_dir", dataset.data_dir,
            "--output_dir", output_dir,
            "--num_train", str(self.num_train),
            "--num_test", str(self.num_test),
            "--max_num_hypotheses", str(num_hypotheses),
            "--embedder", self.embedder,
            "--interpreter_model", self.interpreter_model,
            "--annotator_model", self.annotator_model,
            "--seed", str(self.seed),
        ]
        
        return cmd, output_dir
    
    def is_supported(self, dataset: DatasetInfo) -> bool:
        """Check if dataset is supported by HypotheSAEs."""
        base_name = dataset.name.split('/')[-1] if '/' in dataset.name else dataset.name
        return base_name in self.DATASET_CONFIGS


def generate_batch_commands(
    datasets: List[DatasetInfo],
    method: str,
    num_hypotheses: int = 20,
    model_name: str = "gpt-4o-mini"
) -> List[Tuple[str, List[str]]]:
    """Generate batch commands for all datasets.
    
    Args:
        datasets: List of datasets.
        method: 'hypogenic' or 'hypothesaes'.
        num_hypotheses: Number of hypotheses to generate.
        model_name: Model name for generation.
        
    Returns:
        List of (dataset_name, command) tuples.
    """
    commands = []
    
    if method == 'hypogenic':
        runner = HypoGenicRunner(model_name=model_name)
        for dataset in datasets:
            cmd, _ = runner.generate_hypotheses(dataset, num_hypotheses)
            commands.append((dataset.name, cmd))
    
    elif method == 'hypothesaes':
        runner = HypotheSAEsRunner(interpreter_model=model_name, annotator_model=model_name)
        for dataset in datasets:
            if runner.is_supported(dataset):
                cmd, _ = runner.generate_command(dataset, num_hypotheses)
                commands.append((dataset.name, cmd))
            else:
                print(f"Warning: {dataset.name} not supported by HypotheSAEs, skipping...")
    
    return commands


def print_dataset_list(datasets: List[DatasetInfo]):
    """Print list of all datasets."""
    print("\n" + "=" * 80)
    print("HypoBench Datasets")
    print("=" * 80)
    
    # Group by category
    real_datasets = [d for d in datasets if d.category == 'real']
    synthetic_datasets = [d for d in datasets if d.category == 'synthetic']
    
    print(f"\nReal-World Datasets ({len(real_datasets)}):")
    print("-" * 40)
    for d in sorted(real_datasets, key=lambda x: x.name):
        print(f"  - {d.name}")
    
    print(f"\nSynthetic Datasets ({len(synthetic_datasets)}):")
    print("-" * 40)
    
    # Group synthetic by subcategory
    by_subcategory = {}
    for d in synthetic_datasets:
        subcat = d.subcategory or 'other'
        if subcat not in by_subcategory:
            by_subcategory[subcat] = []
        by_subcategory[subcat].append(d)
    
    for subcat in sorted(by_subcategory.keys()):
        print(f"\n  [{subcat}]")
        for d in sorted(by_subcategory[subcat], key=lambda x: x.name):
            print(f"    - {d.name}")
    
    print(f"\nTotal: {len(datasets)} datasets")
    print("=" * 80)


def save_commands_to_file(
    commands: List[Tuple[str, List[str]]],
    output_file: str,
    method: str
):
    """Save commands to a shell script file.
    
    Args:
        commands: List of (dataset_name, command) tuples.
        output_file: Output file path.
        method: Method name for header.
    """
    with open(output_file, 'w') as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"# HypoBench {method} hypothesis generation commands\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Total datasets: {len(commands)}\n\n")
        
        for dataset_name, cmd in commands:
            f.write(f"# Dataset: {dataset_name}\n")
            f.write(" ".join(cmd) + "\n\n")
    
    print(f"Commands saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="HypoBench Hypothesis Generator - Generate hypotheses using HypoGenic or HypotheSAEs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python hypobench_generator.py --list-datasets
  
  # Generate one-liner commands for HypoGenic on all datasets
  python hypobench_generator.py --method hypogenic --num_hypotheses 20 --save-commands
  
  # Generate commands for HypotheSAEs on supported datasets
  python hypobench_generator.py --method hypothesaes --num_hypotheses 20 --save-commands
  
  # Run HypoGenic on a specific dataset
  python hypobench_generator.py --method hypogenic --dataset deceptive_reviews --run
  
  # Filter by category
  python hypobench_generator.py --method hypogenic --category real --save-commands
"""
    )
    
    parser.add_argument('--list-datasets', action='store_true',
                        help='List all available datasets')
    parser.add_argument('--method', type=str, choices=['hypogenic', 'hypothesaes', 'both'],
                        help='Hypothesis generation method')
    parser.add_argument('--dataset', type=str,
                        help='Specific dataset to run (use full name like "deceptive_reviews" or "synthetic/shoe")')
    parser.add_argument('--category', type=str, choices=['real', 'synthetic', 'all'],
                        default='all', help='Dataset category to process')
    parser.add_argument('--num_hypotheses', type=int, default=20,
                        help='Number of hypotheses to generate')
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini',
                        help='Model name for generation')
    parser.add_argument('--save-commands', action='store_true',
                        help='Save commands to shell script files')
    parser.add_argument('--run', action='store_true',
                        help='Actually run the generation (otherwise just prints commands)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Base output directory for results')
    
    args = parser.parse_args()
    
    # Discover datasets
    discovery = HypoBenchDatasetDiscovery()
    all_datasets = discovery.discover_all_datasets()
    
    # Filter by category
    if args.category != 'all':
        all_datasets = [d for d in all_datasets if d.category == args.category]
    
    # List datasets if requested
    if args.list_datasets:
        print_dataset_list(all_datasets)
        return
    
    # Filter by specific dataset if provided
    if args.dataset:
        all_datasets = [d for d in all_datasets if args.dataset in d.name]
        if not all_datasets:
            print(f"Error: Dataset '{args.dataset}' not found")
            return
    
    # Require method for generation
    if not args.method:
        parser.print_help()
        return
    
    # Generate commands
    methods = ['hypogenic', 'hypothesaes'] if args.method == 'both' else [args.method]
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Generating commands for {method.upper()}")
        print(f"{'='*80}")
        
        commands = generate_batch_commands(
            all_datasets,
            method,
            args.num_hypotheses,
            args.model_name
        )
        
        if args.save_commands:
            output_file = f"run_{method}_all.sh"
            save_commands_to_file(commands, output_file, method)
            
            # Also save PowerShell version
            ps_file = f"run_{method}_all.ps1"
            with open(ps_file, 'w') as f:
                f.write(f"# HypoBench {method} hypothesis generation commands\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Total datasets: {len(commands)}\n\n")
                
                for dataset_name, cmd in commands:
                    f.write(f"# Dataset: {dataset_name}\n")
                    # Convert to PowerShell format
                    cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
                    f.write(f"{cmd_str}\n\n")
            print(f"PowerShell commands saved to: {ps_file}")
        
        if args.run:
            for dataset_name, cmd in commands:
                print(f"\nRunning {method} on {dataset_name}...")
                try:
                    subprocess.run(cmd, check=True)
                    print(f"Completed: {dataset_name}")
                except subprocess.CalledProcessError as e:
                    print(f"Error running {dataset_name}: {e}")
                except FileNotFoundError:
                    print(f"Error: Command not found. Make sure {method} is installed.")
                    break
        else:
            print(f"\nGenerated {len(commands)} commands for {method}")
            print("\nSample commands:")
            for name, cmd in commands[:3]:
                print(f"\n# {name}")
                print(" ".join(cmd))
            if len(commands) > 3:
                print(f"\n... and {len(commands) - 3} more")


if __name__ == '__main__':
    main()

