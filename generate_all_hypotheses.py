#!/usr/bin/env python3
"""
One-liner hypothesis generation for HypoBench.

This script provides simple functions to generate 20 hypotheses for any dataset
using either HypoGenic or HypotheSAEs.

Usage:
    # Generate hypotheses for a dataset using HypoGenic
    python generate_all_hypotheses.py hypogenic <config_path> [--num_hypotheses 20]
    
    # Generate hypotheses for a dataset using HypotheSAEs  
    python generate_all_hypotheses.py hypothesaes <dataset_name> <data_dir> [--num_hypotheses 20]
    
    # Generate for all real datasets
    python generate_all_hypotheses.py all-real --method both
    
    # Generate for all synthetic datasets
    python generate_all_hypotheses.py all-synthetic --method hypogenic
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hypogenic'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hypothesaes'))


def run_hypogenic(
    config_path: str,
    num_hypotheses: int = 20,
    output_dir: str = None,
    model_name: str = "gpt-4o-mini",
    model_type: str = "gpt",
    num_train: int = 200,
    num_test: int = 100,
    num_val: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate hypotheses using HypoGenic.
    
    Args:
        config_path: Path to dataset config.yaml file.
        num_hypotheses: Maximum number of hypotheses to generate.
        output_dir: Output directory for results.
        model_name: LLM model name.
        model_type: Model type (gpt, claude, vllm, huggingface).
        num_train: Number of training samples.
        num_test: Number of test samples.
        num_val: Number of validation samples.
        seed: Random seed.
        
    Returns:
        Dictionary with generated hypotheses and metadata.
    """
    from hypogenic.extract_label import extract_label_register
    from hypogenic.tasks import BaseTask
    from hypogenic.prompt import BasePrompt
    from hypogenic.utils import set_seed
    from hypogenic.LLM_wrapper import llm_wrapper_register
    from hypogenic.algorithm.summary_information import SummaryInformation
    from hypogenic.algorithm.generation import DefaultGeneration
    from hypogenic.algorithm.inference import DefaultInference
    from hypogenic.algorithm.replace import DefaultReplace
    from hypogenic.algorithm.update import DefaultUpdate
    from hypogenic.logger_config import LoggerConfig
    
    LoggerConfig.setup_logger(level=logging.INFO)
    logger = LoggerConfig.get_logger("HypoGenic")
    
    start_time = time.time()
    
    # Load task
    task = BaseTask(config_path, extract_label=None, from_register=extract_label_register)
    
    # Set output directory
    if output_dir is None:
        output_dir = f"./outputs/hypogenic/{task.task_name.replace('/', '_')}/{model_name}/hyp_{num_hypotheses}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize API
    api = llm_wrapper_register.build(model_type)(model=model_name, path_name=None)
    
    # Set seed and get data
    set_seed(seed)
    train_data, _, _ = task.get_data(num_train, num_test, num_val, seed)
    
    # Initialize components
    prompt_class = BasePrompt(task)
    inference_class = DefaultInference(api, prompt_class, train_data, task)
    generation_class = DefaultGeneration(api, prompt_class, inference_class, task)
    
    update_class = DefaultUpdate(
        generation_class=generation_class,
        inference_class=inference_class,
        replace_class=DefaultReplace(num_hypotheses),
        save_path=output_dir,
        num_init=10,
        k=5,
        alpha=0.5,
        update_batch_size=10,
        num_hypotheses_to_update=1,
        save_every_n_examples=10,
    )
    
    # Generate hypotheses
    hypotheses_bank = update_class.batched_initialize_hypotheses(
        num_init=10,
        init_batch_size=10,
        init_hypotheses_per_batch=10,
        cache_seed=None,
        temperature=1e-5,
        max_tokens=4096,
    )
    
    update_class.save_to_json(hypotheses_bank, sample=10, seed=seed, epoch=0)
    
    # Update hypotheses
    hypotheses_bank = update_class.update(
        current_epoch=0,
        hypotheses_bank=hypotheses_bank,
        current_seed=seed,
        cache_seed=None,
    )
    
    update_class.save_to_json(hypotheses_bank, sample="final", seed=seed, epoch=0)
    
    runtime = time.time() - start_time
    logger.info(f"Total time: {runtime:.2f} seconds")
    
    # Compile results
    results = {
        'method': 'HypoGenic',
        'task_name': task.task_name,
        'config_path': config_path,
        'num_hypotheses': len(hypotheses_bank),
        'hypotheses': [
            {
                'hypothesis': hyp,
                'info': {
                    'accuracy': info.acc,
                    'reward': info.reward,
                    'num_correct': info.num_correct,
                    'num_wrong': info.num_wrong,
                }
            }
            for hyp, info in hypotheses_bank.items()
        ],
        'config': {
            'model_name': model_name,
            'model_type': model_type,
            'num_train': num_train,
            'num_test': num_test,
            'num_val': num_val,
            'seed': seed,
        },
        'runtime_seconds': runtime,
        'timestamp': datetime.now().isoformat(),
        'output_dir': output_dir,
    }
    
    # Save results
    results_file = os.path.join(output_dir, 'hypogenic_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"HypoGenic Results: {task.task_name}")
    print(f"{'='*60}")
    print(f"Hypotheses generated: {len(hypotheses_bank)}")
    print(f"Runtime: {runtime:.1f} seconds")
    print(f"Results saved to: {results_file}")
    
    return results


def run_hypothesaes(
    dataset_name: str,
    data_dir: str,
    num_hypotheses: int = 20,
    output_dir: str = None,
    embedder: str = "text-embedding-3-small",
    interpreter_model: str = "gpt-4o-mini",
    annotator_model: str = "gpt-4o-mini",
    num_train: int = 200,
    num_test: int = 300,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate hypotheses using HypotheSAEs.
    
    Args:
        dataset_name: Name of the dataset.
        data_dir: Directory containing dataset files.
        num_hypotheses: Maximum number of hypotheses to generate.
        output_dir: Output directory for results.
        embedder: Embedding model name.
        interpreter_model: LLM for interpreting neurons.
        annotator_model: LLM for annotating texts.
        num_train: Number of training samples.
        num_test: Number of test samples.
        seed: Random seed.
        
    Returns:
        Dictionary with generated hypotheses and metadata.
    """
    from comparison.hypothesaes_runner import HypotheSAEsRunner
    
    if output_dir is None:
        output_dir = f"./outputs/hypothesaes/{dataset_name}"
    
    runner = HypotheSAEsRunner(
        dataset_name=dataset_name,
        data_dir=data_dir,
        output_dir=output_dir,
        num_train=num_train,
        num_test=num_test,
        seed=seed
    )
    
    results = runner.run_pipeline(
        max_num_hypotheses=num_hypotheses,
        embedder=embedder,
        interpreter_model=interpreter_model,
        annotator_model=annotator_model
    )
    
    return results


def get_all_datasets() -> Dict[str, List[Dict[str, str]]]:
    """Get all available datasets organized by category."""
    base_path = Path(__file__).parent / "hypogenic" / "data"
    
    datasets = {
        'real': [],
        'synthetic': []
    }
    
    # Real datasets
    real_path = base_path / "real"
    if real_path.exists():
        for item in real_path.iterdir():
            if item.is_dir():
                config = item / "config.yaml"
                if config.exists():
                    datasets['real'].append({
                        'name': item.name,
                        'config_path': str(config),
                        'data_dir': str(item)
                    })
                else:
                    # Nested (journal datasets)
                    for sub in item.iterdir():
                        if sub.is_dir():
                            sub_config = sub / "config.yaml"
                            if sub_config.exists():
                                datasets['real'].append({
                                    'name': f"{item.name}/{sub.name}",
                                    'config_path': str(sub_config),
                                    'data_dir': str(sub)
                                })
    
    # Synthetic datasets
    synth_path = base_path / "synthetic"
    if synth_path.exists():
        for item in synth_path.iterdir():
            if item.is_dir():
                config = item / "config.yaml"
                if config.exists():
                    datasets['synthetic'].append({
                        'name': f"synthetic/{item.name}",
                        'config_path': str(config),
                        'data_dir': str(item)
                    })
                else:
                    # Discover nested
                    for sub in item.iterdir():
                        if sub.is_dir():
                            sub_config = sub / "config.yaml"
                            if sub_config.exists():
                                datasets['synthetic'].append({
                                    'name': f"synthetic/{item.name}/{sub.name}",
                                    'config_path': str(sub_config),
                                    'data_dir': str(sub)
                                })
                            else:
                                # Even deeper nesting
                                for deep in sub.iterdir():
                                    if deep.is_dir():
                                        deep_config = deep / "config.yaml"
                                        if deep_config.exists():
                                            datasets['synthetic'].append({
                                                'name': f"synthetic/{item.name}/{sub.name}/{deep.name}",
                                                'config_path': str(deep_config),
                                                'data_dir': str(deep)
                                            })
    
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Generate hypotheses using HypoGenic or HypotheSAEs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # HypoGenic command
    hypogenic_parser = subparsers.add_parser('hypogenic', help='Run HypoGenic')
    hypogenic_parser.add_argument('config_path', type=str, help='Path to config.yaml')
    hypogenic_parser.add_argument('--num_hypotheses', type=int, default=20)
    hypogenic_parser.add_argument('--output_dir', type=str, default=None)
    hypogenic_parser.add_argument('--model_name', type=str, default='gpt-4o-mini')
    hypogenic_parser.add_argument('--model_type', type=str, default='gpt')
    hypogenic_parser.add_argument('--num_train', type=int, default=200)
    hypogenic_parser.add_argument('--seed', type=int, default=42)
    
    # HypotheSAEs command
    hypothesaes_parser = subparsers.add_parser('hypothesaes', help='Run HypotheSAEs')
    hypothesaes_parser.add_argument('dataset_name', type=str, help='Dataset name')
    hypothesaes_parser.add_argument('data_dir', type=str, help='Data directory')
    hypothesaes_parser.add_argument('--num_hypotheses', type=int, default=20)
    hypothesaes_parser.add_argument('--output_dir', type=str, default=None)
    hypothesaes_parser.add_argument('--embedder', type=str, default='text-embedding-3-small')
    hypothesaes_parser.add_argument('--model', type=str, default='gpt-4o-mini')
    hypothesaes_parser.add_argument('--num_train', type=int, default=200)
    hypothesaes_parser.add_argument('--seed', type=int, default=42)
    
    # Run all real datasets
    all_real_parser = subparsers.add_parser('all-real', help='Run on all real datasets')
    all_real_parser.add_argument('--method', type=str, choices=['hypogenic', 'hypothesaes', 'both'], default='both')
    all_real_parser.add_argument('--num_hypotheses', type=int, default=20)
    all_real_parser.add_argument('--model', type=str, default='gpt-4o-mini')
    
    # Run all synthetic datasets
    all_synth_parser = subparsers.add_parser('all-synthetic', help='Run on all synthetic datasets')
    all_synth_parser.add_argument('--method', type=str, choices=['hypogenic', 'hypothesaes', 'both'], default='hypogenic')
    all_synth_parser.add_argument('--num_hypotheses', type=int, default=20)
    all_synth_parser.add_argument('--model', type=str, default='gpt-4o-mini')
    
    # List datasets
    list_parser = subparsers.add_parser('list', help='List all datasets')
    
    args = parser.parse_args()
    
    if args.command == 'hypogenic':
        run_hypogenic(
            config_path=args.config_path,
            num_hypotheses=args.num_hypotheses,
            output_dir=args.output_dir,
            model_name=args.model_name,
            model_type=args.model_type,
            num_train=args.num_train,
            seed=args.seed
        )
    
    elif args.command == 'hypothesaes':
        run_hypothesaes(
            dataset_name=args.dataset_name,
            data_dir=args.data_dir,
            num_hypotheses=args.num_hypotheses,
            output_dir=args.output_dir,
            embedder=args.embedder,
            interpreter_model=args.model,
            annotator_model=args.model,
            num_train=args.num_train,
            seed=args.seed
        )
    
    elif args.command == 'all-real':
        datasets = get_all_datasets()['real']
        print(f"Found {len(datasets)} real datasets")
        
        for ds in datasets:
            print(f"\n{'='*60}")
            print(f"Processing: {ds['name']}")
            print(f"{'='*60}")
            
            if args.method in ['hypogenic', 'both']:
                try:
                    run_hypogenic(
                        config_path=ds['config_path'],
                        num_hypotheses=args.num_hypotheses,
                        model_name=args.model
                    )
                except Exception as e:
                    print(f"Error with HypoGenic on {ds['name']}: {e}")
            
            if args.method in ['hypothesaes', 'both']:
                # Only supported datasets for HypotheSAEs
                base_name = ds['name'].split('/')[-1]
                supported = ['deceptive_reviews', 'retweet', 'dreaddit', 'headline_binary', 
                            'gptgc_detect', 'llamagc_detect', 'persuasive_pairs']
                if base_name in supported:
                    try:
                        run_hypothesaes(
                            dataset_name=base_name,
                            data_dir=ds['data_dir'],
                            num_hypotheses=args.num_hypotheses,
                            interpreter_model=args.model,
                            annotator_model=args.model
                        )
                    except Exception as e:
                        print(f"Error with HypotheSAEs on {ds['name']}: {e}")
    
    elif args.command == 'all-synthetic':
        datasets = get_all_datasets()['synthetic']
        print(f"Found {len(datasets)} synthetic datasets")
        
        for ds in datasets:
            print(f"\n{'='*60}")
            print(f"Processing: {ds['name']}")
            print(f"{'='*60}")
            
            if args.method in ['hypogenic', 'both']:
                try:
                    run_hypogenic(
                        config_path=ds['config_path'],
                        num_hypotheses=args.num_hypotheses,
                        model_name=args.model
                    )
                except Exception as e:
                    print(f"Error with HypoGenic on {ds['name']}: {e}")
    
    elif args.command == 'list':
        datasets = get_all_datasets()
        print("\n" + "="*60)
        print("Available Datasets")
        print("="*60)
        
        print(f"\nReal Datasets ({len(datasets['real'])}):")
        for ds in sorted(datasets['real'], key=lambda x: x['name']):
            print(f"  - {ds['name']}")
        
        print(f"\nSynthetic Datasets ({len(datasets['synthetic'])}):")
        for ds in sorted(datasets['synthetic'], key=lambda x: x['name']):
            print(f"  - {ds['name']}")
        
        print(f"\nTotal: {len(datasets['real']) + len(datasets['synthetic'])} datasets")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

