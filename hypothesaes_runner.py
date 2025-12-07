"""
HypotheSAEs Runner for Hypothesis Comparison
Unified interface for running HypotheSAEs on different datasets
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any

""" Add hypothesaes to path """
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hypothesaes'))

from hypothesaes.quickstart import train_sae, generate_hypotheses, evaluate_hypotheses
from hypothesaes.embedding import get_openai_embeddings


class HypotheSAEsRunner:
    """Unified runner for HypotheSAEs across different datasets.
    
    Handles data loading, SAE training, hypothesis generation,
    and evaluation for multiple dataset types.
    """
    
    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        output_dir: str,
        openai_api_key: str = None,
        anthropic_api_key: str = None,
        num_train: int = 200,
        num_test: int = 300,
        seed: int = 42
    ):
        """Initialize HypotheSAEs runner.
        
        Args:
            dataset_name: Name of the dataset to process.
            data_dir: Directory containing dataset files.
            output_dir: Directory for saving results.
            openai_api_key: OpenAI API key for embeddings and LLM calls.
            anthropic_api_key: Anthropic API key for LLM calls.
            num_train: Number of training samples to use.
            num_test: Number of test samples to use.
            seed: Random seed for reproducibility.
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.num_train = num_train
        self.num_test = num_test
        self.seed = seed
        
        """ Set API keys """
        if openai_api_key:
            os.environ['OPENAI_KEY_SAE'] = openai_api_key
            os.environ['OPENAI_API_KEY'] = openai_api_key  # Also set this for compatibility
        if anthropic_api_key:
            os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key
        
        """ Clear any cached OpenAI clients to force re-initialization with new API key """
        try:
            from hypothesaes import llm_api
            llm_api._CLIENT_OPENAI = None
            llm_api._CLIENT_ANTHROPIC = None
        except:
            pass  # If not imported yet, that's fine
            
        """ Load dataset-specific configurations """
        self.config = self._get_dataset_config()
        
        """ Create output directory """
        os.makedirs(output_dir, exist_ok=True)
        
    def _get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset-specific configuration.
        
        Returns:
            Dictionary containing dataset configuration including
            file names, field mappings, SAE parameters, and task instructions.
        
        Raises:
            ValueError: If dataset name is unknown.
        """
        configs = {
            'deceptive_reviews': {
                'train_file': 'hotel_reviews_train.json',
                'test_file': 'hotel_reviews_test.json',
                'val_file': 'hotel_reviews_val.json',
                'text_field': 'review_sentence',
                'label_field': 'label',
                'label_map': {'deceptive': 1, 'truthful': 0},
                'task_type': 'binary_classification',
                'M': 512,  # Number of SAE features
                'K': 8,    # Sparsity level
                'matryoshka_prefix_lengths': [64, 512],
                'task_instructions': """All texts are hotel reviews.
Features should describe specific aspects of review authenticity. For example:
- "mentions spatial details like room layout or bathroom location"
- "uses excessive positive emotion words"
- "contains first-person singular pronouns"
- "describes sensory or perceptual details"
"""
            },
            'retweet': {
                'train_file': 'retweet_train.json',
                'test_file': 'retweet_test.json',
                'val_file': 'retweet_val.json',
                'text_fields': ['first_tweet', 'second_tweet'],
                'label_field': 'label',
                'label_map': {'first': 0, 'second': 1},
                'task_type': 'pairwise_classification',
                'M': 512,
                'K': 8,
                'matryoshka_prefix_lengths': [64, 512],
                'task_instructions': """All texts are pairs of tweets about the same content.
Features should describe linguistic patterns that predict retweet likelihood. For example:
- "expresses strong negative emotion"
- "includes hashtags"
- "uses provocative or surprising language"
- "contains explicit calls to action"
- "mentions other users with @"
"""
            },
            'dreaddit': {
                'train_file': 'dreaddit_ind_train.json',
                'test_file': 'dreaddit_ind_test.json',
                'val_file': 'dreaddit_ind_val.json',
                'text_field': 'text',
                'label_field': 'label',
                'label_map': {'has stress': 1, 'no stress': 0},
                'task_type': 'binary_classification',
                'M': 512,
                'K': 8,
                'matryoshka_prefix_lengths': [64, 512],
                'task_instructions': """All texts are Reddit posts.
Features should describe patterns indicating stress levels. For example:
- "expresses feelings of being overwhelmed"
- "mentions sleep problems or fatigue"
- "uses negative emotional language"
- "describes interpersonal conflicts"
- "mentions work or financial concerns"
"""
            },
            'headline_binary': {
                'train_file': 'headline_binary_train.json',
                'test_file': 'headline_binary_test.json',
                'val_file': None,
                'text_fields': ['headline_1', 'headline_2'],
                'label_field': 'label',
                'label_map': {'Headline 1 has more clicks than Headline 2.': 0, 'Headline 2 has more clicks than Headline 1.': 1},
                'task_type': 'pairwise_classification',
                'M': 512,
                'K': 8,
                'matryoshka_prefix_lengths': [64, 512],
                'task_instructions': """All texts are pairs of news headlines about the same topic.
Features should describe patterns that make headlines more clickable. For example:
- "uses numbers or statistics"
- "poses a question"
- "creates a sense of urgency"
- "uses strong emotional language"
- "includes a surprising or counterintuitive claim"
"""
            },
            'gptgc_detect': {
                'train_file': 'WP_aigc_detect_train.json',
                'test_file': 'WP_aigc_detect_test.json',
                'val_file': 'WP_aigc_detect_val.json',
                'text_field': 'story',
                'label_field': 'label',
                'label_map': {'AI': 1, 'HUMAN': 0},
                'task_type': 'binary_classification',
                'M': 512,
                'K': 8,
                'matryoshka_prefix_lengths': [64, 512],
                'task_instructions': """All texts are creative writing stories (GPT-generated vs human-written).
Features should describe patterns distinguishing AI-generated from human-written text. For example:
- "uses overly formal or polished language"
- "lacks personal anecdotes or experiences"
- "has uniform sentence structure"
- "contains generic or safe opinions"
"""
            },
            'llamagc_detect': {
                'train_file': 'WP_aigc_detect_train.json',
                'test_file': 'WP_aigc_detect_test.json',
                'val_file': 'WP_aigc_detect_val.json',
                'text_field': 'story',
                'label_field': 'label',
                'label_map': {'AI': 1, 'HUMAN': 0},
                'task_type': 'binary_classification',
                'M': 512,
                'K': 8,
                'matryoshka_prefix_lengths': [64, 512],
                'task_instructions': """All texts are creative writing stories (LLaMA-generated vs human-written).
Features should describe patterns distinguishing AI-generated from human-written text. For example:
- "uses overly formal or polished language"
- "lacks personal anecdotes or experiences"
- "has uniform sentence structure"
- "contains generic or safe opinions"
"""
            },
            'persuasive_pairs': {
                'train_file': 'persuasive_pairs_human_gt_ind_train.json',
                'test_file': 'persuasive_pairs_human_gt_ind_test.json',
                'val_file': 'persuasive_pairs_human_gt_ind_val.json',
                'text_fields': ['argument_1', 'argument_2'],
                'label_field': 'label',
                'label_map': {'first': 0, 'second': 1},
                'task_type': 'pairwise_classification',
                'M': 512,
                'K': 8,
                'matryoshka_prefix_lengths': [64, 512],
                'task_instructions': """All texts are pairs of persuasive arguments on the same topic.
Features should describe patterns that make text more persuasive. For example:
- "uses concrete examples and evidence"
- "appeals to emotions"
- "addresses counterarguments"
- "uses confident and assertive language"
- "establishes credibility"
"""
            },
        }
        
        if self.dataset_name not in configs:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        return configs[self.dataset_name]
    
    def load_data(self) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str]]:
        """Load and format dataset.
        
        Returns:
            Tuple containing:
                - train_texts: List of training text samples
                - train_labels: Array of training labels
                - test_texts: List of test text samples
                - test_labels: Array of test labels
                - val_texts: List of validation text samples
        """
        print(f"\n{'='*80}")
        print(f"Loading {self.dataset_name} dataset...")
        print(f"{'='*80}")
        
        """ Load train data """
        train_path = os.path.join(self.data_dir, self.config['train_file'])
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        
        """ Load test data """
        test_path = os.path.join(self.data_dir, self.config['test_file'])
        with open(test_path, 'r') as f:
            test_data = json.load(f)
            
        """ Load validation data if available """
        val_texts = []
        if self.config.get('val_file'):
            val_path = os.path.join(self.data_dir, self.config['val_file'])
            with open(val_path, 'r') as f:
                val_data = json.load(f)
            val_texts = self._extract_texts(val_data)[:200]
        
        """ Extract texts and labels """
        train_texts = self._extract_texts(train_data)
        train_labels = self._extract_labels(train_data)
        test_texts = self._extract_texts(test_data)
        test_labels = self._extract_labels(test_data)
        
        """ Sample if needed """
        np.random.seed(self.seed)
        if len(train_texts) > self.num_train:
            indices = np.random.choice(len(train_texts), self.num_train, replace=False)
            train_texts = [train_texts[i] for i in indices]
            train_labels = train_labels[indices]
            
        if len(test_texts) > self.num_test:
            indices = np.random.choice(len(test_texts), self.num_test, replace=False)
            test_texts = [test_texts[i] for i in indices]
            test_labels = test_labels[indices]
        
        print(f"Train samples: {len(train_texts)}")
        print(f"Test samples: {len(test_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        
        return train_texts, train_labels, test_texts, test_labels, val_texts
    
    def _extract_texts(self, data: Dict) -> List[str]:
        """Extract text from dataset.
        
        Args:
            data: Dataset dictionary.
        
        Returns:
            List of text strings.
        """
        if 'text_field' in self.config:
            """ Single text field """
            field = self.config['text_field']
            return data[field]
        else:
            """ Multiple text fields (pairwise comparison) """
            fields = self.config['text_fields']
            texts = []
            for i in range(len(data[fields[0]])):
                combined = f"Text 1: {data[fields[0]][i]}\n\nText 2: {data[fields[1]][i]}"
                texts.append(combined)
            return texts
    
    def _extract_labels(self, data: Dict) -> np.ndarray:
        """Extract and map labels.
        
        Args:
            data: Dataset dictionary.
        
        Returns:
            Array of numeric labels.
        """
        field = self.config['label_field']
        labels = data[field]
        label_map = self.config['label_map']
        
        """ Map string labels to numeric """
        numeric_labels = np.array([label_map[label] for label in labels])
        return numeric_labels
    
    def run_pipeline(
        self,
        max_num_hypotheses: int = 20,
        selection_method: str = "correlation",
        embedder: str = "text-embedding-3-small",
        interpreter_model: str = "gpt-4o-mini",
        annotator_model: str = "gpt-4o-mini",
        n_workers: int = 5
    ) -> Dict[str, Any]:
        """Run complete HypotheSAEs pipeline.
        
        Args:
            max_num_hypotheses: Maximum number of hypotheses to generate.
            selection_method: Method for selecting neurons (correlation/lasso/separation_score).
            embedder: Embedding model name.
            interpreter_model: LLM for interpreting neurons.
            annotator_model: LLM for annotating texts.
            n_workers: Number of parallel workers for API calls.
        
        Returns:
            Dictionary containing results, hypotheses, and metrics.
        """
        
        start_time = datetime.now()
        
        """ Load data """
        train_texts, train_labels, test_texts, test_labels, val_texts = self.load_data()
        
        """ Step 1: Generate embeddings """
        print(f"\n{'='*80}")
        print("Step 1: Generating embeddings...")
        print(f"{'='*80}")
        
        cache_name = f"{self.dataset_name}_hypothesaes_{embedder}"
        all_texts = train_texts + test_texts + val_texts
        
        text2embedding = get_openai_embeddings(
            all_texts,
            model=embedder,
            cache_name=cache_name
        )
        
        train_embeddings = np.stack([text2embedding[text] for text in train_texts])
        test_embeddings = np.stack([text2embedding[text] for text in test_texts])
        val_embeddings = np.stack([text2embedding[text] for text in val_texts]) if val_texts else None
        
        print(f"Train embeddings shape: {train_embeddings.shape}")
        print(f"Test embeddings shape: {test_embeddings.shape}")
        
        """ Step 2: Train SAE """
        print(f"\n{'='*80}")
        print("Step 2: Training Sparse Autoencoder...")
        print(f"{'='*80}")
        print(f"M={self.config['M']}, K={self.config['K']}, "
              f"prefix_lengths={self.config['matryoshka_prefix_lengths']}")
        
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints", cache_name)
        
        sae = train_sae(
            embeddings=train_embeddings,
            val_embeddings=val_embeddings,
            M=self.config['M'],
            K=self.config['K'],
            matryoshka_prefix_lengths=self.config['matryoshka_prefix_lengths'],
            checkpoint_dir=checkpoint_dir
        )
        
        """ Step 3: Generate hypotheses """
        print(f"\n{'='*80}")
        print("Step 3: Generating hypotheses...")
        print(f"{'='*80}")
        
        hypotheses_df = generate_hypotheses(
            texts=train_texts,
            labels=train_labels,
            embeddings=train_embeddings,
            sae=sae,
            cache_name=cache_name,
            selection_method=selection_method,
            n_selected_neurons=max_num_hypotheses,
            n_candidate_interpretations=1,
            task_specific_instructions=self.config['task_instructions'],
            interpreter_model=interpreter_model,
            annotator_model=annotator_model,
            n_workers_annotation=n_workers
        )
        
        print(f"\nGenerated {len(hypotheses_df)} hypotheses")
        
        """ Step 4: Evaluate on test set """
        print(f"\n{'='*80}")
        print("Step 4: Evaluating hypotheses on test set...")
        print(f"{'='*80}")
        
        metrics, evaluation_df = evaluate_hypotheses(
            hypotheses_df=hypotheses_df,
            texts=test_texts,
            labels=test_labels,
            cache_name=cache_name,
            annotator_model=annotator_model,
            n_workers_annotation=n_workers
        )
        
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        """ Merge hypotheses with evaluation results """
        # Check if we can merge the dataframes
        print(f"\nDebug info:")
        print(f"hypotheses_df columns: {hypotheses_df.columns.tolist()}")
        print(f"evaluation_df columns: {evaluation_df.columns.tolist()}")
        print(f"hypotheses_df shape: {hypotheses_df.shape}")
        print(f"evaluation_df shape: {evaluation_df.shape}")
        
        # If evaluation_df has neuron_idx, merge properly. Otherwise, assume matching order
        if 'neuron_idx' in evaluation_df.columns:
            merged_df = hypotheses_df.merge(
                evaluation_df,
                on='neuron_idx',
                how='left',
                suffixes=('_hyp', '_eval')
            )
        else:
            # Assume same order and just concatenate
            print("Warning: evaluation_df doesn't have neuron_idx, assuming same order as hypotheses_df")
            # Reset index to ensure alignment
            hypotheses_df_copy = hypotheses_df.reset_index(drop=True)
            evaluation_df_copy = evaluation_df.reset_index(drop=True)
            merged_df = pd.concat([hypotheses_df_copy, evaluation_df_copy], axis=1)
        
        """ Compile results """
        results = {
            'dataset': self.dataset_name,
            'method': 'HypotheSAEs',
            'config': {
                'num_train': len(train_texts),
                'num_test': len(test_texts),
                'num_val': len(val_texts),
                'max_num_hypotheses': max_num_hypotheses,
                'selection_method': selection_method,
                'M': self.config['M'],
                'K': self.config['K'],
                'matryoshka_prefix_lengths': self.config['matryoshka_prefix_lengths'],
                'embedder': embedder,
                'interpreter_model': interpreter_model,
                'annotator_model': annotator_model,
                'seed': self.seed
            },
            'hypotheses': [
                {
                    'hypothesis': row['interpretation'],
                    'neuron_idx': int(row['neuron_idx']),
                    'target_score': float(row[f'target_{selection_method}']),
                    'fidelity_score': float(row['f1_fidelity_score']),
                    'separation_score': float(row['separation_score']) if pd.notna(row.get('separation_score')) else None,
                    'separation_pvalue': float(row['separation_pval']) if pd.notna(row.get('separation_pval')) else None,
                    'regression_coef': float(row['regression_coef']) if pd.notna(row.get('regression_coef')) else None,
                    'regression_pval': float(row['regression_pval']) if pd.notna(row.get('regression_pval')) else None,
                    'feature_prevalence': float(row['feature_prevalence']) if pd.notna(row.get('feature_prevalence')) else None
                }
                for _, row in merged_df.iterrows()
            ],
            'metrics': {
                'test_r2': float(metrics['r2']) if 'r2' in metrics else None,
                'test_auc': float(metrics['auc']) if 'auc' in metrics else None,
                'num_significant_hypotheses': int(metrics['Significant'][0]),
                'total_hypotheses': int(metrics['Significant'][1]),
                'significance_threshold': float(metrics['Significant'][2]),
                'runtime_seconds': runtime
            },
            'timestamp': datetime.now().isoformat()
        }
        
        """ Save results """
        output_file = os.path.join(
            self.output_dir,
            f"{self.dataset_name}_hypothesaes_results.json"
        )
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Total hypotheses: {len(results['hypotheses'])}")
        print(f"Significant hypotheses: {metrics['Significant'][0]}/{metrics['Significant'][1]}")
        if 'r2' in metrics:
            print(f"Test R²: {metrics['r2']:.3f}")
        if 'auc' in metrics:
            print(f"Test AUC: {metrics['auc']:.3f}")
        print(f"Runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")
        print(f"\nResults saved to: {output_file}")
        
        return results


def main():
    """Main entry point for HypotheSAEs runner.
    
    Parses command-line arguments and executes the HypotheSAEs pipeline.
    """
    parser = argparse.ArgumentParser(description='Run HypotheSAEs on a dataset')
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['deceptive_reviews', 'retweet', 'dreaddit', 'headline_binary',
                                 'gptgc_detect', 'llamagc_detect', 'persuasive_pairs'],
                        help='Name of the dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing dataset files')
    parser.add_argument('--output_dir', type=str, default='./comparison/results',
                        help='Output directory for results')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key')
    parser.add_argument('--anthropic_api_key', type=str, default=None,
                        help='Anthropic API key')
    parser.add_argument('--num_train', type=int, default=200,
                        help='Number of training samples')
    parser.add_argument('--num_test', type=int, default=300,
                        help='Number of test samples')
    parser.add_argument('--max_num_hypotheses', type=int, default=20,
                        help='Maximum number of hypotheses to generate')
    parser.add_argument('--selection_method', type=str, default='correlation',
                        choices=['correlation', 'lasso', 'separation_score'],
                        help='Method for selecting predictive neurons')
    parser.add_argument('--embedder', type=str, default='text-embedding-3-small',
                        help='Embedding model to use')
    parser.add_argument('--interpreter_model', type=str, default='gpt-4o-mini',
                        help='LLM for interpreting neurons')
    parser.add_argument('--annotator_model', type=str, default='gpt-4o-mini',
                        help='LLM for annotating texts')
    parser.add_argument('--n_workers', type=int, default=5,
                        help='Number of parallel workers for API calls')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    """ Create runner """
    runner = HypotheSAEsRunner(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        openai_api_key=args.openai_api_key,
        anthropic_api_key=args.anthropic_api_key,
        num_train=args.num_train,
        num_test=args.num_test,
        seed=args.seed
    )
    
    """ Run pipeline """
    results = runner.run_pipeline(
        max_num_hypotheses=args.max_num_hypotheses,
        selection_method=args.selection_method,
        embedder=args.embedder,
        interpreter_model=args.interpreter_model,
        annotator_model=args.annotator_model,
        n_workers=args.n_workers
    )
    
    print("\nHypotheSAEs pipeline completed successfully!")


if __name__ == '__main__':
    main()