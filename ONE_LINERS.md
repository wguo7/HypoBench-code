# HypoBench One-Line Hypothesis Generation Commands

This document provides simple one-liner commands to generate 20 hypotheses using HypoGenic and HypotheSAEs for all datasets in HypoBench.

## Quick Start

### List All Available Datasets
```bash
python generate_all_hypotheses.py list
```

### Run on All Real Datasets
```bash
# Using HypoGenic
python generate_all_hypotheses.py all-real --method hypogenic --num_hypotheses 20

# Using HypotheSAEs (supported datasets only)
python generate_all_hypotheses.py all-real --method hypothesaes --num_hypotheses 20

# Using both methods
python generate_all_hypotheses.py all-real --method both --num_hypotheses 20
```

### Run on All Synthetic Datasets
```bash
python generate_all_hypotheses.py all-synthetic --method hypogenic --num_hypotheses 20
```

---

## One-Line Commands for Individual Datasets

### HypoGenic Commands

#### Real-World Datasets

```bash
# Deceptive Reviews
hypogenic_generation --task_config_path ./hypogenic/data/real/deceptive_reviews/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/deceptive_reviews --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# Dreaddit
hypogenic_generation --task_config_path ./hypogenic/data/real/dreaddit/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/dreaddit --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# Retweet
hypogenic_generation --task_config_path ./hypogenic/data/real/retweet/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/retweet --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# Headline Binary
hypogenic_generation --task_config_path ./hypogenic/data/real/headline_binary/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/headline_binary --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# GPT-GC Detect
hypogenic_generation --task_config_path ./hypogenic/data/real/gptgc_detect/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/gptgc_detect --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# LLaMA-GC Detect
hypogenic_generation --task_config_path ./hypogenic/data/real/llamagc_detect/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/llamagc_detect --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# Persuasive Pairs
hypogenic_generation --task_config_path ./hypogenic/data/real/persuasive_pairs/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/persuasive_pairs --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5
```

#### Synthetic Datasets

```bash
# Shoe
hypogenic_generation --task_config_path ./hypogenic/data/synthetic/shoe/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/shoe --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# Admission Level 1 Base
hypogenic_generation --task_config_path ./hypogenic/data/synthetic/admission/level_1/base/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/admission_level1_base --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# Election Level 0
hypogenic_generation --task_config_path ./hypogenic/data/synthetic/election/level0/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/election_level0 --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# Preference Level 0
hypogenic_generation --task_config_path ./hypogenic/data/synthetic/preference/level0/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/preference_level0 --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5
```

---

### HypotheSAEs Commands

```bash
# Deceptive Reviews
python comparison/hypothesaes_runner.py --dataset_name deceptive_reviews --data_dir ./hypogenic/data/real/deceptive_reviews --output_dir ./outputs/hypothesaes/deceptive_reviews --num_train 200 --num_test 300 --max_num_hypotheses 20 --interpreter_model gpt-4o-mini --annotator_model gpt-4o-mini --seed 42

# Retweet
python comparison/hypothesaes_runner.py --dataset_name retweet --data_dir ./hypogenic/data/real/retweet --output_dir ./outputs/hypothesaes/retweet --num_train 200 --num_test 300 --max_num_hypotheses 20 --interpreter_model gpt-4o-mini --annotator_model gpt-4o-mini --seed 42

# Dreaddit
python comparison/hypothesaes_runner.py --dataset_name dreaddit --data_dir ./hypogenic/data/real/dreaddit --output_dir ./outputs/hypothesaes/dreaddit --num_train 200 --num_test 300 --max_num_hypotheses 20 --interpreter_model gpt-4o-mini --annotator_model gpt-4o-mini --seed 42

# Headline Binary
python comparison/hypothesaes_runner.py --dataset_name headline_binary --data_dir ./hypogenic/data/real/headline_binary --output_dir ./outputs/hypothesaes/headline_binary --num_train 200 --num_test 300 --max_num_hypotheses 20 --interpreter_model gpt-4o-mini --annotator_model gpt-4o-mini --seed 42

# GPT-GC Detect
python comparison/hypothesaes_runner.py --dataset_name gptgc_detect --data_dir ./hypogenic/data/real/gptgc_detect --output_dir ./outputs/hypothesaes/gptgc_detect --num_train 200 --num_test 300 --max_num_hypotheses 20 --interpreter_model gpt-4o-mini --annotator_model gpt-4o-mini --seed 42

# LLaMA-GC Detect
python comparison/hypothesaes_runner.py --dataset_name llamagc_detect --data_dir ./hypogenic/data/real/llamagc_detect --output_dir ./outputs/hypothesaes/llamagc_detect --num_train 200 --num_test 300 --max_num_hypotheses 20 --interpreter_model gpt-4o-mini --annotator_model gpt-4o-mini --seed 42

# Persuasive Pairs
python comparison/hypothesaes_runner.py --dataset_name persuasive_pairs --data_dir ./hypogenic/data/real/persuasive_pairs --output_dir ./outputs/hypothesaes/persuasive_pairs --num_train 200 --num_test 300 --max_num_hypotheses 20 --interpreter_model gpt-4o-mini --annotator_model gpt-4o-mini --seed 42
```

---

## Batch Execution Scripts

### Linux/Mac
```bash
# Run HypoGenic on all datasets
./run_all_hypogenic.sh gpt-4o-mini 20

# Run HypotheSAEs on supported datasets
./run_all_hypothesaes.sh gpt-4o-mini 20
```

### Windows (PowerShell)
```powershell
# Run HypoGenic on all datasets
.\run_all_hypogenic.ps1 -ModelName "gpt-4o-mini" -NumHypotheses 20

# Run HypotheSAEs on supported datasets
.\run_all_hypothesaes.ps1 -ModelName "gpt-4o-mini" -NumHypotheses 20
```

---

## Python API

You can also use the Python API directly:

```python
from generate_all_hypotheses import run_hypogenic, run_hypothesaes

# HypoGenic
results = run_hypogenic(
    config_path="./hypogenic/data/real/deceptive_reviews/config.yaml",
    num_hypotheses=20,
    model_name="gpt-4o-mini"
)

# HypotheSAEs
results = run_hypothesaes(
    dataset_name="deceptive_reviews",
    data_dir="./hypogenic/data/real/deceptive_reviews",
    num_hypotheses=20
)
```

