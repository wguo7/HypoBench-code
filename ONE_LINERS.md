# HypoBench One-Line Hypothesis Generation Commands

Generate 20 hypotheses using HypoGenic and HypotheSAEs for all HypoBench datasets.

## Setup

```bash
# Set API keys
export OPENAI_API_KEY="your-key"      # For HypoGenic
export OPENAI_KEY_SAE="your-key"      # For HypotheSAEs
```

---

## HypoGenic Commands

> **Note:** The last two args (`4096 1e-5`) are `max_tokens` and `temperature` (positional args in CLI).

### Real-World Datasets

```bash
# Deceptive Reviews
hypogenic_generation --task_config_path ./real_datasets/deceptive_reviews/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/deceptive_reviews --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# Dreaddit
hypogenic_generation --task_config_path ./real_datasets/dreaddit/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/dreaddit --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# Retweet
hypogenic_generation --task_config_path ./real_datasets/retweet/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/retweet --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# Headline Binary
hypogenic_generation --task_config_path ./real_datasets/headline_binary/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/headline_binary --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# GPT-GC Detect
hypogenic_generation --task_config_path ./real_datasets/gptgc_detect/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/gptgc_detect --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# LLaMA-GC Detect
hypogenic_generation --task_config_path ./real_datasets/llamagc_detect/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/llamagc_detect --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# Persuasive Pairs
hypogenic_generation --task_config_path ./real_datasets/persuasive_pairs/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/persuasive_pairs --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5
```

### Synthetic Datasets

```bash
# Shoe
hypogenic_generation --task_config_path ./synthetic_datasets/shoe/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/shoe --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# Admission
hypogenic_generation --task_config_path ./synthetic_datasets/admission/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/admission --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# Election
hypogenic_generation --task_config_path ./synthetic_datasets/election/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/election --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5

# Preference
hypogenic_generation --task_config_path ./synthetic_datasets/preference/config.yaml --model_name gpt-4o-mini --model_type gpt --max_num_hypotheses 20 --output_folder ./outputs/hypogenic/preference --num_train 200 --num_test 100 --num_val 100 --seed 42 --num_init 10 --k 5 --alpha 0.5 4096 1e-5
```

---

## HypotheSAEs Commands

```bash
# Deceptive Reviews
python hypothesaes_runner.py --dataset_name deceptive_reviews --data_dir ./real_datasets/deceptive_reviews --output_dir ./outputs/hypothesaes/deceptive_reviews --num_train 200 --num_test 300 --max_num_hypotheses 20 --seed 42

# Retweet
python hypothesaes_runner.py --dataset_name retweet --data_dir ./real_datasets/retweet --output_dir ./outputs/hypothesaes/retweet --num_train 200 --num_test 300 --max_num_hypotheses 20 --seed 42

# Dreaddit
python hypothesaes_runner.py --dataset_name dreaddit --data_dir ./real_datasets/dreaddit --output_dir ./outputs/hypothesaes/dreaddit --num_train 200 --num_test 300 --max_num_hypotheses 20 --seed 42

# Headline Binary
python hypothesaes_runner.py --dataset_name headline_binary --data_dir ./real_datasets/headline_binary --output_dir ./outputs/hypothesaes/headline_binary --num_train 200 --num_test 300 --max_num_hypotheses 20 --seed 42

# GPT-GC Detect
python hypothesaes_runner.py --dataset_name gptgc_detect --data_dir ./real_datasets/gptgc_detect --output_dir ./outputs/hypothesaes/gptgc_detect --num_train 200 --num_test 300 --max_num_hypotheses 20 --seed 42

# LLaMA-GC Detect
python hypothesaes_runner.py --dataset_name llamagc_detect --data_dir ./real_datasets/llamagc_detect --output_dir ./outputs/hypothesaes/llamagc_detect --num_train 200 --num_test 300 --max_num_hypotheses 20 --seed 42

# Persuasive Pairs
python hypothesaes_runner.py --dataset_name persuasive_pairs --data_dir ./real_datasets/persuasive_pairs --output_dir ./outputs/hypothesaes/persuasive_pairs --num_train 200 --num_test 300 --max_num_hypotheses 20 --seed 42
```

---

## Batch Execution

### Linux/Mac
```bash
./run_all_hypogenic.sh gpt-4o-mini 20
./run_all_hypothesaes.sh gpt-4o-mini 20
```

### Windows PowerShell
```powershell
.\run_all_hypogenic.ps1 -ModelName "gpt-4o-mini" -NumHypotheses 20
.\run_all_hypothesaes.ps1 -ModelName "gpt-4o-mini" -NumHypotheses 20
```

---

## CLI Tool

```bash
# List datasets
python generate_hypotheses.py --list

# Single dataset command
python generate_hypotheses.py --method hypogenic --dataset deceptive_reviews
python generate_hypotheses.py --method hypothesaes --dataset deceptive_reviews

# All commands
python generate_hypotheses.py --method hypogenic --all
python generate_hypotheses.py --method hypothesaes --all
```
