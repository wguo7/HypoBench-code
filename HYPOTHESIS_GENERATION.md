# HypoBench Hypothesis Generation

Scripts for generating hypotheses using HypoGenic and HypotheSAEs on all HypoBench datasets.

## Setup

### 1. Install HypoGenic
```bash
cd hypogenic
pip install -e .
```

### 2. Install HypotheSAEs (additional packages)
```bash
cd hypothesaes
pip install -e .
pip install -r ../requirements_hypothesaes.txt
```

### 3. Set API Keys
```bash
# For HypoGenic
export OPENAI_API_KEY="your-key"

# For HypotheSAEs (uses different env var)
export OPENAI_KEY_SAE="your-key"
```

---

## File Descriptions

| File | Description |
|------|-------------|
| `generate_hypotheses.py` | CLI tool to generate commands for both methods |
| `hypothesaes_runner.py` | Unified runner for HypotheSAEs across all datasets |
| `ONE_LINERS.md` | Copy-paste ready one-liner commands |
| `run_all_hypogenic.sh/.ps1` | Batch scripts for all HypoGenic runs |
| `run_all_hypothesaes.sh/.ps1` | Batch scripts for all HypotheSAEs runs |
| `requirements_hypothesaes.txt` | Additional packages needed for HypotheSAEs |

---

## Quick Start

### List Available Datasets
```bash
python generate_hypotheses.py --list
```

### Generate Single Dataset Command
```bash
# HypoGenic
python generate_hypotheses.py --method hypogenic --dataset deceptive_reviews

# HypotheSAEs
python generate_hypotheses.py --method hypothesaes --dataset deceptive_reviews
```

### Generate All Commands
```bash
python generate_hypotheses.py --method hypogenic --all
python generate_hypotheses.py --method hypothesaes --all
```

## Method Details

### HypoGenic

Uses the `hypogenic_generation` CLI command. Generates hypotheses through:
1. Initialize hypothesis bank from training examples
2. Iteratively update hypotheses based on prediction accuracy
3. Replace lowest-performing hypotheses

**Command Format:**
```bash
hypogenic_generation \
    --task_config_path ./real_datasets/DATASET/config.yaml \
    --model_name gpt-4o-mini --model_type gpt \
    --max_num_hypotheses 20 \
    --output_folder ./outputs/hypogenic/DATASET \
    --num_train 200 --num_test 100 --num_val 100 --seed 42 \
    --num_init 10 --k 5 --alpha 0.5 \
    4096 1e-5
```

> **Note:** The last two positional arguments (`4096 1e-5`) are `max_tokens` and `temperature`.

**Command Format:**
```bash
python hypothesaes_runner.py \
    --dataset_name DATASET \
    --data_dir ./real_datasets/DATASET \
    --output_dir ./outputs/hypothesaes/DATASET \
    --num_train 200 --num_test 300 \
    --max_num_hypotheses 20 \
    --seed 42
```

## Output Format

### HypoGenic Output
```json
{
    "hypothesis_text": {
        "acc": 0.85,
        "num_wrong": 15,
        "example_correct": [...],
        "example_wrong": [...]
    }
}
```

### HypotheSAEs Output
```json
{
    "dataset": "deceptive_reviews",
    "method": "HypotheSAEs",
    "hypotheses": [
        {
            "hypothesis": "mentions spatial details like room layout",
            "neuron_idx": 42,
            "target_score": 0.45,
            "fidelity_score": 0.82
        }
    ],
    "metrics": {
        "test_r2": 0.32,
        "test_auc": 0.78
    }
}
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

### Windows CMD (one-liner)
```cmd
set OPENAI_KEY_SAE=your-key & python hypothesaes_runner.py --dataset_name deceptive_reviews --data_dir real_datasets/deceptive_reviews --output_dir outputs/hypothesaes/deceptive_reviews --max_num_hypotheses 20 --seed 42
```

