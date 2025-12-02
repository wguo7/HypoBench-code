# HypoBench HypoGenic Hypothesis Generation
# Generate 20 hypotheses for all datasets using HypoGenic
# Usage: .\run_all_hypogenic.ps1 [-ModelName "gpt-4o-mini"] [-NumHypotheses 20]

param(
    [string]$ModelName = "gpt-4o-mini",
    [int]$NumHypotheses = 20
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "HypoBench - HypoGenic Hypothesis Generator" -ForegroundColor Cyan
Write-Host "Model: $ModelName" -ForegroundColor Yellow
Write-Host "Hypotheses: $NumHypotheses" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan

# Base path
$BaseDir = $PSScriptRoot
$DataDir = Join-Path $BaseDir "hypogenic\data"

# ==========================================
# REAL DATASETS
# ==========================================
Write-Host "`n=== REAL DATASETS ===" -ForegroundColor Green

$RealDatasets = @(
    @{Name="deceptive_reviews"; Config="real/deceptive_reviews/config.yaml"},
    @{Name="dreaddit"; Config="real/dreaddit/config.yaml"},
    @{Name="gptgc_detect"; Config="real/gptgc_detect/config.yaml"},
    @{Name="headline_binary"; Config="real/headline_binary/config.yaml"},
    @{Name="llamagc_detect"; Config="real/llamagc_detect/config.yaml"},
    @{Name="persuasive_pairs"; Config="real/persuasive_pairs/config.yaml"},
    @{Name="retweet"; Config="real/retweet/config.yaml"}
)

foreach ($ds in $RealDatasets) {
    Write-Host "Running: $($ds.Name)" -ForegroundColor Yellow
    $ConfigPath = Join-Path $DataDir $ds.Config
    $OutputDir = "./outputs/hypogenic/$($ds.Name)"
    
    hypogenic_generation `
        --task_config_path $ConfigPath `
        --model_name $ModelName --model_type gpt `
        --max_num_hypotheses $NumHypotheses `
        --output_folder $OutputDir `
        --num_train 200 --num_test 100 --num_val 100 --seed 42 `
        --num_init 10 --k 5 --alpha 0.5 `
        4096 1e-5
}

# Journal Cross datasets
$JournalCrossVariants = @(
    "cross_journal_health_nips",
    "cross_journal_health_radiology",
    "cross_journal_nips_health",
    "cross_journal_nips_radiolody",
    "cross_journal_radiology_health",
    "cross_journal_radiology_nips"
)

foreach ($variant in $JournalCrossVariants) {
    Write-Host "Running: journal_cross/$variant" -ForegroundColor Yellow
    $ConfigPath = Join-Path $DataDir "real/journal_cross/$variant/config.yaml"
    $OutputDir = "./outputs/hypogenic/journal_cross_$variant"
    
    if (Test-Path $ConfigPath) {
        hypogenic_generation `
            --task_config_path $ConfigPath `
            --model_name $ModelName --model_type gpt `
            --max_num_hypotheses $NumHypotheses `
            --output_folder $OutputDir `
            --num_train 200 --num_test 100 --num_val 100 --seed 42 `
            --num_init 10 --k 5 --alpha 0.5 `
            4096 1e-5
    }
}

# Journal Same datasets
$JournalSameVariants = @(
    "same_journal_health",
    "same_journal_nips",
    "same_journal_radiology"
)

foreach ($variant in $JournalSameVariants) {
    Write-Host "Running: journal_same/$variant" -ForegroundColor Yellow
    $ConfigPath = Join-Path $DataDir "real/journal_same/$variant/config.yaml"
    $OutputDir = "./outputs/hypogenic/journal_same_$variant"
    
    if (Test-Path $ConfigPath) {
        hypogenic_generation `
            --task_config_path $ConfigPath `
            --model_name $ModelName --model_type gpt `
            --max_num_hypotheses $NumHypotheses `
            --output_folder $OutputDir `
            --num_train 200 --num_test 100 --num_val 100 --seed 42 `
            --num_init 10 --k 5 --alpha 0.5 `
            4096 1e-5
    }
}

# ==========================================
# SYNTHETIC DATASETS
# ==========================================
Write-Host "`n=== SYNTHETIC DATASETS ===" -ForegroundColor Green

# Shoe
Write-Host "Running: synthetic/shoe" -ForegroundColor Yellow
$ConfigPath = Join-Path $DataDir "synthetic/shoe/config.yaml"
if (Test-Path $ConfigPath) {
    hypogenic_generation `
        --task_config_path $ConfigPath `
        --model_name $ModelName --model_type gpt `
        --max_num_hypotheses $NumHypotheses `
        --output_folder "./outputs/hypogenic/synthetic_shoe" `
        --num_train 200 --num_test 100 --num_val 100 --seed 42 `
        --num_init 10 --k 5 --alpha 0.5 `
        4096 1e-5
}

# Shoe Two Level
foreach ($variant in @("hard", "simple")) {
    Write-Host "Running: synthetic/shoe_two_level/$variant" -ForegroundColor Yellow
    $ConfigPath = Join-Path $DataDir "synthetic/shoe_two_level/$variant/config.yaml"
    if (Test-Path $ConfigPath) {
        hypogenic_generation `
            --task_config_path $ConfigPath `
            --model_name $ModelName --model_type gpt `
            --max_num_hypotheses $NumHypotheses `
            --output_folder "./outputs/hypogenic/synthetic_shoe_two_level_$variant" `
            --num_train 200 --num_test 100 --num_val 100 --seed 42 `
            --num_init 10 --k 5 --alpha 0.5 `
            4096 1e-5
    }
}

# Admission - discover all levels and variants
$AdmissionBase = Join-Path $DataDir "synthetic/admission"
if (Test-Path $AdmissionBase) {
    Get-ChildItem $AdmissionBase -Directory | ForEach-Object {
        $level = $_.Name
        Get-ChildItem $_.FullName -Directory | ForEach-Object {
            $variant = $_.Name
            $ConfigPath = Join-Path $_.FullName "config.yaml"
            if (Test-Path $ConfigPath) {
                Write-Host "Running: synthetic/admission/$level/$variant" -ForegroundColor Yellow
                hypogenic_generation `
                    --task_config_path $ConfigPath `
                    --model_name $ModelName --model_type gpt `
                    --max_num_hypotheses $NumHypotheses `
                    --output_folder "./outputs/hypogenic/synthetic_admission_${level}_${variant}" `
                    --num_train 200 --num_test 100 --num_val 100 --seed 42 `
                    --num_init 10 --k 5 --alpha 0.5 `
                    4096 1e-5
            }
        }
    }
}

# Admission Advanced
$AdmissionAdvBase = Join-Path $DataDir "synthetic/admission_adv"
if (Test-Path $AdmissionAdvBase) {
    Get-ChildItem $AdmissionAdvBase -Directory | ForEach-Object {
        $level = $_.Name
        Get-ChildItem $_.FullName -Directory | ForEach-Object {
            $variant = $_.Name
            $ConfigPath = Join-Path $_.FullName "config.yaml"
            if (Test-Path $ConfigPath) {
                Write-Host "Running: synthetic/admission_adv/$level/$variant" -ForegroundColor Yellow
                hypogenic_generation `
                    --task_config_path $ConfigPath `
                    --model_name $ModelName --model_type gpt `
                    --max_num_hypotheses $NumHypotheses `
                    --output_folder "./outputs/hypogenic/synthetic_admission_adv_${level}_${variant}" `
                    --num_train 200 --num_test 100 --num_val 100 --seed 42 `
                    --num_init 10 --k 5 --alpha 0.5 `
                    4096 1e-5
            }
        }
    }
}

# Election
$ElectionBase = Join-Path $DataDir "synthetic/election"
if (Test-Path $ElectionBase) {
    Get-ChildItem $ElectionBase -Directory | ForEach-Object {
        $variant = $_.Name
        $ConfigPath = Join-Path $_.FullName "config.yaml"
        if (Test-Path $ConfigPath) {
            Write-Host "Running: synthetic/election/$variant" -ForegroundColor Yellow
            hypogenic_generation `
                --task_config_path $ConfigPath `
                --model_name $ModelName --model_type gpt `
                --max_num_hypotheses $NumHypotheses `
                --output_folder "./outputs/hypogenic/synthetic_election_$variant" `
                --num_train 200 --num_test 100 --num_val 100 --seed 42 `
                --num_init 10 --k 5 --alpha 0.5 `
                4096 1e-5
        }
    }
}

# Preference
$PreferenceBase = Join-Path $DataDir "synthetic/preference"
if (Test-Path $PreferenceBase) {
    Get-ChildItem $PreferenceBase -Directory | ForEach-Object {
        $variant = $_.Name
        $ConfigPath = Join-Path $_.FullName "config.yaml"
        if (Test-Path $ConfigPath) {
            Write-Host "Running: synthetic/preference/$variant" -ForegroundColor Yellow
            hypogenic_generation `
                --task_config_path $ConfigPath `
                --model_name $ModelName --model_type gpt `
                --max_num_hypotheses $NumHypotheses `
                --output_folder "./outputs/hypogenic/synthetic_preference_$variant" `
                --num_train 200 --num_test 100 --num_val 100 --seed 42 `
                --num_init 10 --k 5 --alpha 0.5 `
                4096 1e-5
        }
    }
}

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "HypoGenic generation complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

