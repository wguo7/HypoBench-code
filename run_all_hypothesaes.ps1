# HypoBench HypotheSAEs Hypothesis Generation
# Generate 20 hypotheses for supported datasets using HypotheSAEs
# Usage: .\run_all_hypothesaes.ps1 [-ModelName "gpt-4o-mini"] [-NumHypotheses 20]

param(
    [string]$ModelName = "gpt-4o-mini",
    [int]$NumHypotheses = 20
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "HypoBench - HypotheSAEs Hypothesis Generator" -ForegroundColor Cyan
Write-Host "Model: $ModelName" -ForegroundColor Yellow
Write-Host "Hypotheses: $NumHypotheses" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan

# Base path
$BaseDir = $PSScriptRoot
$DataDir = Join-Path $BaseDir "hypogenic\data"

# ==========================================
# SUPPORTED REAL DATASETS
# ==========================================
Write-Host "`n=== REAL DATASETS (HypotheSAEs) ===" -ForegroundColor Green

$SupportedDatasets = @(
    @{Name="deceptive_reviews"; DataDir="real/deceptive_reviews"},
    @{Name="retweet"; DataDir="real/retweet"},
    @{Name="dreaddit"; DataDir="real/dreaddit"},
    @{Name="headline_binary"; DataDir="real/headline_binary"},
    @{Name="gptgc_detect"; DataDir="real/gptgc_detect"},
    @{Name="llamagc_detect"; DataDir="real/llamagc_detect"},
    @{Name="persuasive_pairs"; DataDir="real/persuasive_pairs"}
)

foreach ($ds in $SupportedDatasets) {
    Write-Host "Running: $($ds.Name)" -ForegroundColor Yellow
    $DataPath = Join-Path $DataDir $ds.DataDir
    $OutputDir = "./outputs/hypothesaes/$($ds.Name)"
    
    python "$BaseDir\comparison\hypothesaes_runner.py" `
        --dataset_name $ds.Name `
        --data_dir $DataPath `
        --output_dir $OutputDir `
        --num_train 200 --num_test 300 `
        --max_num_hypotheses $NumHypotheses `
        --interpreter_model $ModelName `
        --annotator_model $ModelName `
        --seed 42
}

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "HypotheSAEs generation complete!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

