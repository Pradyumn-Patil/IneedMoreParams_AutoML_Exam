#!/usr/bin/env python3
"""
Demo script showing how to use the AutoML Text Classification system
"""

import logging
from pathlib import Path

# Example 1: One-click complete AutoML solution
print("=" * 80)
print("Example 1: One-Click AutoML Solution")
print("=" * 80)
print("""
# For the complete AutoML experience with automatic configuration:
python run_automl_complete.py --dataset amazon --data-path ./data --budget 24

# For quick testing (1-hour budget):
python run_automl_complete.py --dataset amazon --data-path ./data --quick-test
""")

# Example 2: Basic training with specific approach
print("\n" + "=" * 80)
print("Example 2: Basic Training")
print("=" * 80)
print("""
# Train logistic regression:
python run.py --data-path ./data --dataset amazon --approach logistic --epochs 5

# Train enhanced FFNN:
python run.py --data-path ./data --dataset ag_news --approach ffnn --use-enhanced-ffnn --epochs 10

# Train LSTM with attention:
python run.py --data-path ./data --dataset imdb --approach lstm --epochs 5
""")

# Example 3: Hyperparameter optimization
print("\n" + "=" * 80)
print("Example 3: Hyperparameter Optimization")
print("=" * 80)
print("""
# Run HPO with 50 trials:
python run.py --data-path ./data --dataset amazon --use-hpo --hpo-trials 50 --hpo-sampler tpe

# Multi-fidelity HPO with pruning:
python run.py --data-path ./data --dataset dbpedia --use-hpo --hpo-trials 100 --hpo-pruner median
""")

# Example 4: Advanced features
print("\n" + "=" * 80)
print("Example 4: Advanced Features")
print("=" * 80)
print("""
# With text augmentation:
python run.py --data-path ./data --dataset amazon --use-augmentation --augmentation-strength medium

# Neural Architecture Search:
python run.py --data-path ./data --dataset ag_news --use-nas --nas-trials 20

# Combined NAS + HPO:
python run.py --data-path ./data --dataset imdb --use-nas --use-hpo --hpo-trials 30
""")

# Example 5: Using configuration files
print("\n" + "=" * 80)
print("Example 5: Configuration Files")
print("=" * 80)
print("""
# Run with optimized configuration:
python run_with_config.py --config configs/amazon_optimized.yaml --data-path ./data

# Run comprehensive NEPS optimization:
python run_with_config.py --config configs/neps_comprehensive.yaml --data-path ./data
""")

# Example 6: Batch experiments
print("\n" + "=" * 80)
print("Example 6: Batch Experiments")
print("=" * 80)
print("""
# Run experiments on all datasets:
python run_all_experiments.py

# The script will automatically:
1. Load optimized configurations for each dataset
2. Run experiments with appropriate settings
3. Compare results with baselines
4. Generate a summary report
""")

# Example 7: Custom usage
print("\n" + "=" * 80)
print("Example 7: Custom Usage in Python")
print("=" * 80)
print("""
from automl.core import TextAutoML
from automl.datasets import AmazonReviewsDataset

# Load dataset
dataset = AmazonReviewsDataset('./data')
data_info = dataset.create_dataloaders(val_size=0.2)

# Create AutoML instance
automl = TextAutoML(
    approach='logistic',
    use_preprocessing=True,
    use_augmentation=True,
    seed=42
)

# Train model
val_error = automl.fit(
    data_info['train_df'],
    data_info['val_df'],
    data_info['num_classes']
)

# Make predictions
test_preds, test_labels = automl.predict(data_info['test_df'])
print(f"Test accuracy: {(test_preds == test_labels).mean():.4f}")
""")

print("\n" + "=" * 80)
print("For more information, see:")
print("- README.md: Project overview and setup")
print("- CLAUDE.md: Detailed documentation") 
print("- IMPROVEMENTS_SUMMARY.md: List of enhancements")
print("- FINAL_RESULTS.md: Performance analysis")
print("=" * 80)