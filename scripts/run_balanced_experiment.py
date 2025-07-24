#!/usr/bin/env python3
"""
Run experiments with balanced metrics and class weights.

Usage:
    python scripts/run_balanced_experiment.py --dataset amazon --approach ffnn
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import subprocess
import yaml
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_predictions(test_preds_path, test_df):
    """Evaluate predictions with balanced metrics."""
    # Load predictions
    predictions = np.load(test_preds_path)
    true_labels = test_df['label'].values
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'balanced_accuracy': balanced_accuracy_score(true_labels, predictions),
        'macro_f1': f1_score(true_labels, predictions, average='macro'),
        'weighted_f1': f1_score(true_labels, predictions, average='weighted'),
        'per_class_f1': f1_score(true_labels, predictions, average=None)
    }
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS (Balanced Metrics)")
    print("="*60)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f} (treats all classes equally)")
    print(f"Macro F1: {metrics['macro_f1']:.4f} (average of per-class F1)")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f} (weighted by class frequency)")
    
    print("\nPer-class F1 scores:")
    for i, f1 in enumerate(metrics['per_class_f1']):
        print(f"  Class {i}: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate per-class accuracy
    print("\nPer-class Accuracy:")
    for i in range(len(cm)):
        if cm[i].sum() > 0:
            class_acc = cm[i, i] / cm[i].sum()
            print(f"  Class {i}: {class_acc:.4f} ({cm[i, i]}/{cm[i].sum()})")
    
    return metrics


def run_weighted_experiment(dataset, approach, data_path='.'):
    """Run experiment with class weights."""
    
    # Check if weighted config exists
    weighted_config_path = Path('configs') / f'{dataset}_weighted.yaml'
    
    if weighted_config_path.exists():
        print(f"Using weighted configuration: {weighted_config_path}")
        cmd = [
            'python', 'run_with_config.py',
            '--config', str(weighted_config_path),
            '--data-path', data_path
        ]
        
        # Override approach if specified
        if approach:
            cmd.extend(['--approach', approach])
    else:
        print(f"No weighted config found. Running with default settings.")
        cmd = [
            'python', 'run.py',
            '--data-path', data_path,
            '--dataset', dataset,
            '--approach', approach
        ]
    
    print(f"\nExecuting: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("\nTraining completed successfully!")
        
        # Load test data for evaluation
        from automl.datasets import (
            AGNewsDataset, 
            IMDBDataset, 
            AmazonReviewsDataset, 
            DBpediaDataset
        )
        
        dataset_class = {
            'ag_news': AGNewsDataset,
            'imdb': IMDBDataset,
            'amazon': AmazonReviewsDataset,
            'dbpedia': DBpediaDataset
        }[dataset]
        
        dataset_obj = dataset_class(data_path)
        data_info = dataset_obj.create_dataloaders(val_size=0.2, random_state=42)
        test_df = data_info['test_df']
        
        # Find predictions file
        results_path = Path('results') / f'dataset={dataset}' / 'seed=42'
        test_preds_path = results_path / 'test_preds.npy'
        
        if test_preds_path.exists():
            metrics = evaluate_predictions(test_preds_path, test_df)
            
            # Save balanced metrics
            balanced_metrics_path = results_path / 'balanced_metrics.yaml'
            with open(balanced_metrics_path, 'w') as f:
                yaml.dump(metrics, f)
            print(f"\nBalanced metrics saved to: {balanced_metrics_path}")
        else:
            print(f"Predictions not found at: {test_preds_path}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")


def compare_approaches(dataset, data_path='.'):
    """Compare different approaches on the same dataset."""
    
    approaches = ['logistic', 'ffnn', 'lstm']
    results = []
    
    print(f"\n{'='*80}")
    print(f"COMPARING APPROACHES for {dataset.upper()}")
    print(f"{'='*80}")
    
    for approach in approaches:
        print(f"\n\nTesting {approach}...")
        try:
            run_weighted_experiment(dataset, approach, data_path)
            
            # Load metrics
            metrics_path = Path('results') / f'dataset={dataset}' / 'seed=42' / 'balanced_metrics.yaml'
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = yaml.safe_load(f)
                    results.append({
                        'approach': approach,
                        'accuracy': metrics['accuracy'],
                        'balanced_accuracy': metrics['balanced_accuracy'],
                        'macro_f1': metrics['macro_f1']
                    })
        except Exception as e:
            print(f"Error with {approach}: {e}")
    
    # Print comparison table
    if results:
        print(f"\n{'='*60}")
        print("APPROACH COMPARISON")
        print(f"{'='*60}")
        
        df = pd.DataFrame(results)
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Find best approach
        best_f1 = df.loc[df['macro_f1'].idxmax()]
        print(f"\nBest approach (by Macro F1): {best_f1['approach']} with F1={best_f1['macro_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Run balanced experiments')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ag_news', 'imdb', 'amazon', 'dbpedia'],
                        help='Dataset to use')
    parser.add_argument('--approach', type=str,
                        choices=['ffnn', 'logistic', 'lstm', 'transformer'],
                        help='Approach to use (if not specified, compares all)')
    parser.add_argument('--data-path', type=str, default='.',
                        help='Path to data directory')
    parser.add_argument('--compare-all', action='store_true',
                        help='Compare all approaches')
    
    args = parser.parse_args()
    
    if args.compare_all or not args.approach:
        compare_approaches(args.dataset, args.data_path)
    else:
        run_weighted_experiment(args.dataset, args.approach, args.data_path)


if __name__ == '__main__':
    main()