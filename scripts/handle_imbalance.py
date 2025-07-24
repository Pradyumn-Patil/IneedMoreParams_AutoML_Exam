#!/usr/bin/env python3
"""
Handle class imbalance in text classification datasets.

Usage:
    python scripts/handle_imbalance.py --dataset amazon --data-path . --strategy weighted
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from automl.datasets import (
    AGNewsDataset, 
    IMDBDataset, 
    AmazonReviewsDataset, 
    DBpediaDataset
)


def calculate_class_weights(labels):
    """Calculate class weights for handling imbalance."""
    label_counts = Counter(labels)
    total_samples = len(labels)
    n_classes = len(label_counts)
    
    class_weights = {}
    for label, count in label_counts.items():
        weight = total_samples / (n_classes * count)
        class_weights[label] = weight
    
    return class_weights


def oversample_minority_classes(df, random_state=42):
    """Oversample minority classes to match the majority class."""
    label_counts = Counter(df['label'])
    max_count = max(label_counts.values())
    
    balanced_dfs = []
    for label in label_counts.keys():
        class_df = df[df['label'] == label]
        if len(class_df) < max_count:
            # Oversample minority class
            oversampled_df = resample(
                class_df,
                replace=True,
                n_samples=max_count,
                random_state=random_state
            )
            balanced_dfs.append(oversampled_df)
        else:
            balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    return balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def undersample_majority_classes(df, random_state=42):
    """Undersample majority classes to match the minority class."""
    label_counts = Counter(df['label'])
    min_count = min(label_counts.values())
    
    balanced_dfs = []
    for label in label_counts.keys():
        class_df = df[df['label'] == label]
        if len(class_df) > min_count:
            # Undersample majority class
            undersampled_df = resample(
                class_df,
                replace=False,
                n_samples=min_count,
                random_state=random_state
            )
            balanced_dfs.append(undersampled_df)
        else:
            balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    return balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def create_balanced_dataset(df, strategy='oversample', random_state=42):
    """Create a balanced dataset using the specified strategy."""
    if strategy == 'oversample':
        return oversample_minority_classes(df, random_state)
    elif strategy == 'undersample':
        return undersample_majority_classes(df, random_state)
    elif strategy == 'hybrid':
        # Combination: slight oversampling of minority and slight undersampling of majority
        label_counts = Counter(df['label'])
        median_count = int(np.median(list(label_counts.values())))
        
        balanced_dfs = []
        for label in label_counts.keys():
            class_df = df[df['label'] == label]
            current_count = len(class_df)
            
            if current_count < median_count:
                # Oversample to median
                resampled_df = resample(
                    class_df,
                    replace=True,
                    n_samples=median_count,
                    random_state=random_state
                )
            elif current_count > median_count:
                # Undersample to median
                resampled_df = resample(
                    class_df,
                    replace=False,
                    n_samples=median_count,
                    random_state=random_state
                )
            else:
                resampled_df = class_df
                
            balanced_dfs.append(resampled_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        return balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    else:
        return df


def evaluate_with_balanced_metrics(y_true, y_pred, class_names=None):
    """Evaluate predictions using metrics suitable for imbalanced datasets."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'micro_f1': f1_score(y_true, y_pred, average='micro')
    }
    
    print("\nEvaluation Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  Micro F1: {metrics['micro_f1']:.4f}")
    
    print("\nClassification Report:")
    if class_names:
        print(classification_report(y_true, y_pred, target_names=class_names))
    else:
        print(classification_report(y_true, y_pred))
    
    return metrics


def generate_config_with_class_weights(dataset_name, class_weights, output_path=None):
    """Generate a configuration file with class weights for the dataset."""
    config = {
        'experiment_name': f'{dataset_name}_weighted',
        'description': f'Configuration with class weights for {dataset_name}',
        'global_settings': {
            'dataset': dataset_name,
            'use_preprocessing': True,
            'class_weights': {int(k): float(v) for k, v in class_weights.items()},
            'use_hpo': True,
            'hpo_trials': 30,
            'evaluation_metric': 'macro_f1'  # Use macro F1 for imbalanced datasets
        },
        'recommended_approaches': {
            'primary': 'ffnn',
            'alternatives': ['logistic', 'lstm']
        }
    }
    
    if output_path:
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"\nConfiguration saved to: {output_path}")
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Handle class imbalance in datasets')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ag_news', 'imdb', 'amazon', 'dbpedia'],
                        help='Dataset to analyze')
    parser.add_argument('--data-path', type=str, default='.',
                        help='Path to data directory')
    parser.add_argument('--strategy', type=str, 
                        choices=['weighted', 'oversample', 'undersample', 'hybrid'],
                        default='weighted',
                        help='Strategy to handle imbalance')
    parser.add_argument('--save-balanced', action='store_true',
                        help='Save the balanced dataset')
    parser.add_argument('--create-config', action='store_true',
                        help='Create configuration file with class weights')
    
    args = parser.parse_args()
    
    # Load dataset
    dataset_class = {
        'ag_news': AGNewsDataset,
        'imdb': IMDBDataset,
        'amazon': AmazonReviewsDataset,
        'dbpedia': DBpediaDataset
    }[args.dataset]
    
    dataset = dataset_class(args.data_path)
    data_info = dataset.create_dataloaders(val_size=0.2, random_state=42)
    
    train_df = data_info['train_df']
    test_df = data_info['test_df']
    
    print(f"\nAnalyzing {args.dataset.upper()} dataset...")
    print(f"Strategy: {args.strategy}")
    
    # Original distribution
    print("\nOriginal class distribution:")
    original_counts = Counter(train_df['label'])
    for label, count in sorted(original_counts.items()):
        percentage = (count / len(train_df)) * 100
        print(f"  Class {label}: {count:,} ({percentage:.1f}%)")
    
    if args.strategy == 'weighted':
        # Calculate class weights
        class_weights = calculate_class_weights(train_df['label'])
        print("\nCalculated class weights:")
        for label, weight in sorted(class_weights.items()):
            print(f"  Class {label}: {weight:.3f}")
        
        if args.create_config:
            # Create configuration file
            config_path = Path('configs') / f'{args.dataset}_weighted.yaml'
            generate_config_with_class_weights(args.dataset, class_weights, config_path)
    
    else:
        # Create balanced dataset
        balanced_df = create_balanced_dataset(train_df, strategy=args.strategy)
        
        print("\nBalanced class distribution:")
        balanced_counts = Counter(balanced_df['label'])
        for label, count in sorted(balanced_counts.items()):
            percentage = (count / len(balanced_df)) * 100
            print(f"  Class {label}: {count:,} ({percentage:.1f}%)")
        
        print(f"\nOriginal size: {len(train_df):,}")
        print(f"Balanced size: {len(balanced_df):,}")
        
        if args.save_balanced:
            # Save balanced dataset
            output_dir = Path('data_balanced') / args.dataset
            output_dir.mkdir(parents=True, exist_ok=True)
            
            balanced_df.to_csv(output_dir / 'train_balanced.csv', index=False)
            print(f"\nBalanced dataset saved to: {output_dir / 'train_balanced.csv'}")
    
    # Provide recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print(f"{'='*60}")
    
    imbalance_ratio = max(original_counts.values()) / min(original_counts.values())
    
    if imbalance_ratio > 10:
        print("\n⚠️  SEVERE imbalance detected!")
        print("Recommended approaches:")
        print("1. Use class weights with focal loss")
        print("2. Consider ensemble methods with balanced sampling")
        print("3. Use SMOTE for synthetic minority oversampling")
    elif imbalance_ratio > 3:
        print("\n⚠️  Moderate imbalance detected")
        print("Recommended approaches:")
        print("1. Use class weights in loss function")
        print("2. Try hybrid sampling strategy")
        print("3. Monitor per-class metrics during training")
    else:
        print("\n✓ Relatively balanced dataset")
    
    print("\nMetrics to monitor:")
    print("- Macro F1-score (treats all classes equally)")
    print("- Balanced accuracy")
    print("- Per-class precision and recall")
    print("- Confusion matrix to identify misclassification patterns")


if __name__ == '__main__':
    main()