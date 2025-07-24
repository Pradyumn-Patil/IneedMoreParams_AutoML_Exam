#!/usr/bin/env python3
"""
Analyze text classification datasets for class imbalance and other characteristics.

Usage:
    python scripts/analyze_datasets.py --dataset amazon --data-path .
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from automl.datasets import (
    AGNewsDataset, 
    IMDBDataset, 
    AmazonReviewsDataset, 
    DBpediaDataset
)


def analyze_dataset(dataset_name, data_path):
    """Analyze a single dataset for class distribution and text characteristics."""
    
    # Load dataset
    dataset_class = {
        'ag_news': AGNewsDataset,
        'imdb': IMDBDataset,
        'amazon': AmazonReviewsDataset,
        'dbpedia': DBpediaDataset
    }[dataset_name]
    
    dataset = dataset_class(data_path)
    data_info = dataset.create_dataloaders(val_size=0.2, random_state=42)
    
    train_df = data_info['train_df']
    val_df = data_info['val_df']
    test_df = data_info['test_df']
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Basic statistics
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val: {len(val_df):,}")
    print(f"  Test: {len(test_df):,}")
    print(f"  Total: {len(train_df) + len(val_df) + len(test_df):,}")
    
    # Class distribution
    print(f"\nClass distribution:")
    for split_name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        class_counts = Counter(df['label'])
        total = len(df)
        print(f"\n  {split_name}:")
        for label in sorted(class_counts.keys()):
            count = class_counts[label]
            percentage = (count / total) * 100
            print(f"    Class {label}: {count:,} ({percentage:.1f}%)")
    
    # Class imbalance ratio
    test_counts = Counter(test_df['label'])
    max_class = max(test_counts.values())
    min_class = min(test_counts.values())
    imbalance_ratio = max_class / min_class
    print(f"\nClass imbalance ratio (max/min): {imbalance_ratio:.2f}")
    
    # Text length statistics
    print(f"\nText length statistics (characters):")
    train_df['text_length'] = train_df['text'].str.len()
    
    # Overall statistics
    print(f"\n  Overall:")
    print(f"    Mean: {train_df['text_length'].mean():.1f}")
    print(f"    Median: {train_df['text_length'].median():.1f}")
    print(f"    Std: {train_df['text_length'].std():.1f}")
    print(f"    Min: {train_df['text_length'].min()}")
    print(f"    Max: {train_df['text_length'].max()}")
    
    # Per-class statistics
    print(f"\n  Per class:")
    for label in sorted(train_df['label'].unique()):
        class_texts = train_df[train_df['label'] == label]['text_length']
        print(f"    Class {label}: mean={class_texts.mean():.1f}, median={class_texts.median():.1f}")
    
    # Sample texts from each class
    print(f"\nSample texts from each class:")
    for label in sorted(train_df['label'].unique()):
        print(f"\n  Class {label}:")
        samples = train_df[train_df['label'] == label].sample(min(2, len(train_df[train_df['label'] == label])))
        for idx, row in samples.iterrows():
            text = row['text'][:200] + "..." if len(row['text']) > 200 else row['text']
            print(f"    - {text}")
    
    # Create visualizations
    create_visualizations(dataset_name, train_df, val_df, test_df)
    
    # Recommendations
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS for {dataset_name}:")
    print(f"{'='*60}")
    
    if imbalance_ratio > 2:
        print("\n⚠️  Significant class imbalance detected!")
        print("\nSuggested strategies:")
        print("1. Use class weights in loss function")
        print("2. Try oversampling minority classes or undersampling majority class")
        print("3. Use stratified sampling for train/val splits")
        print("4. Consider using macro F1-score instead of accuracy")
        print("5. Use focal loss for extreme imbalance")
        
        # Calculate class weights
        total_samples = len(train_df)
        n_classes = len(train_df['label'].unique())
        class_weights = {}
        for label, count in Counter(train_df['label']).items():
            weight = total_samples / (n_classes * count)
            class_weights[label] = weight
        
        print(f"\nSuggested class weights:")
        for label, weight in sorted(class_weights.items()):
            print(f"  Class {label}: {weight:.3f}")
    else:
        print("\n✓ Class distribution is relatively balanced.")
    
    return {
        'dataset': dataset_name,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'n_classes': len(train_df['label'].unique()),
        'imbalance_ratio': imbalance_ratio,
        'mean_text_length': train_df['text_length'].mean(),
        'class_distribution': dict(Counter(test_df['label']))
    }


def create_visualizations(dataset_name, train_df, val_df, test_df):
    """Create and save visualization plots."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{dataset_name.upper()} Dataset Analysis', fontsize=16)
    
    # 1. Class distribution bar plot
    ax = axes[0, 0]
    all_data = pd.concat([
        train_df.assign(split='Train'),
        val_df.assign(split='Val'),
        test_df.assign(split='Test')
    ])
    
    class_counts = all_data.groupby(['split', 'label']).size().unstack(fill_value=0)
    class_counts.plot(kind='bar', ax=ax)
    ax.set_title('Class Distribution by Split')
    ax.set_xlabel('Split')
    ax.set_ylabel('Count')
    ax.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Text length distribution
    ax = axes[0, 1]
    train_df['text_length'] = train_df['text'].str.len()
    for label in sorted(train_df['label'].unique()):
        class_lengths = train_df[train_df['label'] == label]['text_length']
        ax.hist(class_lengths, alpha=0.5, label=f'Class {label}', bins=30)
    ax.set_title('Text Length Distribution by Class')
    ax.set_xlabel('Text Length (characters)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.set_xlim(0, np.percentile(train_df['text_length'], 95))  # Clip at 95th percentile
    
    # 3. Class distribution pie chart (test set)
    ax = axes[1, 0]
    test_counts = Counter(test_df['label'])
    labels = [f'Class {k}' for k in sorted(test_counts.keys())]
    sizes = [test_counts[k] for k in sorted(test_counts.keys())]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('Test Set Class Distribution')
    
    # 4. Word count statistics
    ax = axes[1, 1]
    train_df['word_count'] = train_df['text'].str.split().str.len()
    word_stats = train_df.groupby('label')['word_count'].describe()[['mean', 'std', 'min', 'max']]
    word_stats.plot(kind='bar', ax=ax)
    ax.set_title('Word Count Statistics by Class')
    ax.set_xlabel('Class')
    ax.set_ylabel('Word Count')
    ax.legend(['Mean', 'Std', 'Min', 'Max'])
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('scripts/analysis_plots')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f'{dataset_name}_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to: {output_dir / f'{dataset_name}_analysis.png'}")
    
    # Close plot to free memory
    plt.close()


def analyze_all_datasets(data_path):
    """Analyze all datasets and create a comparison table."""
    
    datasets = ['amazon', 'imdb', 'ag_news', 'dbpedia']
    results = []
    
    for dataset in datasets:
        try:
            result = analyze_dataset(dataset, data_path)
            results.append(result)
        except Exception as e:
            print(f"\nError analyzing {dataset}: {e}")
    
    # Create comparison table
    if results:
        print(f"\n{'='*80}")
        print("DATASET COMPARISON")
        print(f"{'='*80}")
        
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        
        # Save to CSV
        df.to_csv('scripts/dataset_comparison.csv', index=False)
        print(f"\nComparison saved to: scripts/dataset_comparison.csv")


def main():
    parser = argparse.ArgumentParser(description='Analyze text classification datasets')
    parser.add_argument('--dataset', type=str, 
                        choices=['ag_news', 'imdb', 'amazon', 'dbpedia', 'all'],
                        default='all',
                        help='Dataset to analyze (default: all)')
    parser.add_argument('--data-path', type=str, default='.',
                        help='Path to data directory')
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        analyze_all_datasets(args.data_path)
    else:
        analyze_dataset(args.dataset, args.data_path)


if __name__ == '__main__':
    main()