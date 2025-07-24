#!/usr/bin/env python3
"""
Run experiments on all datasets with progress tracking.

Usage:
    python scripts/run_all_datasets.py --approach ffnn
"""

import argparse
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Target baselines from exam
BASELINES = {
    'amazon': 81.799,
    'imdb': 86.993,
    'ag_news': 90.265,
    'dbpedia': 97.882
}

def run_dataset(dataset, approach, epochs=5, data_path='.'):
    """Run experiment on a single dataset."""
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset.upper()}")
    print(f"Approach: {approach} | Epochs: {epochs}")
    print(f"Baseline accuracy: {BASELINES[dataset]}%")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    cmd = [
        'python', 'run.py',
        '--data-path', data_path,
        '--dataset', dataset,
        '--approach', approach,
        '--epochs', str(epochs)
    ]
    
    # Add approach-specific parameters
    if approach == 'ffnn':
        cmd.extend(['--ffnn-hidden', '256'])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Parse accuracy from output
        output_lines = result.stdout.split('\n')
        test_acc = None
        for line in output_lines:
            if 'Accuracy on test set:' in line:
                test_acc = float(line.split(':')[-1].strip())
                break
        
        elapsed = time.time() - start_time
        
        if test_acc:
            improvement = (test_acc * 100) - BASELINES[dataset]
            status = "✅ IMPROVED" if improvement > 0 else "❌ BELOW BASELINE"
            
            print(f"\nRESULTS:")
            print(f"  Test Accuracy: {test_acc*100:.2f}%")
            print(f"  Baseline: {BASELINES[dataset]:.2f}%")
            print(f"  Improvement: {improvement:+.2f}% {status}")
            print(f"  Time: {str(timedelta(seconds=int(elapsed)))}")
            
            return {
                'dataset': dataset,
                'accuracy': test_acc * 100,
                'baseline': BASELINES[dataset],
                'improvement': improvement,
                'time_seconds': elapsed
            }
        else:
            print("Failed to parse accuracy from output")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running {dataset}: {e}")
        if e.stdout:
            print(f"Output: {e.stdout[-500:]}")  # Last 500 chars
        return None


def main():
    parser = argparse.ArgumentParser(description='Run experiments on all datasets')
    parser.add_argument('--approach', type=str, default='ffnn',
                        choices=['ffnn', 'logistic', 'lstm', 'transformer'],
                        help='Approach to use')
    parser.add_argument('--data-path', type=str, default='.',
                        help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--datasets', nargs='+', 
                        default=['amazon', 'imdb', 'ag_news', 'dbpedia'],
                        help='Datasets to run')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"RUNNING ALL DATASETS EXPERIMENT")
    print(f"Approach: {args.approach}")
    print(f"Epochs: {args.epochs}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    results = []
    total_start = time.time()
    
    # Adjust epochs per dataset based on size
    dataset_epochs = {
        'amazon': args.epochs,
        'imdb': args.epochs,
        'ag_news': max(3, args.epochs - 2),  # Faster convergence
        'dbpedia': max(2, args.epochs - 3)   # Very fast convergence
    }
    
    for dataset in args.datasets:
        epochs = dataset_epochs.get(dataset, args.epochs)
        result = run_dataset(dataset, args.approach, epochs, args.data_path)
        if result:
            results.append(result)
    
    # Summary
    total_time = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    if results:
        df = pd.DataFrame(results)
        print("\nResults Table:")
        print(df.to_string(index=False, float_format='%.2f'))
        
        print(f"\nOverall Statistics:")
        print(f"  Average Accuracy: {df['accuracy'].mean():.2f}%")
        print(f"  Average Improvement: {df['improvement'].mean():+.2f}%")
        print(f"  Datasets Improved: {len(df[df['improvement'] > 0])}/{len(df)}")
        print(f"  Total Time: {str(timedelta(seconds=int(total_time)))}")
        
        # Save results
        results_file = f'scripts/results_{args.approach}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()