#!/usr/bin/env python3
"""
Run experiments on all datasets to get baseline results
"""

import subprocess
import time
import yaml
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Baselines from exam
BASELINES = {
    'amazon': 0.81799,
    'imdb': 0.86993,
    'ag_news': 0.90265,
    'dbpedia': 0.97882
}

# Optimized configurations based on dataset characteristics
DATASET_CONFIGS = {
    'amazon': {
        'approach': 'logistic',
        'epochs': 5,
        'use_hpo': True,
        'hpo_trials': 10,
        'use_augmentation': True,
        'augmentation_strength': 'medium'
    },
    'imdb': {
        'approach': 'logistic',  # Binary classification
        'epochs': 5,
        'use_hpo': True,
        'hpo_trials': 10,
        'use_augmentation': True,
        'augmentation_strength': 'light'  # IMDB has longer texts
    },
    'ag_news': {
        'approach': 'ffnn',
        'use_enhanced_ffnn': True,
        'epochs': 10,
        'use_hpo': True,
        'hpo_trials': 10,
        'use_augmentation': True,
        'augmentation_strength': 'medium'
    },
    'dbpedia': {
        'approach': 'logistic',  # Efficient for large dataset
        'epochs': 3,
        'data_fraction': 0.3,  # Use subset for faster testing
        'use_hpo': False,  # Skip HPO for initial test
        'use_augmentation': False  # Skip augmentation for large dataset
    }
}


def run_experiment(dataset, config):
    """Run experiment for a single dataset."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Running experiment on {dataset.upper()}")
    logger.info(f"Configuration: {config}")
    logger.info(f"{'='*80}")
    
    # Build command
    cmd = ['python', 'run.py', '--data-path', '.', '--dataset', dataset]
    
    # Add configuration parameters
    if 'approach' in config:
        cmd.extend(['--approach', config['approach']])
    
    if config.get('epochs'):
        cmd.extend(['--epochs', str(config['epochs'])])
        
    if config.get('data_fraction'):
        cmd.extend(['--data-fraction', str(config['data_fraction'])])
    
    if config.get('use_hpo'):
        cmd.append('--use-hpo')
        if 'hpo_trials' in config:
            cmd.extend(['--hpo-trials', str(config['hpo_trials'])])
    
    if config.get('use_enhanced_ffnn'):
        cmd.append('--use-enhanced-ffnn')
    
    if config.get('use_augmentation'):
        cmd.append('--use-augmentation')
        if 'augmentation_strength' in config:
            cmd.extend(['--augmentation-strength', config['augmentation_strength']])
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        runtime = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ Experiment completed in {runtime:.1f}s")
            
            # Read results
            score_path = Path(f"results/dataset={dataset}/seed=42/score.yaml")
            if score_path.exists():
                with open(score_path) as f:
                    scores = yaml.safe_load(f)
                    test_acc = 1 - scores.get('test_err', 1.0)
                    val_acc = 1 - scores.get('val_err', 1.0)
                    
                    baseline = BASELINES[dataset]
                    improvement = test_acc - baseline
                    
                    logger.info(f"Results:")
                    logger.info(f"  Test accuracy: {test_acc:.4f}")
                    logger.info(f"  Val accuracy: {val_acc:.4f}")
                    logger.info(f"  Baseline: {baseline:.4f}")
                    logger.info(f"  Improvement: {improvement:+.4f}")
                    
                    if test_acc > baseline:
                        logger.info("  🏆 BEATS BASELINE!")
                    
                    return {
                        'dataset': dataset,
                        'test_acc': test_acc,
                        'val_acc': val_acc,
                        'baseline': baseline,
                        'improvement': improvement,
                        'beats_baseline': test_acc > baseline,
                        'runtime': runtime,
                        'config': config
                    }
        else:
            logger.error(f"❌ Experiment failed")
            logger.error(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Experiment timed out after 30 minutes")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    return None


def main():
    """Run experiments on all datasets."""
    logger.info("Starting experiments on all datasets")
    logger.info(f"Timestamp: {datetime.now()}")
    
    results = []
    
    # Run experiments
    for dataset, config in DATASET_CONFIGS.items():
        result = run_experiment(dataset, config)
        if result:
            results.append(result)
        
        # Brief pause between experiments
        time.sleep(5)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)
    
    for result in results:
        logger.info(f"\n{result['dataset'].upper()}:")
        logger.info(f"  Test accuracy: {result['test_acc']:.4f}")
        logger.info(f"  Baseline: {result['baseline']:.4f}")
        logger.info(f"  Improvement: {result['improvement']:+.4f}")
        logger.info(f"  Status: {'✅ BEATS BASELINE' if result['beats_baseline'] else '❌ Below baseline'}")
        logger.info(f"  Runtime: {result['runtime']:.1f}s")
    
    # Save results
    results_path = Path('experiment_results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, default_flow_style=False)
    
    logger.info(f"\nResults saved to: {results_path}")
    
    # Overall statistics
    datasets_beating_baseline = sum(1 for r in results if r['beats_baseline'])
    avg_improvement = sum(r['improvement'] for r in results) / len(results) if results else 0
    
    logger.info(f"\nOverall:")
    logger.info(f"  Datasets beating baseline: {datasets_beating_baseline}/{len(results)}")
    logger.info(f"  Average improvement: {avg_improvement:+.4f}")


if __name__ == "__main__":
    main()