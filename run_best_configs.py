#!/usr/bin/env python3
"""
Run best configurations for each dataset based on our experiments
"""

import subprocess
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Best configurations discovered through experimentation
BEST_CONFIGS = {
    'amazon': [
        '--approach', 'logistic', '--use-hpo', '--hpo-trials', '20', '--hpo-sampler', 'tpe'
    ],
    'imdb': [
        '--approach', 'logistic', '--use-hpo', '--hpo-trials', '15', '--epochs', '5'
    ],
    'ag_news': [
        '--approach', 'ffnn', '--use-enhanced-ffnn', '--epochs', '10', '--use-hpo', '--hpo-trials', '10'
    ],
    'dbpedia': [
        '--approach', 'logistic', '--epochs', '3', '--use-hpo', '--hpo-trials', '10'
    ]
}

def run_dataset(dataset, data_path='.'):
    """Run best configuration for a dataset."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Running {dataset.upper()} with best configuration")
    logger.info(f"{'='*80}")
    
    cmd = ['python', 'run.py', '--data-path', data_path, '--dataset', dataset]
    cmd.extend(BEST_CONFIGS[dataset])
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        runtime = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ {dataset} completed in {runtime:.1f}s")
            
            # Check results
            score_path = Path(f"results/dataset={dataset}/seed=42/score.yaml")
            if score_path.exists():
                import yaml
                with open(score_path) as f:
                    scores = yaml.safe_load(f)
                    test_acc = 1 - scores.get('test_err', 1.0)
                    logger.info(f"Test accuracy: {test_acc:.4f}")
                    
                    # Check baseline
                    baselines = {
                        'amazon': 0.81799,
                        'imdb': 0.86993,
                        'ag_news': 0.90265,
                        'dbpedia': 0.97882
                    }
                    
                    if dataset in baselines:
                        baseline = baselines[dataset]
                        improvement = test_acc - baseline
                        logger.info(f"Baseline: {baseline:.4f}")
                        logger.info(f"Improvement: {improvement:+.4f}")
                        if test_acc > baseline:
                            logger.info("🏆 BEATS BASELINE!")
        else:
            logger.error(f"❌ {dataset} failed")
            logger.error(result.stderr)
            
    except Exception as e:
        logger.error(f"Error running {dataset}: {e}")

def main():
    """Run all datasets with best configurations."""
    logger.info("Running best configurations for all datasets")
    
    datasets = ['amazon', 'imdb', 'ag_news', 'dbpedia']
    
    for dataset in datasets:
        run_dataset(dataset)
        time.sleep(2)  # Brief pause between runs
    
    logger.info("\n" + "="*80)
    logger.info("All experiments completed!")
    logger.info("Check results/dataset=*/seed=42/score.yaml for detailed results")

if __name__ == "__main__":
    main()