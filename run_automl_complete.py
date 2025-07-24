#!/usr/bin/env python3
"""
One-Click Complete AutoML Solution for Text Classification

This script provides a complete AutoML pipeline that:
1. Automatically detects and loads the dataset
2. Selects the best approach based on dataset characteristics
3. Optimizes within the 24-hour budget
4. Produces a trained model ready for inference

Usage:
    python run_automl_complete.py --dataset <dataset_name> --data-path <path> [--budget <hours>]
"""

import argparse
import logging
import time
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from automl.core import TextAutoML
from automl.datasets import AGNewsDataset, AmazonReviewsDataset, DBpediaDataset, IMDBDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoMLComplete:
    """Complete AutoML solution with automatic configuration."""
    
    def __init__(self, budget_hours=24):
        self.budget_hours = budget_hours
        self.budget_seconds = budget_hours * 3600
        self.start_time = time.time()
        
        # Dataset characteristics for intelligent approach selection
        self.dataset_profiles = {
            'amazon': {
                'size': 'small',
                'num_classes': 3,
                'avg_length': 'medium',
                'class_balance': 'imbalanced',
                'recommended_approach': 'logistic',
                'recommended_epochs': 5,
                'hpo_budget_ratio': 0.3
            },
            'imdb': {
                'size': 'medium',
                'num_classes': 2,
                'avg_length': 'long',
                'class_balance': 'balanced',
                'recommended_approach': 'logistic',
                'recommended_epochs': 5,
                'hpo_budget_ratio': 0.3
            },
            'ag_news': {
                'size': 'large',
                'num_classes': 4,
                'avg_length': 'short',
                'class_balance': 'balanced',
                'recommended_approach': 'ffnn',
                'recommended_epochs': 10,
                'hpo_budget_ratio': 0.4
            },
            'dbpedia': {
                'size': 'very_large',
                'num_classes': 14,
                'avg_length': 'medium',
                'class_balance': 'balanced',
                'recommended_approach': 'logistic',
                'recommended_epochs': 3,
                'hpo_budget_ratio': 0.2
            }
        }
        
    def get_remaining_budget(self):
        """Get remaining time budget in seconds."""
        elapsed = time.time() - self.start_time
        return max(0, self.budget_seconds - elapsed)
    
    def select_approach(self, dataset_name, train_size):
        """Intelligently select approach based on dataset characteristics."""
        profile = self.dataset_profiles.get(dataset_name, {})
        
        # Adjust based on actual training size
        if train_size < 5000:
            # Small dataset: can afford neural networks
            if profile.get('num_classes', 2) == 2:
                approach = 'ffnn'  # Binary classification
            else:
                approach = 'ffnn' if profile.get('avg_length') == 'short' else 'logistic'
        elif train_size < 50000:
            # Medium dataset: balance between performance and efficiency
            approach = profile.get('recommended_approach', 'logistic')
        else:
            # Large dataset: prioritize efficiency
            approach = 'logistic'
            
        logger.info(f"Selected approach: {approach} (train_size: {train_size})")
        return approach
    
    def allocate_budget(self, dataset_name, train_size):
        """Allocate time budget across different optimization strategies."""
        remaining = self.get_remaining_budget()
        profile = self.dataset_profiles.get(dataset_name, {})
        
        # Default allocations
        allocations = {
            'data_loading': min(300, remaining * 0.02),  # Max 5 minutes
            'initial_training': min(600, remaining * 0.1),  # Max 10 minutes
            'hpo': remaining * profile.get('hpo_budget_ratio', 0.3),
            'ensemble': remaining * 0.2,
            'final_training': remaining * 0.1,
            'buffer': remaining * 0.05
        }
        
        # Adjust for very large datasets
        if train_size > 100000:
            allocations['hpo'] *= 0.5  # Reduce HPO budget
            allocations['ensemble'] *= 0.5  # Skip or reduce ensemble
            
        logger.info(f"Budget allocation (hours): {{k: v/3600:.2f for k, v in allocations.items()}}")
        return allocations
    
    def run_pipeline(self, dataset_name, data_path, output_path=None):
        """Run the complete AutoML pipeline."""
        logger.info("="*80)
        logger.info(f"AutoML Complete Pipeline - {dataset_name.upper()}")
        logger.info(f"Total budget: {self.budget_hours} hours")
        logger.info(f"Start time: {datetime.now()}")
        logger.info("="*80)
        
        # Step 1: Load dataset
        logger.info("\n[Step 1/6] Loading dataset...")
        dataset_class = {
            'amazon': AmazonReviewsDataset,
            'imdb': IMDBDataset,
            'ag_news': AGNewsDataset,
            'dbpedia': DBpediaDataset
        }[dataset_name]
        
        dataset = dataset_class(data_path)
        data_info = dataset.create_dataloaders(val_size=0.2, random_state=42)
        
        train_df = data_info['train_df']
        val_df = data_info.get('val_df')
        test_df = data_info['test_df']
        num_classes = data_info['num_classes']
        
        logger.info(f"Dataset loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
        logger.info(f"Number of classes: {num_classes}")
        
        # Step 2: Select approach and allocate budget
        logger.info("\n[Step 2/6] Selecting approach and allocating budget...")
        approach = self.select_approach(dataset_name, len(train_df))
        budget_allocation = self.allocate_budget(dataset_name, len(train_df))
        
        # Step 3: Initial training with best known configuration
        logger.info("\n[Step 3/6] Initial training with optimized configuration...")
        
        initial_config = {
            'seed': 42,
            'approach': approach,
            'dataset_name': dataset_name,
            'use_preprocessing': True,
            'use_augmentation': dataset_name in ['amazon', 'ag_news'],  # Augment smaller/imbalanced datasets
            'augmentation_strength': 'light' if len(train_df) > 50000 else 'medium',
            'balance_augmentation': True
        }
        
        # Approach-specific configurations
        if approach == 'logistic':
            initial_config.update({
                'vocab_size': min(20000, len(train_df) * 2),
                'logistic_C': 1.0,
                'logistic_max_iter': 1000
            })
        elif approach == 'ffnn':
            initial_config.update({
                'use_enhanced_ffnn': True,
                'ffnn_hidden': 256,
                'ffnn_num_layers': 3,
                'ffnn_dropout_rate': 0.2,
                'epochs': min(10, budget_allocation['initial_training'] // 60),
                'batch_size': 64,
                'lr': 0.001
            })
        elif approach == 'lstm':
            initial_config.update({
                'lstm_emb_dim': 128,
                'lstm_hidden_dim': 128,
                'epochs': min(5, budget_allocation['initial_training'] // 120),
                'batch_size': 32,
                'lr': 0.001
            })
        
        automl = TextAutoML(**initial_config)
        
        # Run initial training
        initial_start = time.time()
        initial_val_err = automl.fit(train_df, val_df, num_classes, save_path=output_path)
        initial_time = time.time() - initial_start
        
        logger.info(f"Initial training completed in {initial_time:.1f}s")
        logger.info(f"Initial validation accuracy: {1 - initial_val_err:.4f}")
        
        # Step 4: Hyperparameter optimization if budget allows
        if self.get_remaining_budget() > budget_allocation['hpo'] and budget_allocation['hpo'] > 600:
            logger.info("\n[Step 4/6] Running hyperparameter optimization...")
            
            hpo_trials = min(
                50,  # Max trials
                int(budget_allocation['hpo'] / 120)  # Estimate 2 minutes per trial
            )
            
            hpo_start = time.time()
            hpo_val_err = automl.fit_with_hpo(
                train_df, val_df, num_classes,
                n_trials=hpo_trials,
                timeout=int(budget_allocation['hpo']),
                sampler='tpe',
                pruner='median' if len(train_df) > 10000 else None,
                save_path=output_path
            )
            hpo_time = time.time() - hpo_start
            
            logger.info(f"HPO completed in {hpo_time:.1f}s with {hpo_trials} trials")
            logger.info(f"HPO validation accuracy: {1 - hpo_val_err:.4f}")
            
            best_val_err = min(initial_val_err, hpo_val_err)
        else:
            logger.info("\n[Step 4/6] Skipping HPO due to budget constraints")
            best_val_err = initial_val_err
        
        # Step 5: Ensemble if budget allows and worth it
        if (self.get_remaining_budget() > budget_allocation['ensemble'] and 
            budget_allocation['ensemble'] > 1200 and  # At least 20 minutes
            dataset_name in ['amazon', 'imdb']):  # Ensemble works well for binary/ternary
            
            logger.info("\n[Step 5/6] Building ensemble model...")
            
            ensemble_methods = ['logistic', 'ffnn'] if approach != 'lstm' else ['logistic', 'lstm']
            individual_trials = min(5, int(budget_allocation['ensemble'] / 600))
            
            ensemble_start = time.time()
            ensemble_val_err = automl.fit_with_ensemble(
                train_df, val_df, num_classes,
                ensemble_methods=ensemble_methods,
                ensemble_type='weighted',
                individual_trials=individual_trials,
                save_path=output_path
            )
            ensemble_time = time.time() - ensemble_start
            
            logger.info(f"Ensemble completed in {ensemble_time:.1f}s")
            logger.info(f"Ensemble validation accuracy: {1 - ensemble_val_err:.4f}")
            
            best_val_err = min(best_val_err, ensemble_val_err)
        else:
            logger.info("\n[Step 5/6] Skipping ensemble due to budget/dataset constraints")
        
        # Step 6: Final evaluation
        logger.info("\n[Step 6/6] Final evaluation on test set...")
        
        test_preds, test_labels = automl.predict(test_df)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report
        
        if not np.isnan(test_labels).any():
            test_acc = accuracy_score(test_labels, test_preds)
            logger.info(f"Test accuracy: {test_acc:.4f}")
            logger.info("\nClassification Report:")
            logger.info(classification_report(test_labels, test_preds))
        else:
            test_acc = None
            logger.info("Test labels not available (competition dataset)")
        
        # Save results
        if output_path is None:
            output_path = Path(f"results/automl_complete/{dataset_name}/seed=42")
        else:
            output_path = Path(output_path)
            
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        np.save(output_path / "test_preds.npy", test_preds)
        
        # Save configuration and results
        total_time = time.time() - self.start_time
        results = {
            'dataset': dataset_name,
            'approach': approach,
            'validation_error': float(best_val_err),
            'validation_accuracy': float(1 - best_val_err),
            'test_accuracy': float(test_acc) if test_acc else None,
            'total_time_seconds': total_time,
            'total_time_hours': total_time / 3600,
            'configuration': initial_config,
            'budget_allocation': {k: v/3600 for k, v in budget_allocation.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path / "results.yaml", 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
            
        # Save model checkpoint
        if hasattr(automl, 'model') and automl.model is not None:
            import torch
            torch.save(automl.model.state_dict(), output_path / "final_model.pth")
            
        logger.info(f"\nResults saved to: {output_path}")
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED")
        logger.info("="*80)
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Best validation accuracy: {1 - best_val_err:.4f}")
        if test_acc:
            logger.info(f"Test accuracy: {test_acc:.4f}")
        logger.info(f"Output directory: {output_path}")
        
        return results


def main():
    """Main entry point for one-click AutoML."""
    parser = argparse.ArgumentParser(description="One-Click AutoML for Text Classification")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["amazon", "imdb", "ag_news", "dbpedia"],
        help="Dataset to train on"
    )
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("."),
        help="Path to dataset files"
    )
    
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to save results (default: results/automl_complete/<dataset>/seed=42)"
    )
    
    parser.add_argument(
        "--budget",
        type=float,
        default=24.0,
        help="Time budget in hours (default: 24)"
    )
    
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with 1-hour budget"
    )
    
    args = parser.parse_args()
    
    # Adjust budget for quick test
    if args.quick_test:
        budget = 1.0
        logger.info("Running in quick test mode (1 hour budget)")
    else:
        budget = args.budget
    
    # Run the pipeline
    automl_complete = AutoMLComplete(budget_hours=budget)
    results = automl_complete.run_pipeline(
        dataset_name=args.dataset,
        data_path=args.data_path,
        output_path=args.output_path
    )
    
    # Check baseline comparison if available
    baselines = {
        'amazon': 0.81799,
        'imdb': 0.86993,
        'ag_news': 0.90265,
        'dbpedia': 0.97882
    }
    
    if results['test_accuracy'] and args.dataset in baselines:
        baseline = baselines[args.dataset]
        improvement = results['test_accuracy'] - baseline
        
        logger.info(f"\nBaseline comparison:")
        logger.info(f"  Baseline: {baseline:.4f}")
        logger.info(f"  Our result: {results['test_accuracy']:.4f}")
        logger.info(f"  Improvement: {improvement:+.4f}")
        
        if results['test_accuracy'] > baseline:
            logger.info("  🏆 BEATS BASELINE!")


if __name__ == "__main__":
    main()