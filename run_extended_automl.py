#!/usr/bin/env python3
"""
Extended AutoML System - Complete Implementation

This script demonstrates the full capabilities of the extended AutoML system:
- Multiple text representations (TF-IDF, GloVe, FastText, Sentence-BERT)
- Traditional ML models (SVM, XGBoost, LightGBM, CatBoost)
- Deep learning models (GRU, CNN, Hierarchical Attention)
- Transformer fine-tuning (BERT, RoBERTa, ALBERT, etc.)
- Advanced augmentation strategies
- Multi-fidelity optimization
- Hybrid models

Usage:
    python run_extended_automl.py --dataset amazon --data-path ./data --approach auto
    python run_extended_automl.py --dataset imdb --approach svm_rbf --use-embeddings glove
    python run_extended_automl.py --dataset ag_news --approach lightgbm --use-hpo
    python run_extended_automl.py --dataset dbpedia --approach bert --use-multi-fidelity
"""

import argparse
import logging
import time
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

# Import the extended AutoML system
try:
    from automl.core_extended import ExtendedTextAutoML
except ImportError:
    # Fallback to regular core with extended features
    from automl.core import TextAutoML as ExtendedTextAutoML
    
from automl.datasets import AGNewsDataset, AmazonReviewsDataset, DBpediaDataset, IMDBDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtendedAutoMLRunner:
    """Runner for the extended AutoML system."""
    
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        
    def run(self):
        """Run the extended AutoML pipeline."""
        logger.info("=" * 80)
        logger.info("Extended AutoML System")
        logger.info(f"Dataset: {self.args.dataset}")
        logger.info(f"Approach: {self.args.approach}")
        logger.info("=" * 80)
        
        # Load dataset
        dataset = self._load_dataset()
        data_info = dataset.create_dataloaders(val_size=0.2, random_state=42)
        
        train_df = data_info['train_df']
        val_df = data_info.get('val_df')
        test_df = data_info['test_df']
        num_classes = data_info['num_classes']
        
        logger.info(f"Dataset loaded: {len(train_df)} train, {len(val_df) if val_df else 0} val, {len(test_df)} test")
        logger.info(f"Number of classes: {num_classes}")
        
        # Configure AutoML system
        config = self._build_config()
        
        # Create AutoML instance
        automl = ExtendedTextAutoML(**config)
        
        # Run optimization or direct training
        if self.args.use_hpo:
            logger.info(f"\nRunning HPO with {self.args.hpo_trials} trials...")
            results = automl.optimize_hyperparameters(
                train_df, val_df, num_classes,
                n_trials=self.args.hpo_trials,
                timeout=self.args.hpo_timeout
            )
            logger.info(f"Best parameters: {results['best_params']}")
            logger.info(f"Best validation error: {results['best_value']:.4f}")
        else:
            logger.info("\nTraining model...")
            val_error = automl.fit(train_df, val_df, num_classes)
            logger.info(f"Validation error: {val_error:.4f}")
            logger.info(f"Validation accuracy: {1 - val_error:.4f}")
        
        # Make predictions
        logger.info("\nMaking predictions on test set...")
        test_preds, test_labels = automl.predict(test_df)
        
        # Calculate metrics
        if test_labels is not None and not np.isnan(test_labels).any():
            from sklearn.metrics import accuracy_score, classification_report
            test_acc = accuracy_score(test_labels, test_preds)
            logger.info(f"Test accuracy: {test_acc:.4f}")
            logger.info("\nClassification Report:")
            logger.info(classification_report(test_labels, test_preds))
        else:
            logger.info("Test labels not available")
            
        # Save results
        self._save_results(test_preds, val_error if not self.args.use_hpo else results['best_value'])
        
        # Print summary
        total_time = time.time() - self.start_time
        logger.info("\n" + "=" * 80)
        logger.info("EXECUTION COMPLETE")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info("=" * 80)
        
    def _load_dataset(self):
        """Load the specified dataset."""
        dataset_map = {
            'amazon': AmazonReviewsDataset,
            'imdb': IMDBDataset,
            'ag_news': AGNewsDataset,
            'dbpedia': DBpediaDataset
        }
        
        dataset_class = dataset_map[self.args.dataset]
        return dataset_class(self.args.data_path)
        
    def _build_config(self):
        """Build configuration for AutoML system."""
        config = {
            'approach': self.args.approach,
            'seed': self.args.seed,
            'device': self.args.device,
            'use_preprocessing': not self.args.no_preprocessing,
            'use_augmentation': self.args.use_augmentation,
            'use_multi_fidelity': self.args.use_multi_fidelity,
        }
        
        # Embedding configuration
        if self.args.use_embeddings:
            config['use_pretrained_embeddings'] = True
            config['embedding_type'] = self.args.use_embeddings
            config['embedding_dim'] = self.args.embedding_dim
            
        # Augmentation configuration
        if self.args.use_augmentation:
            config['augmentation_types'] = self.args.augmentation_types
            config['augmentation_strength'] = self.args.augmentation_strength
            
        # Model-specific configuration
        model_config = {}
        
        # Traditional ML configs
        if self.args.approach.startswith('svm'):
            model_config['C'] = self.args.svm_c
            model_config['gamma'] = self.args.svm_gamma
        elif self.args.approach in ['xgboost', 'lightgbm']:
            model_config['n_estimators'] = self.args.n_estimators
            model_config['max_depth'] = self.args.max_depth
            model_config['learning_rate'] = self.args.learning_rate
            
        # Deep learning configs
        elif self.args.approach in ['gru', 'bigru', 'lstm', 'cnn']:
            model_config['embedding_dim'] = self.args.embedding_dim
            model_config['hidden_dim'] = self.args.hidden_dim
            model_config['num_layers'] = self.args.num_layers
            model_config['dropout'] = self.args.dropout
            model_config['batch_size'] = self.args.batch_size
            model_config['num_epochs'] = self.args.epochs
            
        # Transformer configs
        elif self.args.approach in ['bert', 'roberta', 'albert', 'deberta']:
            model_config['max_length'] = self.args.max_seq_length
            model_config['batch_size'] = self.args.batch_size
            model_config['learning_rate'] = self.args.transformer_lr
            model_config['num_epochs'] = self.args.epochs
            
        config['model_config'] = model_config
        
        return config
        
    def _save_results(self, predictions, val_error):
        """Save results to file."""
        output_path = Path(f"results/extended/{self.args.dataset}/{self.args.approach}/seed={self.args.seed}")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save predictions
        np.save(output_path / "test_preds.npy", predictions)
        
        # Save configuration and results
        results = {
            'dataset': self.args.dataset,
            'approach': self.args.approach,
            'validation_error': float(val_error),
            'validation_accuracy': float(1 - val_error),
            'timestamp': datetime.now().isoformat(),
            'args': vars(self.args)
        }
        
        with open(output_path / "results.yaml", 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
            
        logger.info(f"Results saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Extended AutoML System for Text Classification")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True,
                      choices=["amazon", "imdb", "ag_news", "dbpedia"],
                      help="Dataset to use")
    parser.add_argument("--data-path", type=Path, default=Path("."),
                      help="Path to dataset files")
    
    # Approach arguments
    parser.add_argument("--approach", type=str, default="auto",
                      help="Model approach to use (auto for automatic selection)")
    
    # Preprocessing arguments
    parser.add_argument("--no-preprocessing", action="store_true",
                      help="Disable text preprocessing")
    
    # Embedding arguments
    parser.add_argument("--use-embeddings", type=str, default=None,
                      choices=["glove", "fasttext", "word2vec", "sentence_bert"],
                      help="Use pre-trained embeddings")
    parser.add_argument("--embedding-dim", type=int, default=300,
                      help="Embedding dimension")
    
    # Augmentation arguments
    parser.add_argument("--use-augmentation", action="store_true",
                      help="Enable text augmentation")
    parser.add_argument("--augmentation-types", nargs="+", 
                      default=["synonym", "contextual"],
                      help="Types of augmentation to use")
    parser.add_argument("--augmentation-strength", type=str, default="medium",
                      choices=["light", "medium", "heavy"],
                      help="Augmentation strength")
    
    # Optimization arguments
    parser.add_argument("--use-hpo", action="store_true",
                      help="Use hyperparameter optimization")
    parser.add_argument("--hpo-trials", type=int, default=50,
                      help="Number of HPO trials")
    parser.add_argument("--hpo-timeout", type=int, default=None,
                      help="HPO timeout in seconds")
    parser.add_argument("--use-multi-fidelity", action="store_true",
                      help="Use multi-fidelity optimization")
    
    # Model-specific arguments
    # Traditional ML
    parser.add_argument("--svm-c", type=float, default=1.0,
                      help="SVM regularization parameter")
    parser.add_argument("--svm-gamma", type=str, default="scale",
                      help="SVM kernel coefficient")
    parser.add_argument("--n-estimators", type=int, default=100,
                      help="Number of estimators for tree-based models")
    parser.add_argument("--max-depth", type=int, default=6,
                      help="Max depth for tree-based models")
    
    # Deep learning
    parser.add_argument("--hidden-dim", type=int, default=128,
                      help="Hidden dimension for neural models")
    parser.add_argument("--num-layers", type=int, default=2,
                      help="Number of layers for neural models")
    parser.add_argument("--dropout", type=float, default=0.2,
                      help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=32,
                      help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                      help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10,
                      help="Number of training epochs")
    
    # Transformer-specific
    parser.add_argument("--max-seq-length", type=int, default=512,
                      help="Maximum sequence length for transformers")
    parser.add_argument("--transformer-lr", type=float, default=2e-5,
                      help="Learning rate for transformers")
    
    # General arguments
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                      help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Run the extended AutoML system
    runner = ExtendedAutoMLRunner(args)
    runner.run()


if __name__ == "__main__":
    main()