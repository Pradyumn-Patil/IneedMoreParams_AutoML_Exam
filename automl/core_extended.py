"""
Extended AutoML Core with Full Feature Set

Integrates all advanced components:
- Multiple embeddings (GloVe, FastText, Word2Vec, Sentence-BERT)
- Traditional ML models (SVM, XGBoost, LightGBM, etc.)
- Deep models (GRU, Hierarchical Attention, etc.)
- Proper transformer fine-tuning
- Advanced augmentation strategies
- Multi-dimensional fidelity management
- Hybrid models
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

from .embeddings import EmbeddingManager, create_embedding_matrix
from .traditional_ml import TraditionalMLFactory, get_traditional_ml_search_space
from .transformers_advanced import (
    TransformerModelFactory, TransformerConfig, 
    TransformerTrainer, get_transformer_search_space
)
from .models_extended import (
    GRUClassifier, MultiScaleCNN, HierarchicalAttentionNetwork,
    TextCNNLSTM, WideAndDeepText, HybridDeepMLModel
)
from .augmentation_advanced import create_augmentation_pipeline
from .fidelity_manager import (
    FidelityManager, FidelityType, AdaptiveFidelityScheduler,
    create_default_fidelity_configs
)
from .preprocessing import AdvancedTextPreprocessor
from .datasets import get_dataset
from .progress_utils import DataLoadingProgress, TrainingProgress

logger = logging.getLogger(__name__)


class ExtendedTextAutoML:
    """Extended AutoML system with full feature set."""
    
    SUPPORTED_APPROACHES = [
        # Traditional ML
        'svm_linear', 'svm_rbf', 'svm_poly',
        'xgboost', 'lightgbm', 'catboost',
        'random_forest', 'extra_trees', 'gradient_boosting',
        'naive_bayes',
        # Deep Learning
        'ffnn', 'enhanced_ffnn', 'residual_ffnn',
        'lstm', 'enhanced_lstm', 'cnn_lstm',
        'gru', 'bigru', 'hierarchical_attention',
        'multi_scale_cnn', 'wide_deep',
        # Transformers
        'bert', 'roberta', 'albert', 'deberta',
        'distilbert', 'xlnet', 'electra',
        # Hybrid
        'hybrid_svm_bert', 'hybrid_xgboost_lstm'
    ]
    
    def __init__(
        self,
        approach: str = 'auto',
        seed: int = 42,
        device: str = 'auto',
        # Preprocessing
        use_preprocessing: bool = True,
        preprocessing_config: Optional[Dict] = None,
        # Embeddings
        embedding_type: Optional[str] = None,
        embedding_dim: int = 300,
        use_pretrained_embeddings: bool = False,
        # Augmentation
        use_augmentation: bool = False,
        augmentation_types: List[str] = ['synonym', 'contextual'],
        augmentation_strength: str = 'medium',
        # Fidelity
        use_multi_fidelity: bool = False,
        fidelity_strategy: str = 'progressive',
        # Model specific
        model_config: Optional[Dict] = None,
        **kwargs
    ):
        self.approach = approach
        self.seed = seed
        self.device = self._setup_device(device)
        
        # Preprocessing
        self.use_preprocessing = use_preprocessing
        self.preprocessing_config = preprocessing_config or {}
        self.preprocessor = None
        
        # Embeddings
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.embedding_manager = EmbeddingManager()
        
        # Augmentation
        self.use_augmentation = use_augmentation
        self.augmentation_types = augmentation_types
        self.augmentation_strength = augmentation_strength
        
        # Fidelity
        self.use_multi_fidelity = use_multi_fidelity
        self.fidelity_strategy = fidelity_strategy
        self.fidelity_manager = None
        
        # Model configuration
        self.model_config = model_config or {}
        self.model = None
        self.label_encoder = LabelEncoder()
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
        
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        num_classes: int = None,
        save_path: Optional[Path] = None,
        **kwargs
    ):
        """Fit the AutoML system."""
        logger.info(f"Starting ExtendedTextAutoML fit with approach: {self.approach}")
        
        # Auto-select approach if needed
        if self.approach == 'auto':
            self.approach = self._auto_select_approach(train_df, num_classes)
            logger.info(f"Auto-selected approach: {self.approach}")
            
        # Prepare data
        train_texts, train_labels = self._prepare_data(train_df, 'train')
        val_texts, val_labels = self._prepare_data(val_df, 'val') if val_df is not None else (None, None)
        
        # Encode labels
        train_labels = self.label_encoder.fit_transform(train_labels)
        if val_labels is not None:
            val_labels = self.label_encoder.transform(val_labels)
            
        # Apply preprocessing
        if self.use_preprocessing:
            logger.info("Applying text preprocessing...")
            self.preprocessor = AdvancedTextPreprocessor(**self.preprocessing_config)
            train_texts = self.preprocessor.fit_transform(train_texts)
            if val_texts is not None:
                val_texts = self.preprocessor.transform(val_texts)
                
        # Apply augmentation
        if self.use_augmentation:
            logger.info("Applying text augmentation...")
            augmenter = create_augmentation_pipeline(
                self.augmentation_types,
                self.augmentation_strength,
                str(self.device)
            )
            train_texts, train_labels = augmenter.augment(
                train_texts, train_labels, 
                num_augmentations=augmenter.num_augmentations
            )
            
        # Setup fidelity manager
        if self.use_multi_fidelity:
            fidelity_configs = create_default_fidelity_configs(self.approach)
            self.fidelity_manager = FidelityManager(fidelity_configs)
            
        # Train model based on approach
        if self.approach in ['svm_linear', 'svm_rbf', 'svm_poly', 'xgboost', 
                            'lightgbm', 'catboost', 'random_forest', 
                            'extra_trees', 'gradient_boosting', 'naive_bayes']:
            return self._train_traditional_ml(
                train_texts, train_labels, val_texts, val_labels, 
                num_classes, save_path
            )
        elif self.approach in ['bert', 'roberta', 'albert', 'deberta',
                              'distilbert', 'xlnet', 'electra']:
            return self._train_transformer(
                train_texts, train_labels, val_texts, val_labels,
                num_classes, save_path
            )
        elif self.approach in ['gru', 'bigru', 'hierarchical_attention',
                              'multi_scale_cnn', 'text_cnn_lstm']:
            return self._train_deep_model(
                train_texts, train_labels, val_texts, val_labels,
                num_classes, save_path
            )
        elif self.approach.startswith('hybrid_'):
            return self._train_hybrid_model(
                train_texts, train_labels, val_texts, val_labels,
                num_classes, save_path
            )
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
            
    def _auto_select_approach(self, train_df: pd.DataFrame, num_classes: int) -> str:
        """Automatically select best approach based on dataset characteristics."""
        train_size = len(train_df)
        
        # Simple heuristics for approach selection
        if train_size < 5000:
            # Small dataset - use efficient models
            if num_classes == 2:
                return 'svm_rbf'
            else:
                return 'xgboost'
        elif train_size < 50000:
            # Medium dataset
            return 'lightgbm'
        else:
            # Large dataset
            if torch.cuda.is_available():
                return 'distilbert'
            else:
                return 'lightgbm'
                
    def _prepare_data(self, df: pd.DataFrame, split: str) -> Tuple[List[str], List[int]]:
        """Prepare data from dataframe."""
        if df is None:
            return None, None
            
        texts = df['text'].tolist() if 'text' in df else df.iloc[:, 0].tolist()
        labels = df['label'].tolist() if 'label' in df else df.iloc[:, 1].tolist()
        
        return texts, labels
        
    def _train_traditional_ml(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]],
        val_labels: Optional[np.ndarray],
        num_classes: int,
        save_path: Optional[Path]
    ) -> float:
        """Train traditional ML model."""
        logger.info(f"Training traditional ML model: {self.approach}")
        
        # Get embeddings based on configuration
        if self.use_pretrained_embeddings and self.embedding_type:
            logger.info(f"Using {self.embedding_type} embeddings")
            embedding = self.embedding_manager.get_embedding(
                self.embedding_type,
                dim=self.embedding_dim
            )
            X_train = embedding.encode(train_texts)
            X_val = embedding.encode(val_texts) if val_texts else None
        else:
            # Use TF-IDF
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            vocab_size = self.model_config.get('vocab_size', 10000)
            if self.fidelity_manager:
                vocab_params = self.fidelity_manager.apply_vocabulary_size_fidelity(
                    {'max_features': vocab_size}
                )
                vocab_size = vocab_params['max_features']
                
            vectorizer = TfidfVectorizer(
                max_features=vocab_size,
                ngram_range=(1, 3),
                sublinear_tf=True
            )
            X_train = vectorizer.fit_transform(train_texts).toarray()
            X_val = vectorizer.transform(val_texts).toarray() if val_texts else None
            
        # Apply feature selection if configured
        feature_selection = self.model_config.get('feature_selection')
        
        # Create and train model
        self.model = TraditionalMLFactory.create_model(
            self.approach,
            **self.model_config
        )
        
        # Apply fidelity to training data
        if self.fidelity_manager:
            data_indices = np.arange(len(X_train))
            sampled_indices = self.fidelity_manager.apply_data_fraction_fidelity(data_indices)
            X_train = X_train[sampled_indices]
            train_labels = train_labels[sampled_indices]
            
        # Train model
        self.model.fit(
            X_train, train_labels,
            feature_selection=feature_selection,
            scale_features=self.model_config.get('scale_features', False)
        )
        
        # Evaluate
        if X_val is not None:
            val_preds = self.model.predict(X_val)
            val_error = 1.0 - (val_preds == val_labels).mean()
            logger.info(f"Validation error: {val_error:.4f}")
            return val_error
            
        return 0.0
        
    def _train_transformer(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]],
        val_labels: Optional[np.ndarray],
        num_classes: int,
        save_path: Optional[Path]
    ) -> float:
        """Train transformer model with proper fine-tuning."""
        logger.info(f"Training transformer model: {self.approach}")
        
        # Create transformer configuration
        config = TransformerConfig(
            model_name=self.approach,
            num_labels=num_classes,
            **self.model_config
        )
        
        # Apply fidelity to configuration
        if self.fidelity_manager:
            model_params = self.fidelity_manager.apply_model_complexity_fidelity({})
            if 'freeze_layers' in model_params:
                config.freeze_layers = model_params['freeze_layers']
                
        # Create model
        self.model = TransformerModelFactory.create_model(config)
        
        # Create datasets
        from torch.utils.data import Dataset
        
        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def __len__(self):
                return len(self.texts)
                
            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                }
                
        # Apply sequence length fidelity
        max_length = config.max_length
        if self.fidelity_manager:
            seq_len = self.fidelity_manager.get_fidelity(FidelityType.SEQUENCE_LENGTH)
            if seq_len:
                max_length = min(int(seq_len), max_length)
                
        train_dataset = TextDataset(
            train_texts, train_labels,
            self.model.tokenizer, max_length
        )
        val_dataset = TextDataset(
            val_texts, val_labels,
            self.model.tokenizer, max_length
        ) if val_texts else None
        
        # Create trainer
        trainer = TransformerTrainer(self.model, config)
        
        # Apply training steps fidelity
        num_epochs = self.model_config.get('num_epochs', 3)
        if self.fidelity_manager:
            training_params = self.fidelity_manager.apply_training_steps_fidelity(
                {'epochs': num_epochs}
            )
            num_epochs = training_params['epochs']
            
        # Train
        output_dir = save_path or Path('./transformer_output')
        trainer.train(
            train_dataset,
            val_dataset,
            output_dir,
            num_epochs=num_epochs,
            batch_size=self.model_config.get('batch_size', 16)
        )
        
        # Evaluate
        if val_dataset:
            results = trainer.evaluate(val_dataset)
            val_error = 1.0 - results.get('eval_accuracy', 0.0)
            logger.info(f"Validation error: {val_error:.4f}")
            return val_error
            
        return 0.0
        
    def _train_deep_model(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]],
        val_labels: Optional[np.ndarray],
        num_classes: int,
        save_path: Optional[Path]
    ) -> float:
        """Train deep learning model (GRU, CNN, etc.)."""
        logger.info(f"Training deep model: {self.approach}")
        
        # Build vocabulary and convert to indices
        from sklearn.feature_extraction.text import CountVectorizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        # Create vocabulary
        vocab_size = self.model_config.get('vocab_size', 10000)
        vectorizer = CountVectorizer(max_features=vocab_size, token_pattern=r'\b\w+\b')
        vectorizer.fit(train_texts)
        
        # Convert texts to sequences
        def texts_to_sequences(texts):
            sequences = []
            for text in texts:
                tokens = text.lower().split()
                sequence = [vectorizer.vocabulary_.get(token, 0) + 1 for token in tokens]
                sequences.append(sequence)
            return sequences
            
        train_sequences = texts_to_sequences(train_texts)
        val_sequences = texts_to_sequences(val_texts) if val_texts else None
        
        # Pad sequences
        max_length = self.model_config.get('max_length', 200)
        if self.fidelity_manager:
            seq_len = self.fidelity_manager.get_fidelity(FidelityType.SEQUENCE_LENGTH)
            if seq_len:
                max_length = int(seq_len)
                
        X_train = pad_sequences(train_sequences, maxlen=max_length)
        X_val = pad_sequences(val_sequences, maxlen=max_length) if val_sequences else None
        
        # Create model based on approach
        if self.approach in ['gru', 'bigru']:
            self.model = GRUClassifier(
                vocab_size=vocab_size + 1,
                embedding_dim=self.model_config.get('embedding_dim', 128),
                hidden_dim=self.model_config.get('hidden_dim', 128),
                output_dim=num_classes,
                bidirectional=(self.approach == 'bigru'),
                **self.model_config
            )
        elif self.approach == 'multi_scale_cnn':
            self.model = MultiScaleCNN(
                vocab_size=vocab_size + 1,
                embedding_dim=self.model_config.get('embedding_dim', 128),
                num_filters=self.model_config.get('num_filters', 100),
                filter_sizes=self.model_config.get('filter_sizes', [3, 4, 5]),
                output_dim=num_classes,
                **self.model_config
            )
        else:
            raise ValueError(f"Unknown deep model: {self.approach}")
            
        # Move to device
        self.model = self.model.to(self.device)
        
        # Train model
        from .utils import train_neural_model
        
        val_error = train_neural_model(
            self.model,
            X_train, train_labels,
            X_val, val_labels,
            num_epochs=self.model_config.get('num_epochs', 10),
            batch_size=self.model_config.get('batch_size', 32),
            learning_rate=self.model_config.get('learning_rate', 0.001),
            device=self.device,
            save_path=save_path
        )
        
        return val_error
        
    def _train_hybrid_model(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        val_texts: Optional[List[str]],
        val_labels: Optional[np.ndarray],
        num_classes: int,
        save_path: Optional[Path]
    ) -> float:
        """Train hybrid model combining traditional ML with deep learning."""
        logger.info(f"Training hybrid model: {self.approach}")
        
        # Parse hybrid approach
        _, ml_type, deep_type = self.approach.split('_')
        
        # First train deep model to extract features
        if deep_type == 'bert':
            # Use pre-trained BERT for feature extraction
            from sentence_transformers import SentenceTransformer
            feature_extractor = SentenceTransformer('all-MiniLM-L6-v2')
            X_train = feature_extractor.encode(train_texts, show_progress_bar=True)
            X_val = feature_extractor.encode(val_texts) if val_texts else None
        else:
            # Train custom deep model for features
            # ... (implementation similar to _train_deep_model but returns features)
            raise NotImplementedError(f"Hybrid with {deep_type} not yet implemented")
            
        # Train traditional ML on extracted features
        if ml_type == 'svm':
            ml_approach = 'svm_rbf'
        elif ml_type == 'xgboost':
            ml_approach = 'xgboost'
        else:
            raise ValueError(f"Unknown ML type for hybrid: {ml_type}")
            
        self.model = TraditionalMLFactory.create_model(ml_approach)
        self.model.fit(X_train, train_labels)
        
        # Evaluate
        if X_val is not None:
            val_preds = self.model.predict(X_val)
            val_error = 1.0 - (val_preds == val_labels).mean()
            logger.info(f"Validation error: {val_error:.4f}")
            return val_error
            
        return 0.0
        
    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on test data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        test_texts, test_labels = self._prepare_data(test_df, 'test')
        
        # Apply preprocessing
        if self.preprocessor:
            test_texts = self.preprocessor.transform(test_texts)
            
        # Make predictions based on model type
        if hasattr(self.model, 'predict'):
            # Traditional ML or simple models
            if hasattr(self, 'vectorizer'):
                X_test = self.vectorizer.transform(test_texts).toarray()
            else:
                # Use appropriate feature extraction
                X_test = self._extract_features(test_texts)
            predictions = self.model.predict(X_test)
        else:
            # Deep learning models
            predictions = self._predict_deep(test_texts)
            
        # Decode labels
        predictions = self.label_encoder.inverse_transform(predictions)
        if test_labels:
            test_labels = self.label_encoder.inverse_transform(test_labels)
            
        return predictions, test_labels
        
    def optimize_hyperparameters(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_classes: int,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """Run hyperparameter optimization."""
        logger.info(f"Starting HPO for {self.approach} with {n_trials} trials")
        
        def objective(trial):
            # Get search space based on approach
            if self.approach in ['svm_linear', 'svm_rbf', 'xgboost', 'lightgbm']:
                params = self._sample_traditional_ml_params(trial)
            elif self.approach in ['bert', 'roberta']:
                params = self._sample_transformer_params(trial)
            else:
                params = self._sample_deep_params(trial)
                
            # Update model config
            self.model_config.update(params)
            
            # Train and evaluate
            val_error = self.fit(train_df, val_df, num_classes)
            
            return val_error
            
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Return best parameters
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
        
    def _sample_traditional_ml_params(self, trial: optuna.Trial) -> Dict:
        """Sample hyperparameters for traditional ML models."""
        search_space = get_traditional_ml_search_space(self.approach)
        params = {}
        
        for param, values in search_space.items():
            if isinstance(values, list):
                params[param] = trial.suggest_categorical(param, values)
            elif isinstance(values, tuple) and len(values) == 2:
                if isinstance(values[0], int):
                    params[param] = trial.suggest_int(param, values[0], values[1])
                else:
                    params[param] = trial.suggest_float(param, values[0], values[1])
                    
        return params
        
    def _sample_transformer_params(self, trial: optuna.Trial) -> Dict:
        """Sample hyperparameters for transformer models."""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.2),
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.1),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            'num_epochs': trial.suggest_int('num_epochs', 2, 5)
        }
        
    def _sample_deep_params(self, trial: optuna.Trial) -> Dict:
        """Sample hyperparameters for deep learning models."""
        return {
            'embedding_dim': trial.suggest_categorical('embedding_dim', [64, 128, 256]),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
        }