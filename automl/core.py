import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from automl.models import SimpleFFNN, EnhancedFFNN, LSTMClassifier, NASSearchableFFNN, NASSearchableLSTM, NASSearchSpace
from automl.utils import SimpleTextDataset, TextPreprocessor
from automl.progress_utils import ProgressTracker, DataLoadingProgress, HPOProgress, create_batch_progress_bar
from automl.augmentation import TextAugmenter, DatasetAugmenter, create_augmenter
from pathlib import Path
import logging
import time
from typing import Tuple, List
from collections import Counter
import yaml
from tqdm import tqdm

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, NSGAIISampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        RobertaTokenizer,
        RobertaForSequenceClassification,
        AlbertTokenizer,
        AlbertForSequenceClassification,
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
    )
    TRANSFORMERS_AVAILABLE = True
    try:
        from transformers.adapters import AdapterConfig, AdapterType
        ADAPTERS_AVAILABLE = True
    except ImportError:
        ADAPTERS_AVAILABLE = False
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    ADAPTERS_AVAILABLE = False

try:
    import neps
    NEPS_AVAILABLE = True
except ImportError:
    NEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TextAutoML:
    def __init__(
        self,
        seed=42,
        approach='auto',
        vocab_size=10000,
        token_length=128,
        epochs=5,
        batch_size=64,
        lr=1e-4,
        weight_decay=0.0,
        ffnn_hidden=128,
        use_enhanced_ffnn=False,  # Use EnhancedFFNN with residual connections
        ffnn_num_layers=3,  # Number of layers for enhanced FFNN
        ffnn_dropout_rate=0.1,  # Dropout rate for enhanced FFNN
        ffnn_use_residual=True,  # Use residual connections
        ffnn_use_layer_norm=True,  # Use layer normalization
        lstm_emb_dim=128,
        lstm_hidden_dim=128,
        fraction_layers_to_finetune: float=1.0,
        # LogisticRegression parameters
        logistic_C=1.0, # Regularization strength
        logistic_max_iter=1000,
        # HPO parameters
        hpo_sampler='tpe',  # 'tpe', 'random', 'cmaes', 'nsga2'
        hpo_pruner='median',  # 'median', 'successive_halving', 'hyperband'
        use_multi_fidelity=True,  # Enable/disable multi-fidelity optimization
        # NAS parameters
        use_nas=False,  # Enable/disable neural architecture search
        # Preprocessing parameters
        dataset_name=None,  # For dataset-specific preprocessing
        use_preprocessing=True,  # Enable/disable preprocessing
        custom_preprocessor=None,  # Custom preprocessor instance
        # Transformer parameters
        transformer_model='distilbert-base-uncased',  # Model name or path
        use_adapters=False,  # Use adapter layers for efficient fine-tuning
        adapter_reduction_factor=16,  # Adapter bottleneck dimension
        # Augmentation parameters
        use_augmentation=False,  # Enable text augmentation
        augmentation_strength='medium',  # 'light', 'medium', 'heavy'
        augmentation_factor=0.5,  # Fraction of data to augment
        balance_augmentation=True,  # Augment minority classes more
    ):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.approach = approach
        self.vocab_size = vocab_size
        self.token_length = token_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.ffnn_hidden = ffnn_hidden
        self.use_enhanced_ffnn = use_enhanced_ffnn
        self.ffnn_num_layers = ffnn_num_layers
        self.ffnn_dropout_rate = ffnn_dropout_rate
        self.ffnn_use_residual = ffnn_use_residual
        self.ffnn_use_layer_norm = ffnn_use_layer_norm
        self.lstm_emb_dim = lstm_emb_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fraction_layers_to_finetune = fraction_layers_to_finetune
        
        # LogisticRegression parameters
        self.logistic_C = logistic_C
        self.logistic_max_iter = logistic_max_iter
        
        # HPO parameters
        self.hpo_sampler = hpo_sampler
        self.hpo_pruner = hpo_pruner  # 'median', 'successive_halving', 'hyperband'
        self.use_multi_fidelity = use_multi_fidelity  # Enable/disable multi-fidelity optimization
        self.class_weights = None  # For handling class imbalance
        
        # NAS parameters
        self.use_nas = False  # Enable/disable neural architecture search
        
        # Preprocessing parameters
        self.dataset_name = dataset_name
        self.use_preprocessing = use_preprocessing
        if custom_preprocessor:
            self.preprocessor = custom_preprocessor
        elif use_preprocessing:
            self.preprocessor = TextPreprocessor.get_dataset_specific_preprocessor(dataset_name) if dataset_name else TextPreprocessor()
        else:
            self.preprocessor = None
            
        # Transformer parameters
        self.transformer_model = transformer_model
        self.use_adapters = use_adapters
        self.adapter_reduction_factor = adapter_reduction_factor
        
        # Augmentation parameters
        self.use_augmentation = use_augmentation
        self.augmentation_strength = augmentation_strength
        self.augmentation_factor = augmentation_factor
        self.balance_augmentation = balance_augmentation

        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.char_vectorizer = None  # For character n-grams
        self.num_classes = None
        self.train_texts = []
        self.train_labels = []
        self.val_texts = []
        self.val_labels = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_classes: int,
        approach=None,
        vocab_size=None,
        token_length=None,
        epochs=None,
        batch_size=None,
        lr=None,
        weight_decay=None,
        ffnn_hidden=None,
        lstm_emb_dim=None,
        lstm_hidden_dim=None,
        fraction_layers_to_finetune=None,
        load_path: Path=None,
        save_path: Path=None,
    ):
        """
        Fits a model to the given dataset.

        Parameters:
        - train_df (pd.DataFrame): Training data with 'text' and 'label' columns.
        - val_df (pd.DataFrame): Validation data with 'text' and 'label' columns.
        - num_classes (int): Number of classes in the dataset.
        - seed (int): Random seed for reproducibility.
        - approach (str): Model type - 'ffnn', 'lstm', or 'transformer'. Default is 'auto'.
        - vocab_size (int): Maximum vocabulary size.
        - token_length (int): Maximum token sequence length.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - lr (float): Learning rate.
        - weight_decay (float): Weight decay for optimizer.
        - ffnn_hidden (int): Hidden dimension size for FFNN.
        - lstm_emb_dim (int): Embedding dimension size for LSTM.
        - lstm_hidden_dim (int): Hidden dimension size for LSTM.
        
        Note: 
        - This method performs basic model training without optimization.
        - For hyperparameter optimization, use fit_with_hpo().
        - For neural architecture search, use fit_with_nas().
        - For combined optimization, use fit_with_nas_hpo().
        """
        if approach is not None: self.approach = approach
        if vocab_size is not None: self.vocab_size = vocab_size
        if token_length is not None: self.token_length = token_length
        if epochs is not None: self.epochs = epochs
        if batch_size is not None: self.batch_size = batch_size
        if lr is not None: self.lr = lr
        if weight_decay is not None: self.weight_decay = weight_decay
        if ffnn_hidden is not None: self.ffnn_hidden = ffnn_hidden
        if lstm_emb_dim is not None: self.lstm_emb_dim = lstm_emb_dim
        if lstm_hidden_dim is not None: self.lstm_hidden_dim = lstm_hidden_dim
        if fraction_layers_to_finetune is not None: self.fraction_layers_to_finetune = fraction_layers_to_finetune
        
        # Data loading progress
        data_progress = DataLoadingProgress(self.dataset_name)
        
        self.train_texts = train_df['text'].tolist()
        self.train_labels = train_df['label'].tolist()
        data_progress.update("Training data loaded", len(self.train_texts))
        
        self.val_texts = val_df['text'].tolist()
        self.val_labels = val_df['label'].tolist()
        data_progress.update("Validation data loaded", len(self.val_texts))
        
        # Apply preprocessing if enabled
        if self.preprocessor:
            data_progress.update("Applying text preprocessing...")
            self.train_texts = self.preprocessor.preprocess_batch(self.train_texts)
            self.val_texts = self.preprocessor.preprocess_batch(self.val_texts)
            data_progress.update("Preprocessing completed")
        
        # Apply augmentation if enabled
        if self.use_augmentation:
            data_progress.update("Applying text augmentation...")
            augmenter = create_augmenter(self.dataset_name, self.augmentation_strength)
            dataset_augmenter = DatasetAugmenter(
                augmenter, 
                augmentation_factor=self.augmentation_factor,
                balance_classes=self.balance_augmentation
            )
            
            # Augment training data only (not validation)
            self.train_texts, self.train_labels = dataset_augmenter.augment_dataset(
                self.train_texts, 
                self.train_labels,
                max_augmentations_per_sample=3
            )
            data_progress.update(f"Augmentation completed - {len(self.train_texts)} samples")
        
        data_progress.finish()
            
        self.num_classes = num_classes
        logger.info(f"Train class distribution: {Counter(self.train_labels)}")
        logger.info(f"Val class distribution: {Counter(self.val_labels)}")
        
        # Calculate class weights for imbalanced datasets
        train_counter = Counter(self.train_labels)
        total_samples = len(self.train_labels)
        self.class_weights = {}
        for label, count in train_counter.items():
            weight = total_samples / (len(train_counter) * count)
            self.class_weights[label] = weight
        
        # Check for significant imbalance
        max_weight = max(self.class_weights.values())
        min_weight = min(self.class_weights.values())
        imbalance_ratio = max_weight / min_weight
        
        if imbalance_ratio > 2:
            logger.info(f"Class imbalance detected (ratio: {imbalance_ratio:.2f})")
            logger.info(f"Class weights: {self.class_weights}")
            # Convert to tensor for PyTorch
            weight_tensor = torch.zeros(num_classes)
            for label, weight in self.class_weights.items():
                weight_tensor[label] = weight
            self.class_weight_tensor = weight_tensor.to(self.device)
        else:
            logger.info("Dataset is relatively balanced")
            self.class_weight_tensor = None

        dataset = None
        if self.approach in ['ffnn', 'logistic']:
            # Both use TF-IDF vectorization with enhanced features
            # Adaptive max_features based on dataset size
            dataset_size = len(self.train_texts)
            adaptive_max_features = min(self.vocab_size, max(5000, dataset_size // 10))
            
            self.vectorizer = TfidfVectorizer(
                max_features=adaptive_max_features,
                lowercase=True,
                min_df=2,    # ignore words appearing in less than 2 sentences
                max_df=0.8,  # ignore words appearing in > 80% of sentences
                sublinear_tf=True,  # use log-spaced term-frequency scoring
                ngram_range=(1, 3),  # unigrams, bigrams, and trigrams for better context
                analyzer='word',     # word-level analysis
                strip_accents='unicode',  # handle accented characters
                decode_error='ignore',    # ignore decode errors
                token_pattern=r'\b\w+\b',  # improved token pattern
            )
            
            # Also create a character-level TF-IDF vectorizer
            self.char_vectorizer = TfidfVectorizer(
                max_features=min(5000, adaptive_max_features // 2),  # fewer features for char n-grams
                lowercase=True,
                analyzer='char',
                ngram_range=(3, 5),  # character 3-grams to 5-grams
                min_df=2,
                max_df=0.8,
                sublinear_tf=True,
            )
            # Fit both word and character vectorizers
            X_train_word = self.vectorizer.fit_transform(self.train_texts)
            X_val_word = self.vectorizer.transform(self.val_texts)
            
            X_train_char = self.char_vectorizer.fit_transform(self.train_texts)
            X_val_char = self.char_vectorizer.transform(self.val_texts)
            
            # Combine word and character features
            from scipy.sparse import hstack
            X_train = hstack([X_train_word, X_train_char])
            X_val = hstack([X_val_word, X_val_char])
            
            logger.info(f"Combined TF-IDF features: {X_train.shape[1]} dimensions")
            
            if self.approach == 'ffnn':
                # Convert to dense arrays for PyTorch
                X_train_dense = X_train.toarray()
                X_val_dense = X_val.toarray()
                
                dataset = torch.utils.data.TensorDataset(
                    torch.tensor(X_train_dense, dtype=torch.float32),
                    torch.tensor(self.train_labels)
                )
                train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                _dataset = torch.utils.data.TensorDataset(
                    torch.tensor(X_val_dense, dtype=torch.float32),
                    torch.tensor(self.val_labels)
                )
                val_loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=True)
                if self.use_enhanced_ffnn:
                    self.model = EnhancedFFNN(
                        X_train_dense.shape[1], 
                        hidden=self.ffnn_hidden, 
                        output_dim=self.num_classes,
                        num_layers=self.ffnn_num_layers,
                        dropout_rate=self.ffnn_dropout_rate,
                        use_residual=self.ffnn_use_residual,
                        use_layer_norm=self.ffnn_use_layer_norm
                    )
                else:
                    self.model = SimpleFFNN(
                        X_train_dense.shape[1], hidden=self.ffnn_hidden, output_dim=self.num_classes
                    )
            elif self.approach == 'logistic': 
                # Use sklearn LogisticRegression directly with sparse matrices
                self.model = LogisticRegression(
                    C=self.logistic_C,
                    max_iter=self.logistic_max_iter,
                    random_state=self.seed
                )
                # For logistic regression, we'll handle training differently
                train_loader = val_loader = None  # Not used for sklearn model

        elif self.approach in ['lstm', 'transformer']:
            # Use configured transformer model for tokenizer
            model_name = self.transformer_model if self.approach == 'transformer' else 'distilbert-base-uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.vocab_size = self.tokenizer.vocab_size
            dataset = SimpleTextDataset(
                self.train_texts, self.train_labels, self.tokenizer, self.token_length
            )
            train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            _dataset = SimpleTextDataset(
                self.val_texts, self.val_labels, self.tokenizer, self.token_length
            )
            val_loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=True)

            match self.approach:
                case "lstm":
                    self.model = LSTMClassifier(
                        len(self.tokenizer),
                        self.lstm_emb_dim,
                        self.lstm_hidden_dim,
                        self.num_classes
                    )
                case "transformer":
                    if TRANSFORMERS_AVAILABLE:
                        # Load the appropriate transformer model
                        self.model = self._create_transformer_model(model_name, self.num_classes)
                        
                        # Add adapters if requested
                        if self.use_adapters and ADAPTERS_AVAILABLE:
                            self._add_adapters_to_model(self.model)
                        else:
                            # Standard fine-tuning with layer freezing
                            self._freeze_layers(self.model, self.fraction_layers_to_finetune)
                    else:
                        raise ValueError(
                            "Need `AutoTokenizer`, `AutoModelForSequenceClassification` "
                            "from `transformers` package."
                        )
                case _:
                    raise ValueError("Unsupported approach or missing transformers.")
        else:
            raise ValueError(f"Unrecognized approach: {self.approach}")
        
        # Training and validating
        if self.approach == 'logistic':
            # For sklearn LogisticRegression, training is different
            from scipy.sparse import hstack
            
            # Transform with both word and character features
            X_train_word = self.vectorizer.transform(self.train_texts)
            X_train_char = self.char_vectorizer.transform(self.train_texts)
            X_train = hstack([X_train_word, X_train_char])
            
            X_val_word = self.vectorizer.transform(self.val_texts)
            X_val_char = self.char_vectorizer.transform(self.val_texts)
            X_val = hstack([X_val_word, X_val_char])
            
            print(f"\n🔧 Training Logistic Regression with combined TF-IDF features...")
            print(f"   Feature dimensions: {X_train.shape[1]} (word: {X_train_word.shape[1]}, char: {X_train_char.shape[1]})")
            print(f"   Training samples: {X_train.shape[0]}")
            
            # Simple progress indicator for sklearn
            print("   Training...", end="", flush=True)
            start_time = time.time()
            self.model.fit(X_train, self.train_labels)
            train_time = time.time() - start_time
            print(f" ✓ Completed in {train_time:.1f}s")
            
            # Evaluate on validation set
            print("   Evaluating...", end="", flush=True)
            val_preds = self.model.predict(X_val)
            val_acc = accuracy_score(self.val_labels, val_preds)
            print(f" ✓ Validation Accuracy: {val_acc:.4f}")
            
            logger.info(f"Validation Accuracy: {val_acc:.4f}")
            
            return 1 - val_acc
        else:
            # For PyTorch models
            self.model.to(self.device)
            assert dataset is not None, f"`dataset` cannot be None here!"
            val_acc = self._train_loop(
                train_loader,
                val_loader,
                load_path=load_path,
                save_path=save_path,
                trial=getattr(self, 'current_trial', None),  # Pass trial for multi-fidelity
            )
            return 1 - val_acc

    def _train_loop(
        self, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        load_path: Path=None,
        save_path: Path=None,
        trial=None,  # For multi-fidelity HPO
    ):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Use weighted loss if class imbalance is detected
        if hasattr(self, 'class_weight_tensor') and self.class_weight_tensor is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weight_tensor)
            logger.info("Using weighted CrossEntropyLoss for imbalanced data")
        else:
            criterion = nn.CrossEntropyLoss()

        start_epoch = 0
        # handling checkpoint resume
        if load_path is not None:
            # Try to find checkpoint file
            checkpoint_files = list(Path(load_path).glob("checkpoint*.pth"))
            if checkpoint_files:
                # Prioritize checkpoint selection
                checkpoint_file = None
                
                # First, check if there's a generic checkpoint.pth (for non-HPO usage)
                generic_checkpoint = Path(load_path) / "checkpoint.pth"
                if generic_checkpoint.exists():
                    checkpoint_file = generic_checkpoint
                    logger.info("Found generic checkpoint.pth, using for resume")
                else:
                    # No generic checkpoint, look for trial-specific ones (HPO usage)
                    trial_files = [f for f in checkpoint_files if "trial_" in f.name]
                    if trial_files:
                        # Files are zero-padded, so alphabetical sort = numerical sort
                        trial_files.sort()
                        checkpoint_file = trial_files[-1]  # Use the latest trial
                        logger.info(f"Found trial checkpoints, using latest: {checkpoint_file.name}")
                    else:
                        # Fallback to any checkpoint file
                        checkpoint_files.sort()
                        checkpoint_file = checkpoint_files[-1]
                        logger.info(f"Using fallback checkpoint: {checkpoint_file.name}")
                
                logger.info(f"Loading checkpoint from: {checkpoint_file}")
                _states = torch.load(checkpoint_file, map_location='cpu')
                self.model.load_state_dict(_states["model_state_dict"])
                optimizer.load_state_dict(_states["optimizer_state_dict"])
                start_epoch = _states["epoch"]
                logger.info(f"Resuming from checkpoint at epoch {start_epoch}")
            else:
                logger.warning(f"No checkpoint files found in {load_path}")

        # Initialize progress tracker
        progress = ProgressTracker(
            self.epochs, 
            self.dataset_name, 
            self.approach,
            extra_info=f"batch_size={self.batch_size}, lr={self.lr}"
        )
        
        for epoch in range(start_epoch, self.epochs):
            progress.start_epoch(epoch)
            total_loss = 0
            batch_progress = create_batch_progress_bar(train_loader, f"Epoch {epoch+1}/{self.epochs}")
            
            for batch in batch_progress:
                self.model.train()
                optimizer.zero_grad()

                # if isinstance(batch, dict):
                if isinstance(self.model, AutoModelForSequenceClassification):
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    labels = inputs['labels']
                else:
                    match self.approach:
                        case "ffnn":
                            x, y = batch[0].to(self.device), batch[1].to(self.device)
                            outputs = self.model(x)
                            labels = y
                        case "lstm" | "transformer":
                            inputs = {k: v.to(self.device) for k, v in batch.items()}
                            outputs = self.model(**inputs)
                            labels = inputs["labels"]
                        case _:
                            raise ValueError("Oops! Wrong approach.")

                    outputs = outputs.logits if self.approach == "transformer" else outputs
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            if self.val_texts:
                val_preds, val_labels = self._predict(val_loader)
                val_acc = accuracy_score(val_labels, val_preds)
                val_error = 1 - val_acc
                
                # Update progress with metrics
                metrics = {
                    "Loss": total_loss,
                    "Val Acc": val_acc
                }
                progress.end_epoch(epoch, metrics)
                
                # Multi-fidelity HPO: report intermediate values and check for pruning
                if trial is not None and self.use_multi_fidelity:
                    # Report intermediate value to Optuna for pruning
                    trial.report(val_error, epoch)
                        
                    # Check if trial should be pruned
                    if trial.should_prune():
                        # Get study to compare with other trials
                        study = trial.study
                        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                        
                        # Concise pruning log
                        best_error = min(t.value for t in completed_trials) if completed_trials else "N/A"
                        logger.info(f"Trial #{trial.number} PRUNED at epoch {epoch + 1} | "
                                  f"Error: {val_error:.4f} | Pruner: {self.hpo_pruner} | "
                                  f"Best so far: {best_error if best_error == 'N/A' else f'{best_error:.4f}'}")
                        
                        raise optuna.exceptions.TrialPruned()
            else:
                # No validation data, just show loss
                metrics = {"Loss": total_loss}
                progress.end_epoch(epoch, metrics)

        # Final validation
        if self.val_texts:
            val_preds, val_labels = self._predict(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            progress.finish({"Validation Accuracy": val_acc})
        else:
            progress.finish()

        if save_path is not None:
            save_path = Path(save_path) if not isinstance(save_path, Path) else save_path
            save_path.mkdir(parents=True, exist_ok=True)

            # Use trial-specific checkpoint name if this is during HPO
            if hasattr(self, 'trial_number'):
                checkpoint_name = f"checkpoint_trial_{self.trial_number:02d}.pth"  # Zero-padded 2 digits
            else:
                checkpoint_name = "checkpoint.pth"

            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                },
                save_path / checkpoint_name
            )   
        torch.cuda.empty_cache()
        return val_acc or 0.0

    def _predict(self, val_loader: DataLoader):
        self.model.eval()
        preds = []
        labels = []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(self.model, AutoModelForSequenceClassification):
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                    outputs = self.model(**inputs).logits
                    labels.extend(batch["labels"])
                else:
                    match self.approach:
                        case "ffnn":
                            x, y = batch[0].to(self.device), batch[1].to(self.device)
                            outputs = self.model(x)
                            labels.extend(y)
                        case "lstm" | "transformer":
                            inputs = {k: v.to(self.device) for k, v in batch.items()}
                            outputs = self.model(**inputs)
                            outputs = outputs.logits if self.approach == "transformer" else outputs
                            labels.extend(inputs["labels"])
                        case _:
                            raise ValueError("Oops! Wrong approach.")
                            
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        
        if isinstance(preds, list):
            preds = [p.item() for p in preds]
            labels = [l.item() for l in labels]
            return np.array(preds), np.array(labels)
        else:
            return preds.cpu().numpy(), labels.cpu().numpy()


    def predict(self, test_data: pd.DataFrame | DataLoader) -> Tuple[np.ndarray, np.ndarray]:

        assert isinstance(test_data, DataLoader) or isinstance(test_data, pd.DataFrame), \
            f"Input data type: {type(test_data)}; Expected: pd.DataFrame | DataLoader"

        if isinstance(test_data, DataLoader):
            return self._predict(test_data)
        
        if self.approach == 'logistic':
            # For sklearn LogisticRegression with combined features
            from scipy.sparse import hstack
            test_texts = test_data['text'].tolist()
            
            # Apply preprocessing if enabled
            if self.preprocessor:
                test_texts = self.preprocessor.preprocess_batch(test_texts)
                
            _X_word = self.vectorizer.transform(test_texts)
            _X_char = self.char_vectorizer.transform(test_texts)
            _X = hstack([_X_word, _X_char])
            _labels = test_data['label'].tolist()
            _preds = self.model.predict(_X)
            return np.array(_preds), np.array(_labels)
            
        elif self.approach == 'ffnn':
            # Transform with both word and character features
            from scipy.sparse import hstack
            test_texts = test_data['text'].tolist()
            
            # Apply preprocessing if enabled
            if self.preprocessor:
                test_texts = self.preprocessor.preprocess_batch(test_texts)
                
            _X_word = self.vectorizer.transform(test_texts)
            _X_char = self.char_vectorizer.transform(test_texts)
            _X = hstack([_X_word, _X_char]).toarray()
            _labels = test_data['label'].tolist()
            _dataset = torch.utils.data.TensorDataset(
                torch.tensor(_X, dtype=torch.float32),
                torch.tensor(_labels)
            )
            _loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=True)
            return self._predict(_loader)
            
        elif self.approach in ['lstm', 'transformer']:
            test_texts = test_data['text'].tolist()
            
            # Apply preprocessing if enabled (for LSTM/transformer too)
            if self.preprocessor and self.approach == 'lstm':
                test_texts = self.preprocessor.preprocess_batch(test_texts)
                
            _dataset = SimpleTextDataset(
                test_texts,
                test_data['label'].tolist(),
                self.tokenizer,
                self.token_length
            )
            _loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=True)
            return self._predict(_loader)
        else:
            raise ValueError(f"Unrecognized approach: {self.approach}")
            # handling any possible tokenization

    def _freeze_layers(self, model, fraction_layers_to_finetune: float = 1.0) -> None:
        """Freeze layers in transformer model for partial fine-tuning."""
        # Get the transformer layers based on model architecture
        if hasattr(model, 'distilbert'):
            layers = model.distilbert.transformer.layer
        elif hasattr(model, 'roberta'):
            layers = model.roberta.encoder.layer
        elif hasattr(model, 'albert'):
            layers = model.albert.encoder.albert_layer_groups[0].albert_layers
        elif hasattr(model, 'bert'):
            layers = model.bert.encoder.layer
        else:
            logger.warning("Unknown model architecture, skipping layer freezing")
            return
            
        total_layers = len(layers)
        _num_layers_to_finetune = int(fraction_layers_to_finetune * total_layers)
        layers_to_freeze = total_layers - _num_layers_to_finetune

        for layer in layers[:layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
                
        logger.info(f"Froze {layers_to_freeze}/{total_layers} layers, fine-tuning {_num_layers_to_finetune} layers")
    
    def _create_transformer_model(self, model_name: str, num_labels: int):
        """Create a transformer model with support for different architectures."""
        logger.info(f"Loading transformer model: {model_name}")
        
        # Map of model names to specific model classes for better control
        model_mapping = {
            'distilbert': DistilBertForSequenceClassification,
            'roberta': RobertaForSequenceClassification,
            'albert': AlbertForSequenceClassification,
        }
        
        # Determine which model class to use
        model_class = AutoModelForSequenceClassification
        for key, cls in model_mapping.items():
            if key in model_name.lower():
                model_class = cls
                break
                
        # Load the model
        model = model_class.from_pretrained(model_name, num_labels=num_labels)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model loaded with {total_params:,} total parameters, {trainable_params:,} trainable")
        
        return model
    
    def _add_adapters_to_model(self, model):
        """Add adapter layers to the transformer model for efficient fine-tuning."""
        if not ADAPTERS_AVAILABLE:
            logger.warning("Adapters not available, falling back to standard fine-tuning")
            self._freeze_layers(model, self.fraction_layers_to_finetune)
            return
            
        logger.info(f"Adding adapters with reduction factor: {self.adapter_reduction_factor}")
        
        # Add adapter configuration
        adapter_config = AdapterConfig.load(
            "pfeiffer",  # Use Pfeiffer adapter architecture
            non_linearity="relu",
            reduction_factor=self.adapter_reduction_factor
        )
        
        # Add the adapter
        model.add_adapter("text_classification", config=adapter_config)
        model.set_active_adapters("text_classification")
        
        # Train only the adapter weights
        model.train_adapter("text_classification")
        
        # Log adapter info
        adapter_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Added adapters with {adapter_params:,} trainable parameters")

    def _create_hpo_search_space(self, trial):
        """Create hyperparameter search space for the current approach."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for HPO. Install with: pip install optuna")
            
        params = {}
        
        # Dataset-specific adjustments
        dataset_configs = {
            'amazon': {
                'vocab_range': (5000, 15000),  # Medium vocabulary
                'batch_sizes': [32, 64, 128],  # Larger batches OK
                'epochs_range': (3, 10),
                'token_length_range': (64, 256),  # Medium sequences
            },
            'imdb': {
                'vocab_range': (10000, 30000),  # Larger vocabulary for sentiment
                'batch_sizes': [16, 32, 64],  # Smaller batches due to longer texts
                'epochs_range': (3, 12),
                'token_length_range': (128, 512),  # Longer sequences
            },
            'ag_news': {
                'vocab_range': (5000, 20000),  # News vocabulary
                'batch_sizes': [64, 128, 256],  # Can handle larger batches
                'epochs_range': (2, 8),
                'token_length_range': (64, 192),  # Shorter sequences
            },
            'dbpedia': {
                'vocab_range': (10000, 40000),  # Large vocabulary for ontologies
                'batch_sizes': [64, 128, 256],  # Large dataset, bigger batches
                'epochs_range': (2, 6),
                'token_length_range': (96, 256),  # Medium sequences
            },
        }
        
        # Get dataset-specific config or use defaults
        dataset_config = dataset_configs.get(self.dataset_name, {
            'vocab_range': (5000, 20000),
            'batch_sizes': [32, 64, 128],
            'epochs_range': (3, 10),
            'token_length_range': (64, 256),
        })
        
        if self.approach == 'logistic':
            params['logistic_C'] = trial.suggest_float('logistic_C', 0.001, 100.0, log=True)
            params['logistic_max_iter'] = trial.suggest_int('logistic_max_iter', 100, 2000)
            params['vocab_size'] = trial.suggest_int('vocab_size', *dataset_config['vocab_range'])
            
        elif self.approach == 'ffnn':
            params['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            params['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 0.1)
            params['batch_size'] = trial.suggest_categorical('batch_size', dataset_config['batch_sizes'])
            params['ffnn_hidden'] = trial.suggest_int('ffnn_hidden', 64, 512)
            params['epochs'] = trial.suggest_int('epochs', *dataset_config['epochs_range'])
            params['vocab_size'] = trial.suggest_int('vocab_size', *dataset_config['vocab_range'])
            
            # Enhanced FFNN parameters (only if enhanced FFNN is enabled)
            if self.use_enhanced_ffnn:
                params['ffnn_num_layers'] = trial.suggest_int('ffnn_num_layers', 2, 5)
                params['ffnn_dropout_rate'] = trial.suggest_float('ffnn_dropout_rate', 0.0, 0.5)
                params['ffnn_use_residual'] = trial.suggest_categorical('ffnn_use_residual', [True, False])
                params['ffnn_use_layer_norm'] = trial.suggest_categorical('ffnn_use_layer_norm', [True, False])
            
        elif self.approach == 'lstm':
            params['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            params['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 0.1)
            params['batch_size'] = trial.suggest_categorical('batch_size', 
                [bs // 2 for bs in dataset_config['batch_sizes']])  # LSTM needs smaller batches
            params['lstm_emb_dim'] = trial.suggest_int('lstm_emb_dim', 64, 256)
            params['lstm_hidden_dim'] = trial.suggest_int('lstm_hidden_dim', 64, 256)
            params['epochs'] = trial.suggest_int('epochs', *dataset_config['epochs_range'])
            params['token_length'] = trial.suggest_int('token_length', *dataset_config['token_length_range'])
            
        elif self.approach == 'transformer':
            params['lr'] = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
            params['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 0.1)
            # Transformers need smaller batches
            transformer_batch_sizes = [bs // 4 for bs in dataset_config['batch_sizes'] if bs // 4 >= 4]
            params['batch_size'] = trial.suggest_categorical('batch_size', transformer_batch_sizes or [8, 16])
            params['fraction_layers_to_finetune'] = trial.suggest_float('fraction_layers_to_finetune', 0.1, 1.0)
            params['epochs'] = trial.suggest_int('epochs', 
                max(2, dataset_config['epochs_range'][0] - 1), 
                min(8, dataset_config['epochs_range'][1]))
            params['token_length'] = trial.suggest_int('token_length', *dataset_config['token_length_range'])
            
            # Add transformer model selection
            params['transformer_model'] = trial.suggest_categorical('transformer_model', [
                'distilbert-base-uncased',
                'roberta-base',
                'albert-base-v2',
                'microsoft/deberta-v3-small',  # Efficient DeBERTa variant
            ])
            
            # Adapter configuration
            params['use_adapters'] = trial.suggest_categorical('use_adapters', [False, True])
            if params['use_adapters']:
                params['adapter_reduction_factor'] = trial.suggest_categorical('adapter_reduction_factor', [8, 16, 32, 64])
            
        return params

    def _hpo_objective(self, trial, save_path=None):
        """Objective function for Optuna hyperparameter optimization."""
        # Check if this is combined NAS+HPO optimization
        if getattr(self, 'use_nas', False):
            return self._combined_nas_hpo_objective(trial, save_path)
        
        params = self._create_hpo_search_space(trial)
        
        # Log the hyperparameters for this trial
        logger.info(f"Trial {trial.number} hyperparameters: {params}")
        
        # Create temporary AutoML instance with suggested parameters
        temp_automl = TextAutoML(
            seed=self.seed,
            approach=self.approach,
            vocab_size=params.get('vocab_size', self.vocab_size),
            token_length=params.get('token_length', self.token_length),
            epochs=params.get('epochs', self.epochs),
            batch_size=params.get('batch_size', self.batch_size),
            lr=params.get('lr', self.lr),
            weight_decay=params.get('weight_decay', self.weight_decay),
            ffnn_hidden=params.get('ffnn_hidden', self.ffnn_hidden),
            lstm_emb_dim=params.get('lstm_emb_dim', self.lstm_emb_dim),
            lstm_hidden_dim=params.get('lstm_hidden_dim', self.lstm_hidden_dim),
            fraction_layers_to_finetune=params.get('fraction_layers_to_finetune', self.fraction_layers_to_finetune),
            logistic_C=params.get('logistic_C', self.logistic_C),
            logistic_max_iter=params.get('logistic_max_iter', self.logistic_max_iter),
            dataset_name=self.dataset_name,
            use_preprocessing=self.use_preprocessing,
            transformer_model=params.get('transformer_model', self.transformer_model),
            use_adapters=params.get('use_adapters', self.use_adapters),
            adapter_reduction_factor=params.get('adapter_reduction_factor', self.adapter_reduction_factor),
        )
        
        # Create temporary DataFrames
        train_df = pd.DataFrame({
            'text': self.train_texts,
            'label': self.train_labels
        })
        val_df = pd.DataFrame({
            'text': self.val_texts,
            'label': self.val_labels
        })
        
        try:
            # Set current trial for multi-fidelity HPO
            temp_automl.current_trial = trial
            
            # Train and evaluate with checkpoint saving
            if save_path is not None:
                # Set trial number for checkpoint naming
                temp_automl.trial_number = trial.number
                val_error = temp_automl.fit(train_df, val_df, self.num_classes, save_path=save_path)
            else:
                val_error = temp_automl.fit(train_df, val_df, self.num_classes)
                
            # Update HPO progress
            if hasattr(self, 'hpo_progress'):
                # Get key params for display
                key_params = {}
                if 'lr' in params:
                    key_params['lr'] = params['lr']
                if 'batch_size' in params:
                    key_params['batch_size'] = params['batch_size']
                if 'ffnn_hidden' in params:
                    key_params['hidden'] = params['ffnn_hidden']
                if 'epochs' in params:
                    key_params['epochs'] = params['epochs']
                    
                self.hpo_progress.update_trial(trial.number, val_error, key_params)
            
            logger.info(f"Trial {trial.number}: val_error = {val_error:.4f}")
            return val_error
        except optuna.exceptions.TrialPruned:
            # Re-raise pruned exception to be handled by Optuna
            raise
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float('inf')

    def optimize_hyperparameters(self, n_trials=20, timeout=3600, save_path=None, sampler=None, pruner=None):
        """Run hyperparameter optimization using Optuna (optionally with multi-fidelity support)."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for HPO. Install with: pip install optuna")
            
        # Select sampler
        sampler_name = sampler or self.hpo_sampler
        if sampler_name == 'tpe':
            sampler_obj = TPESampler(seed=self.seed)
        elif sampler_name == 'random':
            sampler_obj = RandomSampler(seed=self.seed)
        elif sampler_name == 'cmaes':
            sampler_obj = CmaEsSampler(seed=self.seed)
        elif sampler_name == 'nsga2':
            sampler_obj = NSGAIISampler(seed=self.seed)
        else:
            logger.warning(f"Unknown sampler '{sampler_name}', falling back to TPE")
            sampler_name = 'tpe'
            sampler_obj = TPESampler(seed=self.seed)
            
        logger.info(f"Using {sampler_name.upper()} sampler for HPO")
        
        # Select pruner for multi-fidelity optimization
        pruner_obj = None
        if self.use_multi_fidelity:
            pruner_name = pruner or getattr(self, 'hpo_pruner', 'median')
            
            # Create only the selected pruner to save memory
            if pruner_name == 'median':
                pruner_obj = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
            elif pruner_name == 'successive_halving':
                pruner_obj = SuccessiveHalvingPruner(min_resource=2, reduction_factor=2)
            elif pruner_name == 'hyperband':
                pruner_obj = HyperbandPruner(min_resource=2, max_resource='auto', reduction_factor=3)
            else:
                logger.warning(f"Unknown pruner '{pruner_name}', falling back to median")
                pruner_name = 'median'
                pruner_obj = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
                
            logger.info(f"Using {pruner_name.upper()} pruner for multi-fidelity HPO")
        else:
            logger.info("Multi-fidelity optimization disabled - using standard HPO")
        
        # Create study with persistent storage
        study_name = f"hpo_{self.approach}_{self.seed}"
        if save_path is not None:
            # Ensure directory exists before creating database
            save_path = Path(save_path) if not isinstance(save_path, Path) else save_path
            save_path.mkdir(parents=True, exist_ok=True)
            # Use SQLite database for persistence
            storage_url = f"sqlite:///{save_path}/optuna_study.db"
            try:
                # Try to load existing study
                study = optuna.load_study(
                    study_name=study_name,
                    storage=storage_url
                )
                logger.info(f"Resumed existing study with {len(study.trials)} completed trials")
            except KeyError:
                # Create new study if it doesn't exist
                study = optuna.create_study(
                    direction='minimize',
                    sampler=sampler_obj,
                    pruner=pruner_obj,
                    study_name=study_name,
                    storage=storage_url
                )
                if self.use_multi_fidelity:
                    logger.info("Created new study with multi-fidelity pruning")
                else:
                    logger.info("Created new study without pruning")
        else:
            # In-memory study (no persistence)
            study = optuna.create_study(
                direction='minimize',
                sampler=sampler_obj,
                pruner=pruner_obj,
                study_name=study_name
            )
            if self.use_multi_fidelity:
                logger.info("Created in-memory study with multi-fidelity pruning")
            else:
                logger.info("Created in-memory study without pruning")
            
        # Initialize HPO progress tracker
        hpo_progress = HPOProgress(n_trials, sampler_name)
        
        if self.use_multi_fidelity:
            logger.info(f"Starting multi-fidelity HPO for {self.approach} with {n_trials} trials (timeout: {timeout}s)")
        else:
            logger.info(f"Starting standard HPO for {self.approach} with {n_trials} trials (timeout: {timeout}s)")
        
        # Store progress tracker for callback
        self.hpo_progress = hpo_progress
        
        study.optimize(
            lambda trial: self._hpo_objective(trial, save_path=save_path),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Finish HPO progress
        if hasattr(self, 'hpo_progress'):
            self.hpo_progress.finish()
        
        best_params = study.best_params
        if self.use_multi_fidelity:
            logger.info(f"Multi-fidelity HPO completed. Best params: {best_params}")
            logger.info(f"Best validation error: {study.best_value:.4f}")
            logger.info(f"Total trials: {len(study.trials)}, Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        else:
            logger.info(f"Standard HPO completed. Best params: {best_params}")
            logger.info(f"Best validation error: {study.best_value:.4f}")
            logger.info(f"Total trials: {len(study.trials)}")
        
        return study.best_value, best_params, study

    def fit_with_hpo(
        self, 
        train_df: pd.DataFrame,
        val_df: pd.DataFrame, 
        num_classes: int,
        n_trials: int = 20,
        timeout: int = 3600,
        save_path: Path = None,
        sampler: str = None,
        pruner: str = None,  # For multi-fidelity HPO
        use_multi_fidelity: bool = None,  # Enable/disable multi-fidelity
        **kwargs
    ):
        """
        Fit model with Hyperparameter Optimization only (no NAS).
        
        This method performs hyperparameter optimization without architecture search.
        For combined NAS+HPO, use fit_with_nas_hpo() instead.
        """
        if use_multi_fidelity is not None:
            self.use_multi_fidelity = use_multi_fidelity
            
        if self.use_multi_fidelity:
            logger.info("Starting fit_with_hpo with multi-fidelity optimization...")
        else:
            logger.info("Starting fit_with_hpo with standard optimization...")
        
        # Ensure NAS is disabled for HPO-only optimization
        original_use_nas = getattr(self, 'use_nas', False)
        self.use_nas = False
        
        # Store data for HPO
        self.train_texts = train_df['text'].tolist()
        self.train_labels = train_df['label'].tolist()
        self.val_texts = val_df['text'].tolist()
        self.val_labels = val_df['label'].tolist()
        
        # Apply preprocessing if enabled
        if self.preprocessor:
            logger.info("Applying text preprocessing for HPO...")
            self.train_texts = self.preprocessor.preprocess_batch(self.train_texts)
            self.val_texts = self.preprocessor.preprocess_batch(self.val_texts)
            
        self.num_classes = num_classes
        
        # Run HPO (with or without multi-fidelity)
        best_score, best_params, study = self.optimize_hyperparameters(n_trials, timeout, save_path, sampler, pruner)
        
        # Apply best hyperparameters to current instance
        logger.info("Applying best hyperparameters to current instance...")
        for param, value in best_params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        
        # Final training with best parameters
        logger.info("Training final model with optimized hyperparameters...")
        final_score = self.fit(train_df, val_df, num_classes, save_path=save_path, **kwargs)
        
        # Restore original NAS setting
        self.use_nas = original_use_nas
        
        if self.use_multi_fidelity:
            logger.info(f"HPO with multi-fidelity completed. Final validation error: {final_score:.4f}")
        else:
            logger.info(f"HPO completed. Final validation error: {final_score:.4f}")
        return final_score

    def fit_with_nas(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_classes: int,
        n_trials: int = 20,
        timeout: int = 3600,
        save_path: Path = None,
        **kwargs
    ):
        """
        Fit model with Neural Architecture Search only (no HPO).
        
        This method performs architecture search without optimizing other hyperparameters.
        For combined NAS+HPO, use fit_with_nas_hpo() instead.
        """
        logger.info("Starting fit_with_nas with architecture search...")
        
        # Ensure HPO is disabled for NAS-only optimization
        original_use_hpo = getattr(self, 'use_hpo', False)
        self.use_hpo = False
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for NAS. Install with: pip install optuna")
            
        if self.approach not in ['ffnn', 'lstm']:
            raise ValueError(f"NAS is not supported for approach '{self.approach}'. Use 'ffnn' or 'lstm'.")
            
        logger.info(f"Starting NAS-only optimization for {self.approach} with {n_trials} trials...")
        
        # Store data for NAS
        self.train_texts = train_df['text'].tolist()
        self.train_labels = train_df['label'].tolist()
        self.val_texts = val_df['text'].tolist()
        self.val_labels = val_df['label'].tolist()
        
        # Apply preprocessing if enabled
        if self.preprocessor:
            logger.info("Applying text preprocessing for NAS...")
            self.train_texts = self.preprocessor.preprocess_batch(self.train_texts)
            self.val_texts = self.preprocessor.preprocess_batch(self.val_texts)
            
        self.num_classes = num_classes
        
        # Run NAS
        best_score, best_params, study = self.optimize_architectures(n_trials, timeout, save_path)
        
        # Apply best architecture to current instance
        logger.info("Applying best architecture to current instance...")
        train_df_for_nas = pd.DataFrame({
            'text': self.train_texts,
            'label': self.train_labels
        })
        val_df_for_nas = pd.DataFrame({
            'text': self.val_texts,
            'label': self.val_labels
        })
        self._setup_nas_model(study.best_trial, train_df_for_nas, val_df_for_nas)
        
        # Final training with best architecture (already applied by optimize_architectures)
        logger.info("Training final model with optimized architecture...")
        final_score = self.fit(train_df, val_df, num_classes, save_path=save_path, **kwargs)
        
        # Restore original HPO setting
        self.use_hpo = original_use_hpo
        
        logger.info(f"NAS completed. Final validation error: {final_score:.4f}")
        return final_score

    def _nas_objective(self, trial, train_df, val_df, save_path=None):
        """Objective function for NAS-only optimization."""
        # Create temporary AutoML instance with current hyperparameters (no HPO)
        temp_automl = TextAutoML(
            seed=self.seed,
            dataset_name=self.dataset_name,
            approach=self.approach,
            vocab_size=self.vocab_size,
            token_length=self.token_length,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            ffnn_hidden=self.ffnn_hidden,
            lstm_emb_dim=self.lstm_emb_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            use_nas=True,  # Enable NAS
        )
        
        try:

            temp_automl.num_classes = self.num_classes
            # Generate architecture for this trial
            temp_automl._setup_nas_model(trial, train_df, val_df)
            
            # Train and evaluate
            if save_path is not None:
                temp_automl.trial_number = trial.number
                val_error = temp_automl.fit(train_df, val_df, self.num_classes, save_path=save_path)
            else:
                val_error = temp_automl.fit(train_df, val_df, self.num_classes)
                
            logger.info(f"NAS Trial {trial.number}: val_error = {val_error:.4f}")
            return val_error
            
        except Exception as e:
            logger.warning(f"NAS Trial {trial.number} failed: {e}")
            return float('inf')

    def optimize_architectures(self, n_trials=20, timeout=3600, save_path=None):
        """Run neural architecture search using Optuna."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for NAS. Install with: pip install optuna")
            
        logger.info(f"Using TPE sampler for NAS")
        
        # Create study with persistent storage
        study_name = f"nas_{self.approach}_{self.seed}"
        if save_path is not None:
            # Ensure directory exists before creating database
            save_path = Path(save_path) if not isinstance(save_path, Path) else save_path
            save_path.mkdir(parents=True, exist_ok=True)
            # Use SQLite database for persistence
            storage_url = f"sqlite:///{save_path}/optuna_nas_study.db"
            try:
                # Try to load existing study
                study = optuna.load_study(
                    study_name=study_name,
                    storage=storage_url
                )
                logger.info(f"Resumed existing NAS study with {len(study.trials)} completed trials")
            except KeyError:
                # Create new study if it doesn't exist
                study = optuna.create_study(
                    direction='minimize',
                    sampler=TPESampler(seed=self.seed),
                    study_name=study_name,
                    storage=storage_url
                )
                logger.info("Created new NAS study")
        else:
            # In-memory study (no persistence)
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=self.seed),
                study_name=study_name
            )
            logger.info("Created in-memory NAS study")
            
        logger.info(f"Starting NAS for {self.approach} with {n_trials} trials (timeout: {timeout}s)")
        
        # Create temporary DataFrames for the objective function
        train_df = pd.DataFrame({
            'text': self.train_texts,
            'label': self.train_labels
        })
        val_df = pd.DataFrame({
            'text': self.val_texts,
            'label': self.val_labels
        })

        study.optimize(
            lambda trial: self._nas_objective(trial, train_df, val_df, save_path=save_path),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        logger.info(f"NAS completed. Best architecture: {best_params}")
        logger.info(f"Best validation error: {study.best_value:.4f}")
        logger.info(f"Total trials: {len(study.trials)}")
        
        return study.best_value, best_params, study

    
    def _setup_nas_model(self, trial, train_df, val_df):
        """Setup NAS model based on trial architecture configuration."""
        if self.approach == 'ffnn':
            # Setup data processing for FFNN with enhanced TF-IDF
            dataset_size = len(train_df)
            adaptive_max_features = min(self.vocab_size, max(5000, dataset_size // 10))
            
            self.vectorizer = TfidfVectorizer(
                max_features=adaptive_max_features,
                lowercase=True,
                min_df=2,
                max_df=0.8,
                sublinear_tf=True,
                ngram_range=(1, 3),  # Include trigrams
                analyzer='word',
                strip_accents='unicode',
                decode_error='ignore',
                token_pattern=r'\b\w+\b',
            )
            
            # Character-level TF-IDF
            self.char_vectorizer = TfidfVectorizer(
                max_features=min(5000, adaptive_max_features // 2),
                lowercase=True,
                analyzer='char',
                ngram_range=(3, 5),
                min_df=2,
                max_df=0.8,
                sublinear_tf=True,
            )
            # Fit both vectorizers and combine features
            from scipy.sparse import hstack
            X_train_word = self.vectorizer.fit_transform(train_df['text'].tolist())
            X_train_char = self.char_vectorizer.fit_transform(train_df['text'].tolist())
            X_train = hstack([X_train_word, X_train_char]).toarray()
            input_dim = X_train.shape[1]
            logger.info(f"NAS FFNN combined features: {input_dim} dimensions")
            
            # Generate architecture
            architecture_config = NASSearchSpace.generate_ffnn_architecture(trial, input_dim, self.num_classes)
            logger.info(f"NAS FFNN Architecture: {architecture_config}")
            
            # Create model
            self.model = NASSearchableFFNN(input_dim, self.num_classes, architecture_config)
            self.nas_input_dim = input_dim
            
        elif self.approach == 'lstm':
            # Setup tokenizer for LSTM
            model_name = 'distilbert-base-uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            vocab_size = len(self.tokenizer)
            
            # Generate architecture
            architecture_config = NASSearchSpace.generate_lstm_architecture(trial, vocab_size, self.num_classes)
            logger.info(f"NAS LSTM Architecture: {architecture_config}")
            
            # Create model
            self.model = NASSearchableLSTM(vocab_size, self.num_classes, architecture_config)
    
    def _combined_nas_hpo_objective(self, trial, save_path=None):
        """Objective function for combined NAS+HPO optimization."""
        # Create both hyperparameter and architecture search spaces
        hpo_params = self._create_hpo_search_space(trial)
        
        # Log the parameters for this trial
        logger.info(f"NAS+HPO Trial {trial.number} hyperparameters: {hpo_params}")
        
        # Create temporary AutoML instance with suggested hyperparameters
        temp_automl = TextAutoML(
            seed=self.seed,
            approach=self.approach,
            dataset_name=self.dataset_name, 
            vocab_size=hpo_params.get('vocab_size', self.vocab_size),
            token_length=hpo_params.get('token_length', self.token_length),
            epochs=hpo_params.get('epochs', self.epochs),
            batch_size=hpo_params.get('batch_size', self.batch_size),
            lr=hpo_params.get('lr', self.lr),
            weight_decay=hpo_params.get('weight_decay', self.weight_decay),
            ffnn_hidden=hpo_params.get('ffnn_hidden', self.ffnn_hidden),
            lstm_emb_dim=hpo_params.get('lstm_emb_dim', self.lstm_emb_dim),
            lstm_hidden_dim=hpo_params.get('lstm_hidden_dim', self.lstm_hidden_dim),
            fraction_layers_to_finetune=hpo_params.get('fraction_layers_to_finetune', self.fraction_layers_to_finetune),
            logistic_C=hpo_params.get('logistic_C', self.logistic_C),
            logistic_max_iter=hpo_params.get('logistic_max_iter', self.logistic_max_iter),
            use_nas=True,  # Enable NAS for this trial        
        )
        
        # Create temporary DataFrames
        train_df = pd.DataFrame({
            'text': self.train_texts,
            'label': self.train_labels
        })
        val_df = pd.DataFrame({
            'text': self.val_texts,
            'label': self.val_labels
        })
        
        try:
            temp_automl.num_classes = self.num_classes
            # Set current trial for multi-fidelity HPO
            temp_automl.current_trial = trial
            
            # Generate and apply architecture for this trial
            temp_automl._setup_nas_model(trial, train_df, val_df)
            
            # Train and evaluate
            if save_path is not None:
                temp_automl.trial_number = trial.number
                val_error = temp_automl.fit(train_df, val_df, self.num_classes, save_path=save_path)
            else:
                val_error = temp_automl.fit(train_df, val_df, self.num_classes)
                
            logger.info(f"NAS+HPO Trial {trial.number}: val_error = {val_error:.4f}")
            return val_error
            
        except optuna.exceptions.TrialPruned:
            # Re-raise pruned exception to be handled by Optuna
            raise
        except Exception as e:
            logger.warning(f"NAS+HPO Trial {trial.number} failed: {e}")
            return float('inf')

    def fit_with_nas_hpo(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_classes: int,
        n_trials: int = 20,
        timeout: int = 3600,
        save_path: Path = None,
        sampler: str = None,
        pruner: str = None,  # For multi-fidelity HPO
        use_multi_fidelity: bool = None,  # Enable/disable multi-fidelity
        **kwargs
    ):
        """
        Fit model with both Neural Architecture Search AND Hyperparameter Optimization.
        
        This method performs comprehensive optimization by searching for both optimal
        network architectures and optimal hyperparameters simultaneously.
        
        Parameters:
        - train_df, val_df: Training and validation data
        - num_classes: Number of output classes
        - n_trials: Number of combined NAS+HPO trials to evaluate
        - timeout: Maximum time for combined optimization
        - save_path: Path to save best results
        - sampler: Optuna sampler ('tpe', 'random', 'cmaes', 'nsga2')
        - pruner: Optuna pruner for multi-fidelity ('median', 'successive_halving', 'hyperband')
        - use_multi_fidelity: Enable/disable multi-fidelity optimization
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for NAS+HPO. Install with: pip install optuna")
            
        if self.approach not in ['ffnn', 'lstm']:
            raise ValueError(f"NAS is not supported for approach '{self.approach}'. Use 'ffnn' or 'lstm'.")
            
        # Force enable NAS for this method
        original_use_nas = getattr(self, 'use_nas', False)
        self.use_nas = True
        
        if use_multi_fidelity is not None:
            self.use_multi_fidelity = use_multi_fidelity
            
        if self.use_multi_fidelity:
            logger.info("Starting fit_with_nas_hpo with multi-fidelity optimization...")
        else:
            logger.info("Starting fit_with_nas_hpo with standard optimization...")
        
        logger.info(f"Starting combined NAS+HPO optimization for {self.approach} with {n_trials} trials...")
        
        # Store data for optimization
        self.train_texts = train_df['text'].tolist()
        self.train_labels = train_df['label'].tolist()
        self.val_texts = val_df['text'].tolist()
        self.val_labels = val_df['label'].tolist()
        
        # Apply preprocessing if enabled
        if self.preprocessor:
            logger.info("Applying text preprocessing...")
            self.train_texts = self.preprocessor.preprocess_batch(self.train_texts)
            self.val_texts = self.preprocessor.preprocess_batch(self.val_texts)
            
        self.num_classes = num_classes
        
        # Run combined NAS+HPO optimization
        best_score, best_params, study = self.optimize_hyperparameters(n_trials, timeout, save_path, sampler, pruner)
        
        # Apply best hyperparameters to current instance
        logger.info("Applying best hyperparameters to current instance...")
        for param, value in best_params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        
        # Apply best architecture to current instance
        logger.info("Applying best architecture to current instance...")
        train_df_for_nas = pd.DataFrame({
            'text': self.train_texts,
            'label': self.train_labels
        })
        val_df_for_nas = pd.DataFrame({
            'text': self.val_texts,
            'label': self.val_labels
        })
        self._setup_nas_model(study.best_trial, train_df_for_nas, val_df_for_nas)
        
        # Final training with best architecture and hyperparameters
        logger.info("Training final model with optimized architecture and hyperparameters...")
        final_score = self.fit(train_df, val_df, num_classes, save_path=save_path, **kwargs)
        
        # Restore original NAS setting
        self.use_nas = original_use_nas
        
        if self.use_multi_fidelity:
            logger.info(f"Combined NAS+HPO with multi-fidelity completed. Final validation error: {final_score:.4f}")
        else:
            logger.info(f"Combined NAS+HPO completed. Final validation error: {final_score:.4f}")
        return final_score

    # ========== NEPS Auto-Approach Selection Pipeline ==========
    
    def _create_neps_approach_search_space(self):
        """Create NEPS search space for approach selection."""
        if not NEPS_AVAILABLE:
            raise ImportError("NEPS is required. Install with: pip install neps")
            
        search_space = {
            'approach': neps.CategoricalParameter(choices=['logistic', 'ffnn', 'lstm', 'transformer']),
            'optimization_strategy': neps.CategoricalParameter(choices=['basic', 'hpo', 'nas', 'nas_hpo']),
            'use_multi_fidelity': neps.CategoricalParameter(choices=[True, False]),
            'hpo_sampler': neps.CategoricalParameter(choices=['tpe', 'random', 'cmaes', 'nsga2']),
            'hpo_pruner': neps.CategoricalParameter(choices=['median', 'successive_halving', 'hyperband']),
        }
        
        logger.info("Created NEPS approach selection search space with multi-fidelity options")
        return search_space

    def _neps_approach_objective(self, approach: str, optimization_strategy: str, use_multi_fidelity: bool, hpo_sampler: str, hpo_pruner: str):
        """NEPS objective function that uses existing Optuna-based methods."""
        try:
            start_time = time.time()
            
            logger.info(f"NEPS trying approach='{approach}' with strategy='{optimization_strategy}', "
                       f"multi_fidelity={use_multi_fidelity}, sampler={hpo_sampler}, pruner={hpo_pruner}")
            
            # Create temporary AutoML instance with the suggested approach and settings
            temp_automl = TextAutoML(
                seed=self.seed,
                approach=approach,
                dataset_name=self.dataset_name, 
                vocab_size=self.vocab_size,
                token_length=self.token_length,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                weight_decay=self.weight_decay,
                ffnn_hidden=self.ffnn_hidden,
                lstm_emb_dim=self.lstm_emb_dim,
                lstm_hidden_dim=self.lstm_hidden_dim,
                fraction_layers_to_finetune=self.fraction_layers_to_finetune,
                logistic_C=self.logistic_C,
                logistic_max_iter=self.logistic_max_iter,
                hpo_sampler=hpo_sampler,
                hpo_pruner=hpo_pruner,
                use_multi_fidelity=use_multi_fidelity,
            )
            
            # Create DataFrames for training
            train_df = pd.DataFrame({
                'text': self.train_texts,
                'label': self.train_labels
            })
            val_df = pd.DataFrame({
                'text': self.val_texts,
                'label': self.val_labels
            })
            
            # Use existing Optuna-based methods based on optimization strategy
            if optimization_strategy == 'basic':
                # Just basic training
                val_error = temp_automl.fit(train_df, val_df, self.num_classes)
                
            elif optimization_strategy == 'hpo':
                # Use existing HPO method
                val_error = temp_automl.fit_with_hpo(
                    train_df, val_df, self.num_classes,
                    n_trials=10,  # Reduced for NEPS efficiency
                    timeout=600,  # 10 minutes per approach
                    sampler=hpo_sampler,
                    pruner=hpo_pruner,
                    use_multi_fidelity=use_multi_fidelity,
                )
                
            elif optimization_strategy == 'nas':
                # Use existing NAS method (only for supported approaches)
                if approach in ['ffnn', 'lstm']:
                    val_error = temp_automl.fit_with_nas(
                        train_df, val_df, self.num_classes,
                        n_trials=8,   # Reduced for NEPS efficiency
                        timeout=600,  # 10 minutes per approach
                    )
                else:
                    # Fallback to HPO for non-NAS approaches
                    logger.info(f"NAS not supported for {approach}, falling back to HPO")
                    val_error = temp_automl.fit_with_hpo(
                        train_df, val_df, self.num_classes,
                        n_trials=10,
                        timeout=600,
                        sampler=hpo_sampler,
                        pruner=hpo_pruner,
                        use_multi_fidelity=use_multi_fidelity,
                    )
                    
            elif optimization_strategy == 'nas_hpo':
                # Use existing combined NAS+HPO method
                if approach in ['ffnn', 'lstm']:
                    val_error = temp_automl.fit_with_nas_hpo(
                        train_df, val_df, self.num_classes,
                        n_trials=12,  # Reduced for NEPS efficiency
                        timeout=800,  # A bit more time for combined optimization
                        sampler=hpo_sampler,
                        pruner=hpo_pruner,
                        use_multi_fidelity=use_multi_fidelity,
                    )
                else:
                    # Fallback to HPO for non-NAS approaches
                    logger.info(f"NAS not supported for {approach}, falling back to HPO")
                    val_error = temp_automl.fit_with_hpo(
                        train_df, val_df, self.num_classes,
                        n_trials=10,
                        timeout=600,
                        sampler=hpo_sampler,
                        pruner=hpo_pruner,
                        use_multi_fidelity=use_multi_fidelity,
                    )
            else:
                raise ValueError(f"Unknown optimization strategy: {optimization_strategy}")
            
            training_time = time.time() - start_time
            
            logger.info(f"NEPS approach trial completed: approach={approach}, strategy={optimization_strategy}, "
                       f"multi_fidelity={use_multi_fidelity}, sampler={hpo_sampler}, pruner={hpo_pruner}, "
                       f"val_error={val_error:.4f}, time={training_time:.2f}s")
            
            return {
                'loss': val_error,
                'cost': training_time,
                'approach': approach,
                'strategy': optimization_strategy,
                'use_multi_fidelity': use_multi_fidelity,
                'hpo_sampler': hpo_sampler,
                'hpo_pruner': hpo_pruner,
            }
            
        except Exception as e:
            training_time = time.time() - start_time if 'start_time' in locals() else 0.0
            logger.error(f"NEPS approach trial failed: approach={approach}, strategy={optimization_strategy}, "
                        f"multi_fidelity={use_multi_fidelity}, sampler={hpo_sampler}, pruner={hpo_pruner}, error={e}")
            return {
                'loss': float('inf'),
                'cost': training_time,
                'approach': approach,
                'strategy': optimization_strategy,
                'use_multi_fidelity': use_multi_fidelity,
                'hpo_sampler': hpo_sampler,
                'hpo_pruner': hpo_pruner,
            }

    def fit_with_neps_auto_approach(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_classes: int,
        max_evaluations: int = 16,  # 4 approaches × 4 strategies = 16 combinations
        timeout: int = 7200,  # 2 hours total
        searcher: str = 'bayesian_optimization',
        working_directory: str = "./neps_auto_approach",
        save_path: Path = None,
        **kwargs
    ):
        """
        NEPS-based automatic approach selection that uses existing Optuna methods.
        
        This method uses NEPS at the top level to select the best approach and optimization strategy,
        while leveraging all the existing Optuna-based optimization methods internally.
        
        Parameters:
        - train_df, val_df: Training and validation data
        - num_classes: Number of output classes
        - max_evaluations: Maximum number of approach+strategy combinations to try
        - timeout: Total timeout in seconds
        - searcher: NEPS searcher algorithm
        - working_directory: NEPS working directory
        - save_path: Path to save final model
        """
        if not NEPS_AVAILABLE:
            raise ImportError("NEPS is required for auto-approach selection. Install with: pip install neps")
            
        logger.info("Starting NEPS automatic approach selection...")
        logger.info("This will use existing Optuna methods (fit_with_hpo, fit_with_nas, fit_with_nas_hpo) internally")
        logger.info(f"Max evaluations: {max_evaluations}, Total timeout: {timeout}s")
        
        # Store data for optimization
        self.train_texts = train_df['text'].tolist()
        self.train_labels = train_df['label'].tolist()
        self.val_texts = val_df['text'].tolist()
        self.val_labels = val_df['label'].tolist()
        
        # Apply preprocessing if enabled
        if self.preprocessor:
            logger.info("Applying text preprocessing...")
            self.train_texts = self.preprocessor.preprocess_batch(self.train_texts)
            self.val_texts = self.preprocessor.preprocess_batch(self.val_texts)
            
        self.num_classes = num_classes
        
        # Create search space for approach selection
        search_space = self._create_neps_approach_search_space()
        
        # Setup working directory
        working_dir = Path(working_directory)
        working_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure NEPS run
        neps_config = {
            'pipeline_space': search_space,
            'working_directory': str(working_dir),
            'max_evaluations_total': max_evaluations,
            'searcher': searcher,
        }
        
        if timeout > 0:
            neps_config['max_cost_total'] = timeout
            
        logger.info(f"NEPS configuration: {neps_config}")
        
        # Run NEPS optimization
        neps.run(
            run_pipeline=self._neps_approach_objective,
            **neps_config
        )
        
        # Get best result
        try:
            best_result = neps.status(working_dir)
            best_config = best_result.best_config
            best_score = best_result.best_loss
            
            logger.info(f"NEPS auto-approach selection completed!")
            logger.info(f"Best approach: {best_config.get('approach', 'unknown')}")
            logger.info(f"Best optimization strategy: {best_config.get('optimization_strategy', 'unknown')}")
            logger.info(f"Best multi-fidelity setting: {best_config.get('use_multi_fidelity', 'unknown')}")
            logger.info(f"Best HPO sampler: {best_config.get('hpo_sampler', 'unknown')}")
            logger.info(f"Best HPO pruner: {best_config.get('hpo_pruner', 'unknown')}")
            logger.info(f"Best validation error: {best_score:.4f}")
            
        except Exception as e:
            logger.warning(f"Could not retrieve NEPS results: {e}")
            best_config = {
                'approach': 'ffnn', 
                'optimization_strategy': 'hpo',
                'use_multi_fidelity': True,
                'hpo_sampler': 'tpe',
                'hpo_pruner': 'median'
            }  # Safe fallback
            best_score = float('inf')
        
        # Train final model with best approach and strategy
        if best_config and best_score < float('inf'):
            logger.info("Training final model with best approach and optimization strategy...")
            
            best_approach = best_config.get('approach', 'ffnn')
            best_strategy = best_config.get('optimization_strategy', 'hpo')
            best_multi_fidelity = best_config.get('use_multi_fidelity', True)
            best_sampler = best_config.get('hpo_sampler', 'tpe')
            best_pruner = best_config.get('hpo_pruner', 'median')
            
            # Update current instance with best approach and settings
            self.approach = best_approach
            self.use_multi_fidelity = best_multi_fidelity
            self.hpo_sampler = best_sampler
            self.hpo_pruner = best_pruner
            
            # Train final model using the best strategy with existing methods
            if best_strategy == 'basic':
                final_score = self.fit(train_df, val_df, num_classes, save_path=save_path, **kwargs)
                
            elif best_strategy == 'hpo':
                final_score = self.fit_with_hpo(
                    train_df, val_df, num_classes,
                    n_trials=20,  # Full trials for final model
                    timeout=1800,  # 30 minutes for final optimization
                    sampler=best_sampler,
                    pruner=best_pruner,
                    use_multi_fidelity=best_multi_fidelity,
                    save_path=save_path,
                    **kwargs
                )
                
            elif best_strategy == 'nas':
                if best_approach in ['ffnn', 'lstm']:
                    final_score = self.fit_with_nas(
                        train_df, val_df, num_classes,
                        n_trials=15,
                        timeout=1800,
                        save_path=save_path,
                        **kwargs
                    )
                else:
                    logger.info(f"NAS not supported for {best_approach}, using HPO for final model")
                    final_score = self.fit_with_hpo(
                        train_df, val_df, num_classes,
                        n_trials=20,
                        timeout=1800,
                        sampler=best_sampler,
                        pruner=best_pruner,
                        use_multi_fidelity=best_multi_fidelity,
                        save_path=save_path,
                        **kwargs
                    )
                    
            elif best_strategy == 'nas_hpo':
                if best_approach in ['ffnn', 'lstm']:
                    final_score = self.fit_with_nas_hpo(
                        train_df, val_df, num_classes,
                        n_trials=25,
                        timeout=2400,  # 40 minutes for combined optimization
                        sampler=best_sampler,
                        pruner=best_pruner,
                        use_multi_fidelity=best_multi_fidelity,
                        save_path=save_path,
                        **kwargs
                    )
                else:
                    logger.info(f"NAS not supported for {best_approach}, using HPO for final model")
                    final_score = self.fit_with_hpo(
                        train_df, val_df, num_classes,
                        n_trials=20,
                        timeout=1800,
                        sampler=best_sampler,
                        pruner=best_pruner,
                        use_multi_fidelity=best_multi_fidelity,
                        save_path=save_path,
                        **kwargs
                    )
            else:
                raise ValueError(f"Unknown optimization strategy: {best_strategy}")
            
            logger.info(f"NEPS auto-approach pipeline completed!")
            logger.info(f"Final approach: {best_approach}")
            logger.info(f"Final strategy: {best_strategy}")
            logger.info(f"Final multi-fidelity: {best_multi_fidelity}")
            logger.info(f"Final sampler: {best_sampler}")
            logger.info(f"Final pruner: {best_pruner}")
            logger.info(f"Final validation error: {final_score:.4f}")
            
            # Save results summary
            if save_path is not None:
                results_summary = {
                    'neps_auto_approach': True,
                    'best_approach': best_approach,
                    'best_optimization_strategy': best_strategy,
                    'best_use_multi_fidelity': best_multi_fidelity,
                    'best_hpo_sampler': best_sampler,
                    'best_hpo_pruner': best_pruner,
                    'neps_best_validation_error': float(best_score),
                    'final_validation_error': float(final_score),
                    'max_evaluations': max_evaluations,
                    'total_timeout': timeout,
                    'searcher': searcher,
                    'seed': self.seed,
                }
                
                with open(save_path / "neps_auto_approach_summary.yaml", 'w') as f:
                    yaml.dump(results_summary, f, default_flow_style=False)
                
                logger.info(f"Results summary saved to {save_path / 'neps_auto_approach_summary.yaml'}")
            
            return final_score
        else:
            logger.error("NEPS auto-approach selection failed")
            return float('inf')
    
    def fit_with_ensemble(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_classes: int,
        ensemble_methods: List[str] = ['logistic', 'ffnn', 'lstm'],
        ensemble_type: str = 'auto',  # 'voting', 'stacking', 'weighted', 'auto'
        individual_trials: int = 10,  # HPO trials per individual model
        save_path: Path = None,
        **kwargs
    ):
        """
        Fit ensemble of multiple models for better performance.
        
        Args:
            train_df: Training data
            val_df: Validation data
            num_classes: Number of classes
            ensemble_methods: List of approaches to include in ensemble
            ensemble_type: Type of ensemble ('voting', 'stacking', 'weighted', 'auto')
            individual_trials: HPO trials for each individual model
            save_path: Path to save ensemble results
        
        Returns:
            Validation error of the ensemble
        """
        from automl.ensemble import EnsembleBuilder
        
        logger.info(f"Starting ensemble training with methods: {ensemble_methods}")
        logger.info(f"Ensemble type: {ensemble_type}")
        
        # Track individual model results
        individual_models = []
        individual_scores = []
        individual_names = []
        
        # Train each individual model with HPO
        for approach in ensemble_methods:
            logger.info(f"Training individual model: {approach}")
            
            # Create a new instance for this approach
            individual_automl = TextAutoML(
                seed=self.seed,
                approach=approach,
                vocab_size=self.vocab_size,
                token_length=self.token_length,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                weight_decay=self.weight_decay,
                ffnn_hidden=self.ffnn_hidden,
                use_enhanced_ffnn=self.use_enhanced_ffnn,
                ffnn_num_layers=self.ffnn_num_layers,
                ffnn_dropout_rate=self.ffnn_dropout_rate,
                ffnn_use_residual=self.ffnn_use_residual,
                ffnn_use_layer_norm=self.ffnn_use_layer_norm,
                lstm_emb_dim=self.lstm_emb_dim,
                lstm_hidden_dim=self.lstm_hidden_dim,
                hpo_sampler=self.hpo_sampler,
                hpo_pruner=self.hpo_pruner,
                use_multi_fidelity=self.use_multi_fidelity,
                dataset_name=self.dataset_name,
                use_preprocessing=self.use_preprocessing,
            )
            
            try:
                # Train with HPO for better individual performance
                if individual_trials > 0:
                    val_error = individual_automl.fit_with_hpo(
                        train_df, val_df, num_classes,
                        n_trials=individual_trials,
                        timeout=1800  # 30 minutes per model
                    )
                else:
                    val_error = individual_automl.fit(
                        train_df, val_df, num_classes
                    )
                
                # Store the trained model
                individual_models.append(individual_automl)
                individual_scores.append(1 - val_error)  # Convert error to accuracy
                individual_names.append(approach)
                
                logger.info(f"{approach} trained successfully. Validation accuracy: {1-val_error:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {approach}: {e}")
                continue
        
        if len(individual_models) < 2:
            raise ValueError(f"Need at least 2 successful models for ensemble. Got {len(individual_models)}")
        
        # Build ensemble using EnsembleBuilder
        logger.info("Building ensemble...")
        builder = EnsembleBuilder()
        
        for model, name, score in zip(individual_models, individual_names, individual_scores):
            builder.add_model(model, name, score)
        
        # Prepare validation data for ensemble fitting
        val_texts = val_df['text'].tolist()
        val_labels = val_df['label'].values
        
        # Create ensemble based on specified type
        if ensemble_type == 'voting':
            self.ensemble = builder.build_voting_ensemble(voting='soft')
        elif ensemble_type == 'stacking':
            self.ensemble = builder.build_stacking_ensemble()
            # For stacking, we need to fit it properly
            X_val_combined = self._prepare_ensemble_features(val_texts, individual_models)
            self.ensemble.fit(X_val_combined, val_labels)
        elif ensemble_type == 'weighted':
            self.ensemble = builder.build_weighted_ensemble(weight_method='accuracy')
            self.ensemble.fit()  # Uses stored validation scores
        elif ensemble_type == 'auto':
            # Automatically select best ensemble method
            X_val_combined = self._prepare_ensemble_features(val_texts, individual_models)
            self.ensemble, ensemble_name, ensemble_score = builder.auto_select_best_ensemble(
                X_val_combined, val_labels
            )
            logger.info(f"Auto-selected ensemble: {ensemble_name} (score: {ensemble_score:.4f})")
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        # Store ensemble information
        self.individual_models = individual_models
        self.ensemble_type = ensemble_type
        self.approach = 'ensemble'  # Mark as ensemble approach
        
        # Evaluate ensemble on validation data
        val_predictions = self.predict_ensemble(val_df)
        from sklearn.metrics import accuracy_score
        ensemble_accuracy = accuracy_score(val_labels, val_predictions)
        ensemble_error = 1 - ensemble_accuracy
        
        logger.info(f"Ensemble validation accuracy: {ensemble_accuracy:.4f}")
        logger.info(f"Individual model accuracies: {dict(zip(individual_names, individual_scores))}")
        
        # Save ensemble results if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save ensemble summary
            ensemble_summary = {
                'ensemble_type': ensemble_type,
                'individual_methods': individual_names,
                'individual_scores': individual_scores,
                'ensemble_accuracy': float(ensemble_accuracy),
                'ensemble_error': float(ensemble_error),
                'improvement_over_best': float(ensemble_accuracy - max(individual_scores)),
                'individual_trials': individual_trials,
            }
            
            with open(save_path / "ensemble_summary.yaml", 'w') as f:
                yaml.dump(ensemble_summary, f, default_flow_style=False)
                
            logger.info(f"Ensemble summary saved to {save_path / 'ensemble_summary.yaml'}")
        
        return ensemble_error
    
    def predict_ensemble(self, test_data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained ensemble."""
        if not hasattr(self, 'ensemble'):
            raise ValueError("No ensemble has been trained. Call fit_with_ensemble first.")
        
        test_texts = test_data['text'].tolist()
        
        # Prepare features for ensemble prediction
        X_test_combined = self._prepare_ensemble_features(test_texts, self.individual_models)
        
        # Make ensemble prediction
        return self.ensemble.predict(X_test_combined)
    
    def _prepare_ensemble_features(self, texts: List[str], models: List) -> np.ndarray:
        """
        Prepare features for ensemble by getting predictions from individual models.
        
        This creates a feature matrix where each column represents predictions from one model.
        """
        predictions_list = []
        
        for model in models:
            # Create a temporary dataframe for prediction
            temp_df = pd.DataFrame({'text': texts, 'label': [0] * len(texts)})
            
            try:
                if model.approach == 'logistic':
                    # For logistic regression, get decision function scores
                    if model.preprocessor:
                        processed_texts = model.preprocessor.preprocess_batch(texts)
                    else:
                        processed_texts = texts
                    
                    X_word = model.vectorizer.transform(processed_texts)
                    if hasattr(model, 'char_vectorizer'):
                        X_char = model.char_vectorizer.transform(processed_texts)
                        from scipy.sparse import hstack
                        X = hstack([X_word, X_char])
                    else:
                        X = X_word
                    
                    if hasattr(model.model, 'decision_function'):
                        scores = model.model.decision_function(X)
                        if scores.ndim == 1:
                            predictions_list.append(scores)
                        else:
                            predictions_list.append(np.max(scores, axis=1))
                    elif hasattr(model.model, 'predict_proba'):
                        proba = model.model.predict_proba(X)
                        predictions_list.append(np.max(proba, axis=1))
                    else:
                        pred = model.model.predict(X)
                        predictions_list.append(pred.astype(float))
                        
                elif model.approach in ['ffnn', 'lstm', 'transformer']:
                    # For neural models, get raw predictions using DataFrame
                    preds, _ = model.predict(temp_df)
                    predictions_list.append(preds.astype(float))
                    
                else:
                    # Fallback: use predict method with DataFrame
                    preds, _ = model.predict(temp_df)
                    predictions_list.append(preds.astype(float))
                    
            except Exception as e:
                logger.warning(f"Failed to get predictions from {model.approach}: {e}")
                # Use dummy predictions as fallback
                predictions_list.append(np.zeros(len(texts)))
        
        # Stack predictions as features
        return np.column_stack(predictions_list)

# end of file