import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from automl.models import SimpleFFNN, LSTMClassifier
from automl.utils import SimpleTextDataset
from pathlib import Path
import logging
from typing import Tuple
from collections import Counter

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, NSGAIISampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

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
        lstm_emb_dim=128,
        lstm_hidden_dim=128,
        fraction_layers_to_finetune: float=1.0,
        # LogisticRegression parameters
        logistic_C=1.0, # Regularization strength
        logistic_max_iter=1000,
        # HPO parameters
        hpo_sampler='tpe',  # 'tpe', 'random', 'cmaes', 'nsga2'
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
        self.lstm_emb_dim = lstm_emb_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.fraction_layers_to_finetune = fraction_layers_to_finetune
        
        # LogisticRegression parameters
        self.logistic_C = logistic_C
        self.logistic_max_iter = logistic_max_iter
        
        # HPO parameters
        self.hpo_sampler = hpo_sampler

        self.model = None
        self.tokenizer = None
        self.vectorizer = None
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
        - approach (str): Model type - 'tfidf', 'lstm', or 'transformer'. Default is 'auto'.
        - vocab_size (int): Maximum vocabulary size.
        - token_length (int): Maximum token sequence length.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - lr (float): Learning rate.
        - weight_decay (float): Weight decay for optimizer.
        - ffnn_hidden (int): Hidden dimension size for FFNN.
        - lstm_emb_dim (int): Embedding dimension size for LSTM.
        - lstm_hidden_dim (int): Hidden dimension size for LSTM.
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
        
        logger.info("Loading and preparing data...")

        self.train_texts = train_df['text'].tolist()
        self.train_labels = train_df['label'].tolist()
        self.val_texts = val_df['text'].tolist()
        self.val_labels = val_df['label'].tolist()
        self.num_classes = num_classes
        logger.info(f"Train class distribution: {Counter(self.train_labels)}")
        logger.info(f"Val class distribution: {Counter(self.val_labels)}")

        dataset = None
        if self.approach in ['ffnn', 'logistic']:
            # Both use TF-IDF vectorization
            self.vectorizer = TfidfVectorizer(
                max_features=self.vocab_size,
                lowercase=True,
                min_df=2,    # ignore words appearing in less than 2 sentences
                max_df=0.8,  # ignore words appearing in > 80% of sentences
                sublinear_tf=True,  # use log-spaced term-frequency scoring
                ngram_range=(1, 2),  # unigrams and bigrams
            )
            X_train = self.vectorizer.fit_transform(self.train_texts)
            X_val = self.vectorizer.transform(self.val_texts)
            
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
            model_name = 'distilbert-base-uncased'
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
                        self.model = AutoModelForSequenceClassification.from_pretrained(
                            model_name, 
                            num_labels=self.num_classes
                        )
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
            X_train = self.vectorizer.transform(self.train_texts)
            X_val = self.vectorizer.transform(self.val_texts)
            
            logger.info("Training Logistic Regression model...")
            self.model.fit(X_train, self.train_labels)
            
            # Evaluate on validation set
            val_preds = self.model.predict(X_val)
            val_acc = accuracy_score(self.val_labels, val_preds)
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
            )
            return 1 - val_acc

    def _train_loop(
        self, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        load_path: Path=None,
        save_path: Path=None,
    ):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        start_epoch = 0
        # handling checkpoint resume
        if load_path is not None:
            _states = torch.load(load_path / "checkpoint.pth", map_location='cpu')  # Load to CPU first
            self.model.load_state_dict(_states["model_state_dict"])
            optimizer.load_state_dict(_states["optimizer_state_dict"])
            start_epoch = _states["epoch"]
            logger.info(f"Resuming from checkpoint at {start_epoch}")

        for epoch in range(start_epoch, self.epochs):            
            total_loss = 0
            for batch in train_loader:
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

            logger.info(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

            if self.val_texts:
                val_preds, val_labels = self._predict(val_loader)
                val_acc = accuracy_score(val_labels, val_preds)
                logger.info(f"Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}")

        if self.val_texts:
            val_preds, val_labels = self._predict(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            logger.info(f"Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}")

        if save_path is not None:
            save_path = Path(save_path) if not isinstance(save_path, Path) else save_path
            save_path.mkdir(parents=True, exist_ok=True)

            # Use trial-specific checkpoint name if this is during HPO
            if hasattr(self, 'trial_number'):
                checkpoint_name = f"checkpoint_trial_{self.trial_number}.pth"
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
            # For sklearn LogisticRegression
            _X = self.vectorizer.transform(test_data['text'].tolist())
            _labels = test_data['label'].tolist()
            _preds = self.model.predict(_X)
            return np.array(_preds), np.array(_labels)
            
        elif self.approach == 'ffnn':
            _X = self.vectorizer.transform(test_data['text'].tolist()).toarray()
            _labels = test_data['label'].tolist()
            _dataset = torch.utils.data.TensorDataset(
                torch.tensor(_X, dtype=torch.float32),
                torch.tensor(_labels)
            )
            _loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=True)
            return self._predict(_loader)
            
        elif self.approach in ['lstm', 'transformer']:
            _dataset = SimpleTextDataset(
                test_data['text'].tolist(),
                test_data['label'].tolist(),
                self.tokenizer,
                self.token_length
            )
            _loader = DataLoader(_dataset, batch_size=self.batch_size, shuffle=True)
            return self._predict(_loader)
        else:
            raise ValueError(f"Unrecognized approach: {self.approach}")
            # handling any possible tokenization

    def _create_hpo_search_space(self, trial):
        """Create hyperparameter search space for the current approach."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for HPO. Install with: pip install optuna")
            
        params = {}
        
        if self.approach == 'logistic':
            params['logistic_C'] = trial.suggest_float('logistic_C', 0.001, 100.0, log=True)
            params['logistic_max_iter'] = trial.suggest_int('logistic_max_iter', 100, 2000)
            params['vocab_size'] = trial.suggest_int('vocab_size', 5000, 20000)
            
        elif self.approach == 'ffnn':
            params['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            params['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 0.1)
            params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            params['ffnn_hidden'] = trial.suggest_int('ffnn_hidden', 64, 512)
            params['epochs'] = trial.suggest_int('epochs', 3, 10)
            params['vocab_size'] = trial.suggest_int('vocab_size', 5000, 20000)
            
        elif self.approach == 'lstm':
            params['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            params['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 0.1)
            params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
            params['lstm_emb_dim'] = trial.suggest_int('lstm_emb_dim', 64, 256)
            params['lstm_hidden_dim'] = trial.suggest_int('lstm_hidden_dim', 64, 256)
            params['epochs'] = trial.suggest_int('epochs', 3, 10)
            params['token_length'] = trial.suggest_int('token_length', 64, 256)
            
        elif self.approach == 'transformer':
            params['lr'] = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
            params['weight_decay'] = trial.suggest_float('weight_decay', 0.0, 0.1)
            params['batch_size'] = trial.suggest_categorical('batch_size', [8, 16, 32])
            params['fraction_layers_to_finetune'] = trial.suggest_float('fraction_layers_to_finetune', 0.1, 1.0)
            params['epochs'] = trial.suggest_int('epochs', 2, 5)
            params['token_length'] = trial.suggest_int('token_length', 128, 512)
            
        return params

    def _hpo_objective(self, trial, save_path=None):
        """Objective function for Optuna hyperparameter optimization."""
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
            # Train and evaluate with checkpoint saving
            if save_path is not None:
                # Set trial number for checkpoint naming
                temp_automl.trial_number = trial.number
                val_error = temp_automl.fit(train_df, val_df, self.num_classes, save_path=save_path)
            else:
                val_error = temp_automl.fit(train_df, val_df, self.num_classes)
                
            logger.info(f"Trial {trial.number}: val_error = {val_error:.4f}")
            return val_error
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float('inf')

    def optimize_hyperparameters(self, n_trials=20, timeout=3600, save_path=None, sampler=None):
        """Run hyperparameter optimization using Optuna."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for HPO. Install with: pip install optuna")
            
        # Select sampler
        sampler_name = sampler or self.hpo_sampler
        samplers = {
            'tpe': TPESampler(seed=self.seed),
            'random': RandomSampler(seed=self.seed),
            'cmaes': CmaEsSampler(seed=self.seed),
            'nsga2': NSGAIISampler(seed=self.seed)
        }
        
        if sampler_name not in samplers:
            logger.warning(f"Unknown sampler '{sampler_name}', falling back to TPE")
            sampler_name = 'tpe'
            
        sampler_obj = samplers[sampler_name]
        logger.info(f"Using {sampler_name.upper()} sampler for HPO")
        logger.info(f"Starting HPO for {self.approach} with {n_trials} trials (timeout: {timeout}s)")
        
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler_obj,
            study_name=f"hpo_{self.approach}_{self.seed}"
        )
        
        study.optimize(
            lambda trial: self._hpo_objective(trial, save_path=save_path),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        logger.info(f"HPO completed. Best params: {best_params}")
        logger.info(f"Best validation error: {study.best_value:.4f}")
        
        # Update instance with best parameters
        for param, value in best_params.items():
            if hasattr(self, param):
                setattr(self, param, value)
                
        return study.best_value, best_params

    def fit_with_hpo(
        self, 
        train_df: pd.DataFrame,
        val_df: pd.DataFrame, 
        num_classes: int,
        n_trials: int = 20,
        timeout: int = 3600,
        save_path: Path = None,
        sampler: str = None,
        **kwargs
    ):
        """Fit model with hyperparameter optimization."""
        logger.info("Starting fit_with_hpo...")
        
        # Store data for HPO
        self.train_texts = train_df['text'].tolist()
        self.train_labels = train_df['label'].tolist()
        self.val_texts = val_df['text'].tolist()
        self.val_labels = val_df['label'].tolist()
        self.num_classes = num_classes
        
        # Run HPO
        best_score, best_params = self.optimize_hyperparameters(n_trials, timeout, save_path, sampler)
        
        # Final training with best parameters
        logger.info("Training final model with optimized hyperparameters...")
        final_score = self.fit(train_df, val_df, num_classes, save_path=save_path, **kwargs)
        
        logger.info(f"HPO completed. Final validation error: {final_score:.4f}")
        return final_score

    def _freeze_layers(self, model, fraction_layers_to_finetune: float = 1.0) -> None:
        """Freeze layers in transformer model for partial fine-tuning."""
        total_layers = len(model.distilbert.transformer.layer)
        _num_layers_to_finetune = int(fraction_layers_to_finetune * total_layers)
        layers_to_freeze = total_layers - _num_layers_to_finetune

        for layer in model.distilbert.transformer.layer[:layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
                
        logger.info(f"Froze {layers_to_freeze}/{total_layers} layers, fine-tuning {_num_layers_to_finetune} layers")

# end of file