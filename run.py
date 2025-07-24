"""A STARTER KIT SCRIPT for SS25 AutoML Exam --- Modality III: Text

You are not expected to follow this script or be constrained to it.

For a test run:
1) Download datasets (see, README) at chosen path
2) Run the script: 
```
# Basic training
python run.py --data-path <path-to-downloaded-data> --dataset amazon --epochs 1

# HPO with different samplers
python run.py --data-path <path-to-downloaded-data> --dataset amazon --use-hpo --hpo-trials 50 --hpo-sampler tpe
python run.py --data-path <path-to-downloaded-data> --dataset amazon --use-hpo --hpo-trials 50 --hpo-sampler random
python run.py --data-path <path-to-downloaded-data> --dataset amazon --use-hpo --hpo-trials 50 --hpo-sampler cmaes
python run.py --data-path <path-to-downloaded-data> --dataset amazon --use-hpo --hpo-trials 50 --hpo-sampler nsga2

# NEPS Auto-Approach Selection (Automatic Everything!)
# NEPS will automatically select the best combination of:
# - Model approach (logistic, ffnn, lstm, transformer)
# - Optimization strategy (basic, hpo, nas, nas_hpo)
# - Multi-fidelity setting (True/False)
# - HPO sampler (tpe, random, cmaes)
# - HPO pruner (median, successive_halving, hyperband)
python run.py --data-path <path-to-downloaded-data> --dataset amazon --use-neps-auto-approach --neps-max-evaluations 16 --neps-timeout 7200

# NEPS with more evaluations for better results
python run.py --data-path <path-to-downloaded-data> --dataset amazon --use-neps-auto-approach --neps-max-evaluations 32 --neps-timeout 14400 --neps-searcher bayesian_optimization

# NEPS quick test with fewer evaluations
python run.py --data-path <path-to-downloaded-data> --dataset amazon --use-neps-auto-approach --neps-max-evaluations 8 --neps-timeout 3600 --neps-searcher random_search
```

"""
from __future__ import annotations

import argparse
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from typing import List
import yaml

from automl.core import TextAutoML
from automl.datasets import (
    AGNewsDataset,
    AmazonReviewsDataset,
    DBpediaDataset,
    IMDBDataset,
)

logger = logging.getLogger(__name__)

FINAL_TEST_DATASET=...  # TBA later


def main_loop(
        dataset: str,
        output_path: Path,
        data_path: Path,
        seed: int,
        approach: str,
        val_size: float = 0.2,
        vocab_size: int = 10000,
        token_length: int = 128,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 0.0001,
        weight_decay: float = 0.01,
        ffnn_hidden: int = 128,
        use_enhanced_ffnn: bool = False,
        ffnn_num_layers: int = 3,
        ffnn_dropout_rate: float = 0.1,
        ffnn_use_residual: bool = True,
        ffnn_use_layer_norm: bool = True,
        lstm_emb_dim: int = 128,
        lstm_hidden_dim: int = 128,
        fraction_layers_to_finetune: float = 1.0,
        data_fraction: int = 1.0,
        load_path: Path = None,
        use_hpo: bool = False,
        hpo_trials: int = 20,
        hpo_timeout: int = 3600,
        hpo_sampler: str = 'tpe',
        hpo_pruner: str = 'median',
        use_multi_fidelity: bool = False,
        use_nas: bool = False,
        nas_trials: int = 20,
        nas_timeout: int = 3600,
        # NEPS Auto-Approach Selection
        use_neps_auto_approach: bool = False,
        neps_max_evaluations: int = 16,
        neps_timeout: int = 7200,
        neps_searcher: str = 'bayesian_optimization',
        # Ensemble parameters
        use_ensemble: bool = False,
        ensemble_methods: List[str] = None,
        ensemble_type: str = 'auto',
        individual_trials: int = 5,
        # Augmentation parameters
        use_augmentation: bool = False,
        augmentation_strength: str = 'medium',
        augmentation_factor: float = 0.5,
        balance_augmentation: bool = True,
    ) -> None:
    match dataset:
        case "ag_news":
            dataset_class = AGNewsDataset
        case "imdb":
            dataset_class = IMDBDataset
        case "amazon":
            dataset_class = AmazonReviewsDataset
        case "dbpedia":
            dataset_class = DBpediaDataset
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

    logger.info("Fitting Text AutoML")

    # You do not need to follow this setup or API it's merely here to provide
    # an example of how your AutoML system could be used.
    # As a general rule of thumb, you should **never** pass in any
    # test data to your AutoML solution other than to generate predictions.

    # Get the dataset and create dataloaders
    data_path = Path(data_path) if isinstance(data_path, str) else data_path
    data_info = dataset_class(data_path).create_dataloaders(val_size=val_size, random_state=seed)
    train_df = data_info['train_df']
    
    _subsample = np.random.choice(
        list(range(len(train_df))),
        size=int(data_fraction * len(train_df)),
        replace=False,
    )
    train_df = train_df.iloc[_subsample]
    
    val_df = data_info.get('val_df', None)
    test_df = data_info['test_df']
    num_classes = data_info['num_classes']
    logger.info(
        f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}"
    )
    logger.info(f"Number of classes: {num_classes}")

    # Initialize the TextAutoML instance with the best parameters
    automl = TextAutoML(
        seed=seed,
        approach=approach,
        vocab_size=vocab_size,
        token_length=token_length,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        ffnn_hidden=ffnn_hidden,
        use_enhanced_ffnn=use_enhanced_ffnn,
        ffnn_num_layers=ffnn_num_layers,
        ffnn_dropout_rate=ffnn_dropout_rate,
        ffnn_use_residual=ffnn_use_residual,
        ffnn_use_layer_norm=ffnn_use_layer_norm,
        lstm_emb_dim=lstm_emb_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        fraction_layers_to_finetune=fraction_layers_to_finetune,
        hpo_sampler=hpo_sampler,
        hpo_pruner=hpo_pruner,
        use_multi_fidelity=use_multi_fidelity,
        dataset_name=dataset,  # Pass dataset name for dataset-specific optimization
        use_augmentation=use_augmentation,
        augmentation_strength=augmentation_strength,
        augmentation_factor=augmentation_factor,
        balance_augmentation=balance_augmentation,
    )

    # Check if ensemble training should be used
    if use_ensemble:
        logger.info("Using ensemble training")
        
        # Set default ensemble methods if not provided
        if ensemble_methods is None:
            ensemble_methods = ['logistic', 'ffnn']  # Conservative default for speed
            
        val_err = automl.fit_with_ensemble(
            train_df,
            val_df,
            num_classes=num_classes,
            ensemble_methods=ensemble_methods,
            ensemble_type=ensemble_type,
            individual_trials=individual_trials,
            save_path=output_path,
        )
    # Check if NEPS auto-approach selection should be used
    elif use_neps_auto_approach:
        logger.info("Using NEPS automatic approach selection")
        logger.info("NEPS will select the best approach and optimization strategy automatically")
        logger.info("Existing Optuna methods will be used internally for optimization")
        
        val_err = automl.fit_with_neps_auto_approach(
            train_df=train_df,
            val_df=val_df,
            num_classes=num_classes,
            max_evaluations=neps_max_evaluations,
            timeout=neps_timeout,
            searcher=neps_searcher,
            working_directory=str(output_path / "neps_auto_approach"),
            save_path=output_path,
        )
        
        logger.info(f"NEPS auto-approach selection completed!")
        logger.info(f"Final validation error: {val_err:.4f}")
        
    # Standard Optuna-based pipeline with manual approach selection
    elif use_hpo and use_nas:
        val_err = automl.fit_with_nas_hpo(
            train_df,
            val_df,
            num_classes=num_classes,
            n_trials=max(hpo_trials, nas_trials),  # Use the larger of the two
            timeout=max(hpo_timeout, nas_timeout),  # Use the larger of the two
            sampler=hpo_sampler,
            pruner=hpo_pruner,
            use_multi_fidelity=use_multi_fidelity,
            load_path=load_path,
            save_path=output_path,
        )
    elif use_hpo:
        val_err = automl.fit_with_hpo(
            train_df,
            val_df,
            num_classes=num_classes,
            n_trials=hpo_trials,
            timeout=hpo_timeout,
            sampler=hpo_sampler,
            pruner=hpo_pruner,
            use_multi_fidelity=use_multi_fidelity,
            load_path=load_path,
            save_path=output_path,
        )
    elif use_nas:
        val_err = automl.fit_with_nas(
            train_df,
            val_df,
            num_classes=num_classes,
            n_trials=nas_trials,
            timeout=nas_timeout,
            load_path=load_path,
            save_path=output_path,
        )
    else:
        val_err = automl.fit(
            train_df,
            val_df,
            num_classes=num_classes,
            load_path=load_path,
            save_path=output_path,
        )
    logger.info("Training complete")

    # Predict on the test set
    if use_ensemble and hasattr(automl, 'ensemble'):
        test_preds = automl.predict_ensemble(test_df)
        test_labels = test_df['label'].values
    else:
        test_preds, test_labels = automl.predict(test_df)

    # Write the predictions of X_test to disk
    logger.info("Writing predictions to disk")
    with (output_path / "score.yaml").open("w") as f:
        yaml.safe_dump({"val_err": float(val_err)}, f)
    logger.info(f"Saved validataion score at {output_path / 'score.yaml'}")
    with (output_path / "test_preds.npy").open("wb") as f:
        np.save(f, test_preds)
    logger.info(f"Saved tet prediction at {output_path / 'test_preds.npy'}")

    # In case of running on the final exam data, also add the predictions.npy
    # to the correct location for auto evaluation.
    if dataset == FINAL_TEST_DATASET: 
        test_output_path = output_path / "predictions.npy"
        test_output_path.parent.mkdir(parents=True, exist_ok=True)
        with test_output_path.open("wb") as f:
            np.save(f, test_preds)

    # Check if test_labels has missing data
    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        logger.info(f"Accuracy on test set: {acc}")
        with (output_path / "score.yaml").open("a+") as f:
            yaml.safe_dump({"test_err": float(1-acc)}, f)
        
        # Log detailed classification report for better insight
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(test_labels, test_preds)}")
    else:
        # This is the setting for the exam dataset, you will not have access to the labels
        logger.info(f"No test labels available for dataset '{dataset}'")

    return val_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to run on.",
        choices=["ag_news", "imdb", "amazon", "dbpedia",]
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "The path to save the predictions to."
            " By default this will just save to the cwd as `./results`."
        )
    )
    parser.add_argument(
        "--load-path",
        type=Path,
        default=None,
        help="The path to resume checkpoint from."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "The path to laod the data from."
            " By default this will look up cwd for `./.data/`."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility if you are using any randomness,"
            " i.e. torch, numpy, pandas, sklearn, etc."
        )
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="transformer",
        choices=["ffnn", "logistic", "lstm", "transformer"],
        help=(
            "The approach to use for the AutoML system. "
            "Options are 'tfidf', 'logistic', 'lstm', or 'transformer'."
        )
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="The size of the vocabulary to use for the text dataset."
    )
    parser.add_argument(
        "--token-length",
        type=int,
        default=128,
        help="The maximum length of tokens to use for the text dataset."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="The number of epochs to train the model for."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size to use for training and evaluation."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="The learning rate to use for the optimizer."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="The weight decay to use for the optimizer."
    )

    parser.add_argument(
        "--lstm-emb-dim",
        type=int,
        default=64,
        help="The embedding dimension to use for the LSTM model."
    )

    parser.add_argument(
        "--lstm-hidden-dim",
        type=int,
        default=64,
        help="The hidden size to use for the LSTM model."
    )

    parser.add_argument(
        "--ffnn-hidden-layer-dim",
        type=int,
        default=64,
        help="The hidden size to use for the model."
    )
    
    # Enhanced FFNN arguments
    parser.add_argument(
        "--use-enhanced-ffnn",
        action="store_true",
        help="Use enhanced FFNN with residual connections and layer normalization."
    )
    parser.add_argument(
        "--ffnn-num-layers",
        type=int,
        default=3,
        help="Number of layers for enhanced FFNN."
    )
    parser.add_argument(
        "--ffnn-dropout-rate",
        type=float,
        default=0.1,
        help="Dropout rate for enhanced FFNN."
    )
    parser.add_argument(
        "--ffnn-use-residual",
        action="store_true",
        default=True,
        help="Use residual connections in enhanced FFNN."
    )
    parser.add_argument(
        "--ffnn-use-layer-norm",
        action="store_true", 
        default=True,
        help="Use layer normalization in enhanced FFNN."
    )

    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1,
        help="Subsampling of training set, in fraction (0, 1]."
    )
    parser.add_argument(
        "--use-hpo",
        action="store_true",
        help="Enable hyperparameter optimization using Optuna."
    )
    parser.add_argument(
        "--hpo-trials",
        type=int,
        default=20,
        help="Number of trials for hyperparameter optimization."
    )
    parser.add_argument(
        "--hpo-timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for hyperparameter optimization."
    )
    parser.add_argument(
        "--hpo-sampler",
        type=str,
        default="tpe",
        choices=["tpe", "random", "cmaes", "nsga2"],
        help="Sampler type for hyperparameter optimization."
    )
    parser.add_argument(
        "--hpo-pruner",
        type=str,
        default="median",
        choices=["median", "successive_halving", "hyperband"],
        help="Pruner type for multi-fidelity hyperparameter optimization."
    )
    parser.add_argument(
        "--use-multi-fidelity",
        action="store_true",
        help="Enable multi-fidelity optimization with pruning."
    )
    parser.add_argument(
        "--use-nas",
        action="store_true",
        help="Enable Neural Architecture Search using Optuna."
    )
    parser.add_argument(
        "--nas-trials",
        type=int,
        default=20,
        help="Number of trials for Neural Architecture Search."
    )
    parser.add_argument(
        "--nas-timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for Neural Architecture Search."
    )
    
    # NEPS Auto-Approach Selection arguments
    parser.add_argument(
        "--use-neps-auto-approach",
        action="store_true",
        help="Use NEPS for automatic approach selection (uses existing Optuna methods internally)."
    )
    parser.add_argument(
        "--neps-max-evaluations",
        type=int,
        default=16,
        help="Maximum number of approach+strategy combinations for NEPS to evaluate."
    )
    parser.add_argument(
        "--neps-timeout",
        type=int,
        default=7200,
        help="Total timeout in seconds for NEPS auto-approach selection."
    )
    parser.add_argument(
        "--neps-searcher",
        type=str,
        default="bayesian_optimization",
        choices=["bayesian_optimization", "evolutionary_search", "random_search"],
        help="NEPS searcher algorithm for approach selection."
    )
    
    # Ensemble arguments
    parser.add_argument(
        "--use-ensemble",
        action="store_true",
        help="Train ensemble of multiple models for better performance."
    )
    parser.add_argument(
        "--ensemble-methods",
        type=str,
        nargs="+",
        default=None,
        choices=["logistic", "ffnn", "lstm", "transformer"],
        help="List of methods to include in ensemble (default: logistic ffnn)."
    )
    parser.add_argument(
        "--ensemble-type",
        type=str,
        default="auto",
        choices=["voting", "stacking", "weighted", "auto"],
        help="Type of ensemble method to use."
    )
    parser.add_argument(
        "--individual-trials",
        type=int,
        default=5,
        help="Number of HPO trials for each individual model in ensemble."
    )
    
    # Augmentation arguments
    parser.add_argument(
        "--use-augmentation",
        action="store_true",
        help="Enable text augmentation for training data."
    )
    parser.add_argument(
        "--augmentation-strength",
        type=str,
        default="medium",
        choices=["light", "medium", "heavy"],
        help="Strength of text augmentation."
    )
    parser.add_argument(
        "--augmentation-factor",
        type=float,
        default=0.5,
        help="Fraction of training data to augment (0.0-1.0)."
    )
    parser.add_argument(
        "--balance-augmentation",
        action="store_true",
        default=True,
        help="Augment minority classes more to balance dataset."
    )

    args = parser.parse_args()

    logger.info(f"Running text dataset {args.dataset}\n{args}")

    if args.output_path is None:
        args.output_path =  (
            Path.cwd().absolute() / 
            "results" / 
            f"dataset={args.dataset}" / 
            f"seed={args.seed}"
        )
    if args.data_path is None:
        args.data_path = Path.cwd().absolute() / ".data"

    args.output_path = Path(args.output_path).absolute()
    args.output_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, filename=args.output_path / "run.log")

    main_loop(
        dataset=args.dataset,
        output_path=Path(args.output_path).absolute(),
        data_path=Path(args.data_path).absolute(),
        seed=args.seed,
        approach=args.approach,
        vocab_size=args.vocab_size,
        token_length=args.token_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ffnn_hidden=args.ffnn_hidden_layer_dim,
        use_enhanced_ffnn=args.use_enhanced_ffnn,
        ffnn_num_layers=args.ffnn_num_layers,
        ffnn_dropout_rate=args.ffnn_dropout_rate,
        ffnn_use_residual=args.ffnn_use_residual,
        ffnn_use_layer_norm=args.ffnn_use_layer_norm,
        lstm_emb_dim=args.lstm_emb_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        data_fraction=args.data_fraction,
        load_path=Path(args.load_path) if args.load_path is not None else None,
        use_hpo=args.use_hpo,
        hpo_trials=args.hpo_trials,
        hpo_timeout=args.hpo_timeout,
        hpo_sampler=args.hpo_sampler,
        hpo_pruner=args.hpo_pruner,
        use_multi_fidelity=args.use_multi_fidelity,
        use_nas=args.use_nas,
        nas_trials=args.nas_trials,
        nas_timeout=args.nas_timeout,
        # NEPS Auto-Approach Selection
        use_neps_auto_approach=args.use_neps_auto_approach,
        neps_max_evaluations=args.neps_max_evaluations,
        neps_timeout=args.neps_timeout,
        neps_searcher=args.neps_searcher,
        # Ensemble parameters
        use_ensemble=args.use_ensemble,
        ensemble_methods=args.ensemble_methods,
        ensemble_type=args.ensemble_type,
        individual_trials=args.individual_trials,
        # Augmentation parameters
        use_augmentation=args.use_augmentation,
        augmentation_strength=args.augmentation_strength,
        augmentation_factor=args.augmentation_factor,
        balance_augmentation=args.balance_augmentation,
    )
# end of file