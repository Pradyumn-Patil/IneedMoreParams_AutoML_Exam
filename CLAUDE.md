# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AutoML system for text classification developed for the SS25 AutoML course at the University of Freiburg. The system supports multiple approaches (Logistic Regression, FFNN, LSTM, Transformer) and optimization strategies (basic training, HPO, NAS, combined NAS+HPO, multi-fidelity, and NEPS auto-approach selection).

## Common Commands

### Installation
```bash
# Create virtual environment
python3 -m venv automl-text-env
source automl-text-env/bin/activate

# Install package in editable mode
pip install -e .
```

### Running Experiments

Basic training:
```bash
python run.py --data-path <path> --dataset amazon --epochs 5 --approach transformer
```

With hyperparameter optimization (HPO):
```bash
python run.py --data-path <path> --dataset amazon --use-hpo --hpo-trials 50 --hpo-sampler tpe
```

With neural architecture search (NAS):
```bash
python run.py --data-path <path> --dataset amazon --use-nas --nas-trials 20
```

Combined NAS+HPO:
```bash
python run.py --data-path <path> --dataset amazon --use-hpo --use-nas
```

NEPS auto-approach selection (fully automatic):
```bash
python run.py --data-path <path> --dataset amazon --use-neps-auto-approach --neps-max-evaluations 16
```

Enhanced FFNN with residual connections:
```bash
python run.py --data-path <path> --dataset amazon --approach ffnn --use-enhanced-ffnn --ffnn-num-layers 4 --ffnn-dropout-rate 0.2
```

### Quick Testing
```bash
# Test with small data fraction
python run.py --data-path <path> --dataset amazon --epochs 1 --data-fraction 0.2

# Test enhanced FFNN
python run.py --data-path <path> --dataset amazon --approach ffnn --use-enhanced-ffnn --epochs 1
```

## High-Level Architecture

The codebase follows a modular design with clear separation of concerns:

### Core Components

1. **TextAutoML** (`automl/core.py`): Main AutoML interface that orchestrates the entire pipeline
   - Supports multiple fitting methods: `fit()`, `fit_with_hpo()`, `fit_with_nas()`, `fit_with_nas_hpo()`, `fit_with_neps_auto_approach()`
   - Handles data preprocessing, model selection, training, and evaluation
   - Implements checkpoint saving/loading for resumable training
   - Multi-fidelity optimization with trial pruning strategies

2. **Model Architectures** (`automl/models.py`):
   - **SimpleFFNN**: Basic feed-forward network with configurable layers
   - **EnhancedFFNN**: Advanced FFNN with residual connections, layer normalization, and GELU activation
   - **LSTMClassifier**: LSTM with optional bidirectional and attention mechanisms
   - **NASSearchableFFNN/LSTM**: Architecture-searchable variants for NAS
   - All models inherit from `nn.Module` and follow PyTorch conventions

3. **Dataset Loaders** (`automl/datasets.py`):
   - Unified interface for 4 text classification datasets (AG News, IMDB, Amazon, DBpedia)
   - Handles train/validation/test splits
   - Returns pandas DataFrames with 'text' and 'label' columns

### Optimization Strategies

The system implements a hierarchy of optimization approaches:

1. **Basic Training**: Standard model training with fixed hyperparameters
2. **HPO**: Uses Optuna with multiple samplers (TPE, Random, CMA-ES, NSGA-II)
3. **NAS**: Searches over architecture configurations (layers, dimensions, activations)
4. **NAS+HPO**: Simultaneous architecture and hyperparameter search
5. **Multi-fidelity**: Efficient search with early stopping (Median, Successive Halving, Hyperband pruners)
6. **NEPS Auto-Approach**: Automatically selects best approach, optimization strategy, and configurations

### Search Spaces

**FFNN NAS Space**:
- Number of layers (1-4)
- Hidden dimensions per layer
- Activation functions (relu, gelu, tanh)
- Batch normalization
- Dropout rates

**LSTM NAS Space**:
- Embedding dimensions
- Hidden dimensions
- Number of layers (1-3)
- Bidirectional flag
- Attention mechanism
- Dropout rates

**HPO Space** (approach-specific):
- Learning rate
- Batch size
- Weight decay
- Model-specific parameters (e.g., hidden dims, dropout)

## Key Implementation Details

1. **Data Processing**:
   - TF-IDF vectorization for logistic regression and FFNN
   - Word embeddings for LSTM
   - Pre-trained tokenizers for transformers
   - Automatic train/validation split if not provided

2. **Training Loop**:
   - Early stopping based on validation loss
   - Checkpoint saving for best models
   - Progress tracking with TQDM
   - GPU acceleration when available

3. **Evaluation**:
   - Accuracy as primary metric
   - Classification reports for detailed analysis
   - Test predictions saved as `.npy` files

4. **Configuration**:
   - YAML config file support for batch experiments
   - Command-line arguments for all major parameters
   - Sensible defaults for quick experimentation

## Dataset Information

| Dataset | Classes | Train Size | Test Size | Avg. Sequence Length |
|---------|---------|------------|-----------|---------------------|
| Amazon  | 3       | ~25k       | ~5k       | 512 chars          |
| IMDB    | 2       | ~25k       | ~25k      | 1300 chars         |
| AG News | 4       | ~120k      | ~7.6k     | 235 chars          |
| DBpedia | 14      | ~560k      | ~70k      | 300 chars          |

Reference accuracies: Amazon (81.8%), IMDB (87.0%), AG News (90.3%), DBpedia (97.9%)

## Recent Improvements

Major enhancements have been implemented to achieve higher accuracy:

1. **Enhanced TF-IDF**: Trigrams (1,3) and character n-grams for better feature extraction
2. **Advanced Preprocessing**: Dataset-specific text preprocessing with stemming/lemmatization
3. **Enhanced FFNN**: Residual connections, layer normalization, and GELU activation for better neural performance
4. **Multiple Transformers**: Support for DistilBERT, RoBERTa, ALBERT with optional adapters
5. **Dataset-Specific HPO**: Optimized search spaces based on dataset characteristics
6. **Configuration System**: Pre-optimized configs for each dataset in `configs/` directory

See `IMPROVEMENTS_SUMMARY.md` for detailed information.

## Running Optimized Experiments

Use the configuration system for best results:
```bash
# Run optimized config for specific dataset
python run_with_config.py --config configs/amazon_optimized.yaml --data-path ./data

# Run comprehensive NEPS optimization
python run_with_config.py --config configs/neps_comprehensive.yaml --data-path ./data
```

## Dependencies

Key libraries (from `pyproject.toml`):
- PyTorch (via transformers)
- Transformers >=4.20
- Optuna ^3.0
- NEPS (neural-pipeline-search) 0.13.0
- Scikit-learn ^1.3
- Datasets >=2.0
- NLTK (for preprocessing)