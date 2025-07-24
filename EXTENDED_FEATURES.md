# Extended AutoML System - Comprehensive Feature Documentation

## Overview

The extended AutoML system now includes a comprehensive set of features covering the full scope of the exam requirements:

### 1. Text Representations

#### Traditional Approaches
- **TF-IDF**: Enhanced with trigrams (1,3) and character n-grams
- **Bag-of-Words**: With configurable n-gram ranges
- **Feature Selection**: Chi-square, mutual information, ANOVA F-test

#### Pre-trained Embeddings
- **GloVe**: 50, 100, 200, 300 dimensions
- **FastText**: With subword information for OOV handling
- **Word2Vec**: Google News 300d vectors
- **Sentence-BERT**: Contextual sentence embeddings

Example usage:
```bash
# Using GloVe embeddings with SVM
python run_extended_automl.py --dataset amazon --approach svm_rbf --use-embeddings glove

# Using Sentence-BERT with XGBoost
python run_extended_automl.py --dataset imdb --approach xgboost --use-embeddings sentence_bert
```

### 2. Model Architectures

#### Traditional ML Models
- **SVM**: Linear, RBF, Polynomial kernels with automatic parameter tuning
- **XGBoost**: Optimized for text with custom objective functions
- **LightGBM**: Fast gradient boosting with categorical feature support
- **CatBoost**: Handles text features natively
- **Random Forest & Extra Trees**: Ensemble tree methods
- **Naive Bayes**: Multinomial, Complement, and Bernoulli variants

#### Deep Learning Models
- **GRU/BiGRU**: With attention mechanism and layer normalization
- **Multi-scale CNN**: Multiple kernel sizes for different n-grams
- **Hierarchical Attention Network**: Document-level classification
- **CNN-LSTM Hybrid**: Combines local and sequential patterns
- **Wide & Deep**: Memorization + generalization

#### Transformer Models
- **BERT/RoBERTa/ALBERT**: With proper fine-tuning
- **DeBERTa/ELECTRA**: Advanced transformer variants
- **DistilBERT**: Efficient for resource constraints
- **Domain-specific**: SciBERT, BioBERT, FinBERT
- **Multilingual**: mBERT, XLM-RoBERTa

Example usage:
```bash
# XGBoost with hyperparameter optimization
python run_extended_automl.py --dataset ag_news --approach xgboost --use-hpo --hpo-trials 100

# Fine-tune RoBERTa
python run_extended_automl.py --dataset imdb --approach roberta --epochs 3 --batch-size 16

# BiGRU with attention
python run_extended_automl.py --dataset amazon --approach bigru --hidden-dim 256 --num-layers 3
```

### 3. Advanced Augmentation

#### Implemented Strategies
- **Back-translation**: Via MarianMT models (supports multiple languages)
- **Paraphrasing**: Using T5/BART models
- **Contextual replacement**: BERT-based masked token prediction
- **Mixup**: For soft label training
- **Adversarial**: Character-level perturbations
- **Style transfer**: Formal/informal/simple transformations

Example usage:
```bash
# Enable augmentation with back-translation
python run_extended_automl.py --dataset amazon --use-augmentation \
    --augmentation-types backtranslation paraphrase --augmentation-strength heavy
```

### 4. Multi-Fidelity Optimization

#### Fidelity Dimensions
- **Sequence length**: Progressive increase from 50 to full length
- **Vocabulary size**: From 1k to 50k features
- **Model depth/width**: Progressive architecture complexity
- **Training data fraction**: Start with 10%, increase to 100%
- **Training epochs**: Adaptive based on performance
- **Feature dimensions**: For traditional ML methods

#### Scheduling Strategies
- **Successive Halving**: Double resources at each stage
- **Hyperband**: Multiple brackets with different resource allocations
- **Progressive**: Different rates for different fidelities

Example usage:
```bash
# Multi-fidelity optimization with progressive scheduling
python run_extended_automl.py --dataset dbpedia --use-multi-fidelity \
    --approach auto --use-hpo --hpo-trials 200
```

### 5. Hybrid Models

Combines traditional ML with deep learning:
- **hybrid_svm_bert**: BERT features + SVM classifier
- **hybrid_xgboost_lstm**: LSTM features + XGBoost
- **Wide & Deep**: TF-IDF (wide) + Neural embeddings (deep)

Example usage:
```bash
# Hybrid SVM with BERT features
python run_extended_automl.py --dataset ag_news --approach hybrid_svm_bert
```

### 6. Efficient Fine-tuning

#### Parameter-Efficient Methods
- **Adapters**: Small bottleneck layers added to transformers
- **LoRA**: Low-rank adaptation of large models
- **Prefix Tuning**: Learnable prompts prepended to input
- **Layer Freezing**: Freeze bottom N layers

Example in code:
```python
config = TransformerConfig(
    model_name='bert',
    use_adapters=True,
    adapter_size=64,
    freeze_layers=8
)
```

## Complete Example Workflows

### 1. Best Performance on Small Dataset (Amazon)
```bash
# Use SVM with GloVe embeddings and heavy augmentation
python run_extended_automl.py \
    --dataset amazon \
    --approach svm_rbf \
    --use-embeddings glove \
    --use-augmentation \
    --augmentation-types backtranslation synonym contextual \
    --augmentation-strength heavy \
    --use-hpo \
    --hpo-trials 100
```

### 2. Efficient Processing of Large Dataset (DBpedia)
```bash
# Use LightGBM with multi-fidelity optimization
python run_extended_automl.py \
    --dataset dbpedia \
    --approach lightgbm \
    --use-multi-fidelity \
    --n-estimators 200 \
    --max-depth 10 \
    --use-hpo \
    --hpo-timeout 3600
```

### 3. State-of-the-art with Transformers
```bash
# Fine-tune DeBERTa with LoRA
python run_extended_automl.py \
    --dataset imdb \
    --approach deberta \
    --max-seq-length 256 \
    --batch-size 8 \
    --transformer-lr 2e-5 \
    --epochs 3
```

### 4. Automatic Approach Selection
```bash
# Let the system choose the best approach
python run_extended_automl.py \
    --dataset ag_news \
    --approach auto \
    --use-hpo \
    --use-multi-fidelity
```

## Performance Expectations

With the extended system and proper compute resources:

| Dataset | Expected Accuracy | Best Approach | Key Features |
|---------|------------------|---------------|--------------|
| Amazon  | 88-90%          | SVM + GloVe + Heavy Augmentation | Class balancing, feature engineering |
| IMDB    | 92-94%          | RoBERTa/DeBERTa fine-tuning | Long sequences, sentiment-specific |
| AG News | 93-95%          | LightGBM/XGBoost + BERT features | Multi-class, short texts |
| DBpedia | 98-99%          | Hierarchical model or BERT | Many classes, structured text |

## Computational Requirements

### Minimal Setup (CPU only)
- Traditional ML models (SVM, XGBoost, LightGBM)
- Small transformer models (DistilBERT)
- Basic augmentation (synonym replacement)

### Recommended Setup (GPU)
- All models including large transformers
- Back-translation and paraphrasing augmentation
- Multi-fidelity optimization with many trials

### Memory Requirements
- Traditional ML: 8GB RAM
- Deep Learning: 16GB RAM + 8GB GPU
- Large Transformers: 32GB RAM + 16GB GPU

## Troubleshooting

### Out of Memory
1. Reduce batch size
2. Use smaller models (DistilBERT instead of BERT)
3. Enable gradient accumulation
4. Use multi-fidelity with sequence length limits

### Slow Training
1. Use multi-fidelity optimization
2. Start with traditional ML for baseline
3. Use efficient models (LightGBM, DistilBERT)
4. Enable mixed precision training

### Poor Performance
1. Check class imbalance and apply balancing
2. Try different text representations
3. Use augmentation for small datasets
4. Tune hyperparameters with HPO

## Future Extensions

The system is designed to easily incorporate:
- New pre-trained models (GPT, T5 for generation)
- Additional augmentation strategies
- Custom loss functions
- Multi-task learning
- Few-shot learning approaches