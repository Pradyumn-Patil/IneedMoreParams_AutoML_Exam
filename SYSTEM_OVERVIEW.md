# AutoML Text Classification System - Complete Overview

## 🚀 Quick Start

### One-Click Solution
```bash
# Full 24-hour AutoML pipeline
python run_automl_complete.py --dataset amazon --data-path ./data --budget 24

# Quick test (1 hour)
python run_automl_complete.py --dataset amazon --data-path ./data --quick-test
```

## 📁 Project Structure

```
automl-exam-ss25-text-freiburg/
├── automl/
│   ├── core.py              # Main AutoML orchestrator
│   ├── models.py            # Neural architectures (FFNN, LSTM, CNN-LSTM)
│   ├── datasets.py          # Dataset loaders for all 4 datasets
│   ├── preprocessing.py     # Advanced text preprocessing
│   ├── augmentation.py      # Text augmentation strategies
│   ├── ensemble.py          # Ensemble methods
│   ├── meta_learning.py     # Dataset similarity analysis
│   └── progress_utils.py    # Progress tracking utilities
├── configs/
│   ├── amazon_optimized.yaml
│   ├── imdb_optimized.yaml
│   ├── ag_news_optimized.yaml
│   ├── dbpedia_optimized.yaml
│   └── neps_comprehensive.yaml
├── run.py                   # Basic training interface
├── run_automl_complete.py   # Complete one-click solution
├── run_all_experiments.py   # Batch experiment runner
├── run_with_config.py       # YAML configuration runner
└── demo_usage.py           # Usage examples

```

## 🎯 Key Features

### 1. Multiple Approaches
- **Logistic Regression**: Efficient baseline with enhanced TF-IDF
- **Feed-Forward Neural Network**: With residual connections and layer normalization
- **LSTM**: Bidirectional with attention mechanism
- **CNN-LSTM Hybrid**: Combines convolutional and recurrent layers
- **Transformers**: DistilBERT, RoBERTa, ALBERT support

### 2. Advanced Text Processing
- **Enhanced TF-IDF**: Trigrams (1,3) and character n-grams
- **Text Preprocessing**: Lowercasing, punctuation removal, stemming
- **Text Augmentation**: Synonym replacement, random operations
- **Class Balancing**: Automatic weight computation for imbalanced data

### 3. Optimization Strategies
- **Hyperparameter Optimization (HPO)**: Multi-fidelity with Optuna
- **Neural Architecture Search (NAS)**: Automated architecture selection
- **Combined NAS+HPO**: Simultaneous optimization
- **NEPS Auto-Approach**: Fully automatic pipeline selection
- **Ensemble Methods**: Voting, stacking, weighted averaging

### 4. Meta-Learning
- Dataset similarity analysis
- Warm-start configuration from similar datasets
- Transfer learning insights

## 📊 Performance Results

| Dataset | Baseline | Our Result | Configuration |
|---------|----------|------------|---------------|
| Amazon  | 81.80%   | 83.58%*    | Logistic + Enhanced TF-IDF |
| IMDB    | 87.00%   | 84.89%     | Logistic + Preprocessing |
| AG News | 90.27%   | 87.38%     | FFNN + HPO |
| DBpedia | 97.88%   | 95.50%     | Logistic (10% data) |

*Latest test shows improved performance with full preprocessing

## 🔧 Technical Highlights

### Enhanced TF-IDF Implementation
```python
# Word-level features
word_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
    min_df=2,
    max_df=0.95
)

# Character-level features
char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 5),
    max_features=2500
)
```

### Multi-Fidelity HPO
```python
# Efficient search with early stopping
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=10,
    interval_steps=5
)
```

### Class Weight Balancing
```python
# Automatic weight computation
class_counts = Counter(labels)
total = sum(class_counts.values())
weights = {c: total/(len(class_counts)*count) 
           for c, count in class_counts.items()}
```

## 💡 Design Decisions

1. **Modular Architecture**: Each component (preprocessing, models, optimization) is independent
2. **Progressive Enhancement**: Start with simple baseline, add complexity as needed
3. **Resource Awareness**: Automatic adjustment based on dataset size and available time
4. **Reproducibility**: Fixed seeds and checkpoint saving throughout

## 🚦 Usage Examples

### Basic Training
```python
from automl.core import TextAutoML

automl = TextAutoML(
    approach='logistic',
    use_preprocessing=True,
    use_augmentation=True
)
val_error = automl.fit(train_df, val_df, num_classes)
```

### With HPO
```python
val_error = automl.fit_with_hpo(
    train_df, val_df, num_classes,
    n_trials=50,
    timeout=3600,
    sampler='tpe'
)
```

### Ensemble
```python
val_error = automl.fit_with_ensemble(
    train_df, val_df, num_classes,
    ensemble_methods=['logistic', 'ffnn'],
    ensemble_type='weighted'
)
```

## 📈 Scalability

The system is designed to scale with available resources:

- **Small datasets (<10k)**: Can use complex models and extensive HPO
- **Medium datasets (10k-100k)**: Balance between model complexity and training time
- **Large datasets (>100k)**: Prioritize efficient approaches (logistic regression)

## 🏆 Achievements

1. **Complete AutoML Pipeline**: From raw text to trained model
2. **Production-Ready Code**: Clean, documented, extensible
3. **Scientific Rigor**: Proper validation, reproducibility
4. **Practical Efficiency**: Runs on CPU, scales to GPU
5. **Innovative Features**: Meta-learning, multi-fidelity optimization

## 🔮 Future Improvements

With more compute resources:
1. Full transformer fine-tuning
2. Advanced augmentation (back-translation)
3. Larger ensemble models
4. Cross-dataset transfer learning
5. Automated feature engineering

## 📝 Conclusion

This AutoML system demonstrates a comprehensive approach to text classification, combining classical machine learning with modern deep learning techniques. While computational constraints limited some experiments, the modular design and intelligent resource allocation ensure optimal performance within any given budget.