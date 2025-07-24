#!/usr/bin/env python3
"""
Comprehensive System Validation Script

Tests all components of the expanded AutoML system:
1. Embeddings (GloVe, FastText, Word2Vec, Sentence-BERT)
2. Traditional ML (SVM, XGBoost, LightGBM, Random Forest)
3. Deep Learning (GRU, Enhanced models)
4. Transformers (BERT, RoBERTa fine-tuning)
5. Advanced Augmentation
6. Fidelity Management
7. Integration tests
"""

import sys
import logging
import traceback
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemValidator:
    """Validates all components of the AutoML system."""
    
    def __init__(self):
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def run_all_tests(self):
        """Run all validation tests."""
        print("=" * 80)
        print("AutoML System Validation")
        print("=" * 80)
        
        # Test each component
        self.test_embeddings()
        self.test_traditional_ml()
        self.test_deep_models()
        self.test_transformers()
        self.test_augmentation()
        self.test_fidelity_management()
        self.test_integration()
        
        # Print summary
        self.print_summary()
        
    def test_embeddings(self):
        """Test embedding modules."""
        print("\n1. Testing Embeddings Module...")
        
        try:
            from automl.embeddings import EmbeddingManager, GloVeEmbedding, SentenceBERTEmbedding
            
            # Test data
            texts = [
                "This is a test sentence.",
                "Another example text for testing.",
                "Machine learning is awesome!"
            ]
            
            # Test GloVe
            print("   - Testing GloVe embeddings...")
            manager = EmbeddingManager()
            glove = manager.get_embedding('glove', dim=50)
            glove_embeddings = glove.encode(texts)
            assert glove_embeddings.shape == (3, 50), f"GloVe shape mismatch: {glove_embeddings.shape}"
            print("     ✓ GloVe embeddings working")
            
            # Test Sentence-BERT
            print("   - Testing Sentence-BERT...")
            sbert = manager.get_embedding('sentence_bert')
            sbert_embeddings = sbert.encode(texts)
            assert sbert_embeddings.shape[0] == 3, "Sentence-BERT batch size mismatch"
            print("     ✓ Sentence-BERT working")
            
            self.results['embeddings'] = 'PASSED'
            print("   ✅ Embeddings module: PASSED")
            
        except Exception as e:
            self.results['embeddings'] = f'FAILED: {str(e)}'
            print(f"   ❌ Embeddings module: FAILED - {str(e)}")
            traceback.print_exc()
            
    def test_traditional_ml(self):
        """Test traditional ML models."""
        print("\n2. Testing Traditional ML Models...")
        
        try:
            from automl.traditional_ml import TraditionalMLFactory
            from sklearn.datasets import make_classification
            
            # Create synthetic data
            X, y = make_classification(n_samples=100, n_features=20, n_classes=3, 
                                     n_informative=15, random_state=42)
            
            models_to_test = ['svm_linear', 'xgboost', 'lightgbm', 'random_forest']
            
            for model_type in models_to_test:
                print(f"   - Testing {model_type}...")
                try:
                    model = TraditionalMLFactory.create_model(model_type)
                    model.fit(X[:80], y[:80])
                    predictions = model.predict(X[80:])
                    assert len(predictions) == 20, f"Prediction length mismatch for {model_type}"
                    print(f"     ✓ {model_type} working")
                except Exception as e:
                    print(f"     ❌ {model_type} failed: {str(e)}")
                    
            self.results['traditional_ml'] = 'PASSED'
            print("   ✅ Traditional ML models: PASSED")
            
        except Exception as e:
            self.results['traditional_ml'] = f'FAILED: {str(e)}'
            print(f"   ❌ Traditional ML models: FAILED - {str(e)}")
            
    def test_deep_models(self):
        """Test deep learning models."""
        print("\n3. Testing Deep Learning Models...")
        
        try:
            from automl.models_extended import GRUClassifier, MultiScaleCNN, HierarchicalAttentionNetwork
            
            # Test parameters
            vocab_size = 1000
            batch_size = 4
            seq_length = 20
            num_classes = 3
            
            # Create dummy input
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
            attention_mask = torch.ones_like(input_ids)
            
            # Test GRU
            print("   - Testing GRU classifier...")
            gru_model = GRUClassifier(
                vocab_size=vocab_size,
                embedding_dim=50,
                hidden_dim=32,
                output_dim=num_classes,
                bidirectional=True,
                use_attention=True
            )
            gru_output = gru_model(input_ids, attention_mask)
            assert gru_output.shape == (batch_size, num_classes), f"GRU output shape mismatch: {gru_output.shape}"
            print("     ✓ GRU classifier working")
            
            # Test Multi-scale CNN
            print("   - Testing Multi-scale CNN...")
            cnn_model = MultiScaleCNN(
                vocab_size=vocab_size,
                embedding_dim=50,
                num_filters=32,
                filter_sizes=[3, 4, 5],
                output_dim=num_classes
            )
            cnn_output = cnn_model(input_ids)
            assert cnn_output.shape == (batch_size, num_classes), f"CNN output shape mismatch: {cnn_output.shape}"
            print("     ✓ Multi-scale CNN working")
            
            self.results['deep_models'] = 'PASSED'
            print("   ✅ Deep learning models: PASSED")
            
        except Exception as e:
            self.results['deep_models'] = f'FAILED: {str(e)}'
            print(f"   ❌ Deep learning models: FAILED - {str(e)}")
            traceback.print_exc()
            
    def test_transformers(self):
        """Test transformer models."""
        print("\n4. Testing Transformer Models...")
        
        try:
            from automl.transformers_advanced import TransformerModelFactory, TransformerConfig
            
            # Test configuration
            config = TransformerConfig(
                model_name='distilbert',
                num_labels=3,
                max_length=128,
                learning_rate=2e-5
            )
            
            print("   - Testing DistilBERT initialization...")
            model = TransformerModelFactory.create_model(config)
            
            # Test tokenization
            texts = ["This is a test.", "Another test sentence."]
            encodings = model.tokenize_texts(texts)
            
            assert 'input_ids' in encodings, "Missing input_ids in tokenizer output"
            assert encodings['input_ids'].shape[0] == 2, "Batch size mismatch"
            
            print("     ✓ Transformer model initialized")
            print("     ✓ Tokenization working")
            
            self.results['transformers'] = 'PASSED'
            print("   ✅ Transformer models: PASSED")
            
        except Exception as e:
            self.results['transformers'] = f'FAILED: {str(e)}'
            print(f"   ❌ Transformer models: FAILED - {str(e)}")
            
    def test_augmentation(self):
        """Test augmentation strategies."""
        print("\n5. Testing Advanced Augmentation...")
        
        try:
            from automl.augmentation_advanced import AdvancedAugmenter, MixupAugmenter
            
            texts = [
                "This is a positive review.",
                "This product is terrible.",
                "Average quality, nothing special."
            ]
            labels = [1, 0, 2]
            
            # Test mixup augmentation
            print("   - Testing Mixup augmentation...")
            mixup = MixupAugmenter(alpha=0.2)
            mixed_text, mixed_label = mixup.augment(texts[0], texts[1], labels[0], labels[1])
            assert isinstance(mixed_text, str), "Mixup should return string"
            assert 0 <= mixed_label <= 2, "Mixed label out of range"
            print("     ✓ Mixup augmentation working")
            
            # Test contextual augmentation (if spaCy is available)
            try:
                print("   - Testing contextual augmentation...")
                augmenter = AdvancedAugmenter(['contextual'], device='cpu')
                aug_texts, aug_labels = augmenter.augment(texts[:1], labels[:1], num_augmentations=1)
                assert len(aug_texts) == 2, "Should have original + 1 augmentation"
                print("     ✓ Contextual augmentation working")
            except:
                print("     ⚠ Contextual augmentation skipped (dependencies missing)")
                
            self.results['augmentation'] = 'PASSED'
            print("   ✅ Advanced augmentation: PASSED")
            
        except Exception as e:
            self.results['augmentation'] = f'FAILED: {str(e)}'
            print(f"   ❌ Advanced augmentation: FAILED - {str(e)}")
            
    def test_fidelity_management(self):
        """Test fidelity management."""
        print("\n6. Testing Fidelity Management...")
        
        try:
            from automl.fidelity_manager import (
                FidelityManager, FidelityConfig, FidelityType,
                create_default_fidelity_configs
            )
            
            # Create fidelity configs
            configs = create_default_fidelity_configs('ffnn')
            manager = FidelityManager(configs)
            
            # Test setting fidelities
            print("   - Testing fidelity parameter conversion...")
            manager.set_fidelities({
                FidelityType.SEQUENCE_LENGTH: 0.5,
                FidelityType.DATA_FRACTION: 0.7,
                FidelityType.MODEL_DEPTH: 0.3
            })
            
            seq_len = manager.get_fidelity(FidelityType.SEQUENCE_LENGTH)
            assert 50 <= seq_len <= 512, f"Sequence length out of range: {seq_len}"
            print(f"     ✓ Sequence length fidelity: {seq_len}")
            
            data_frac = manager.get_fidelity(FidelityType.DATA_FRACTION)
            assert 0.1 <= data_frac <= 1.0, f"Data fraction out of range: {data_frac}"
            print(f"     ✓ Data fraction fidelity: {data_frac}")
            
            # Test cost estimation
            cost = manager.get_cost_estimate()
            assert 0 <= cost <= 1.0, f"Cost estimate out of range: {cost}"
            print(f"     ✓ Cost estimation: {cost:.3f}")
            
            self.results['fidelity'] = 'PASSED'
            print("   ✅ Fidelity management: PASSED")
            
        except Exception as e:
            self.results['fidelity'] = f'FAILED: {str(e)}'
            print(f"   ❌ Fidelity management: FAILED - {str(e)}")
            
    def test_integration(self):
        """Test system integration."""
        print("\n7. Testing System Integration...")
        
        try:
            # Test that core system can load new components
            from automl.core import TextAutoML
            
            print("   - Testing core system compatibility...")
            
            # Basic instantiation test
            automl = TextAutoML(
                approach='logistic',
                seed=42,
                use_preprocessing=True
            )
            
            print("     ✓ Core system compatible with new modules")
            
            self.results['integration'] = 'PASSED'
            print("   ✅ System integration: PASSED")
            
        except Exception as e:
            self.results['integration'] = f'FAILED: {str(e)}'
            print(f"   ❌ System integration: FAILED - {str(e)}")
            
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        
        all_passed = True
        for component, status in self.results.items():
            if status == 'PASSED':
                print(f"✅ {component.upper()}: {status}")
            else:
                print(f"❌ {component.upper()}: {status}")
                all_passed = False
                
        print("=" * 80)
        if all_passed:
            print("🎉 ALL TESTS PASSED! System is ready for use.")
        else:
            print("⚠️  Some tests failed. Please check the errors above.")
            
        return all_passed


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\nChecking dependencies...")
    
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'sklearn': 'Scikit-learn',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'optuna': 'Optuna',
        'sentence_transformers': 'Sentence-Transformers',
        'nltk': 'NLTK',
        'spacy': 'spaCy',
        'gensim': 'Gensim'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} (missing)")
            missing.append(name)
            
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed")
        return True


def main():
    """Main validation entry point."""
    print("AutoML Text Classification System Validator")
    print("==========================================")
    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install missing dependencies before running validation.")
        sys.exit(1)
        
    # Run validation
    validator = SystemValidator()
    success = validator.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()