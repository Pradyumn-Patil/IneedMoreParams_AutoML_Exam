"""
Meta-Learning Module for Dataset Similarity Analysis

Provides functionality to:
1. Extract meta-features from datasets
2. Analyze dataset similarity
3. Transfer configurations from similar datasets
4. Warm-start HPO with meta-learned priors
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class MetaFeatureExtractor:
    """Extract meta-features from text datasets for similarity analysis."""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """
        Extract comprehensive meta-features from a text dataset.
        
        Args:
            texts: List of text samples
            labels: List of corresponding labels
            
        Returns:
            Dictionary of meta-features
        """
        features = {}
        
        # Basic statistics
        features['n_samples'] = len(texts)
        features['n_classes'] = len(set(labels))
        
        # Text length statistics
        text_lengths = [len(text.split()) for text in texts]
        features['avg_text_length'] = np.mean(text_lengths)
        features['std_text_length'] = np.std(text_lengths)
        features['min_text_length'] = np.min(text_lengths)
        features['max_text_length'] = np.max(text_lengths)
        features['median_text_length'] = np.median(text_lengths)
        
        # Character-level statistics
        char_lengths = [len(text) for text in texts]
        features['avg_char_length'] = np.mean(char_lengths)
        features['std_char_length'] = np.std(char_lengths)
        
        # Vocabulary statistics
        all_words = ' '.join(texts).lower().split()
        word_freq = Counter(all_words)
        features['vocabulary_size'] = len(word_freq)
        features['vocabulary_ratio'] = len(word_freq) / len(all_words) if all_words else 0
        
        # Word frequency statistics
        frequencies = list(word_freq.values())
        features['avg_word_frequency'] = np.mean(frequencies)
        features['max_word_frequency'] = np.max(frequencies)
        features['word_frequency_skew'] = self._calculate_skewness(frequencies)
        
        # Class balance statistics
        label_counts = Counter(labels)
        class_frequencies = list(label_counts.values())
        features['class_imbalance_ratio'] = max(class_frequencies) / min(class_frequencies)
        features['class_entropy'] = self._calculate_entropy(class_frequencies)
        
        # Text complexity features
        features['avg_unique_words_ratio'] = np.mean([
            len(set(text.lower().split())) / len(text.split()) 
            for text in texts[:1000]  # Sample for efficiency
        ])
        
        # Special character statistics
        features['avg_special_char_ratio'] = np.mean([
            sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
            for text in texts[:1000] if len(text) > 0
        ])
        
        # Numeric content
        features['contains_numbers_ratio'] = np.mean([
            any(c.isdigit() for c in text) for text in texts
        ])
        
        # Store feature names for later reference
        self.feature_names = list(features.keys())
        
        return features
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of a distribution."""
        if len(values) < 3:
            return 0.0
        
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
            
        return np.mean(((values - mean) / std) ** 3)
    
    def _calculate_entropy(self, frequencies: List[int]) -> float:
        """Calculate entropy of a distribution."""
        total = sum(frequencies)
        if total == 0:
            return 0.0
            
        probs = [f / total for f in frequencies]
        return -sum(p * np.log2(p) if p > 0 else 0 for p in probs)


class DatasetSimilarityAnalyzer:
    """Analyze similarity between datasets based on meta-features."""
    
    def __init__(self):
        self.extractor = MetaFeatureExtractor()
        self.scaler = StandardScaler()
        self.known_datasets = {}
        self.load_known_datasets()
        
    def load_known_datasets(self):
        """Load meta-features for known datasets."""
        # Pre-computed meta-features for reference datasets
        self.known_datasets = {
            'amazon': {
                'n_samples': 25000,
                'n_classes': 3,
                'avg_text_length': 85,
                'vocabulary_size': 45000,
                'class_imbalance_ratio': 8.0,
                'best_approach': 'logistic',
                'best_config': {
                    'vocab_size': 15000,
                    'logistic_C': 1.0,
                    'use_augmentation': True
                }
            },
            'imdb': {
                'n_samples': 25000,
                'n_classes': 2,
                'avg_text_length': 230,
                'vocabulary_size': 80000,
                'class_imbalance_ratio': 1.0,
                'best_approach': 'logistic',
                'best_config': {
                    'vocab_size': 20000,
                    'logistic_C': 0.5,
                    'use_augmentation': False
                }
            },
            'ag_news': {
                'n_samples': 120000,
                'n_classes': 4,
                'avg_text_length': 40,
                'vocabulary_size': 60000,
                'class_imbalance_ratio': 1.1,
                'best_approach': 'ffnn',
                'best_config': {
                    'vocab_size': 20000,
                    'ffnn_hidden': 256,
                    'epochs': 10,
                    'use_enhanced_ffnn': True
                }
            },
            'dbpedia': {
                'n_samples': 560000,
                'n_classes': 14,
                'avg_text_length': 55,
                'vocabulary_size': 150000,
                'class_imbalance_ratio': 1.0,
                'best_approach': 'logistic',
                'best_config': {
                    'vocab_size': 30000,
                    'logistic_C': 0.1,
                    'use_augmentation': False
                }
            }
        }
        
    def find_similar_datasets(
        self, 
        target_features: Dict[str, float], 
        k: int = 3
    ) -> List[Tuple[str, float, Dict]]:
        """
        Find k most similar datasets based on meta-features.
        
        Args:
            target_features: Meta-features of target dataset
            k: Number of similar datasets to return
            
        Returns:
            List of (dataset_name, similarity_score, best_config) tuples
        """
        if not self.known_datasets:
            logger.warning("No known datasets available for comparison")
            return []
            
        # Extract relevant features for comparison
        feature_keys = ['n_samples', 'n_classes', 'avg_text_length', 
                       'vocabulary_size', 'class_imbalance_ratio']
        
        # Prepare feature vectors
        target_vector = []
        dataset_vectors = {}
        
        for key in feature_keys:
            if key in target_features:
                target_vector.append(target_features[key])
                
        for name, meta in self.known_datasets.items():
            vector = []
            for key in feature_keys:
                if key in meta:
                    vector.append(meta[key])
            if len(vector) == len(target_vector):
                dataset_vectors[name] = vector
                
        if not dataset_vectors:
            return []
            
        # Normalize features
        all_vectors = [target_vector] + list(dataset_vectors.values())
        all_vectors = self.scaler.fit_transform(all_vectors)
        
        target_norm = all_vectors[0:1]
        dataset_norms = {
            name: all_vectors[i+1:i+2] 
            for i, name in enumerate(dataset_vectors.keys())
        }
        
        # Calculate similarities
        similarities = []
        for name, vector in dataset_norms.items():
            sim = cosine_similarity(target_norm, vector)[0, 0]
            config = self.known_datasets[name].get('best_config', {})
            similarities.append((name, sim, config))
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def get_warm_start_config(
        self, 
        target_features: Dict[str, float],
        approach: str = None
    ) -> Dict:
        """
        Get warm-start configuration based on similar datasets.
        
        Args:
            target_features: Meta-features of target dataset
            approach: Optional specific approach to use
            
        Returns:
            Suggested configuration dictionary
        """
        similar_datasets = self.find_similar_datasets(target_features, k=3)
        
        if not similar_datasets:
            logger.warning("No similar datasets found, using default config")
            return {}
            
        # Weighted average of configurations from similar datasets
        config = {}
        total_weight = 0
        
        for name, similarity, dataset_config in similar_datasets:
            if similarity > 0.7:  # Only use highly similar datasets
                weight = similarity
                total_weight += weight
                
                logger.info(f"Using config from {name} (similarity: {similarity:.3f})")
                
                # Merge configurations with weighting
                for key, value in dataset_config.items():
                    if isinstance(value, (int, float)):
                        if key not in config:
                            config[key] = 0
                        config[key] += value * weight
                    elif key not in config:
                        config[key] = value
                        
        # Normalize weighted values
        if total_weight > 0:
            for key, value in config.items():
                if isinstance(value, (int, float)):
                    config[key] = type(value)(config[key] / total_weight)
                    
        # Override approach if specified
        if approach:
            config['approach'] = approach
        elif similar_datasets:
            # Use most similar dataset's approach
            best_dataset = similar_datasets[0][0]
            config['approach'] = self.known_datasets[best_dataset].get('best_approach', 'logistic')
            
        return config


class MetaLearningOptimizer:
    """Meta-learning optimizer for AutoML pipeline."""
    
    def __init__(self, history_path: Path = None):
        self.history_path = history_path or Path("meta_learning_history")
        self.history_path.mkdir(exist_ok=True)
        self.analyzer = DatasetSimilarityAnalyzer()
        self.extractor = MetaFeatureExtractor()
        
    def analyze_new_dataset(
        self, 
        texts: List[str], 
        labels: List[int],
        dataset_name: str = None
    ) -> Dict:
        """
        Analyze a new dataset and provide recommendations.
        
        Args:
            texts: Text samples
            labels: Labels
            dataset_name: Optional dataset name
            
        Returns:
            Dictionary with analysis results and recommendations
        """
        logger.info("Extracting meta-features from dataset...")
        
        # Extract meta-features
        features = self.extractor.extract_features(texts, labels)
        
        # Find similar datasets
        similar_datasets = self.analyzer.find_similar_datasets(features)
        
        # Get warm-start configuration
        warm_start_config = self.analyzer.get_warm_start_config(features)
        
        # Prepare recommendations
        recommendations = {
            'meta_features': features,
            'similar_datasets': [
                {'name': name, 'similarity': sim} 
                for name, sim, _ in similar_datasets
            ],
            'suggested_approach': warm_start_config.get('approach', 'logistic'),
            'suggested_config': warm_start_config,
            'analysis': self._generate_analysis(features, similar_datasets)
        }
        
        # Save to history
        if dataset_name:
            self.save_analysis(dataset_name, recommendations)
            
        return recommendations
    
    def _generate_analysis(
        self, 
        features: Dict[str, float], 
        similar_datasets: List[Tuple[str, float, Dict]]
    ) -> Dict[str, str]:
        """Generate human-readable analysis of the dataset."""
        analysis = {}
        
        # Dataset size analysis
        n_samples = features.get('n_samples', 0)
        if n_samples < 5000:
            analysis['size'] = "Small dataset - can afford complex models"
        elif n_samples < 50000:
            analysis['size'] = "Medium dataset - balance complexity and efficiency"
        else:
            analysis['size'] = "Large dataset - prioritize efficiency"
            
        # Class balance analysis
        imbalance_ratio = features.get('class_imbalance_ratio', 1.0)
        if imbalance_ratio > 3:
            analysis['balance'] = "Highly imbalanced - use class weights and augmentation"
        elif imbalance_ratio > 1.5:
            analysis['balance'] = "Moderately imbalanced - consider class weights"
        else:
            analysis['balance'] = "Well balanced - standard training should work"
            
        # Text length analysis
        avg_length = features.get('avg_text_length', 0)
        if avg_length < 50:
            analysis['text_length'] = "Short texts - consider character n-grams"
        elif avg_length < 200:
            analysis['text_length'] = "Medium texts - standard approaches work well"
        else:
            analysis['text_length'] = "Long texts - consider efficient models or truncation"
            
        # Similarity analysis
        if similar_datasets and similar_datasets[0][1] > 0.8:
            analysis['similarity'] = f"Very similar to {similar_datasets[0][0]} - use proven configurations"
        elif similar_datasets and similar_datasets[0][1] > 0.6:
            analysis['similarity'] = f"Somewhat similar to {similar_datasets[0][0]} - adapt configurations"
        else:
            analysis['similarity'] = "Unique dataset - extensive experimentation recommended"
            
        return analysis
    
    def save_analysis(self, dataset_name: str, analysis: Dict):
        """Save analysis results for future reference."""
        output_path = self.history_path / f"{dataset_name}_analysis.yaml"
        
        with open(output_path, 'w') as f:
            yaml.dump(analysis, f, default_flow_style=False)
            
        logger.info(f"Analysis saved to {output_path}")
    
    def load_analysis(self, dataset_name: str) -> Optional[Dict]:
        """Load previous analysis for a dataset."""
        input_path = self.history_path / f"{dataset_name}_analysis.yaml"
        
        if input_path.exists():
            with open(input_path) as f:
                return yaml.safe_load(f)
        return None


def create_meta_learning_optimizer() -> MetaLearningOptimizer:
    """Create and return a meta-learning optimizer instance."""
    return MetaLearningOptimizer()