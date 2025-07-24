"""
Advanced Embeddings Module for Text Representation

Supports multiple embedding strategies:
1. Pre-trained word embeddings (GloVe, FastText, Word2Vec)
2. Contextual embeddings (Sentence-BERT, Universal Sentence Encoder)
3. Custom embeddings with various aggregation methods
4. Hybrid embeddings combining multiple representations
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union, Tuple
import logging
from pathlib import Path
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages various embedding strategies for text representation."""
    
    def __init__(self, cache_dir: Path = Path("~/.cache/embeddings").expanduser()):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_embeddings = {}
        
    def get_embedding(self, embedding_type: str, **kwargs) -> 'BaseEmbedding':
        """Factory method to get appropriate embedding instance."""
        if embedding_type == 'glove':
            return GloVeEmbedding(self.cache_dir, **kwargs)
        elif embedding_type == 'fasttext':
            return FastTextEmbedding(self.cache_dir, **kwargs)
        elif embedding_type == 'word2vec':
            return Word2VecEmbedding(self.cache_dir, **kwargs)
        elif embedding_type == 'sentence_bert':
            return SentenceBERTEmbedding(**kwargs)
        elif embedding_type == 'tfidf':
            return TFIDFEmbedding(**kwargs)
        elif embedding_type == 'hybrid':
            return HybridEmbedding(self.cache_dir, **kwargs)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")


class BaseEmbedding:
    """Base class for all embedding strategies."""
    
    def __init__(self, dim: int):
        self.dim = dim
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        raise NotImplementedError
        
    def encode_words(self, words: List[str]) -> np.ndarray:
        """Encode individual words (for word-level embeddings)."""
        raise NotImplementedError


class GloVeEmbedding(BaseEmbedding):
    """GloVe pre-trained embeddings with various aggregation methods."""
    
    def __init__(self, cache_dir: Path, dim: int = 100, aggregation: str = 'mean'):
        super().__init__(dim)
        self.cache_dir = cache_dir
        self.aggregation = aggregation
        self.model_name = f'glove-wiki-gigaword-{dim}'
        self._load_model()
        
    def _load_model(self):
        """Load GloVe model, downloading if necessary."""
        cache_file = self.cache_dir / f"{self.model_name}.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached GloVe model from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.model = pickle.load(f)
        else:
            logger.info(f"Downloading GloVe model: {self.model_name}")
            self.model = api.load(self.model_name)
            # Cache for faster loading next time
            with open(cache_file, 'wb') as f:
                pickle.dump(self.model, f)
                
    def encode_words(self, words: List[str]) -> np.ndarray:
        """Encode individual words."""
        embeddings = []
        for word in words:
            if word in self.model:
                embeddings.append(self.model[word])
            else:
                # Random embedding for OOV words
                embeddings.append(np.random.randn(self.dim) * 0.1)
        return np.array(embeddings)
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using specified aggregation method."""
        embeddings = []
        
        for text in texts:
            words = text.lower().split()
            word_embeddings = []
            
            for word in words:
                if word in self.model:
                    word_embeddings.append(self.model[word])
                    
            if not word_embeddings:
                # If no words found, use zero vector
                embeddings.append(np.zeros(self.dim))
            else:
                word_embeddings = np.array(word_embeddings)
                
                if self.aggregation == 'mean':
                    embeddings.append(np.mean(word_embeddings, axis=0))
                elif self.aggregation == 'max':
                    embeddings.append(np.max(word_embeddings, axis=0))
                elif self.aggregation == 'sum':
                    embeddings.append(np.sum(word_embeddings, axis=0))
                elif self.aggregation == 'concat_mean_max':
                    # Concatenate mean and max pooling
                    mean_emb = np.mean(word_embeddings, axis=0)
                    max_emb = np.max(word_embeddings, axis=0)
                    embeddings.append(np.concatenate([mean_emb, max_emb]))
                else:
                    raise ValueError(f"Unknown aggregation method: {self.aggregation}")
                    
        return np.array(embeddings)


class FastTextEmbedding(BaseEmbedding):
    """FastText embeddings with subword information."""
    
    def __init__(self, cache_dir: Path, dim: int = 300):
        super().__init__(dim)
        self.cache_dir = cache_dir
        self.model_name = f'fasttext-wiki-news-subwords-{dim}'
        self._load_model()
        
    def _load_model(self):
        """Load FastText model."""
        cache_file = self.cache_dir / f"{self.model_name}.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached FastText model from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.model = pickle.load(f)
        else:
            logger.info(f"Downloading FastText model: {self.model_name}")
            self.model = api.load(self.model_name)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.model, f)
                
    def encode_words(self, words: List[str]) -> np.ndarray:
        """Encode individual words."""
        # FastText can handle OOV words through subword information
        return np.array([self.model[word] for word in words])
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using mean pooling."""
        embeddings = []
        
        for text in texts:
            words = text.lower().split()
            if words:
                # FastText handles OOV words automatically
                word_embeddings = [self.model[word] for word in words]
                embeddings.append(np.mean(word_embeddings, axis=0))
            else:
                embeddings.append(np.zeros(self.dim))
                
        return np.array(embeddings)


class Word2VecEmbedding(BaseEmbedding):
    """Word2Vec embeddings."""
    
    def __init__(self, cache_dir: Path, dim: int = 300):
        super().__init__(dim)
        self.cache_dir = cache_dir
        self.model_name = 'word2vec-google-news-300'
        self._load_model()
        
    def _load_model(self):
        """Load Word2Vec model."""
        cache_file = self.cache_dir / f"{self.model_name}.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached Word2Vec model from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.model = pickle.load(f)
        else:
            logger.info(f"Downloading Word2Vec model: {self.model_name}")
            self.model = api.load(self.model_name)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.model, f)
                
    def encode_words(self, words: List[str]) -> np.ndarray:
        """Encode individual words."""
        embeddings = []
        for word in words:
            if word in self.model:
                embeddings.append(self.model[word])
            else:
                embeddings.append(np.random.randn(self.dim) * 0.1)
        return np.array(embeddings)
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using mean pooling."""
        embeddings = []
        
        for text in texts:
            words = text.lower().split()
            word_embeddings = []
            
            for word in words:
                if word in self.model:
                    word_embeddings.append(self.model[word])
                    
            if word_embeddings:
                embeddings.append(np.mean(word_embeddings, axis=0))
            else:
                embeddings.append(np.zeros(self.dim))
                
        return np.array(embeddings)


class SentenceBERTEmbedding(BaseEmbedding):
    """Sentence-BERT for contextual sentence embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Get dimension from model
        self.model = SentenceTransformer(model_name)
        super().__init__(self.model.get_sentence_embedding_dimension())
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into sentence embeddings."""
        return self.model.encode(texts, show_progress_bar=True)
        
    def encode_words(self, words: List[str]) -> np.ndarray:
        """Encode individual words as sentences."""
        return self.encode(words)


class TFIDFEmbedding(BaseEmbedding):
    """TF-IDF embeddings with dimensionality reduction."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 3)):
        super().__init__(max_features)
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        
    def fit(self, texts: List[str]):
        """Fit TF-IDF vectorizer on texts."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            min_df=2
        )
        self.vectorizer.fit(texts)
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using TF-IDF."""
        if self.vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts).toarray()


class HybridEmbedding(BaseEmbedding):
    """Combines multiple embedding strategies."""
    
    def __init__(self, cache_dir: Path, embedding_types: List[Dict[str, any]]):
        self.embeddings = []
        total_dim = 0
        
        manager = EmbeddingManager(cache_dir)
        for emb_config in embedding_types:
            emb_type = emb_config.pop('type')
            embedding = manager.get_embedding(emb_type, **emb_config)
            self.embeddings.append(embedding)
            total_dim += embedding.dim
            
        super().__init__(total_dim)
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using all embedding strategies and concatenate."""
        all_embeddings = []
        
        for embedding in self.embeddings:
            if hasattr(embedding, 'fit') and not hasattr(embedding, 'vectorizer'):
                # Fit TF-IDF if needed
                embedding.fit(texts)
            emb = embedding.encode(texts)
            all_embeddings.append(emb)
            
        return np.concatenate(all_embeddings, axis=1)


class EmbeddingLayer(nn.Module):
    """PyTorch embedding layer that can use pre-trained embeddings."""
    
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int,
                 pretrained_embeddings: Optional[np.ndarray] = None,
                 freeze: bool = False,
                 padding_idx: int = 0):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            
        if freeze:
            self.embedding.weight.requires_grad = False
            
    def forward(self, input_ids):
        return self.embedding(input_ids)


def create_embedding_matrix(word_index: Dict[str, int], 
                           embedding: BaseEmbedding,
                           embedding_dim: int) -> np.ndarray:
    """Create embedding matrix for vocabulary."""
    vocab_size = len(word_index) + 1  # +1 for padding
    embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.1
    
    # Set padding to zeros
    embedding_matrix[0] = 0
    
    for word, idx in word_index.items():
        if hasattr(embedding, 'model') and word in embedding.model:
            embedding_matrix[idx] = embedding.model[word]
        elif hasattr(embedding, 'encode_words'):
            # For models that can handle OOV words
            word_emb = embedding.encode_words([word])[0]
            embedding_matrix[idx] = word_emb
            
    return embedding_matrix