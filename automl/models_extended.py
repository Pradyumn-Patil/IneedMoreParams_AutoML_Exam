"""
Extended Neural Network Models

Includes:
1. GRU and BiGRU networks
2. Hybrid models combining traditional ML with deep learning
3. Hierarchical attention networks
4. Multi-scale CNN models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class GRUClassifier(nn.Module):
    """GRU-based text classifier with optional bidirectional and attention."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_attention: bool = True,
        use_layer_norm: bool = True,
        freeze_embeddings: bool = False
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_layer_norm = use_layer_norm
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
            
        # GRU layers
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Determine output dimension
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Optional layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(gru_output_dim)
            
        # Attention mechanism
        if use_attention:
            self.attention = SelfAttention(gru_output_dim)
            
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_output_dim, output_dim)
        
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        embedded = self.embedding(input_ids)
        embedded = self.embed_dropout(embedded)
        
        # Pack sequences for efficiency (if using attention mask)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            gru_out, hidden = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        else:
            gru_out, hidden = self.gru(embedded)
            
        # Apply layer norm
        if self.use_layer_norm:
            gru_out = self.layer_norm(gru_out)
            
        # Pooling
        if self.use_attention:
            pooled, _ = self.attention(gru_out, attention_mask)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward last hidden states
                pooled = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                pooled = hidden[-1]
                
        # Output
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        
        return logits


class SelfAttention(nn.Module):
    """Self-attention mechanism for sequence pooling."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, hidden_states, attention_mask=None):
        # Calculate attention scores
        scores = self.attention(hidden_states).squeeze(-1)
        
        # Apply mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            
        # Calculate attention weights
        weights = F.softmax(scores, dim=1)
        
        # Weighted sum
        attended = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        
        return attended, weights


class HierarchicalAttentionNetwork(nn.Module):
    """Hierarchical Attention Network for document classification."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        word_hidden_dim: int,
        sentence_hidden_dim: int,
        output_dim: int,
        max_words_per_sentence: int = 50,
        max_sentences: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_words = max_words_per_sentence
        self.max_sentences = max_sentences
        
        # Word-level encoder
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_gru = nn.GRU(
            embedding_dim, word_hidden_dim, 
            bidirectional=True, batch_first=True
        )
        self.word_attention = HierarchicalAttention(word_hidden_dim * 2)
        
        # Sentence-level encoder
        self.sentence_gru = nn.GRU(
            word_hidden_dim * 2, sentence_hidden_dim,
            bidirectional=True, batch_first=True
        )
        self.sentence_attention = HierarchicalAttention(sentence_hidden_dim * 2)
        
        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(sentence_hidden_dim * 2, output_dim)
        
    def forward(self, input_ids, word_attention_mask=None, sentence_attention_mask=None):
        batch_size = input_ids.size(0)
        
        # Reshape for word-level processing
        words = input_ids.view(-1, self.max_words)  # [batch*sentences, words]
        
        # Word-level encoding
        word_embeds = self.word_embedding(words)
        word_output, _ = self.word_gru(word_embeds)
        word_attn_out, _ = self.word_attention(word_output, word_attention_mask)
        
        # Reshape for sentence-level processing
        sentence_embeds = word_attn_out.view(batch_size, self.max_sentences, -1)
        
        # Sentence-level encoding
        sentence_output, _ = self.sentence_gru(sentence_embeds)
        doc_embed, _ = self.sentence_attention(sentence_output, sentence_attention_mask)
        
        # Classification
        doc_embed = self.dropout(doc_embed)
        logits = self.fc(doc_embed)
        
        return logits


class HierarchicalAttention(nn.Module):
    """Attention mechanism for hierarchical models."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, hidden_states, mask=None):
        scores = self.attention(hidden_states).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        weights = F.softmax(scores, dim=1)
        attended = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        
        return attended, weights


class MultiScaleCNN(nn.Module):
    """Multi-scale CNN for text classification with different kernel sizes."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        filter_sizes: List[int],
        output_dim: int,
        dropout: float = 0.5,
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.use_batch_norm = use_batch_norm
        
        # Multiple convolution layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(num_filters) for _ in filter_sizes
            ])
            
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_dim)
        
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        embedded = self.embedding(input_ids)
        embedded = embedded.transpose(1, 2)  # [batch, embed_dim, seq_len]
        
        # Apply convolutions
        conv_outputs = []
        for i, conv in enumerate(self.convs):
            conv_out = conv(embedded)
            if self.use_batch_norm:
                conv_out = self.batch_norms[i](conv_out)
            conv_out = F.relu(conv_out)
            
            # Max pooling over time
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
            
        # Concatenate all features
        features = torch.cat(conv_outputs, dim=1)
        features = self.dropout(features)
        
        # Classification
        logits = self.fc(features)
        return logits


class HybridDeepMLModel(nn.Module):
    """
    Hybrid model combining deep features with traditional ML.
    Extracts features using deep models and classifies with traditional ML.
    """
    
    def __init__(
        self,
        deep_model: nn.Module,
        feature_dim: int,
        output_dim: int,
        ml_classifier_type: str = 'svm'
    ):
        super().__init__()
        self.deep_model = deep_model
        self.feature_dim = feature_dim
        self.ml_classifier_type = ml_classifier_type
        
        # Feature extraction layer (replaces classification layer)
        self.feature_extractor = nn.Linear(
            self._get_deep_model_output_dim(), 
            feature_dim
        )
        
        # Will be set during training
        self.ml_classifier = None
        
    def _get_deep_model_output_dim(self):
        """Get output dimension of deep model before classification layer."""
        # This is model-specific - simplified version
        if hasattr(self.deep_model, 'fc'):
            return self.deep_model.fc.in_features
        elif hasattr(self.deep_model, 'classifier'):
            return self.deep_model.classifier.in_features
        else:
            return 768  # Default transformer hidden size
            
    def extract_features(self, input_ids, attention_mask=None):
        """Extract features using deep model."""
        # Get deep model representations
        if hasattr(self.deep_model, 'fc'):
            # Remove final classification layer temporarily
            original_fc = self.deep_model.fc
            self.deep_model.fc = nn.Identity()
            
        with torch.no_grad():
            deep_features = self.deep_model(input_ids, attention_mask)
            
        if hasattr(self.deep_model, 'fc'):
            self.deep_model.fc = original_fc
            
        # Project to desired feature dimension
        features = self.feature_extractor(deep_features)
        return features.cpu().numpy()
        
    def fit_ml_classifier(self, features: np.ndarray, labels: np.ndarray):
        """Fit the traditional ML classifier on extracted features."""
        from .traditional_ml import TraditionalMLFactory
        
        self.ml_classifier = TraditionalMLFactory.create_model(
            self.ml_classifier_type,
            random_state=42
        )
        self.ml_classifier.fit(features, labels)
        
    def forward(self, input_ids, attention_mask=None):
        """Forward pass - extract features only (ML classifier used separately)."""
        return self.extract_features(input_ids, attention_mask)


class TextCNNLSTM(nn.Module):
    """Combined CNN-LSTM model for capturing both local and sequential patterns."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        filter_sizes: List[int],
        lstm_hidden_dim: int,
        output_dim: int,
        dropout: float = 0.5
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # LSTM layer
        cnn_output_dim = num_filters * len(filter_sizes)
        self.lstm = nn.LSTM(
            cnn_output_dim, lstm_hidden_dim,
            batch_first=True, bidirectional=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_dim * 2, output_dim)
        
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        embedded = self.embedding(input_ids)
        batch_size, seq_len, _ = embedded.shape
        
        # CNN feature extraction
        embedded_t = embedded.transpose(1, 2)
        
        conv_features = []
        for conv in self.convs:
            # Apply convolution
            conv_out = F.relu(conv(embedded_t))  # [batch, num_filters, new_len]
            conv_features.append(conv_out.transpose(1, 2))  # [batch, new_len, num_filters]
            
        # Concatenate CNN features
        cnn_out = torch.cat(conv_features, dim=2)  # [batch, seq_len, total_filters]
        
        # LSTM processing
        lstm_out, (hidden, _) = self.lstm(cnn_out)
        
        # Use last hidden state
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Classification
        output = self.dropout(final_hidden)
        logits = self.fc(output)
        
        return logits


class WideAndDeepText(nn.Module):
    """Wide & Deep model for text: combines memorization and generalization."""
    
    def __init__(
        self,
        wide_input_dim: int,  # For TF-IDF or n-gram features
        deep_vocab_size: int,
        deep_embedding_dim: int,
        deep_hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Wide component (linear model)
        self.wide = nn.Linear(wide_input_dim, output_dim)
        
        # Deep component
        self.embedding = nn.Embedding(deep_vocab_size, deep_embedding_dim, padding_idx=0)
        
        # Deep layers
        layers = []
        input_dim = deep_embedding_dim
        for hidden_dim in deep_hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, output_dim))
        self.deep = nn.Sequential(*layers)
        
    def forward(self, wide_features, deep_input_ids):
        # Wide path
        wide_out = self.wide(wide_features)
        
        # Deep path
        embedded = self.embedding(deep_input_ids)
        # Average pooling over sequence
        embedded = embedded.mean(dim=1)
        deep_out = self.deep(embedded)
        
        # Combine
        logits = wide_out + deep_out
        return logits