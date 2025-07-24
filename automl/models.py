import numpy as np
import torch
import torch.nn as nn


class SimpleFFNN(nn.Module):
    def __init__(self, input_dim, hidden=128, output_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x): return self.model(x)


class ResidualBlock(nn.Module):
    """Residual block with layer normalization and dropout."""
    
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, x):
        return x + self.ff(x)


class EnhancedFFNN(nn.Module):
    """Enhanced FFNN with residual connections, layer normalization, and adaptive architecture."""
    
    def __init__(self, input_dim, hidden=128, output_dim=2, num_layers=3, dropout_rate=0.1, 
                 use_residual=True, use_layer_norm=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden)
        
        if use_layer_norm:
            self.input_norm = nn.LayerNorm(hidden)
        
        # Hidden layers
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            if use_residual and i > 0:  # Start residual connections from 2nd layer
                self.layers.append(ResidualBlock(hidden, dropout_rate))
            else:
                layer = nn.Sequential(
                    nn.LayerNorm(hidden) if use_layer_norm else nn.Identity(),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout_rate)
                )
                self.layers.append(layer)
        
        # Output layer
        self.output_norm = nn.LayerNorm(hidden) if use_layer_norm else nn.Identity()
        self.output_proj = nn.Linear(hidden, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        if self.use_layer_norm:
            x = self.input_norm(x)
        x = torch.relu(x)
        
        # Hidden layers
        for layer in self.layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x)
            else:
                x = layer(x)
        
        # Output
        x = self.output_norm(x)
        x = self.dropout(x)
        x = self.output_proj(x)
        
        return x


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, **kwargs):
        x = kwargs["input_ids"]
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        logits = self.fc(h[-1])
        return logits


class AttentionPooling(nn.Module):
    """Self-attention pooling layer for sequence representations."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output, attention_mask=None):
        # lstm_output: [batch_size, seq_len, hidden_dim]
        attention_scores = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        if attention_mask is not None:
            # Mask padding tokens
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
            
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Weighted sum of LSTM outputs
        pooled = torch.sum(lstm_output * attention_weights, dim=1)  # [batch_size, hidden_dim]
        
        return pooled, attention_weights.squeeze(-1)


class EnhancedLSTMClassifier(nn.Module):
    """Enhanced LSTM with attention pooling and layer normalization."""
    
    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        hidden_dim, 
        output_dim,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
        use_attention=True,
        use_layer_norm=True
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_layer_norm = use_layer_norm
        self.bidirectional = bidirectional
        
        # Embedding layer with dropout
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        
        # LSTM with multiple layers
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Determine output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Optional layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # Attention pooling
        if use_attention:
            self.attention_pool = AttentionPooling(lstm_output_dim)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs.get("attention_mask", None)
        
        # Embedding
        x = self.embed(input_ids)
        x = self.embed_dropout(x)
        
        # LSTM
        lstm_out, (h, c) = self.lstm(x)
        
        # Apply layer norm to LSTM output
        if self.use_layer_norm:
            lstm_out = self.layer_norm(lstm_out)
        
        # Pooling
        if self.use_attention:
            pooled, attention_weights = self.attention_pool(lstm_out, attention_mask)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward last hidden states
                pooled = torch.cat([h[-2], h[-1]], dim=1)
            else:
                pooled = h[-1]
        
        # Output
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        
        return logits


class CNNTextEncoder(nn.Module):
    """CNN encoder for text feature extraction."""
    
    def __init__(self, embedding_dim, num_filters, filter_sizes, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_filters * len(filter_sizes)
        
    def forward(self, embedded):
        # embedded: [batch_size, seq_len, embedding_dim]
        embedded = embedded.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # [batch_size, num_filters, new_seq_len]
            pooled = torch.max(conv_out, dim=2)[0]  # [batch_size, num_filters]
            conv_outputs.append(pooled)
            
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        
        return self.dropout(concatenated)


class CNNLSTMClassifier(nn.Module):
    """Hybrid CNN-LSTM model for text classification."""
    
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_filters=100,
        filter_sizes=[3, 4, 5],
        lstm_layers=1,
        dropout=0.1,
        use_attention=True
    ):
        super().__init__()
        self.use_attention = use_attention
        
        # Embedding
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        
        # CNN encoder
        self.cnn_encoder = CNNTextEncoder(embedding_dim, num_filters, filter_sizes, dropout)
        
        # LSTM on top of CNN features
        self.lstm = nn.LSTM(
            self.cnn_encoder.output_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = hidden_dim * 2  # bidirectional
        
        # Attention pooling
        if use_attention:
            self.attention_pool = AttentionPooling(lstm_output_dim)
        
        # Output
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs.get("attention_mask", None)
        
        # Embedding
        embedded = self.embed(input_ids)
        embedded = self.embed_dropout(embedded)
        
        # CNN encoding
        batch_size, seq_len, _ = embedded.shape
        cnn_features = []
        
        # Process each position with CNN
        for i in range(seq_len):
            if i + max(self.cnn_encoder.convs[0].kernel_size[0] for conv in self.cnn_encoder.convs) <= seq_len:
                window = embedded[:, i:i+max(5, seq_len-i), :]
                cnn_out = self.cnn_encoder(window)
                cnn_features.append(cnn_out)
        
        if len(cnn_features) == 0:
            # Fallback for very short sequences
            cnn_out = self.cnn_encoder(embedded)
            cnn_features = [cnn_out]
            
        # Stack CNN features
        cnn_features = torch.stack(cnn_features, dim=1)  # [batch_size, new_seq_len, cnn_output_dim]
        
        # LSTM processing
        lstm_out, (h, c) = self.lstm(cnn_features)
        
        # Pooling
        if self.use_attention and lstm_out.size(1) > 1:
            pooled, _ = self.attention_pool(lstm_out, None)  # No mask for CNN features
        else:
            # Use last hidden state
            pooled = torch.cat([h[-2], h[-1]], dim=1)
        
        # Output
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        
        return logits


class NASSearchSpace:
    """Search space definitions for Neural Architecture Search."""
    
    @staticmethod
    def generate_ffnn_architecture(trial, input_dim, output_dim):
        """Generate FFNN architecture configuration using Optuna trial."""
        config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'num_hidden_layers': trial.suggest_int('num_hidden_layers', 1, 4),
            'hidden_dims': [],
            'dropout_rates': [],
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid', 'gelu']),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False])
        }
        
        # Generate hidden layer dimensions
        for i in range(config['num_hidden_layers']):
            hidden_dim = trial.suggest_int(f'hidden_dim_{i}', 32, 512)
            dropout_rate = trial.suggest_float(f'dropout_rate_{i}', 0.0, 0.5)
            config['hidden_dims'].append(hidden_dim)
            config['dropout_rates'].append(dropout_rate)
            
        return config
    
    @staticmethod
    def generate_lstm_architecture(trial, vocab_size, output_dim):
        """Generate LSTM architecture configuration using Optuna trial."""
        config = {
            'vocab_size': vocab_size,
            'output_dim': output_dim,
            'embedding_dim': trial.suggest_int('embedding_dim', 64, 256),
            'hidden_dim': trial.suggest_int('hidden_dim', 64, 256),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
            'use_attention': trial.suggest_categorical('use_attention', [True, False])
        }
        
        return config


class NASSearchableFFNN(nn.Module):
    """FFNN model with searchable architecture."""
    
    def __init__(self, input_dim, output_dim, architecture_config):
        super().__init__()
        self.config = architecture_config
        
        layers = []
        current_dim = input_dim
        
        # Build hidden layers
        for i, (hidden_dim, dropout_rate) in enumerate(zip(
            architecture_config['hidden_dims'], 
            architecture_config['dropout_rates']
        )):
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            # Add batch normalization if enabled
            if architecture_config['use_batch_norm']:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Add activation function
            activation = architecture_config['activation']
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            
            # Add dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class NASSearchableLSTM(nn.Module):
    """LSTM model with searchable architecture."""
    
    def __init__(self, vocab_size, output_dim, architecture_config):
        super().__init__()
        self.config = architecture_config
        
        # Embedding layer
        self.embed = nn.Embedding(
            vocab_size, 
            architecture_config['embedding_dim'], 
            padding_idx=0
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            architecture_config['embedding_dim'],
            architecture_config['hidden_dim'],
            num_layers=architecture_config['num_layers'],
            dropout=architecture_config['dropout'] if architecture_config['num_layers'] > 1 else 0,
            bidirectional=architecture_config['bidirectional'],
            batch_first=True
        )
        
        # Determine LSTM output dimension
        lstm_output_dim = architecture_config['hidden_dim']
        if architecture_config['bidirectional']:
            lstm_output_dim *= 2
        
        # Attention mechanism (optional)
        self.use_attention = architecture_config['use_attention']
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                lstm_output_dim, 
                num_heads=4, 
                batch_first=True
            )
        
        # Output layer
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
        # Dropout
        if architecture_config['dropout'] > 0:
            self.dropout = nn.Dropout(architecture_config['dropout'])
        else:
            self.dropout = None
    
    def forward(self, **kwargs):
        x = kwargs["input_ids"]
        attention_mask = kwargs.get("attention_mask", None)
        
        # Embedding
        x = self.embed(x)
        
        # LSTM
        lstm_out, (h, c) = self.lstm(x)
        
        if self.use_attention:
            # Apply attention mechanism
            if attention_mask is not None:
                # Convert attention mask to key padding mask
                key_padding_mask = attention_mask == 0
            else:
                key_padding_mask = None
                
            attn_out, _ = self.attention(
                lstm_out, lstm_out, lstm_out,
                key_padding_mask=key_padding_mask
            )
            
            # Use mean pooling over sequence length
            if attention_mask is not None:
                # Mask out padding tokens before pooling
                attn_out = attn_out * attention_mask.unsqueeze(-1)
                output = attn_out.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                output = attn_out.mean(dim=1)
        else:
            # Use last hidden state
            if self.config['bidirectional']:
                # Concatenate forward and backward hidden states
                output = torch.cat([h[-2], h[-1]], dim=1)
            else:
                output = h[-1]
        
        # Apply dropout
        if self.dropout is not None:
            output = self.dropout(output)
        
        # Output layer
        logits = self.fc(output)
        
        return logits