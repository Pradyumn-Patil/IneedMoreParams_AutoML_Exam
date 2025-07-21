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