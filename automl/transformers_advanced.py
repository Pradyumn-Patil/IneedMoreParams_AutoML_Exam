"""
Advanced Transformer Models for Text Classification

Implements proper fine-tuning of various pre-trained language models:
1. BERT, RoBERTa, ALBERT, DeBERTa, ELECTRA
2. DistilBERT, XLNet, Longformer
3. Domain-specific models (SciBERT, BioBERT, FinBERT)
4. Multi-lingual models (mBERT, XLM-RoBERTa)
5. Efficient fine-tuning strategies (adapters, LoRA, prefix tuning)
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    BertForSequenceClassification, RobertaForSequenceClassification,
    AlbertForSequenceClassification, DebertaForSequenceClassification,
    ElectraForSequenceClassification, XLNetForSequenceClassification,
    LongformerForSequenceClassification, DistilBertForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_linear_schedule_with_warmup, AdamW
)
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for transformer models."""
    model_name: str
    num_labels: int
    max_length: int = 512
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    freeze_layers: Optional[int] = None  # Number of layers to freeze
    use_adapters: bool = False
    adapter_size: int = 64
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    use_prefix_tuning: bool = False
    prefix_length: int = 10


class TransformerModelFactory:
    """Factory for creating transformer models."""
    
    # Model name mappings to HuggingFace model identifiers
    MODEL_MAPPINGS = {
        'bert': 'bert-base-uncased',
        'bert-large': 'bert-large-uncased',
        'roberta': 'roberta-base',
        'roberta-large': 'roberta-large',
        'albert': 'albert-base-v2',
        'albert-large': 'albert-large-v2',
        'deberta': 'microsoft/deberta-base',
        'deberta-large': 'microsoft/deberta-large',
        'electra': 'google/electra-base-discriminator',
        'xlnet': 'xlnet-base-cased',
        'distilbert': 'distilbert-base-uncased',
        'longformer': 'allenai/longformer-base-4096',
        'scibert': 'allenai/scibert_scivocab_uncased',
        'biobert': 'dmis-lab/biobert-base-cased-v1.1',
        'finbert': 'yiyanghkust/finbert-tone',
        'mbert': 'bert-base-multilingual-cased',
        'xlm-roberta': 'xlm-roberta-base',
        'xlm-roberta-large': 'xlm-roberta-large'
    }
    
    @classmethod
    def create_model(cls, config: TransformerConfig):
        """Create a transformer model based on configuration."""
        model_identifier = cls.MODEL_MAPPINGS.get(config.model_name, config.model_name)
        
        # Create base model
        if config.use_adapters:
            return AdapterTransformer(model_identifier, config)
        elif config.use_lora:
            return LoRATransformer(model_identifier, config)
        elif config.use_prefix_tuning:
            return PrefixTuningTransformer(model_identifier, config)
        else:
            return StandardTransformer(model_identifier, config)


class BaseTransformerModel(nn.Module):
    """Base class for transformer models."""
    
    def __init__(self, model_identifier: str, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.model_identifier = model_identifier
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_identifier,
            num_labels=config.num_labels,
            hidden_dropout_prob=config.dropout_rate,
            attention_probs_dropout_prob=config.dropout_rate
        )
        
        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
        # Apply configuration
        self._apply_configuration()
        
    def _apply_configuration(self):
        """Apply model-specific configuration."""
        # Freeze layers if specified
        if self.config.freeze_layers is not None:
            self._freeze_layers(self.config.freeze_layers)
            
    def _freeze_layers(self, num_layers: int):
        """Freeze the first n layers of the model."""
        # Get the encoder/transformer layers
        if hasattr(self.model, 'bert'):
            layers = self.model.bert.encoder.layer
        elif hasattr(self.model, 'roberta'):
            layers = self.model.roberta.encoder.layer
        elif hasattr(self.model, 'albert'):
            layers = self.model.albert.encoder.albert_layer_groups
        elif hasattr(self.model, 'deberta'):
            layers = self.model.deberta.encoder.layer
        elif hasattr(self.model, 'electra'):
            layers = self.model.electra.encoder.layer
        elif hasattr(self.model, 'distilbert'):
            layers = self.model.distilbert.transformer.layer
        else:
            logger.warning(f"Cannot freeze layers for model type: {type(self.model)}")
            return
            
        # Freeze embeddings
        if hasattr(self.model, 'embeddings'):
            for param in self.model.embeddings.parameters():
                param.requires_grad = False
                
        # Freeze specified layers
        for i in range(min(num_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False
                
        logger.info(f"Froze {num_layers} layers")
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
        
    def tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize texts for model input."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        )


class StandardTransformer(BaseTransformerModel):
    """Standard transformer fine-tuning."""
    pass


class AdapterTransformer(BaseTransformerModel):
    """Transformer with adapter layers for efficient fine-tuning."""
    
    def _apply_configuration(self):
        """Add adapter layers to the model."""
        super()._apply_configuration()
        
        # Add adapters to each transformer layer
        if hasattr(self.model, 'bert'):
            base_model = self.model.bert
        elif hasattr(self.model, 'roberta'):
            base_model = self.model.roberta
        elif hasattr(self.model, 'distilbert'):
            base_model = self.model.distilbert
        else:
            logger.warning("Adapter not supported for this model type")
            return
            
        # Add adapter modules
        for layer in base_model.encoder.layer:
            layer.adapter = AdapterLayer(
                input_size=layer.output.dense.in_features,
                adapter_size=self.config.adapter_size
            )
            
        # Freeze all parameters except adapters and classifier
        for name, param in self.model.named_parameters():
            if 'adapter' not in name and 'classifier' not in name:
                param.requires_grad = False
                
        logger.info(f"Added adapter layers with size {self.config.adapter_size}")


class AdapterLayer(nn.Module):
    """Adapter layer for efficient fine-tuning."""
    
    def __init__(self, input_size: int, adapter_size: int):
        super().__init__()
        self.down_project = nn.Linear(input_size, adapter_size)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_size, input_size)
        self.layer_norm = nn.LayerNorm(input_size)
        
    def forward(self, x):
        residual = x
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        x = self.layer_norm(x + residual)
        return x


class LoRATransformer(BaseTransformerModel):
    """Transformer with LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
    
    def _apply_configuration(self):
        """Apply LoRA to attention layers."""
        super()._apply_configuration()
        
        # Apply LoRA to query and value projections
        self._apply_lora_to_model()
        
    def _apply_lora_to_model(self):
        """Apply LoRA to attention layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(key in name for key in ['query', 'value']):
                # Replace with LoRA linear layer
                lora_layer = LoRALinear(
                    module.in_features,
                    module.out_features,
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha
                )
                # Copy original weights
                lora_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_layer.bias = module.bias
                    
                # Replace module
                parent_name, child_name = name.rsplit('.', 1)
                parent = self.model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, lora_layer)
                
        # Freeze original parameters
        for name, param in self.model.named_parameters():
            if 'lora' not in name and 'classifier' not in name:
                param.requires_grad = False
                
        logger.info(f"Applied LoRA with r={self.config.lora_r}, alpha={self.config.lora_alpha}")


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Original weight (frozen)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
    def forward(self, x):
        # Original linear transformation
        result = F.linear(x, self.weight)
        
        # Add LoRA adaptation
        lora_output = x @ self.lora_A.T @ self.lora_B.T
        result += self.scaling * lora_output
        
        return result


class PrefixTuningTransformer(BaseTransformerModel):
    """Transformer with prefix tuning for efficient fine-tuning."""
    
    def __init__(self, model_identifier: str, config: TransformerConfig):
        super().__init__(model_identifier, config)
        
        # Get hidden size from model config
        self.hidden_size = self.model.config.hidden_size
        
        # Create prefix embeddings
        self.prefix_embeddings = nn.Parameter(
            torch.randn(config.prefix_length, self.hidden_size) * 0.01
        )
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass with prefix."""
        batch_size = input_ids.shape[0]
        
        # Expand prefix for batch
        prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Get embeddings
        if hasattr(self.model, 'bert'):
            embeddings = self.model.bert.embeddings(input_ids)
        elif hasattr(self.model, 'roberta'):
            embeddings = self.model.roberta.embeddings(input_ids)
        else:
            # Fallback to standard forward
            return super().forward(input_ids, attention_mask, labels)
            
        # Concatenate prefix with embeddings
        embeddings = torch.cat([prefix, embeddings], dim=1)
        
        # Update attention mask
        if attention_mask is not None:
            prefix_mask = torch.ones(batch_size, self.config.prefix_length).to(attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
        # Forward through model
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs


class TransformerTrainer:
    """Trainer for transformer models."""
    
    def __init__(self, model: BaseTransformerModel, config: TransformerConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train(self, 
              train_dataset, 
              val_dataset,
              output_dir: Path,
              num_epochs: int = 3,
              batch_size: int = 16,
              eval_steps: int = 100,
              save_steps: int = 500,
              logging_steps: int = 50,
              early_stopping_patience: int = 3):
        """Train the transformer model."""
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            fp16=self.config.fp16,
            logging_dir=str(output_dir / 'logs'),
            logging_steps=logging_steps,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none"
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.model.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        
        # Train
        trainer.train()
        
        # Save best model
        trainer.save_model(str(output_dir / 'best_model'))
        self.model.tokenizer.save_pretrained(str(output_dir / 'best_model'))
        
        return trainer
        
    def evaluate(self, test_dataset):
        """Evaluate the model on test dataset."""
        trainer = Trainer(
            model=self.model.model,
            tokenizer=self.model.tokenizer,
        )
        
        results = trainer.evaluate(test_dataset)
        return results
        
    def predict(self, texts: List[str]) -> np.ndarray:
        """Make predictions on texts."""
        self.model.eval()
        
        # Tokenize texts
        encodings = self.model.tokenize_texts(texts)
        
        # Move to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
        return predictions.cpu().numpy()


def get_transformer_search_space(model_type: str) -> Dict:
    """Get hyperparameter search space for transformer models."""
    base_space = {
        'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
        'warmup_ratio': [0.0, 0.1, 0.2],
        'weight_decay': [0.0, 0.01, 0.1],
        'dropout_rate': [0.1, 0.2, 0.3],
    }
    
    if 'large' in model_type:
        # Smaller learning rates for large models
        base_space['learning_rate'] = [5e-6, 1e-5, 2e-5]
        
    # Add model-specific parameters
    if model_type in ['bert', 'roberta', 'albert']:
        base_space['freeze_layers'] = [0, 4, 8, 10]
        
    return base_space