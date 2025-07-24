"""
Advanced Fidelity Management for Multi-Fidelity Optimization

Implements various fidelity dimensions:
1. Sequence length (truncation)
2. Vocabulary size
3. Model complexity (depth, width)
4. Training data fraction
5. Feature dimensions
6. Embedding dimensions
7. Training iterations/epochs
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FidelityType(Enum):
    """Types of fidelity dimensions."""
    SEQUENCE_LENGTH = "sequence_length"
    VOCABULARY_SIZE = "vocabulary_size"
    MODEL_DEPTH = "model_depth"
    MODEL_WIDTH = "model_width"
    DATA_FRACTION = "data_fraction"
    FEATURE_DIM = "feature_dimension"
    EMBEDDING_DIM = "embedding_dimension"
    TRAINING_STEPS = "training_steps"
    BATCH_SIZE = "batch_size"


@dataclass
class FidelityConfig:
    """Configuration for a fidelity dimension."""
    fidelity_type: FidelityType
    min_value: float
    max_value: float
    scaling: str = "linear"  # linear, log, exponential
    discrete: bool = False
    step_size: Optional[float] = None
    
    def get_value(self, fidelity_param: float) -> Union[int, float]:
        """Convert fidelity parameter (0-1) to actual value."""
        if self.scaling == "linear":
            value = self.min_value + fidelity_param * (self.max_value - self.min_value)
        elif self.scaling == "log":
            log_min = np.log(self.min_value) if self.min_value > 0 else 0
            log_max = np.log(self.max_value)
            value = np.exp(log_min + fidelity_param * (log_max - log_min))
        elif self.scaling == "exponential":
            value = self.min_value * (self.max_value / self.min_value) ** fidelity_param
        else:
            raise ValueError(f"Unknown scaling: {self.scaling}")
            
        # Handle discrete values
        if self.discrete:
            if self.step_size:
                value = self.min_value + round((value - self.min_value) / self.step_size) * self.step_size
            else:
                value = round(value)
                
        return int(value) if self.discrete else value


class FidelityManager:
    """Manages multiple fidelity dimensions for optimization."""
    
    def __init__(self, fidelity_configs: List[FidelityConfig]):
        self.configs = {config.fidelity_type: config for config in fidelity_configs}
        self.current_fidelities = {}
        
    def set_fidelities(self, fidelity_params: Dict[FidelityType, float]):
        """Set fidelity parameters (0-1 scale) for each dimension."""
        self.current_fidelities = {}
        for fidelity_type, param in fidelity_params.items():
            if fidelity_type in self.configs:
                config = self.configs[fidelity_type]
                self.current_fidelities[fidelity_type] = config.get_value(param)
            else:
                logger.warning(f"Unknown fidelity type: {fidelity_type}")
                
    def get_fidelity(self, fidelity_type: FidelityType) -> Optional[Union[int, float]]:
        """Get current value for a fidelity dimension."""
        return self.current_fidelities.get(fidelity_type)
        
    def apply_sequence_length_fidelity(self, texts: List[str]) -> List[str]:
        """Apply sequence length truncation."""
        max_length = self.get_fidelity(FidelityType.SEQUENCE_LENGTH)
        if max_length is None:
            return texts
            
        truncated_texts = []
        for text in texts:
            words = text.split()
            if len(words) > max_length:
                truncated_text = ' '.join(words[:int(max_length)])
                truncated_texts.append(truncated_text)
            else:
                truncated_texts.append(text)
                
        logger.info(f"Applied sequence length fidelity: max_length={max_length}")
        return truncated_texts
        
    def apply_vocabulary_size_fidelity(self, vocab_params: Dict) -> Dict:
        """Apply vocabulary size constraints."""
        vocab_size = self.get_fidelity(FidelityType.VOCABULARY_SIZE)
        if vocab_size is not None:
            vocab_params['max_features'] = int(vocab_size)
            logger.info(f"Applied vocabulary size fidelity: vocab_size={vocab_size}")
        return vocab_params
        
    def apply_model_complexity_fidelity(self, model_params: Dict) -> Dict:
        """Apply model depth and width constraints."""
        # Model depth
        depth = self.get_fidelity(FidelityType.MODEL_DEPTH)
        if depth is not None:
            if 'num_layers' in model_params:
                model_params['num_layers'] = int(depth)
            elif 'n_layers' in model_params:
                model_params['n_layers'] = int(depth)
            elif 'depth' in model_params:
                model_params['depth'] = int(depth)
                
        # Model width
        width = self.get_fidelity(FidelityType.MODEL_WIDTH)
        if width is not None:
            if 'hidden_dim' in model_params:
                model_params['hidden_dim'] = int(width)
            elif 'hidden_size' in model_params:
                model_params['hidden_size'] = int(width)
            elif 'n_hidden' in model_params:
                model_params['n_hidden'] = int(width)
                
        logger.info(f"Applied model complexity: depth={depth}, width={width}")
        return model_params
        
    def apply_data_fraction_fidelity(self, data_indices: np.ndarray) -> np.ndarray:
        """Apply data fraction subsampling."""
        fraction = self.get_fidelity(FidelityType.DATA_FRACTION)
        if fraction is None or fraction >= 1.0:
            return data_indices
            
        n_samples = int(len(data_indices) * fraction)
        sampled_indices = np.random.choice(data_indices, size=n_samples, replace=False)
        
        logger.info(f"Applied data fraction fidelity: {fraction:.2f} ({n_samples} samples)")
        return sampled_indices
        
    def apply_training_steps_fidelity(self, training_params: Dict) -> Dict:
        """Apply training steps/epochs constraints."""
        steps = self.get_fidelity(FidelityType.TRAINING_STEPS)
        if steps is not None:
            if 'epochs' in training_params:
                training_params['epochs'] = int(steps)
            elif 'n_epochs' in training_params:
                training_params['n_epochs'] = int(steps)
            elif 'max_iter' in training_params:
                training_params['max_iter'] = int(steps)
                
        logger.info(f"Applied training steps fidelity: {steps}")
        return training_params
        
    def get_cost_estimate(self) -> float:
        """Estimate computational cost based on current fidelities."""
        cost = 1.0
        
        # Sequence length cost (linear)
        seq_length = self.get_fidelity(FidelityType.SEQUENCE_LENGTH)
        if seq_length is not None:
            max_seq = self.configs[FidelityType.SEQUENCE_LENGTH].max_value
            cost *= (seq_length / max_seq)
            
        # Vocabulary size cost (sub-linear)
        vocab_size = self.get_fidelity(FidelityType.VOCABULARY_SIZE)
        if vocab_size is not None:
            max_vocab = self.configs[FidelityType.VOCABULARY_SIZE].max_value
            cost *= (vocab_size / max_vocab) ** 0.5
            
        # Model complexity cost (quadratic for transformers)
        depth = self.get_fidelity(FidelityType.MODEL_DEPTH)
        width = self.get_fidelity(FidelityType.MODEL_WIDTH)
        if depth is not None and width is not None:
            max_depth = self.configs[FidelityType.MODEL_DEPTH].max_value
            max_width = self.configs[FidelityType.MODEL_WIDTH].max_value
            cost *= (depth / max_depth) * (width / max_width) ** 2
            
        # Data fraction cost (linear)
        data_frac = self.get_fidelity(FidelityType.DATA_FRACTION)
        if data_frac is not None:
            cost *= data_frac
            
        # Training steps cost (linear)
        steps = self.get_fidelity(FidelityType.TRAINING_STEPS)
        if steps is not None:
            max_steps = self.configs[FidelityType.TRAINING_STEPS].max_value
            cost *= (steps / max_steps)
            
        return cost


class AdaptiveFidelityScheduler:
    """Adaptive scheduler for multi-fidelity optimization."""
    
    def __init__(self, 
                 total_budget: float,
                 n_iterations: int,
                 strategy: str = "successive_halving"):
        self.total_budget = total_budget
        self.n_iterations = n_iterations
        self.strategy = strategy
        self.iteration = 0
        self.history = []
        
    def get_next_fidelity(self, performance_history: List[float]) -> Dict[FidelityType, float]:
        """Get next fidelity configuration based on performance history."""
        self.iteration += 1
        
        if self.strategy == "successive_halving":
            return self._successive_halving_schedule()
        elif self.strategy == "hyperband":
            return self._hyperband_schedule()
        elif self.strategy == "progressive":
            return self._progressive_schedule()
        else:
            # Default: linear increase
            fidelity = min(1.0, self.iteration / self.n_iterations)
            return {
                FidelityType.SEQUENCE_LENGTH: fidelity,
                FidelityType.VOCABULARY_SIZE: fidelity,
                FidelityType.MODEL_DEPTH: fidelity,
                FidelityType.DATA_FRACTION: fidelity,
                FidelityType.TRAINING_STEPS: fidelity
            }
            
    def _successive_halving_schedule(self) -> Dict[FidelityType, float]:
        """Successive halving schedule."""
        # Start with low fidelity, double at each stage
        stage = int(np.log2(self.iteration + 1))
        fidelity = min(1.0, 2 ** stage / 2 ** 5)  # Max 32 stages
        
        return {
            FidelityType.SEQUENCE_LENGTH: min(1.0, fidelity * 2),  # Sequence length increases faster
            FidelityType.VOCABULARY_SIZE: fidelity,
            FidelityType.MODEL_DEPTH: fidelity,
            FidelityType.DATA_FRACTION: fidelity,
            FidelityType.TRAINING_STEPS: fidelity
        }
        
    def _hyperband_schedule(self) -> Dict[FidelityType, float]:
        """Hyperband schedule with multiple brackets."""
        # Simplified hyperband - use geometric progression
        s_max = int(np.log2(self.n_iterations))
        s = s_max - (self.iteration % (s_max + 1))
        n = int(np.ceil(self.n_iterations / (s + 1)))
        r = 1.0 / (s + 1)
        
        fidelity = min(1.0, r * (self.iteration % n + 1))
        
        return {
            FidelityType.SEQUENCE_LENGTH: fidelity,
            FidelityType.VOCABULARY_SIZE: fidelity,
            FidelityType.MODEL_DEPTH: fidelity,
            FidelityType.DATA_FRACTION: fidelity,
            FidelityType.TRAINING_STEPS: fidelity
        }
        
    def _progressive_schedule(self) -> Dict[FidelityType, float]:
        """Progressive schedule - different rates for different fidelities."""
        progress = self.iteration / self.n_iterations
        
        return {
            # Sequence length increases quickly
            FidelityType.SEQUENCE_LENGTH: min(1.0, progress * 2),
            # Vocabulary size increases moderately
            FidelityType.VOCABULARY_SIZE: min(1.0, progress * 1.5),
            # Model complexity increases slowly
            FidelityType.MODEL_DEPTH: min(1.0, progress * 0.8),
            FidelityType.MODEL_WIDTH: min(1.0, progress * 0.8),
            # Data fraction increases linearly
            FidelityType.DATA_FRACTION: progress,
            # Training steps increase slowly at first
            FidelityType.TRAINING_STEPS: min(1.0, progress ** 0.5)
        }


def create_default_fidelity_configs(approach: str) -> List[FidelityConfig]:
    """Create default fidelity configurations for an approach."""
    configs = []
    
    # Common fidelities
    configs.extend([
        FidelityConfig(
            fidelity_type=FidelityType.SEQUENCE_LENGTH,
            min_value=50,
            max_value=512,
            scaling="linear",
            discrete=True
        ),
        FidelityConfig(
            fidelity_type=FidelityType.DATA_FRACTION,
            min_value=0.1,
            max_value=1.0,
            scaling="linear",
            discrete=False
        ),
        FidelityConfig(
            fidelity_type=FidelityType.TRAINING_STEPS,
            min_value=1,
            max_value=10,
            scaling="linear",
            discrete=True
        )
    ])
    
    # Approach-specific fidelities
    if approach in ['logistic', 'svm', 'xgboost', 'lightgbm']:
        configs.append(
            FidelityConfig(
                fidelity_type=FidelityType.VOCABULARY_SIZE,
                min_value=1000,
                max_value=20000,
                scaling="log",
                discrete=True,
                step_size=1000
            )
        )
    elif approach in ['ffnn', 'lstm', 'gru']:
        configs.extend([
            FidelityConfig(
                fidelity_type=FidelityType.MODEL_DEPTH,
                min_value=1,
                max_value=4,
                scaling="linear",
                discrete=True
            ),
            FidelityConfig(
                fidelity_type=FidelityType.MODEL_WIDTH,
                min_value=32,
                max_value=256,
                scaling="exponential",
                discrete=True,
                step_size=32
            ),
            FidelityConfig(
                fidelity_type=FidelityType.EMBEDDING_DIM,
                min_value=50,
                max_value=300,
                scaling="linear",
                discrete=True,
                step_size=50
            )
        ])
    elif approach in ['bert', 'roberta', 'transformer']:
        configs.extend([
            FidelityConfig(
                fidelity_type=FidelityType.MODEL_DEPTH,
                min_value=2,
                max_value=12,
                scaling="linear",
                discrete=True
            ),
            FidelityConfig(
                fidelity_type=FidelityType.BATCH_SIZE,
                min_value=8,
                max_value=32,
                scaling="exponential",
                discrete=True,
                step_size=8
            )
        ])
        
    return configs