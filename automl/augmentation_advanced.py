"""
Advanced Text Augmentation Module

Implements sophisticated augmentation strategies:
1. Back-translation using machine translation models
2. Paraphrasing using T5/BART models
3. Contextual word replacement using BERT
4. Mixup and manifold mixup for text
5. Adversarial augmentation
6. Style transfer augmentation
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Union
import logging
from transformers import (
    MarianMTModel, MarianTokenizer,
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    BertForMaskedLM, BertTokenizer,
    pipeline
)
import random
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet
import spacy

logger = logging.getLogger(__name__)


class AdvancedAugmenter:
    """Manager for advanced augmentation strategies."""
    
    def __init__(self, augmentation_types: List[str], device: str = 'cpu'):
        self.device = device
        self.augmenters = {}
        
        # Initialize requested augmenters
        for aug_type in augmentation_types:
            if aug_type == 'backtranslation':
                self.augmenters['backtranslation'] = BackTranslationAugmenter(device)
            elif aug_type == 'paraphrase':
                self.augmenters['paraphrase'] = ParaphraseAugmenter(device)
            elif aug_type == 'contextual':
                self.augmenters['contextual'] = ContextualAugmenter(device)
            elif aug_type == 'mixup':
                self.augmenters['mixup'] = MixupAugmenter()
            elif aug_type == 'adversarial':
                self.augmenters['adversarial'] = AdversarialAugmenter()
            else:
                logger.warning(f"Unknown augmentation type: {aug_type}")
                
    def augment(self, texts: List[str], labels: Optional[List[int]] = None, 
                num_augmentations: int = 1) -> Tuple[List[str], Optional[List[int]]]:
        """Apply all configured augmentations."""
        augmented_texts = []
        augmented_labels = []
        
        for i, text in enumerate(tqdm(texts, desc="Augmenting")):
            # Original text
            augmented_texts.append(text)
            if labels is not None:
                augmented_labels.append(labels[i])
                
            # Generate augmentations
            for _ in range(num_augmentations):
                # Randomly select an augmenter
                aug_name = random.choice(list(self.augmenters.keys()))
                augmenter = self.augmenters[aug_name]
                
                try:
                    if aug_name == 'mixup' and labels is not None:
                        # Mixup requires pairs of texts
                        j = random.randint(0, len(texts) - 1)
                        aug_text, aug_label = augmenter.augment(
                            text, texts[j], labels[i], labels[j]
                        )
                        augmented_texts.append(aug_text)
                        augmented_labels.append(aug_label)
                    else:
                        aug_text = augmenter.augment(text)
                        augmented_texts.append(aug_text)
                        if labels is not None:
                            augmented_labels.append(labels[i])
                except Exception as e:
                    logger.warning(f"Augmentation failed: {e}")
                    # Fall back to original
                    augmented_texts.append(text)
                    if labels is not None:
                        augmented_labels.append(labels[i])
                        
        return augmented_texts, augmented_labels if labels is not None else None


class BackTranslationAugmenter:
    """Augmentation through back-translation."""
    
    def __init__(self, device: str = 'cpu', intermediate_lang: str = 'fr'):
        self.device = device
        self.intermediate_lang = intermediate_lang
        
        # Load translation models
        self._load_models()
        
    def _load_models(self):
        """Load translation models."""
        # English to intermediate language
        model_name_fwd = f'Helsinki-NLP/opus-mt-en-{self.intermediate_lang}'
        self.tokenizer_fwd = MarianTokenizer.from_pretrained(model_name_fwd)
        self.model_fwd = MarianMTModel.from_pretrained(model_name_fwd).to(self.device)
        
        # Intermediate language to English
        model_name_back = f'Helsinki-NLP/opus-mt-{self.intermediate_lang}-en'
        self.tokenizer_back = MarianTokenizer.from_pretrained(model_name_back)
        self.model_back = MarianMTModel.from_pretrained(model_name_back).to(self.device)
        
    def augment(self, text: str) -> str:
        """Augment text through back-translation."""
        try:
            # Translate to intermediate language
            inputs_fwd = self.tokenizer_fwd(text, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                translated_fwd = self.model_fwd.generate(**inputs_fwd, max_length=512)
            intermediate_text = self.tokenizer_fwd.decode(translated_fwd[0], skip_special_tokens=True)
            
            # Translate back to English
            inputs_back = self.tokenizer_back(intermediate_text, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                translated_back = self.model_back.generate(**inputs_back, max_length=512)
            augmented_text = self.tokenizer_back.decode(translated_back[0], skip_special_tokens=True)
            
            return augmented_text
        except Exception as e:
            logger.warning(f"Back-translation failed: {e}")
            return text


class ParaphraseAugmenter:
    """Augmentation through paraphrasing using T5."""
    
    def __init__(self, device: str = 'cpu', model_name: str = 't5-small'):
        self.device = device
        self.model_name = model_name
        
        # Load paraphrase model
        self._load_model()
        
    def _load_model(self):
        """Load paraphrase generation model."""
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        
    def augment(self, text: str, num_variations: int = 1) -> str:
        """Generate paraphrases of the text."""
        # Prepare input
        input_text = f"paraphrase: {text} </s>"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)
        
        # Generate paraphrases
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                num_return_sequences=num_variations,
                temperature=1.2,
                do_sample=True
            )
            
        # Decode and return first paraphrase
        paraphrases = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return paraphrases[0] if paraphrases else text


class ContextualAugmenter:
    """Contextual word replacement using BERT."""
    
    def __init__(self, device: str = 'cpu', model_name: str = 'bert-base-uncased'):
        self.device = device
        
        # Load BERT for masked language modeling
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        
        # Load spaCy for POS tagging
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
    def augment(self, text: str, mask_prob: float = 0.15, top_k: int = 5) -> str:
        """Augment text by replacing words contextually."""
        if self.nlp is None:
            return text
            
        # Tokenize and identify content words
        doc = self.nlp(text)
        words = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]
        
        # Select words to mask (focus on content words)
        content_pos = ['NOUN', 'VERB', 'ADJ', 'ADV']
        mask_indices = []
        for i, (word, pos) in enumerate(zip(words, pos_tags)):
            if pos in content_pos and random.random() < mask_prob:
                mask_indices.append(i)
                
        if not mask_indices:
            return text
            
        # Create masked text
        masked_words = words.copy()
        for idx in mask_indices:
            masked_words[idx] = '[MASK]'
        masked_text = ' '.join(masked_words)
        
        # Predict replacements
        inputs = self.tokenizer(masked_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
            
        # Replace masked tokens
        for idx in mask_indices:
            # Find the position of this mask in the tokenized input
            mask_token_index = self._find_mask_position(inputs.input_ids[0], idx)
            if mask_token_index is None:
                continue
                
            # Get top k predictions
            top_k_tokens = torch.topk(predictions[0, mask_token_index], top_k).indices
            
            # Select a replacement (not the original word if possible)
            original_token = self.tokenizer.encode(words[idx], add_special_tokens=False)[0]
            replacements = [token.item() for token in top_k_tokens if token.item() != original_token]
            
            if replacements:
                replacement_token = random.choice(replacements[:3])  # Choose from top 3
                words[idx] = self.tokenizer.decode([replacement_token])
                
        return ' '.join(words)
        
    def _find_mask_position(self, input_ids, word_index):
        """Find the position of a mask token in the input."""
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if word_index < len(mask_positions):
            return mask_positions[word_index].item()
        return None


class MixupAugmenter:
    """Mixup augmentation for text embeddings."""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        
    def augment(self, text1: str, text2: str, label1: int, label2: int) -> Tuple[str, float]:
        """
        Apply mixup at the text level (interpolation of representations).
        Returns interpolated text and soft label.
        """
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # For text, we'll create a mixed representation by sampling words
        words1 = text1.split()
        words2 = text2.split()
        
        # Create mixed text by probabilistically selecting words
        mixed_words = []
        max_len = max(len(words1), len(words2))
        
        for i in range(max_len):
            if random.random() < lam:
                if i < len(words1):
                    mixed_words.append(words1[i])
            else:
                if i < len(words2):
                    mixed_words.append(words2[i])
                    
        mixed_text = ' '.join(mixed_words)
        
        # Mixed label (for soft labels in training)
        mixed_label = lam * label1 + (1 - lam) * label2
        
        return mixed_text, mixed_label


class AdversarialAugmenter:
    """Adversarial augmentation by adding noise to embeddings."""
    
    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon
        
    def augment(self, text: str) -> str:
        """
        Add adversarial noise through character-level perturbations.
        """
        chars = list(text)
        num_changes = max(1, int(len(chars) * 0.05))  # Change 5% of characters
        
        # Types of perturbations
        perturbations = ['swap', 'delete', 'insert', 'replace']
        
        for _ in range(num_changes):
            if len(chars) < 2:
                break
                
            perturbation = random.choice(perturbations)
            idx = random.randint(0, len(chars) - 1)
            
            if perturbation == 'swap' and idx < len(chars) - 1:
                # Swap adjacent characters
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            elif perturbation == 'delete' and len(chars) > 10:
                # Delete character
                del chars[idx]
            elif perturbation == 'insert':
                # Insert random character
                chars.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz '))
            elif perturbation == 'replace':
                # Replace with visually similar character
                if chars[idx] in self._get_similar_chars():
                    chars[idx] = random.choice(self._get_similar_chars()[chars[idx]])
                    
        return ''.join(chars)
        
    def _get_similar_chars(self):
        """Get visually similar characters for replacement."""
        return {
            'a': ['@', '4'],
            'e': ['3'],
            'i': ['1', '!'],
            'o': ['0'],
            's': ['$', '5'],
            'l': ['1', '|'],
            'g': ['9'],
            'b': ['8'],
        }


class StyleTransferAugmenter:
    """Augmentation through style transfer."""
    
    def __init__(self, styles: List[str] = ['formal', 'informal', 'simple']):
        self.styles = styles
        self.style_prompts = {
            'formal': "Rewrite the following text in a formal style: ",
            'informal': "Rewrite the following text in an informal, casual style: ",
            'simple': "Rewrite the following text using simple words: ",
            'technical': "Rewrite the following text using technical language: ",
            'emotional': "Rewrite the following text with more emotion: "
        }
        
    def augment(self, text: str, target_style: Optional[str] = None) -> str:
        """Transfer text to a different style."""
        if target_style is None:
            target_style = random.choice(self.styles)
            
        if target_style not in self.style_prompts:
            return text
            
        # In practice, this would use a style transfer model
        # For now, we'll use simple rule-based transformations
        if target_style == 'formal':
            # Simple formal transformations
            text = text.replace("don't", "do not")
            text = text.replace("can't", "cannot")
            text = text.replace("won't", "will not")
            text = text.replace("it's", "it is")
        elif target_style == 'informal':
            # Simple informal transformations
            text = text.replace("do not", "don't")
            text = text.replace("cannot", "can't")
            text = text.replace("will not", "won't")
            text = text.replace("it is", "it's")
        elif target_style == 'simple':
            # Would use a simplification model in practice
            pass
            
        return text


def create_augmentation_pipeline(
    augmentation_types: List[str],
    augmentation_strength: str = 'medium',
    device: str = 'cpu'
) -> AdvancedAugmenter:
    """Create an augmentation pipeline with specified strategies."""
    
    # Map strength to number of augmentations
    strength_map = {
        'light': 1,
        'medium': 2,
        'heavy': 3
    }
    
    augmenter = AdvancedAugmenter(augmentation_types, device)
    augmenter.num_augmentations = strength_map.get(augmentation_strength, 2)
    
    return augmenter