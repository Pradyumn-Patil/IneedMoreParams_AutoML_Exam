"""
Text Augmentation Module for AutoML Text Classification

Implements various text augmentation techniques to improve model robustness and performance.
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from collections import defaultdict

# NLTK imports
try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextAugmenter:
    """
    Text augmentation class with multiple augmentation techniques.
    
    Supports:
    - Synonym replacement
    - Random insertion
    - Random swap
    - Random deletion
    - Contextual augmentation
    """
    
    def __init__(
        self,
        augmentation_prob: float = 0.1,
        max_augmentations: int = 3,
        preserve_length: bool = False,
        dataset_name: Optional[str] = None
    ):
        """
        Initialize text augmenter.
        
        Args:
            augmentation_prob: Probability of augmenting each word
            max_augmentations: Maximum number of augmentations per sentence
            preserve_length: Whether to preserve original text length
            dataset_name: Dataset name for dataset-specific strategies
        """
        self.augmentation_prob = augmentation_prob
        self.max_augmentations = max_augmentations
        self.preserve_length = preserve_length
        self.dataset_name = dataset_name
        
        # Download required NLTK data if needed
        if NLTK_AVAILABLE:
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                logger.info("Downloading WordNet data...")
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                
        # Dataset-specific stop words to preserve
        self.dataset_stopwords = {
            'ag_news': {'reuters', 'ap', 'afp', 'bloomberg'},  # News agencies
            'imdb': {'movie', 'film', 'actor', 'director'},   # Movie terms
            'amazon': {'product', 'item', 'quality', 'price'}, # Product terms
            'dbpedia': {'wikipedia', 'article', 'category'}    # Encyclopedia terms
        }
        
    def augment(self, text: str, num_augmentations: int = 1) -> List[str]:
        """
        Augment text using multiple techniques.
        
        Args:
            text: Input text to augment
            num_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented texts
        """
        augmented_texts = []
        
        for _ in range(num_augmentations):
            # Choose augmentation technique randomly
            technique = random.choice([
                self.synonym_replacement,
                self.random_insertion,
                self.random_swap,
                self.random_deletion,
                self.mixed_augmentation
            ])
            
            augmented = technique(text)
            if augmented != text:  # Only add if actually augmented
                augmented_texts.append(augmented)
                
        return augmented_texts
    
    def augment_batch(self, texts: List[str], num_augmentations: int = 1) -> List[List[str]]:
        """Augment a batch of texts."""
        return [self.augment(text, num_augmentations) for text in texts]
    
    def synonym_replacement(self, text: str) -> str:
        """Replace words with synonyms from WordNet."""
        if not NLTK_AVAILABLE:
            return text
            
        words = text.split()
        new_words = words.copy()
        random_word_list = list(set([word for word in words if self._can_augment_word(word)]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self._get_synonyms(random_word)
            if len(synonyms) > 0:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
                
            if num_replaced >= self.max_augmentations:
                break
                
        return ' '.join(new_words)
    
    def random_insertion(self, text: str) -> str:
        """Randomly insert synonyms of random words."""
        words = text.split()
        if len(words) == 0:
            return text
            
        for _ in range(min(self.max_augmentations, len(words))):
            if random.random() < self.augmentation_prob:
                new_word = self._get_random_synonym_from_text(words)
                if new_word:
                    random_idx = random.randint(0, len(words))
                    words.insert(random_idx, new_word)
                    
        return ' '.join(words)
    
    def random_swap(self, text: str) -> str:
        """Randomly swap two words in the sentence."""
        words = text.split()
        if len(words) < 2:
            return text
            
        for _ in range(min(self.max_augmentations, len(words) // 2)):
            if random.random() < self.augmentation_prob:
                idx1 = random.randint(0, len(words) - 1)
                idx2 = random.randint(0, len(words) - 1)
                while idx1 == idx2:
                    idx2 = random.randint(0, len(words) - 1)
                words[idx1], words[idx2] = words[idx2], words[idx1]
                
        return ' '.join(words)
    
    def random_deletion(self, text: str) -> str:
        """Randomly delete words from the sentence."""
        words = text.split()
        if len(words) <= 1:
            return text
            
        # Keep at least half of the words
        max_deletions = min(self.max_augmentations, len(words) // 2)
        
        new_words = []
        deleted = 0
        for word in words:
            if random.random() < self.augmentation_prob and deleted < max_deletions:
                deleted += 1
                continue
            new_words.append(word)
            
        # If all words were deleted, return original
        return ' '.join(new_words) if len(new_words) > 0 else text
    
    def mixed_augmentation(self, text: str) -> str:
        """Apply multiple augmentation techniques."""
        augmented = text
        
        # Apply 2-3 random techniques
        num_techniques = random.randint(2, 3)
        techniques = random.sample([
            self.synonym_replacement,
            self.random_insertion,
            self.random_swap,
            self.random_deletion
        ], num_techniques)
        
        for technique in techniques:
            augmented = technique(augmented)
            
        return augmented
    
    def contextual_augmentation(self, text: str, context_words: List[str] = None) -> str:
        """
        Augment text while preserving context-specific words.
        
        Args:
            text: Input text
            context_words: Important words to preserve
            
        Returns:
            Augmented text
        """
        if context_words is None:
            context_words = []
            
        words = text.split()
        new_words = []
        
        for word in words:
            if word.lower() in context_words:
                new_words.append(word)
            elif random.random() < self.augmentation_prob:
                synonyms = self._get_synonyms(word)
                if synonyms:
                    new_words.append(random.choice(list(synonyms)))
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
                
        return ' '.join(new_words)
    
    def _can_augment_word(self, word: str) -> bool:
        """Check if a word can be augmented."""
        # Don't augment very short words
        if len(word) < 3:
            return False
            
        # Don't augment numbers
        if word.isdigit():
            return False
            
        # Don't augment dataset-specific important words
        if self.dataset_name and self.dataset_name in self.dataset_stopwords:
            if word.lower() in self.dataset_stopwords[self.dataset_name]:
                return False
                
        return True
    
    def _get_synonyms(self, word: str) -> set:
        """Get synonyms for a word using WordNet."""
        if not NLTK_AVAILABLE:
            return set()
            
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym.isalpha():
                    synonyms.add(synonym)
                    
        return synonyms
    
    def _get_random_synonym_from_text(self, words: List[str]) -> Optional[str]:
        """Get a random synonym from any word in the text."""
        random_word_list = list(set([word for word in words if self._can_augment_word(word)]))
        random.shuffle(random_word_list)
        
        for word in random_word_list:
            synonyms = self._get_synonyms(word)
            if synonyms:
                return random.choice(list(synonyms))
                
        return None


class BackTranslationAugmenter:
    """
    Simplified back-translation augmenter using basic transformations.
    
    Note: Real back-translation would require translation models.
    This is a simplified version that simulates the effect.
    """
    
    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level
        
        # Common paraphrasing patterns
        self.paraphrase_rules = {
            "don't": ["do not", "don't"],
            "can't": ["cannot", "can not"],
            "won't": ["will not", "won't"],
            "it's": ["it is", "it's"],
            "that's": ["that is", "that's"],
            "good": ["great", "nice", "fine", "good"],
            "bad": ["poor", "terrible", "awful", "bad"],
            "very": ["really", "quite", "very", "extremely"],
            "big": ["large", "huge", "big", "enormous"],
            "small": ["little", "tiny", "small", "minor"]
        }
        
    def augment(self, text: str) -> str:
        """Simulate back-translation by paraphrasing."""
        words = text.split()
        new_words = []
        
        for word in words:
            lower_word = word.lower()
            if lower_word in self.paraphrase_rules and random.random() < self.noise_level:
                replacement = random.choice(self.paraphrase_rules[lower_word])
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                new_words.append(replacement)
            else:
                new_words.append(word)
                
        return ' '.join(new_words)


class DatasetAugmenter:
    """
    High-level augmenter that applies augmentation to entire datasets.
    """
    
    def __init__(
        self,
        augmenter: TextAugmenter,
        augmentation_factor: float = 0.5,
        balance_classes: bool = True
    ):
        """
        Initialize dataset augmenter.
        
        Args:
            augmenter: Text augmenter instance
            augmentation_factor: Fraction of dataset to augment
            balance_classes: Whether to augment minority classes more
        """
        self.augmenter = augmenter
        self.augmentation_factor = augmentation_factor
        self.balance_classes = balance_classes
        
    def augment_dataset(
        self,
        texts: List[str],
        labels: List[int],
        max_augmentations_per_sample: int = 2
    ) -> Tuple[List[str], List[int]]:
        """
        Augment entire dataset with class balancing.
        
        Args:
            texts: List of texts
            labels: List of labels
            max_augmentations_per_sample: Maximum augmentations per text
            
        Returns:
            Augmented texts and labels
        """
        augmented_texts = list(texts)
        augmented_labels = list(labels)
        
        # Calculate class distribution
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
            
        max_count = max(label_counts.values())
        
        # Augment samples
        for i, (text, label) in enumerate(zip(texts, labels)):
            # Calculate number of augmentations needed
            if self.balance_classes:
                # Augment minority classes more
                class_ratio = max_count / label_counts[label]
                num_augmentations = min(
                    int(class_ratio - 1),
                    max_augmentations_per_sample
                )
            else:
                # Random augmentation
                if random.random() < self.augmentation_factor:
                    num_augmentations = random.randint(1, max_augmentations_per_sample)
                else:
                    num_augmentations = 0
                    
            if num_augmentations > 0:
                augmented = self.augmenter.augment(text, num_augmentations)
                augmented_texts.extend(augmented)
                augmented_labels.extend([label] * len(augmented))
                
        logger.info(f"Dataset augmented from {len(texts)} to {len(augmented_texts)} samples")
        
        return augmented_texts, augmented_labels


def create_augmenter(dataset_name: str = None, augmentation_strength: str = 'medium') -> TextAugmenter:
    """
    Create a text augmenter with dataset-specific configuration.
    
    Args:
        dataset_name: Name of the dataset
        augmentation_strength: 'light', 'medium', or 'heavy'
        
    Returns:
        Configured TextAugmenter instance
    """
    strength_configs = {
        'light': {'augmentation_prob': 0.05, 'max_augmentations': 2},
        'medium': {'augmentation_prob': 0.1, 'max_augmentations': 3},
        'heavy': {'augmentation_prob': 0.2, 'max_augmentations': 5}
    }
    
    config = strength_configs.get(augmentation_strength, strength_configs['medium'])
    
    return TextAugmenter(
        augmentation_prob=config['augmentation_prob'],
        max_augmentations=config['max_augmentations'],
        dataset_name=dataset_name
    )