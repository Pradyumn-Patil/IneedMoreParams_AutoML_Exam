# src/utils.py
import torch
from torch.utils.data import Dataset
import re
import string
import nltk
from typing import List, Optional, Dict
import unicodedata

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False


class TextPreprocessor:
    """Advanced text preprocessing with multiple options."""
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        remove_short_words: int = 2,  # Remove words shorter than this
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True,
        stem: bool = False,
        lemmatize: bool = False,
        custom_stopwords: Optional[List[str]] = None,
        preserve_emojis: bool = True,
    ):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.remove_short_words = remove_short_words
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.stem = stem
        self.lemmatize = lemmatize
        self.preserve_emojis = preserve_emojis
        
        # Initialize NLTK components
        if NLTK_AVAILABLE:
            self.stop_words = set(stopwords.words('english'))
            if custom_stopwords:
                self.stop_words.update(custom_stopwords)
            self.stemmer = PorterStemmer() if stem else None
            self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        else:
            # Basic stopwords if NLTK not available
            self.stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then', 'once'}
            self.stemmer = None
            self.lemmatizer = None
            
        # Compile regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Emoji pattern to preserve emojis
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
    
    def preprocess(self, text: str) -> str:
        """Apply all preprocessing steps to the text."""
        if not text:
            return ""
            
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
            
        # Remove emails
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
            
        # Preserve emojis by temporarily replacing them
        emoji_placeholder = {}
        if self.preserve_emojis:
            emojis = self.emoji_pattern.findall(text)
            for i, emoji in enumerate(emojis):
                placeholder = f"EMOJI_{i}"
                emoji_placeholder[placeholder] = emoji
                text = text.replace(emoji, f" {placeholder} ")
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
            
        # Remove numbers
        if self.remove_numbers:
            text = self.number_pattern.sub(' ', text)
            
        # Remove punctuation (but keep it for some cases like contractions)
        if self.remove_punctuation:
            # Keep apostrophes for contractions
            text = re.sub(r"[^\w\s']", ' ', text)
            # Handle contractions
            text = re.sub(r"'s\b", " is", text)
            text = re.sub(r"'re\b", " are", text)
            text = re.sub(r"'ve\b", " have", text)
            text = re.sub(r"'ll\b", " will", text)
            text = re.sub(r"'d\b", " would", text)
            text = re.sub(r"n't\b", " not", text)
            text = re.sub(r"'m\b", " am", text)
            
        # Tokenize
        if NLTK_AVAILABLE:
            tokens = word_tokenize(text)
        else:
            tokens = text.split()
            
        # Remove stopwords
        if self.remove_stopwords and self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
            
        # Remove short words
        if self.remove_short_words > 0:
            tokens = [token for token in tokens if len(token) >= self.remove_short_words]
            
        # Stemming
        if self.stem and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
            
        # Lemmatization
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
        # Restore emojis
        if self.preserve_emojis:
            tokens = [emoji_placeholder.get(token, token) for token in tokens]
            
        # Join tokens
        text = ' '.join(tokens)
        
        # Normalize whitespace
        if self.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text).strip()
            
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts."""
        return [self.preprocess(text) for text in texts]
    
    @staticmethod
    def get_dataset_specific_preprocessor(dataset_name: str) -> 'TextPreprocessor':
        """Get a preprocessor with settings optimized for specific datasets."""
        if dataset_name == 'imdb':
            # IMDB: Keep more context, don't remove stopwords aggressively
            return TextPreprocessor(
                lowercase=True,
                remove_punctuation=True,
                remove_numbers=False,
                remove_stopwords=False,  # Keep stopwords for sentiment
                remove_short_words=2,
                stem=False,  # Keep word forms for sentiment
                lemmatize=True
            )
        elif dataset_name == 'ag_news':
            # AG News: More aggressive preprocessing for news
            return TextPreprocessor(
                lowercase=True,
                remove_punctuation=True,
                remove_numbers=True,
                remove_stopwords=True,
                remove_short_words=3,
                stem=True,  # Stem for news categorization
                lemmatize=False
            )
        elif dataset_name == 'dbpedia':
            # DBpedia: Keep structure for ontology classification
            return TextPreprocessor(
                lowercase=True,
                remove_punctuation=False,  # Keep some punctuation
                remove_numbers=False,
                remove_stopwords=True,
                remove_short_words=2,
                stem=False,
                lemmatize=True
            )
        elif dataset_name == 'amazon':
            # Amazon: Balance for product reviews
            return TextPreprocessor(
                lowercase=True,
                remove_punctuation=True,
                remove_numbers=False,
                remove_stopwords=True,
                remove_short_words=2,
                stem=False,
                lemmatize=True,
                preserve_emojis=True  # Keep emojis for sentiment
            )
        else:
            # Default preprocessor
            return TextPreprocessor()


class SimpleTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if isinstance(self.tokenizer, dict):
            tokens = text.split()
            ids = [self.tokenizer.get(tok, 1) for tok in tokens[:self.max_length]]
            ids += [0] * (self.max_length - len(ids))
            return torch.tensor(ids), torch.tensor(label)

        elif TRANSFORMERS_AVAILABLE and hasattr(self.tokenizer, 'encode_plus'):
            encoded = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': torch.tensor(label)
            }
        else:
            raise ValueError("Tokenizer not defined or unsupported.")