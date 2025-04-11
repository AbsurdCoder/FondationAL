import os
import json
import regex as re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set, Union, Optional
import pickle
import numpy as np
from tqdm import tqdm
import logging

class BaseTokenizer:
    """Base class for all tokenizers."""
    
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.special_tokens = {
            "<PAD>": 0,   # Padding token
            "<UNK>": 1,   # Unknown token
            "<BOS>": 2,   # Beginning of sequence token
            "<EOS>": 3    # End of sequence token
        }
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
    
    def train(self, texts: List[str]) -> None:
        """Train the tokenizer on a list of texts."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to a list of token IDs."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def decode(self, ids: List[int]) -> str:
        """Convert a list of token IDs back to text."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def save(self, path: str) -> None:
        """Save the tokenizer to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        tokenizer_data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()},
            "special_tokens": self.special_tokens,
            "tokenizer_type": self.__class__.__name__
        }
        
        # Add additional data for specific tokenizer types
        if hasattr(self, 'merges'):
            tokenizer_data["merges"] = self.merges
            
        with open(path, 'wb') as f:
            pickle.dump(tokenizer_data, f)
            
        logging.info(f"Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseTokenizer':
        """Load a tokenizer from a file."""
        with open(path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        # Determine the correct tokenizer class
        tokenizer_type = tokenizer_data.get("tokenizer_type", cls.__name__)
        if tokenizer_type == "CharacterTokenizer":
            tokenizer = CharacterTokenizer(tokenizer_data["vocab_size"])
        elif tokenizer_type == "BPETokenizer":
            tokenizer = BPETokenizer(tokenizer_data["vocab_size"])
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
        
        # Load common attributes
        tokenizer.token_to_id = tokenizer_data["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in tokenizer_data["id_to_token"].items()}
        tokenizer.special_tokens = tokenizer_data["special_tokens"]
        
        # Load additional data for specific tokenizer types
        if tokenizer_type == "BPETokenizer" and "merges" in tokenizer_data:
            tokenizer.merges = tokenizer_data["merges"]
        
        return tokenizer
    
    def batch_encode(self, texts: List[str], max_length: Optional[int] = None, 
                     padding: bool = False, truncation: bool = False) -> List[List[int]]:
        """Encode a batch of texts."""
        batch_ids = []
        
        for text in texts:
            ids = self.encode(text)
            
            # Apply truncation if needed
            if truncation and max_length and len(ids) > max_length:
                ids = ids[:max_length]
            
            batch_ids.append(ids)
        
        # Apply padding if needed
        if padding and max_length:
            for i in range(len(batch_ids)):
                if len(batch_ids[i]) < max_length:
                    batch_ids[i] = batch_ids[i] + [self.token_to_id["<PAD>"]] * (max_length - len(batch_ids[i]))
        
        return batch_ids
    
    def prepare_for_model(self, texts: Union[str, List[str]], max_length: Optional[int] = None,
                           padding: bool = True, truncation: bool = True, 
                           add_special_tokens: bool = True) -> Dict[str, np.ndarray]:
        """Prepare inputs for model (similar to HuggingFace's tokenizer interface)."""
        if isinstance(texts, str):
            texts = [texts]
        
        # First encode all texts
        if add_special_tokens:
            encoded_texts = [self.encode_with_special_tokens(text) for text in texts]
        else:
            encoded_texts = [self.encode(text) for text in texts]
        
        # Apply truncation if needed
        if truncation and max_length:
            encoded_texts = [ids[:max_length] for ids in encoded_texts]
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [[1] * len(ids) for ids in encoded_texts]
        
        # Apply padding if needed
        if padding and max_length:
            for i in range(len(encoded_texts)):
                padding_length = max_length - len(encoded_texts[i])
                if padding_length > 0:
                    encoded_texts[i] = encoded_texts[i] + [self.token_to_id["<PAD>"]] * padding_length
                    attention_mask[i] = attention_mask[i] + [0] * padding_length
        
        return {
            "input_ids": np.array(encoded_texts),
            "attention_mask": np.array(attention_mask)
        }
    
    def encode_with_special_tokens(self, text: str) -> List[int]:
        """Encode text with special tokens (BOS/EOS)."""
        ids = self.encode(text)
        return [self.token_to_id["<BOS>"]] + ids + [self.token_to_id["<EOS>"]]
    
    def __len__(self) -> int:
        """Return the vocabulary size."""
        return len(self.token_to_id)


class BPETokenizer(BaseTokenizer):
    """A Byte-Pair Encoding (BPE) tokenizer implementation."""
    
    def __init__(self, vocab_size: int = 1000):
        logging.info("0 - Initializing BPE")
        super().__init__(vocab_size)
        self.merges: List[Tuple[str, str]] = []
        
    def get_base_vocab(self, texts: List[str]) -> None:
        """Create initial character vocabulary."""
        # Count character frequencies
        char_counter = Counter()
        for text in texts:
            char_counter.update(text)
        
        # Add all unique characters to vocabulary
        for char in char_counter:
            if char not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
        logging.info(f"1.5 - {self.token_to_id}")
        
    
    def train(self, texts: List[str], min_frequency: int = 2, num_iterations: Optional[int] = None) -> None:
        """Train BPE tokenizer on texts."""
        # Start with character vocabulary
        logging.info("1 - Getting base vocabs")
        self.get_base_vocab(texts)
        
        word_freqs = Counter()
        for text in texts:
            word_freqs[" ".join(list(text))] += 1
        logging.info("2 - Convert texts to lists of characters (tokens)")
        
        if num_iterations is None:
            num_iterations = self.vocab_size - len(self.token_to_id)

        logging.info(f"3 - Training BPE tokenizer with {num_iterations} merges")
        
        # Perform BPE training
        for i in tqdm(range(num_iterations)):
            # Find the most frequent pair
            pairs = Counter()

            for word, freq in word_freqs.items():
                if freq < min_frequency:
                    continue
                    
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pairs[(symbols[j], symbols[j + 1])] += freq
            
            if not pairs:
                break
                
            # Get most common pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair into a new token
            new_token = "".join(best_pair)
            if new_token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[new_token] = idx
                self.id_to_token[idx] = new_token
            
            # Add to merges
            self.merges.append(best_pair)
            
            # Update words with the new merged pair
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = word.replace(" ".join(best_pair), new_token)
                new_word_freqs[new_word] += freq
            
            word_freqs = new_word_freqs
            
            # Stop if we've reached vocab size
            if len(self.token_to_id) >= self.vocab_size:
                break
        
        logging.info(f"BPE training complete. Vocabulary size: {len(self.token_to_id)}")
        logging.info(f"Vocabulary: {self.token_to_id}")
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs using BPE encoding."""
        # Start with characters
        tokens = list(text)
        
        # Apply merges
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens[i] = pair[0] + pair[1]
                    del tokens[i + 1]
                else:
                    i += 1
        
        # Convert tokens to IDs
        return [self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        # Filter out special tokens
        filtered_ids = [id for id in ids if id not in [
            self.special_tokens["<PAD>"],
            self.special_tokens["<BOS>"],
            self.special_tokens["<EOS>"]
        ]]
        
        # Convert IDs to tokens and join
        tokens = [self.id_to_token.get(id, "<UNK>") for id in filtered_ids]
        return "".join(tokens)


logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

