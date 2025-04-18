"""
Foundation Language Model Training Workflow 

This document outlines the complete workflow for training a small foundation language model 
using a custom tokenizer. The process follows these key stages:

1. DATA PREPARATION
   - Load raw text data from files or datasets
   - Clean and preprocess text (remove irrelevant content, normalize text, handle special characters)
   - Split data into training and validation sets

2. CUSTOM TOKENIZER CREATION
   - Implement tokenizer variants (Character, BPE, WordPiece)
   - Train the selected tokenizer on corpus data:
     * For BPE: Find most frequent character pairs and merge iteratively
     * For Character: Create vocabulary from all unique characters
     * For WordPiece: Build vocabulary based on likelihood scoring
   - Save tokenizer for reuse during model training and inference
   - Implement encode/decode methods for text conversion

3. MODEL CONFIGURATION
   - Define architecture parameters (embedding size, layers, attention heads)
   - Set up training hyperparameters (learning rate, batch size, epochs)
   - Configure model with custom tokenizer vocabulary size
   - Initialize model weights appropriately

4. DATASET PREPARATION
   - Tokenize text data using the custom tokenizer
   - Create sequence batches of appropriate length
   - Set up data loaders with efficient batching
   - Apply necessary padding and masking

5. TRAINING PROCESS
   - Initialize optimizer and learning rate scheduler
   - Set up training loop with gradient computation and updates
   - Implement validation checking at regular intervals
   - Track and log relevant metrics (loss, perplexity)
   - Save model checkpoints periodically
   - Apply early stopping based on validation performance

6. EVALUATION
   - Compute metrics on held-out test data
   - Generate text samples to assess qualitative performance
   - Compare with baseline models or different tokenizer configurations

7. DEPLOYMENT AND USAGE
   - Package model with its custom tokenizer
   - Create inference utilities for text generation
   - Document tokenizer characteristics for proper usage
"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union, Tuple
import wandb
import json
from pathlib import Path
from classic.tokenizerAL import BPETokenizer

class LLMConfig:
    """Configuration class for the LLM training process."""
    
    def __init__(
        self,
        model_name: str = "tiny-gpt",
        model_type: str = "gpt2",
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 6,
        n_head: int = 12,
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        epochs: int = 1,
        warmup_steps: int = 1000,
        max_seq_length: int = 512,
        output_dir: str = "./output",
        logging_steps: int = 100,
        save_steps: int = 1000,
        eval_steps: int = 500,
        use_wandb: bool = False,
        data_path: str = None,
        tokenizer_path: str = None,
        tokenizer_type: str = None,
        pretrained_model_path: str = None,
        fp16: bool = False,
        device: str = None
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.max_seq_length = max_seq_length
        self.output_dir = output_dir
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.use_wandb = use_wandb
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.tokenizer_type = tokenizer_type
        self.pretrained_model_path = pretrained_model_path
        self.fp16 = fp16
        
        # Set device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def from_json(cls, json_path: str) -> 'LLMConfig':
        """Load configuration from a JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: str) -> None:
        """Save configuration to a JSON file."""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    


class TextDatasetHandler:
    """Handles data preprocessing and loading."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.tokenizer = BPETokenizer(vocab_size=1000)
        self.train_tokenizer_if_needed()
    
    def train_tokenizer_if_needed(self):
        """Train tokenizer if it doesn't have merges yet"""
        if not hasattr(self.tokenizer, 'merges') or not self.tokenizer.merges:
            # Load some data for tokenizer training
            dataset = self.load_dataset("train")
            # Take a subset for tokenizer training
            texts = dataset["text"][:10000] if len(dataset) > 10000 else dataset["text"]
            # Train the tokenizer
            self.tokenizer.train(texts)
            # Save the trained tokenizer
            os.makedirs("tokenizers", exist_ok=True)
            self.tokenizer.save("tokenizers/custom_bpe_tokenizer.pkl")
            print("Tokenizer trained and saved.")

    def load_dataset(self, split: str = "train") -> Dataset:
        """Load dataset from local or HuggingFace's datasets."""
        if self.config.data_path and os.path.exists(self.config.data_path):
            # Load from local data
            if self.config.data_path.endswith('.json'):
                dataset = load_dataset('json', data_files=self.config.data_path)
            elif self.config.data_path.endswith('.txt'):
                dataset = self._load_text_file(self.config.data_path)
            else:
                dataset = load_dataset(self.config.data_path)
        else:
            # Use a small subset of a public dataset for POC
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        return dataset
    
    def _load_text_file(self, file_path: str) -> Dataset:
        """Load a simple text file line by line."""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        return Dataset.from_dict({"text": lines})
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize examples and prepare them for model input."""
        # Assuming examples have a 'text' field
        input_ids_list = []
        attention_mask_list = []
        
        for text in examples["text"]:
            # Encode text using our custom tokenizer
            input_ids = self.tokenizer.encode(text)
            
            # Handle truncation
            if len(input_ids) > self.config.max_seq_length:
                input_ids = input_ids[:self.config.max_seq_length]
            
            # Create attention mask (1 for tokens, 0 for padding)
            attention_mask = [1] * len(input_ids)
            
            # Handle padding
            padding_length = self.config.max_seq_length - len(input_ids)
            if padding_length > 0:
                # Assuming 0 is the padding token id
                input_ids = input_ids + [0] * padding_length
                attention_mask = attention_mask + [0] * padding_length
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
        
        # Convert lists to tensors
        result = {
            "input_ids": torch.tensor(input_ids_list),
            "attention_mask": torch.tensor(attention_mask_list)
        }
        
        # For causal language modeling, we need input_ids and labels
        result["labels"] = result["input_ids"].clone()
        
        return result
    
    def prepare_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Prepare train and validation dataloaders."""
        # Make sure tokenizer is loaded
        if not self.tokenizer:
            self.load_or_create_tokenizer()
            
        # Load and tokenize datasets
        train_dataset = self.load_dataset("train")
        # Tokenize dataset
        if "text" in train_dataset.features:
            tokenized_train = train_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=['text']
            )
        else:
            # Adjust column names if needed
            tokenized_train = train_dataset.map(
                self.tokenize_function,
                batched=True
            )
            
        # Create dataloader
        train_dataloader = DataLoader(
            tokenized_train,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Optionally create validation dataloader
        val_dataloader = None
        try:
            val_dataset = self.load_dataset("validation")
            if "text" in val_dataset.features:
                tokenized_val = val_dataset.map(
                    self.tokenize_function,
                    batched=True,
                    remove_columns=['text']
                )
            else:
                tokenized_val = val_dataset.map(
                    self.tokenize_function,
                    batched=True
                )
            
            val_dataloader = DataLoader(
                tokenized_val,
                batch_size=self.config.batch_size
            )
        except:
            logging.info("No validation dataset found.")
            
        return train_dataloader, val_dataloader



if __name__ == "__main__":
    # Create a configuration for a tiny model suitable for limited computing resources
    configo = LLMConfig(
        model_name="tiny-gpt",
        tokenizer_type="bpe",
        n_positions=512,  # Reduced context length
        n_embd=384,       # Reduced embedding dimension
        n_layer=6,        # Fewer layers
        n_head=6,         # Fewer attention heads
        batch_size=4,     # Small batch size for limited VRAM
        learning_rate=5e-5,
        epochs=3,
        max_seq_length=512,
        output_dir="./tiny_model_output",
        logging_steps=10,
        save_steps=500,
        # Using a small subset of wikitext for quick training
        data_path=None,  # will use wikitext-2-raw-v1 by default
        fp16=True        # Use mixed precision if available
    )
    

data_handler = TextDatasetHandler(configo)
print(data_handler.load_dataset())
data_handler.prepare_dataloaders()
