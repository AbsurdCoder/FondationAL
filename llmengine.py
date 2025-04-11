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
        self.tokenizer = None
    
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
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.config.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # For causal language modeling, we need input_ids and labels
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
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


class LLMTrainer:
    """Main class to orchestrate the training process."""
    
    def __init__(self, config_path: str = None, **kwargs):
        """Initialize with either a config file or parameters."""
        if config_path:
            self.config = LLMConfig.from_json(config_path)
        else:
            self.config = LLMConfig(**kwargs)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.output_dir}/training.log"),
                logging.StreamHandler()
            ]
        )
        
        # Create model manager
        self.model_manager = ModelManager(self.config)
    
    def train(self) -> None:
        """Train the model."""
        self.model_manager.train()
    
    def save_config(self, path: str = None) -> None:
        """Save configuration to file."""
        if not path:
            path = os.path.join(self.config.output_dir, "config.json")
        self.config.to_json(path)
    
    def generate_sample(self, prompt: str, max_length: int = 100, temperature: float = 1.0) -> str:
        """Generate a text sample using the trained model."""
        return self.model_manager.generate_text(prompt, max_length, temperature)
    
    def load_model(self, model_path: str) -> None:
        """Load a pre-trained model."""
        self.config.pretrained_model_path = model_path
        self.model_manager.initialize_model()

