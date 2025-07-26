import os
import sys
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from BitText.logs.logger import setup_logger
logger = setup_logger("Dataset","logs","info")

class CustomTextDataset(Dataset):
    """
    Custom dataset class for loading and processing Hugging Face datasets.
    """

    def __init__(self, dataset_name, tokenizer_name="bert-base-uncased", max_length=512, split="train"):
        """
        Initialize the dataset.

        Args:
            dataset_name (str): Name of the Hugging Face dataset to load.
            tokenizer_name (str): Name of the tokenizer to use.
            max_length (int): Maximum sequence length to truncate/pad to.
            split (str): Which split to use ('train' or 'validation').
        """
        self.dataset = load_dataset(dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.split = split

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Log dataset loading
        logger.info(f"Loaded dataset: {dataset_name}")
        logger.info(f"Number of training samples: {len(self.dataset['train'])}")
        logger.info(f"Number of validation samples: {len(self.dataset['validation'])}")

        # Process the data
        self.processed_data = self._process_data(self.dataset[split])

    def _process_data(self, data):
        """
        Process the raw dataset into tokenized format.

        Args:
            data (datasets.Dataset): Raw dataset (either train or validation).
        
        Returns:
            list: List of processed samples.
        """
        logger.info(f"Processing {self.split} data with {len(data)} samples")
        
        processed_samples = []
        
        for item in data:
            text = item.get('text', '')
            
            # Skip empty texts
            if not text or len(text.strip()) == 0:
                continue
                
            # Tokenize the text
            try:
                # Try the callable interface first
                tokenized = self.tokenizer(
                    text, 
                    truncation=True, 
                    padding="max_length", 
                    max_length=self.max_length,
                    return_tensors="pt"
                )
            except TypeError:
                # Fallback to encode_plus if callable fails
                tokenized = self.tokenizer.encode_plus(
                    text, 
                    truncation=True, 
                    padding="max_length", 
                    max_length=self.max_length,
                    return_tensors="pt"
                )
            
            # Create input and target sequences for language modeling
            input_ids = tokenized["input_ids"].squeeze(0)
            attention_mask = tokenized["attention_mask"].squeeze(0)
            
            # For language modeling, labels are the same as input_ids shifted by one
            # We'll handle this in the collate function or here
            labels = input_ids.clone()
            
            processed_samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })
        
        logger.info(f"Processed {len(processed_samples)} valid samples from {self.split} split")
        return processed_samples

    def __getitem__(self, idx):
        """
        Get an item by index from the dataset.

        Args:
            idx (int): Index for the sample.
        
        Returns:
            dict: Tokenized input and output tensors.
        """
        return self.processed_data[idx]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.processed_data)


def create_data_loaders(dataset_name="iohadrubin/wikitext-103-raw-v1", 
                       tokenizer_name="bert-base-uncased", 
                       max_length=512, 
                       batch_size=16):
    """
    Create DataLoader objects for training and validation sets.

    Args:
        dataset_name (str): Name of the Hugging Face dataset.
        tokenizer_name (str): Name of the tokenizer to use.
        max_length (int): Maximum sequence length.
        batch_size (int): Batch size for DataLoader.
    
    Returns:
        tuple: DataLoaders for training and validation datasets.
    """
    
    # Create datasets
    train_dataset = CustomTextDataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        split="train"
    )
    
    val_dataset = CustomTextDataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        split="validation"
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid tokenizer parallelism issues
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid tokenizer parallelism issues
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def initialize_tokenizer(tokenizer_name="bert-base-uncased"):
    """
    Initialize a tokenizer from Hugging Face's library.

    Args:
        tokenizer_name (str): The name of the tokenizer to use.
    
    Returns:
        transformers.PreTrainedTokenizer: The tokenizer.
    """
    logger.info(f"Initializing tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer


if __name__ == "__main__":
    # Example usage
    train_loader, val_loader = create_data_loaders(
        dataset_name="iohadrubin/wikitext-103-raw-v1",
        batch_size=16
    )

    # Test the data loaders
    logger.info(f"Number of batches in training loader: {len(train_loader)}")
    logger.info(f"Number of batches in validation loader: {len(val_loader)}")
    
    # Test a batch
    for batch in train_loader:
        logger.info(f"Batch keys: {batch.keys()}")
        logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"Labels shape: {batch['labels'].shape}")
        break