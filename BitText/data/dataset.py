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

    def __init__(self, dataset_name, tokenizer_name="bert-base-uncased", max_length=512, split="train", min_val_samples=1000):
        """
        Initialize the dataset.

        Args:
            dataset_name (str): Name of the Hugging Face dataset to load.
            tokenizer_name (str): Name of the tokenizer to use.
            max_length (int): Maximum sequence length to truncate/pad to.
            split (str): Which split to use ('train' or 'validation').
            min_val_samples (int): Minimum number of validation samples required.
        """
        self.dataset = load_dataset(dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.split = split
        self.min_val_samples = min_val_samples

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Log original dataset loading
        logger.info(f"Loaded dataset: {dataset_name}")
        logger.info(f"Original number of training samples: {len(self.dataset['train'])}")
        logger.info(f"Original number of validation samples: {len(self.dataset['validation'])}")

        # Check if validation set is too small and create custom split if needed
        self._ensure_adequate_validation_split()

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

    def _ensure_adequate_validation_split(self):
        """
        Ensure we have at least min_val_samples validation samples.
        If the original validation set is too small, create a custom split from training data.
        """
        original_val_size = len(self.dataset['validation'])
        
        if original_val_size < self.min_val_samples:
            logger.info(f"Original validation set has only {original_val_size} samples, need at least {self.min_val_samples}")
            logger.info("Creating custom validation split from training data...")
            
            # Calculate how many samples to take from training for validation
            train_data = self.dataset['train']
            samples_needed = self.min_val_samples - original_val_size
            
            # Take the last samples_needed samples from training for additional validation
            additional_val_indices = list(range(len(train_data) - samples_needed, len(train_data)))
            remaining_train_indices = list(range(len(train_data) - samples_needed))
            
            # Create new splits
            additional_val_data = train_data.select(additional_val_indices)
            new_train_data = train_data.select(remaining_train_indices)
            
            # Combine original validation with additional samples
            from datasets import concatenate_datasets
            combined_val_data = concatenate_datasets([self.dataset['validation'], additional_val_data])
            
            # Update the dataset splits
            self.dataset['train'] = new_train_data
            self.dataset['validation'] = combined_val_data
            
            logger.info(f"Created custom validation split:")
            logger.info(f"  - New training samples: {len(self.dataset['train'])}")
            logger.info(f"  - New validation samples: {len(self.dataset['validation'])}")
        else:
            logger.info(f"Validation set has adequate samples: {original_val_size}")

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
                       batch_size=16,
                       min_val_samples=1000):
    """
    Create DataLoader objects for training and validation sets.

    Args:
        dataset_name (str): Name of the Hugging Face dataset.
        tokenizer_name (str): Name of the tokenizer to use.
        max_length (int): Maximum sequence length.
        batch_size (int): Batch size for DataLoader.
        min_val_samples (int): Minimum number of validation samples required.
    
    Returns:
        tuple: DataLoaders for training and validation datasets.
    """
    
    # Create datasets
    train_dataset = CustomTextDataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        split="train",
        min_val_samples=min_val_samples
    )
    
    val_dataset = CustomTextDataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        split="validation",
        min_val_samples=min_val_samples
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
    # Example usage with minimum 1000 validation samples
    train_loader, val_loader = create_data_loaders(
        dataset_name="iohadrubin/wikitext-103-raw-v1",
        batch_size=16,
        min_val_samples=1000
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