from datasets import load_dataset
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from BitText.logs.logger import setup_logger
logger = setup_logger("Prepare Data","logs","info")

class TextPreprocessor:
    """
    A class to handle text preprocessing before tokenization.
    """

    def __init__(self):
        """
        Initializes the text preprocessor.
        """
        logger.info("Text Preprocessor initialized")

    def clean_text(self, text):
        """
        Clean the text by removing unwanted characters (e.g., punctuation, extra spaces).
        
        Args:
            text (str): The input text to clean.
        
        Returns:
            str: The cleaned text.
        """
        # Remove unwanted characters (punctuation, numbers, extra spaces)
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text)     # Remove extra spaces
        text = text.strip().lower()          # Strip spaces and convert to lower case
        
        logger.debug(f"Cleaned text: {text[:100]}...")  # Log the first 100 chars of cleaned text
        return text

    def preprocess_data(self, dataset):
        """
        Preprocess the entire dataset.
        
        Args:
            dataset (datasets.Dataset): The raw Hugging Face dataset to preprocess.
        
        Returns:
            datasets.Dataset: The preprocessed dataset.
        """
        logger.info(f"Preprocessing dataset with {len(dataset)} samples")

        # Clean the text data
        dataset = dataset.map(
            lambda x: {"text": self.clean_text(x["text"])}, 
            remove_columns=["text"]  # After cleaning, text is overwritten
        )
        
        logger.info(f"Dataset preprocessing complete: {len(dataset)} samples")
        return dataset

class DatasetLoader:
    """
    A class to load and preprocess datasets from Hugging Face.
    """

    def __init__(self, dataset_name):
        """
        Load the dataset using Hugging Face's library.
        
        Args:
            dataset_name (str): Name of the dataset to load (e.g., 'wikitext').
        """
        self.dataset_name = dataset_name
        logger.info(f"Loading dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name)

        # Initialize the preprocessor
        self.preprocessor = TextPreprocessor()

    def prepare_data(self):
        """
        Preprocess the loaded dataset.
        
        Returns:
            datasets.Dataset: The processed dataset.
        """
        # Preprocess the training and validation sets
        self.dataset["train"] = self.preprocessor.preprocess_data(self.dataset["train"])
        self.dataset["validation"] = self.preprocessor.preprocess_data(self.dataset["validation"])

        return self.dataset

    def save_data(self, path):
        """
        Save the preprocessed dataset to a specified directory.
        
        Args:
            path (str): The path to save the preprocessed dataset.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        self.dataset.save_to_disk(path)
        logger.info(f"Dataset saved to: {path}")


# Example usage:
if __name__ == "__main__":
    dataset_name = "wikitext"  # Replace with your dataset name from Hugging Face
    loader = DatasetLoader(dataset_name)
    preprocessed_data = loader.prepare_data()
    loader.save_data(f"BitText/data/processed_data/{dataset_name}")
    logger.info(f"Dataset {dataset_name} is ready for training!")
