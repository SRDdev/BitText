import torch
import torch.nn as nn
from .layers import BitNetTransformer
from torch.optim import AdamW
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from BitText.logs.logger import setup_logger
logger = setup_logger("Transformer","logs","info")


class BitNetTextModel(nn.Module):
    """
    BitNet-based text model using the BitNetTransformer architecture.
    """

    def __init__(self, vocab_size, num_layers=2, d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1):
        """
        Initialize the BitNet model for text generation or classification.

        Args:
            vocab_size (int): Size of the vocabulary.
            num_layers (int): Number of transformer layers.
            d_model (int): The dimension of the model (embedding size).
            nhead (int): Number of attention heads.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
        """
        super(BitNetTextModel, self).__init__()
        self.model = BitNetTransformer(
            vocab_size=vocab_size,
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

    def forward(self, x):
        """
        Forward pass through the BitNet model.

        Args:
            x (Tensor): Input tensor (batch_size, seq_len).
        
        Returns:
            Tensor: Output tensor with predicted logits.
        """
        return self.model(x)


def initialize_model(vocab_size, num_layers=2, d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1):
    """
    Initializes the BitNet model for training.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        num_layers (int): Number of transformer layers.
        d_model (int): Dimension of the model (embedding size).
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
    
    Returns:
        model (nn.Module): The initialized BitNet model.
        optimizer (torch.optim.Optimizer): AdamW optimizer.
    """
    model = BitNetTextModel(vocab_size, num_layers, d_model, nhead, dim_feedforward, dropout)
    optimizer = AdamW(model.parameters(), lr=1e-4)  # You can adjust the learning rate here.
    
    logger.info("Model and optimizer initialized.")
    return model, optimizer


def load_pretrained_model(model, optimizer, checkpoint_path="models/model_checkpoint.pth"):
    """
    Loads a pretrained model and its optimizer state from a checkpoint.

    Args:
        model (nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        checkpoint_path (str): Path to the checkpoint file.
    
    Returns:
        model (nn.Module): The model with the loaded weights.
        optimizer (torch.optim.Optimizer): The optimizer with the loaded state.
        epoch (int): The epoch at which the checkpoint was saved.
        loss (float): The loss value from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    logger.info(f"Pretrained model loaded from {checkpoint_path}, epoch {epoch}, loss {loss:.4f}")
    return model, optimizer, epoch, loss
