import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.normalization import RMSNorm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from BitText.logs.logger import setup_logger
logger = setup_logger("Utils","logs","info")

def activation_quant(x: Tensor):
    """
    Per token quantization to 8bits. Scales activations to the range [-128, 127] 
    for more efficient representation.
    
    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Quantized tensor.
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: Tensor):
    """
    Quantizes weights by scaling and applying a sign function.

    Args:
        w (Tensor): The weight tensor.

    Returns:
        Tensor: Quantized weight tensor.
    """
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u


class BitLinear(nn.Linear):
    """
    Custom linear layer with bit quantization for weights and activations.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        training (bool, optional): Whether the layer is in training mode or not. Defaults to False.
        *args: Variable length argument list for additional parameters.
        **kwargs: Arbitrary keyword arguments for customization.
    """

    def __init__(self, in_features: int, out_features: int, *args, **kwargs):
        super(BitLinear, self).__init__(in_features, out_features, *args, **kwargs)
        
        # Create RMSNorm layer as a module parameter so it gets moved to device properly
        self.rms_norm = RMSNorm(in_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer, applying quantization to both activations and weights.

        Args:
            x (Tensor): The input tensor.
        
        Returns:
            Tensor: The output tensor after applying the quantized weights and activations.
        """
        # Apply RMS normalization to input tensor using the stored layer
        x_norm = self.rms_norm(x)

        # Apply activation quantization
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()

        # Apply weight quantization
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()

        # Perform the linear transformation using quantized values
        y = F.linear(x_quant, w_quant, self.bias)
        return y


class BitTransformerLayer(nn.Module):
    """
    A single transformer layer with BitLinear-based fully connected layers for BitNet models.

    Args:
        d_model (int): The dimension of the model (embedding size).
        nhead (int): Number of attention heads.
        dim_feedforward (int): Hidden dimension of the feedforward network.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super(BitTransformerLayer, self).__init__()

        # Multihead Self Attention
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        # Feedforward network with BitLinear layers
        self.ffn = nn.Sequential(
            BitLinear(d_model, dim_feedforward),
            nn.ReLU(),
            BitLinear(dim_feedforward, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the BitTransformer layer.

        Args:
            x (Tensor): Input tensor (batch, seq_len, d_model).
        
        Returns:
            Tensor: Output tensor after applying attention and binary feedforward layers.
        """
        # Multihead Self Attention
        attn_output, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.norm1(x)

        # Feedforward network (with BitLinear layers)
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)  # Residual connection
        x = self.norm2(x)

        return x


class BitNetTransformer(nn.Module):
    """
    BitNet-based Transformer model using BitLinear layers.

    Args:
        vocab_size (int): Size of the vocabulary.
        num_layers (int): Number of transformer layers.
        d_model (int): Dimension of the model (embedding size).
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
    """

    def __init__(self, vocab_size: int, num_layers: int = 2, d_model: int = 256, nhead: int = 4,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super(BitNetTransformer, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            BitTransformerLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])

        # Output layer (using BitLinear)
        self.fc_out = BitLinear(d_model, vocab_size)  # Final output layer to predict next token

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the entire BitNet Transformer model.

        Args:
            x (Tensor): Input tensor (batch_size, seq_len).
        
        Returns:
            Tensor: Output logits for the next token prediction.
        """
        # Embedding
        x = self.embedding(x)

        # Pass through each transformer layer
        for layer in self.transformer_layers:
            x = layer(x)

        # Output layer
        output = self.fc_out(x)
        return output