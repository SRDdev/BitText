import torch
import wandb
from torch.utils.data import DataLoader
from models.utils import compute_loss
from models.transformer import BitNetTextModel
from data.dataset import CustomTextDataset
from training.config import Config
import logging

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from BitText.logs.logger import setup_logger
logger = setup_logger("Eval","logs","info")

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on a dataset.

    Args:
        model (nn.Module): The trained model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        criterion (nn.Module): The loss function to use.

    Returns:
        float: Average loss over the evaluation dataset.
        float: Accuracy over the evaluation dataset.
    """
    model.eval()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = compute_loss(outputs, targets, criterion)
            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, dim=-1)
            correct_preds += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_preds / total_samples * 100

    logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Log metrics to W&B
    wandb.log({"eval_loss": avg_loss, "eval_accuracy": accuracy})

    return avg_loss, accuracy


def main():
    # Initialize W&B
    wandb.init(project="BitText", entity="your-wandb-username", job_type="evaluation")

    # Load configuration
    config = Config()

    # Initialize model
    model = BitNetTextModel(vocab_size=config.vocab_size, num_layers=config.num_layers, 
                            d_model=config.d_model, nhead=config.nhead, 
                            dim_feedforward=config.dim_feedforward, dropout=config.dropout)
    model = model.to(config.device)

    # Load evaluation data
    eval_data = CustomTextDataset(config.val_data_path, tokenizer=config.tokenizer)
    eval_loader = DataLoader(eval_data, batch_size=config.batch_size, shuffle=False)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Evaluate the model
    eval_loss, eval_accuracy = evaluate_model(model, eval_loader, criterion, config.device)

    # Finish W&B logging
    wandb.finish()


if __name__ == "__main__":
    main()
