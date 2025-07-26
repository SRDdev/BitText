import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from BitText.logs.logger import setup_logger
logger = setup_logger("Utils","logs","info")

def save_model(model, optimizer, epoch, loss, save_dir="models", filename="model_checkpoint.pth"):
    """
    Save the model checkpoint along with optimizer state and training progress.
    
    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state.
        epoch (int): The current training epoch.
        loss (float): The current training loss.
        save_dir (str): Directory to save the checkpoint.
        filename (str): The filename to save the model as.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved at epoch {epoch} to {save_path}")


def load_model(model, optimizer, load_dir="models", filename="model_checkpoint.pth"):
    """
    Load a model checkpoint and optimizer state.
    
    Args:
        model (nn.Module): The model to load the weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        load_dir (str): Directory where the checkpoint is saved.
        filename (str): The filename of the model checkpoint.
    
    Returns:
        model (nn.Module): The model with loaded weights.
        optimizer (torch.optim.Optimizer): The optimizer with loaded state.
        epoch (int): The epoch at which training was last saved.
        loss (float): The loss value from the checkpoint.
    """
    checkpoint_path = os.path.join(load_dir, filename)

    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
        return model, optimizer, 0, None

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    logger.info(f"Loaded model checkpoint from {checkpoint_path}, epoch {epoch}, loss {loss:.4f}")
    return model, optimizer, epoch, loss


def compute_loss(output, target, criterion=torch.nn.CrossEntropyLoss()):
    """
    Compute the loss between the model output and the target.

    Args:
        output (Tensor): Model predictions (logits).
        target (Tensor): Ground truth labels.
        criterion (nn.Module): The loss function to use. Default is CrossEntropyLoss.

    Returns:
        Tensor: Computed loss value.
    """
    loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
    return loss


def evaluate_model(model, data_loader, criterion):
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
    total_loss = 0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["input_ids"]
            targets = batch["labels"]
            outputs = model(inputs)

            loss = compute_loss(outputs, targets, criterion)
            total_loss += loss.item()

            # Compute accuracy (for classification tasks)
            _, predicted = torch.max(outputs, dim=-1)
            correct_preds += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_preds / total_samples * 100

    logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def adjust_learning_rate(optimizer, epoch, initial_lr=1e-3, lr_decay=0.1, lr_decay_epoch=10):
    """
    Adjust learning rate based on epoch number.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate to adjust.
        epoch (int): The current epoch.
        initial_lr (float): The initial learning rate. Default is 1e-3.
        lr_decay (float): Factor by which to decay the learning rate. Default is 0.1.
        lr_decay_epoch (int): The number of epochs after which to decay the learning rate. Default is 10.
    """
    if epoch % lr_decay_epoch == 0:
        new_lr = initial_lr * (lr_decay ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        logger.info(f"Learning rate adjusted to {new_lr:.6f}")


def save_results(results, result_dir="results"):
    """
    Save evaluation results to disk.

    Args:
        results (dict): Dictionary containing the results to save.
        result_dir (str): Directory to save the results.
    """
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = os.path.join(result_dir, "evaluation_results.txt")
    with open(result_path, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Evaluation results saved to {result_path}")
