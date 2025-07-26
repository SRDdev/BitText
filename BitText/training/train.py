import torch
import torch.nn as nn
import wandb
import os
import sys
import time
import json
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from torch.utils.data import DataLoader
from BitText.models.utils import save_model, compute_loss
from BitText.models.transformer import initialize_model
from BitText.data.dataset import create_data_loaders
from BitText.training.config import Config
from tqdm import tqdm
from BitText.logs.logger import setup_logger


#-----------------------------------------------------------------#
@dataclass
class TrainingMetrics:
    """Container for training metrics and statistics."""
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_perplexity: float = 0.0
    val_perplexity: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    epoch_time: float = 0.0
    tokens_per_second: float = 0.0
    gpu_memory_used: float = 0.0

#-----------------------------------------------------------------#
class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop early."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def restore_best_model(self, model: nn.Module) -> None:
        """Restore the best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

#-----------------------------------------------------------------#
class AdvancedScheduler:
    """Advanced learning rate scheduler with warmup and multiple strategies."""
    
    def __init__(self, optimizer, strategy: str = "cosine", warmup_steps: int = 1000, 
                 total_steps: int = 10000, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.strategy = strategy
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.initial_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
        if strategy == "cosine":
            self.scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)
        elif strategy == "plateau":
            self.scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    def step(self, val_loss: Optional[float] = None):
        """Step the scheduler."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Warmup phase
            lr = self.initial_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Main scheduling phase
            if self.strategy == "cosine":
                self.scheduler.step()
            elif self.strategy == "plateau" and val_loss is not None:
                self.scheduler.step(val_loss)
    
    def get_last_lr(self) -> float:
        """Get the current learning rate."""
        return self.optimizer.param_groups[0]['lr']

#-----------------------------------------------------------------#
def setup_directories(config: Config) -> None:
    """Create necessary directories for training artifacts."""
    directories = [
        config.checkpoints_dir,
        config.results_dir,
        "logs",
        "artifacts",
        "visualizations"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
#-----------------------------------------------------------------#
def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss."""
    return torch.exp(torch.tensor(loss)).item()

def count_tokens(batch: Dict[str, torch.Tensor]) -> int:
    """Count the number of tokens in a batch (excluding padding)."""
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        return attention_mask.sum().item()
    else:
        # Fallback: assume all tokens are valid
        return batch["input_ids"].numel()

def get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

def create_target_sequence(input_ids: torch.Tensor, shift: bool = True) -> torch.Tensor:
    """
    Create target sequence for language modeling.
    
    For causal language modeling, targets should be input_ids shifted by one position:
    Input:  [BOS, w1, w2, w3, EOS, PAD]
    Target: [w1,  w2, w3, EOS, PAD, PAD]
    """
    if shift:
        # Shift targets by one position for next token prediction
        targets = torch.cat([input_ids[:, 1:], torch.zeros_like(input_ids[:, :1])], dim=1)
    else:
        # Use same sequence (for debugging or specific tasks)
        targets = input_ids.clone()
    
    return targets

def compute_language_modeling_loss(outputs: torch.Tensor, targets: torch.Tensor, 
                                 criterion: nn.Module, ignore_index: int = -100) -> torch.Tensor:
    """
    Compute language modeling loss with proper masking.
    
    Args:
        outputs: Model predictions [batch_size, seq_len, vocab_size]
        targets: Target token ids [batch_size, seq_len]
        criterion: Loss function
        ignore_index: Token id to ignore in loss computation
    """
    # Flatten for loss computation
    outputs_flat = outputs.view(-1, outputs.size(-1))  # [batch_size * seq_len, vocab_size]
    targets_flat = targets.view(-1)                     # [batch_size * seq_len]
    
    # Compute loss
    loss = criterion(outputs_flat, targets_flat)
    
    return loss

#-----------------------------------------------------------------#
def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, scheduler: AdvancedScheduler, scaler: GradScaler,
                device: torch.device, config: Config, epoch: int) -> TrainingMetrics:
    """Train one epoch with advanced features."""
    
    logger = setup_logger("Trainer", "logs", "info")
    model.train()
    
    total_loss = 0.0
    total_tokens = 0
    gradient_norms = []
    batch_times = []
    
    epoch_start_time = time.time()
    
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", unit="batch")
    
    for batch_idx, batch in enumerate(progress_bar):
        batch_start_time = time.time()
        
        # Move data to device
        inputs = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
        
        # Create proper targets for language modeling
        targets = create_target_sequence(inputs, shift=config.shift_targets)
        targets = targets.to(device, non_blocking=True)
        
        # Count tokens for throughput calculation
        batch_tokens = count_tokens(batch)
        total_tokens += batch_tokens
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=config.use_amp):
            outputs = model(inputs)
            loss = compute_language_modeling_loss(
                outputs, targets, criterion, ignore_index=config.pad_token_id
            )
            
            # Scale loss if using gradient accumulation
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
        
        # Backward pass with mixed precision
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.max_grad_norm
            )
            gradient_norms.append(grad_norm.item())
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Scheduler step
            scheduler.step()
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{scheduler.get_last_lr():.2e}',
            'gpu_mem': f'{get_gpu_memory_usage():.0f}MB'
        })
        
        # Log metrics to W&B
        if batch_idx % config.log_interval == 0:
            current_lr = scheduler.get_last_lr()
            current_grad_norm = gradient_norms[-1] if gradient_norms else 0.0
            
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/learning_rate": current_lr,
                "train/gradient_norm": current_grad_norm,
                "train/epoch": epoch + 1,
                "train/batch": batch_idx,
                "train/tokens_per_second": batch_tokens / batch_time,
                "train/gpu_memory_mb": get_gpu_memory_usage(),
                "train/batch_time": batch_time
            })
    
    # Calculate epoch metrics
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / len(train_loader)
    avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0
    tokens_per_second = total_tokens / epoch_time
    
    metrics = TrainingMetrics(
        train_loss=avg_loss,
        train_perplexity=calculate_perplexity(avg_loss),
        learning_rate=scheduler.get_last_lr(),
        gradient_norm=avg_grad_norm,
        epoch_time=epoch_time,
        tokens_per_second=tokens_per_second,
        gpu_memory_used=get_gpu_memory_usage()
    )
    
    logger.info(f"Epoch {epoch+1} Training - Loss: {avg_loss:.4f}, "
                f"Perplexity: {metrics.train_perplexity:.2f}, "
                f"Tokens/sec: {tokens_per_second:.0f}")
    
    return metrics

def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
                  device: torch.device, config: Config, epoch: int) -> TrainingMetrics:
    """Validate the model with comprehensive metrics."""
    
    logger = setup_logger("Trainer", "logs", "info")
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", unit="batch")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move data to device
            inputs = batch["input_ids"].to(device, non_blocking=True)
            targets = create_target_sequence(inputs, shift=config.shift_targets)
            targets = targets.to(device, non_blocking=True)
            
            # Count tokens
            batch_tokens = count_tokens(batch)
            total_tokens += batch_tokens
            
            # Forward pass
            with autocast(enabled=config.use_amp):
                outputs = model(inputs)
                loss = compute_language_modeling_loss(
                    outputs, targets, criterion, ignore_index=config.pad_token_id
                )
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(val_loader)
    perplexity = calculate_perplexity(avg_loss)
    
    metrics = TrainingMetrics(
        val_loss=avg_loss,
        val_perplexity=perplexity,
        gpu_memory_used=get_gpu_memory_usage()
    )
    
    logger.info(f"Epoch {epoch+1} Validation - Loss: {avg_loss:.4f}, "
                f"Perplexity: {perplexity:.2f}")
    
    return metrics

def save_training_state(model: nn.Module, optimizer: torch.optim.Optimizer, 
                       scheduler: AdvancedScheduler, scaler: GradScaler,
                       epoch: int, metrics: TrainingMetrics, config: Config,
                       is_best: bool = False) -> str:
    """Save complete training state for resuming."""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': {
            'current_step': scheduler.current_step,
            'strategy': scheduler.strategy
        },
        'scaler_state_dict': scaler.state_dict(),
        'metrics': metrics.__dict__,
        'config': config.__dict__,
        'timestamp': time.time()
    }
    
    filename = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch+1}.pth"
    filepath = os.path.join(config.checkpoints_dir, filename)
    
    torch.save(checkpoint, filepath)
    
    return filepath

def load_training_state(filepath: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: AdvancedScheduler, scaler: GradScaler) -> Tuple[int, TrainingMetrics]:
    """Load training state for resuming."""
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    scheduler.current_step = checkpoint['scheduler_state_dict']['current_step']
    
    metrics = TrainingMetrics(**checkpoint['metrics'])
    epoch = checkpoint['epoch']
    
    return epoch, metrics

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
          optimizer: torch.optim.Optimizer, criterion: nn.Module, config: Config) -> None:
    """Main training loop with all advanced features."""
    
    logger = setup_logger("Trainer", "logs", "info")
    
    # Initialize advanced components
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = AdvancedScheduler(
        optimizer, 
        strategy=config.scheduler_strategy,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        min_lr=config.min_lr
    )
    
    scaler = GradScaler(enabled=config.use_amp)
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        restore_best_weights=True
    )
    
    # Print configuration
    config.print_config()
    
    best_val_loss = float('inf')
    training_history = []
    
    # Resume training if checkpoint exists
    resume_from_epoch = 0
    if config.resume_training and os.path.exists(os.path.join(config.checkpoints_dir, "latest_checkpoint.pth")):
        try:
            resume_from_epoch, _ = load_training_state(
                os.path.join(config.checkpoints_dir, "latest_checkpoint.pth"),
                model, optimizer, scheduler, scaler
            )
            logger.info(f"Resumed training from epoch {resume_from_epoch + 1}")
        except Exception as e:
            logger.warning(f"Failed to resume training: {e}")
    
    # Training loop
    for epoch in range(resume_from_epoch, config.num_epochs):
        logger.info(f"Starting Epoch {epoch+1}/{config.num_epochs}")
        
        # Training phase
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, scaler,
            config.device, config, epoch
        )
        
        # Validation phase
        val_metrics = validate_epoch(
            model, val_loader, criterion, config.device, config, epoch
        )
        
        # Combine metrics
        combined_metrics = TrainingMetrics(
            train_loss=train_metrics.train_loss,
            val_loss=val_metrics.val_loss,
            train_perplexity=train_metrics.train_perplexity,
            val_perplexity=val_metrics.val_perplexity,
            learning_rate=train_metrics.learning_rate,
            gradient_norm=train_metrics.gradient_norm,
            epoch_time=train_metrics.epoch_time,
            tokens_per_second=train_metrics.tokens_per_second,
            gpu_memory_used=val_metrics.gpu_memory_used
        )
        
        training_history.append(combined_metrics)
        
        # Log epoch metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train/epoch_loss": combined_metrics.train_loss,
            "train/epoch_perplexity": combined_metrics.train_perplexity,
            "val/epoch_loss": combined_metrics.val_loss,
            "val/epoch_perplexity": combined_metrics.val_perplexity,
            "train/learning_rate": combined_metrics.learning_rate,
            "train/gradient_norm": combined_metrics.gradient_norm,
            "train/epoch_time": combined_metrics.epoch_time,
            "train/tokens_per_second": combined_metrics.tokens_per_second,
            "system/gpu_memory_mb": combined_metrics.gpu_memory_used,
            "train/best_val_loss": best_val_loss
        })
        
        # Check for best model
        is_best = val_metrics.val_loss < best_val_loss
        if is_best:
            best_val_loss = val_metrics.val_loss
            logger.info(f"New best model! Validation loss: {best_val_loss:.4f}")
        
        # Save checkpoints
        if (epoch + 1) % config.model_save_freq == 0 or is_best:
            filepath = save_training_state(
                model, optimizer, scheduler, scaler, epoch, combined_metrics, config, is_best
            )
            logger.info(f"Checkpoint saved: {filepath}")
            
            # Save as W&B artifact
            if is_best:
                try:
                    artifact = wandb.Artifact(
                        name=f"bitnet-best-model-epoch-{epoch+1}", 
                        type="model",
                        description=f"Best BitNet model at epoch {epoch+1} with val_loss {best_val_loss:.4f}"
                    )
                    artifact.add_file(filepath)
                    wandb.log_artifact(artifact)
                    logger.info("Best model uploaded to W&B")
                except Exception as e:
                    logger.warning(f"Failed to upload model artifact: {e}")
        
        # Save latest checkpoint for resuming
        save_training_state(
            model, optimizer, scheduler, scaler, epoch, combined_metrics, config, False
        )
        os.rename(
            os.path.join(config.checkpoints_dir, f"checkpoint_epoch_{epoch+1}.pth"),
            os.path.join(config.checkpoints_dir, "latest_checkpoint.pth")
        )
        
        # Early stopping check
        if early_stopping(val_metrics.val_loss, model):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            early_stopping.restore_best_model(model)
            break
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    # Save training history
    history_path = os.path.join(config.results_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump([metrics.__dict__ for metrics in training_history], f, indent=2)
    
    logger.info("Training completed successfully!")

def setup_model_and_optimizer(config: Config) -> Tuple[nn.Module, torch.optim.Optimizer]:
    """Initialize model and optimizer with advanced configurations."""
    
    from transformers import AutoTokenizer
    logger = setup_logger("Trainer", "logs", "info")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        actual_vocab_size = len(tokenizer)
        config.pad_token_id = tokenizer.pad_token_id
        
        logger.info(f"Tokenizer loaded: {config.tokenizer}")
        logger.info(f"Vocabulary size: {actual_vocab_size}")
        logger.info(f"Pad token ID: {config.pad_token_id}")
        
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        actual_vocab_size = 30522
        config.pad_token_id = 0
        logger.warning(f"Using fallback vocabulary size: {actual_vocab_size}")
    
    # Initialize model
    model, _ = initialize_model(
        vocab_size=actual_vocab_size,
        num_layers=config.num_layers,
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout
    )
    
    # Advanced optimizer with proper weight decay
    no_decay = ["bias", "norm", "rms_norm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon
    )
    
    return model, optimizer

def setup_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    """Setup data loaders with advanced configurations."""
    
    train_loader, val_loader = create_data_loaders(
        dataset_name=config.dataset_name,
        tokenizer_name=config.tokenizer,
        max_length=config.max_sequence_length,
        batch_size=config.batch_size
    )
    
    return train_loader, val_loader

def main():
    """Production-ready main training function."""
    
    # Load configuration
    config = Config()
    
    # Setup directories
    setup_directories(config)
    
    # Setup logging
    logger = setup_logger("Main", "logs", "info")
    logger.info("Starting BitNet training pipeline")
    
    # Get tokenizer info for W&B config
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        actual_vocab_size = len(tokenizer)
    except Exception:
        actual_vocab_size = 30522
    
    # Initialize W&B with comprehensive config
    wandb.init(
        project=config.project_name,
        entity=config.entity,
        name=f"bitnet-{config.d_model}d-{config.num_layers}l-{int(time.time())}",
        config={
            # Model architecture
            "vocab_size": actual_vocab_size,
            "d_model": config.d_model,
            "nhead": config.nhead,
            "num_layers": config.num_layers,
            "dim_feedforward": config.dim_feedforward,
            "dropout": config.dropout,
            
            # Training hyperparameters
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "weight_decay": config.weight_decay,
            "max_grad_norm": config.max_grad_norm,
            
            # Advanced settings
            "scheduler_strategy": config.scheduler_strategy,
            "warmup_steps": config.warmup_steps,
            "use_amp": config.use_amp,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            
            # System info
            "device": str(config.device),
            "tokenizer": config.tokenizer,
            "dataset": config.dataset_name
        },
        tags=["bitnet", "transformer", "language-model"]
    )
    
    # Initialize model and optimizer
    logger.info("Initializing model and optimizer")
    model, optimizer = setup_model_and_optimizer(config)
    model = model.to(config.device)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    # Setup W&B model watching
    wandb.watch(model, log="all", log_freq=config.log_interval)
    
    # Log model statistics
    wandb.log({
        "model/total_params": total_params,
        "model/trainable_params": trainable_params,
        "model/size_mb": total_params * 4 / 1024 / 1024
    })
    
    # Setup data loaders
    logger.info("Setting up data loaders")
    train_loader, val_loader = setup_data_loaders(config)
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")
    
    # Setup loss function
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
    
    # Start training
    logger.info("Starting training")
    try:
        train(model, train_loader, val_loader, optimizer, criterion, config)
        
        wandb.log({
            "training_completed": True,
            "final_model_saved": True
        })
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        wandb.log({
            "training_failed": True,
            "error_message": str(e)
        })
        raise
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()