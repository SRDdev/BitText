import torch
from typing import Union

class Config:
    """
    Production-ready configuration class for BitNet transformer training.
    Contains all hyperparameters, paths, and advanced training settings.
    """
    
    def __init__(self):
        # =================================================================
        # MODEL ARCHITECTURE
        # =================================================================
        self.d_model = 256                    # Hidden dimension size (embedding size)
        self.nhead = 4                        # Number of attention heads
        self.num_layers = 2                   # Number of transformer layers
        self.dim_feedforward = 1024           # Feedforward network dimension
        self.dropout = 0.1                    # Dropout rate
        self.max_sequence_length = 512        # Maximum sequence length
        
        # =================================================================
        # TRAINING HYPERPARAMETERS
        # =================================================================
        self.batch_size = 32                   # Batch size per GPU
        self.gradient_accumulation_steps = 4  # Effective batch size = batch_size * gradient_accumulation_steps
        self.learning_rate = 5e-5             # Peak learning rate
        self.min_lr = 1e-6                    # Minimum learning rate for scheduler
        self.weight_decay = 0.01              # Weight decay for regularization
        self.num_epochs = 100                  # Maximum number of epochs
        
        # =================================================================
        # OPTIMIZER SETTINGS
        # =================================================================
        self.adam_beta1 = 0.9                 # Adam beta1 parameter
        self.adam_beta2 = 0.999               # Adam beta2 parameter
        self.adam_epsilon = 1e-8              # Adam epsilon for numerical stability
        
        # =================================================================
        # LEARNING RATE SCHEDULER
        # =================================================================
        self.scheduler_strategy = "cosine"    # Scheduler type: "cosine", "plateau", "linear"
        self.warmup_steps = 1000              # Number of warmup steps
        self.warmup_ratio = 0.1               # Warmup steps as ratio of total steps (alternative to warmup_steps)
        
        # =================================================================
        # REGULARIZATION & STABILITY
        # =================================================================
        self.max_grad_norm = 1.0              # Gradient clipping norm
        self.early_stopping_patience = 7     # Early stopping patience
        self.early_stopping_min_delta = 1e-4 # Minimum improvement for early stopping
        self.label_smoothing = 0.1            # Label smoothing factor
        
        # =================================================================
        # MIXED PRECISION & PERFORMANCE
        # =================================================================
        self.use_amp = True                   # Use Automatic Mixed Precision
        self.compile_model = False            # Use torch.compile (PyTorch 2.0+)
        self.dataloader_num_workers = 4       # Number of dataloader workers
        self.pin_memory = True                # Pin memory for faster GPU transfer
        
        # =================================================================
        # LANGUAGE MODELING SPECIFIC
        # =================================================================
        self.shift_targets = True             # Shift targets by one for next token prediction
        self.tie_word_embeddings = False      # Tie input/output embeddings
        self.vocab_size = None                # Will be set dynamically from tokenizer
        self.pad_token_id = 0                 # Padding token ID (will be set from tokenizer)
        
        # =================================================================
        # DATA & TOKENIZATION
        # =================================================================
        self.dataset_name = "iohadrubin/wikitext-103-raw-v1"  # HuggingFace dataset name
        self.tokenizer = "bert-base-uncased"  # Tokenizer to use
        self.preprocessing_num_workers = 8    # Workers for data preprocessing
        self.cache_dir = "./cache"            # Cache directory for datasets
        
        # =================================================================
        # DEVICE & DISTRIBUTED TRAINING
        # =================================================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ddp = False                  # Use Distributed Data Parallel
        self.local_rank = -1                  # Local rank for distributed training
        self.world_size = 1                   # World size for distributed training
        
        # =================================================================
        # LOGGING & MONITORING
        # =================================================================
        self.project_name = "bitvision"  # W&B project name
        self.entity = "srddev"                # W&B entity/username
        self.experiment_name = None           # Experiment name (auto-generated if None)
        self.log_interval = 50                # Logging interval (batches)
        self.eval_interval = 500              # Evaluation interval (batches)
        self.log_gradients = True             # Log gradients to W&B
        self.log_learning_rate = True         # Log learning rate
        self.log_gpu_memory = True            # Log GPU memory usage
        
        # =================================================================
        # CHECKPOINTING & SAVING
        # =================================================================
        self.checkpoints_dir = "training/checkpoints"
        self.results_dir = "results"
        self.logs_dir = "logs"
        self.model_save_freq = 2              # Save model every N epochs
        self.keep_last_n_checkpoints = 5      # Keep only last N checkpoints
        self.save_optimizer_state = True      # Save optimizer state in checkpoints
        self.resume_training = True           # Resume from latest checkpoint if available
        
        # =================================================================
        # VALIDATION & EVALUATION
        # =================================================================
        self.eval_batch_size = None           # Evaluation batch size (uses train batch_size if None)
        self.eval_accumulation_steps = 1      # Gradient accumulation for evaluation
        self.compute_metrics_on_train = False # Compute detailed metrics on training set
        self.save_predictions = False         # Save model predictions for analysis
        
        # =================================================================
        # QUANTIZATION SPECIFIC (BitNet)
        # =================================================================
        self.quantization_bits = 1            # Number of bits for quantization
        self.quantization_scheme = "sign"     # Quantization scheme: "sign", "linear"
        self.activation_quantization = True   # Quantize activations
        self.weight_quantization = True       # Quantize weights
        
        # =================================================================
        # DEBUGGING & DEVELOPMENT
        # =================================================================
        self.debug_mode = False               # Enable debug mode
        self.profile_model = False            # Profile model performance
        self.detect_anomaly = False           # Detect autograd anomalies
        self.deterministic = False            # Use deterministic algorithms
        self.seed = 42                        # Random seed
        
        # =================================================================
        # ADVANCED FEATURES
        # =================================================================
        self.use_flash_attention = False      # Use Flash Attention (if available)
        self.use_gradient_checkpointing = False  # Use gradient checkpointing to save memory
        self.freeze_embeddings = False        # Freeze embedding layers
        self.freeze_first_n_layers = 0        # Freeze first N transformer layers
        
        # =================================================================
        # LOSS FUNCTION SETTINGS
        # =================================================================
        self.loss_function = "cross_entropy"  # Loss function type
        self.ignore_index = -100              # Index to ignore in loss computation
        self.reduction = "mean"               # Loss reduction method
        
        # =================================================================
        # VALIDATION SETTINGS
        # =================================================================
        self.validate_every_n_steps = None    # Validate every N steps (None = end of epoch)
        self.validation_split_ratio = 0.1     # Validation split ratio if no val set provided
        self.stratify_validation = False      # Stratify validation split
        
        # Set eval batch size if not specified
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size
        
        # Set ignore index from pad token
        self.ignore_index = self.pad_token_id
        
        # Auto-calculate total steps for warmup
        self.total_training_steps = None      # Will be calculated during training
        
        # Set experiment name if not provided
        if self.experiment_name is None:
            import time
            self.experiment_name = f"bitnet-{self.d_model}d-{self.num_layers}l-{int(time.time())}"
    
    def print_config(self):
        """Print the configuration in a formatted way."""
        print("\n" + "="*80)
        print("🚀 BITNET TRANSFORMER TRAINING CONFIGURATION")
        print("="*80)
        
        sections = {
            "Model Architecture": [
                "d_model", "nhead", "num_layers", "dim_feedforward", 
                "dropout", "max_sequence_length"
            ],
            "Training Hyperparameters": [
                "batch_size", "gradient_accumulation_steps", "learning_rate", 
                "weight_decay", "num_epochs"
            ],
            "Optimizer & Scheduler": [
                "adam_beta1", "adam_beta2", "scheduler_strategy", 
                "warmup_steps", "max_grad_norm"
            ],
            "Performance & Precision": [
                "use_amp", "compile_model", "dataloader_num_workers", 
                "pin_memory"
            ],
            "Data & Tokenization": [
                "dataset_name", "tokenizer", "shift_targets"
            ],
            "Logging & Monitoring": [
                "project_name", "log_interval", "log_gradients"
            ],
            "Checkpointing": [
                "checkpoints_dir", "model_save_freq", "resume_training"
            ],
            "System": [
                "device", "use_ddp", "world_size"
            ]
        }
        
        for section_name, params in sections.items():
            print(f"\n📋 {section_name}:")
            print("-" * 40)
            for param in params:
                if hasattr(self, param):
                    value = getattr(self, param)
                    print(f"  {param:<25}: {value}")
        
        print("\n" + "="*80 + "\n")
    
    def to_dict(self):
        """Convert config to dictionary for W&B logging."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
    
    def save_config(self, filepath: str):
        """Save configuration to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from JSON file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def validate_config(self):
        """Validate configuration parameters."""
        assert self.d_model % self.nhead == 0, "d_model must be divisible by nhead"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.num_epochs > 0, "num_epochs must be positive"
        assert self.max_grad_norm > 0, "max_grad_norm must be positive"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        
        print("✅ Configuration validation passed!")
    
    def update_from_args(self, args):
        """Update config from command line arguments."""
        for key, value in vars(args).items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
    
    def get_effective_batch_size(self):
        """Get the effective batch size considering gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps * self.world_size
    
    def get_total_steps(self, num_training_samples: int):
        """Calculate total training steps."""
        steps_per_epoch = num_training_samples // self.get_effective_batch_size()
        return steps_per_epoch * self.num_epochs
    
    def setup_for_distributed(self, local_rank: int, world_size: int):
        """Setup configuration for distributed training."""
        self.local_rank = local_rank
        self.world_size = world_size
        self.use_ddp = world_size > 1
        self.device = torch.device(f"cuda:{local_rank}")
        
        # Adjust batch size for distributed training
        if self.use_ddp:
            self.batch_size = self.batch_size // world_size
    
    def enable_debug_mode(self):
        """Enable debug mode with appropriate settings."""
        self.debug_mode = True
        self.detect_anomaly = True
        self.deterministic = True
        self.log_interval = 1
        self.model_save_freq = 1
        self.num_epochs = min(self.num_epochs, 2)
        print("🐛 Debug mode enabled!")
    
    def enable_production_mode(self):
        """Enable production mode with optimized settings."""
        self.use_amp = True
        self.compile_model = True
        self.pin_memory = True
        self.dataloader_num_workers = 8
        self.gradient_accumulation_steps = max(4, self.gradient_accumulation_steps)
        print("🚀 Production mode enabled!")