"""Utility functions for reproducible experiments and device management."""

import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    return OmegaConf.load(config_path)


def save_config(config: Union[Dict[str, Any], DictConfig], config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary or OmegaConf object.
        config_path: Path to save configuration file.
    """
    OmegaConf.save(config, config_path)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    import logging
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def print_system_info() -> None:
    """Print system information for debugging."""
    print("System Information:")
    print(f"Python version: {torch.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    print(f"Device: {get_device()}")


def validate_config(config: DictConfig) -> None:
    """Validate configuration parameters.
    
    Args:
        config: Configuration to validate.
        
    Raises:
        ValueError: If configuration is invalid.
    """
    required_keys = ["data", "model", "training", "evaluation"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate data configuration
    if config.data.synthetic_size <= 0:
        raise ValueError("synthetic_size must be positive")
    
    # Validate model configuration
    valid_models = ["logistic", "xgboost", "lightgbm", "deep"]
    if config.model.type not in valid_models:
        raise ValueError(f"Invalid model type: {config.model.type}. Must be one of {valid_models}")
    
    # Validate training configuration
    if config.training.cv_folds <= 0:
        raise ValueError("cv_folds must be positive")
    
    if not 0 < config.training.test_size < 1:
        raise ValueError("test_size must be between 0 and 1")


def create_experiment_name(config: DictConfig) -> str:
    """Create experiment name from configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Experiment name string.
    """
    model_type = config.model.type
    data_size = config.data.synthetic_size
    timestamp = torch.cuda.Event(enable_timing=True).elapsed_time() if torch.cuda.is_available() else "cpu"
    
    return f"{model_type}_size{data_size}_{timestamp}"


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path.
    """
    import os
    os.makedirs(path, exist_ok=True)
