import torch
import numpy as np
from typing import Dict, Any, Optional
import logging
import random
import yaml


logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


LOG_FORMAT = "%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s"


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Thiết lập logging.

    Args:
        log_file: Đường dẫn đến file log (optional)
        level: Mức độ logging
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=handlers,
    )


def switch_log_file(new_log_file: str) -> Optional[str]:
    """
    Thay đổi file handler của root logger sang file mới.

    Args:
        new_log_file: Đường dẫn đến file log mới

    Returns:
        old_log_file: Đường dẫn file log cũ (hoặc None nếu không có)
    """
    root_logger = logging.getLogger()
    old_log_file = None

    # Tìm và xóa FileHandler cũ
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            old_log_file = handler.baseFilename
            handler.close()
            root_logger.removeHandler(handler)

    # Thêm FileHandler mới
    new_handler = logging.FileHandler(new_log_file)
    new_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(new_handler)

    return old_log_file


def log_separator():
    """Write a blank line directly to all log handler streams (no metadata)."""
    root_handlers = logging.getLogger().handlers
    own_handlers = logger.handlers
    for handler in root_handlers + own_handlers:
        if hasattr(handler, "stream"):
            handler.stream.write("\n")
            handler.flush()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        config: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Lưu configuration ra file YAML.

    Args:
        config: Dictionary chứa configuration
        config_path: Đường dẫn để lưu file
    """
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Config saved to {config_path}")
