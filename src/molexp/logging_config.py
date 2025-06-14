"""
Logging configuration for MolExp.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", logger_name: str = "molexp") -> logging.Logger:
    """
    Setup logging configuration for MolExp.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Name of the logger
        
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(logger_name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Set level
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str, parent: Optional[str] = "molexp") -> logging.Logger:
    """
    Get a child logger with the given name.
    
    Args:
        name: Name of the child logger
        parent: Parent logger name
        
    Returns:
        Child logger instance
    """
    if parent:
        full_name = f"{parent}.{name}"
    else:
        full_name = name
    
    return logging.getLogger(full_name)


# Initialize the main logger
main_logger = setup_logging()
