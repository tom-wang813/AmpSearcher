"""
Logging utilities for AmpSearcher.
"""

import logging
import sys
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO, 
                 log_file: Optional[str] = None, 
                 format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    """Set up a logger with the given name and configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(format)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Usage example:
# logger = setup_logger('amp_searcher', level=logging.DEBUG, log_file='amp_searcher.log')
# logger.debug('This is a debug message')
# logger.info('This is an info message')
# logger.warning('This is a warning message')
# logger.error('This is an error message')
