"""
Performance monitoring utilities for AmpSearcher.
"""

import time
import psutil
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)

def timer(func: Callable) -> Callable:
    """A decorator that prints the execution time for the decorated function."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class MemoryTracker:
    """A context manager for tracking memory usage."""
    def __init__(self, label: str):
        self.label = label
        self.process = psutil.Process()

    def __enter__(self):
        self.start_memory = self.process.memory_info().rss
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_memory = self.process.memory_info().rss
        memory_used = self.end_memory - self.start_memory
        logger.info(f"Memory used in {self.label}: {memory_used / 1024 / 1024:.2f} MB")
