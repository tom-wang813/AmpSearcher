# AmpSearcher Utilities

This directory contains utility modules that provide essential functionality for the AmpSearcher project.

## Contents

1. `config.py`: Configuration management utilities
   - Provides a `Config` class for loading, accessing, and saving configuration data from YAML files.

2. `logging_utils.py`: Logging utilities
   - Offers a `setup_logger` function to easily configure logging for different parts of the project.

3. `performance_monitoring.py`: Performance monitoring tools
   - Includes a `timer` decorator for measuring function execution time.
   - Provides a `MemoryTracker` context manager for tracking memory usage of code blocks.

4. `constants.py`: Constant values used throughout the project

5. `sequence_decoder.py` and `sequence_decoder_factory.py`: Utilities for decoding sequences

## Usage

To use these utilities in your AmpSearcher modules, import them as needed. For example:

```python
from amp_searcher.utils.config import Config
from amp_searcher.utils.logging_utils import setup_logger
from amp_searcher.utils.performance_monitoring import timer, MemoryTracker

# Set up configuration
config = Config('path/to/config.yaml')

# Set up logging
logger = setup_logger('my_module')

# Use performance monitoring
@timer
def my_function():
    # Function code here
    pass

with MemoryTracker("data processing"):
    # Code block to track memory usage
    pass
```

For more detailed usage instructions, refer to the docstrings in each utility module.
