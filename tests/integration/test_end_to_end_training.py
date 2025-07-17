import subprocess
import sys
import os
import shutil

def test_end_to_end_training_run():
    """
    Tests the full end-to-end training pipeline by running the main train.py script.
    It uses a dedicated lightweight config file to ensure the test is fast.
    """
    config_path = "configs/test_e2e_training.yaml"
    log_dir = "lightning_logs/e2e_test_run"

    # Ensure the log directory is clean before the test
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    # Command to execute the training script
    command = [
        sys.executable, # Use the same python interpreter that runs pytest
        "train.py",
        "--config_path",
        config_path
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Assert that the script ran successfully
    assert result.returncode == 0, f"Training script failed with exit code {result.returncode}\nStderr: {result.stderr}"

    # Assert that the log directory and some log files were created
    assert os.path.isdir(log_dir), f"Log directory not found at {log_dir}"
    
    # Check for the presence of a version directory (e.g., 'version_0')
    version_dirs = [d for d in os.listdir(log_dir) if d.startswith('version_')]
    assert len(version_dirs) > 0, "No version directory found in the log directory."
    
    # Clean up the created log directory after the test
    shutil.rmtree(log_dir)

