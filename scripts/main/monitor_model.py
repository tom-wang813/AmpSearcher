import pandas as pd
import os
import argparse
from typing import List, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from amp_searcher.utils.logging_utils import setup_logger
from amp_searcher.utils.config import Config
from amp_searcher.utils.performance_monitoring import timer, MemoryTracker

logger = setup_logger("monitor_model")

def calculate_data_drift_metrics(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict:
    drift_metrics = {}
    for col in reference_data.columns:
        if pd.api.types.is_numeric_dtype(reference_data[col]):
            drift_metrics[col] = {
                "reference_mean": reference_data[col].mean(),
                "current_mean": current_data[col].mean(),
                "reference_std": reference_data[col].std(),
                "current_std": current_data[col].std(),
            }
        else:
            ref_counts = reference_data[col].value_counts(normalize=True)
            curr_counts = current_data[col].value_counts(normalize=True)
            drift_metrics[col] = {
                "reference_value_counts": ref_counts.to_dict(),
                "current_value_counts": curr_counts.to_dict(),
            }
    return drift_metrics

def calculate_classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred
    return {
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, average="binary", zero_division=0),
        "f1_score": f1_score(y_true, y_pred_binary, average="binary", zero_division=0),
    }

@timer
def monitor_model(config: Config):
    try:
        reference_data_path = config.get("reference_data_path")
        current_data_path = config.get("current_data_path")
        output_dir = config.get("output_dir")
        model_type = config.get("model_type", "classification")
        target_column = config.get("target_column", "label")
        prediction_column = config.get("prediction_column", "prediction")

        logger.info("--- Starting Model Monitoring ---")
        logger.info(f"Loading reference data from {reference_data_path}")
        reference_data = pd.read_csv(reference_data_path)
        logger.info(f"Loading current data from {current_data_path}")
        current_data = pd.read_csv(current_data_path)

        with MemoryTracker("Data Drift Analysis"):
            logger.info("--- Data Drift Metrics ---")
            data_drift_metrics = calculate_data_drift_metrics(reference_data, current_data)
            for col, metrics in data_drift_metrics.items():
                logger.info(f"Column: {col}")
                for metric_name, value in metrics.items():
                    logger.info(f"  {metric_name}: {value}")

        logger.info("--- Model Performance Metrics ---")
        if model_type == "classification":
            if target_column not in current_data.columns or prediction_column not in current_data.columns:
                logger.warning(f"Skipping classification metrics: '{target_column}' or '{prediction_column}' column not found in current data.")
            else:
                with MemoryTracker("Classification Metrics Calculation"):
                    classification_metrics = calculate_classification_metrics(current_data[target_column], current_data[prediction_column])
                    for metric_name, value in classification_metrics.items():
                        logger.info(f"  {metric_name}: {value}")
        elif model_type == "regression":
            logger.warning("Regression metrics not yet implemented.")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        logger.info("--- Model Monitoring Complete ---")

    except Exception as e:
        logger.error(f"An error occurred during model monitoring: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor model performance and data drift.")
    parser.add_argument("--config_path", type=str, default="configs/monitor_model_config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    config = Config(args.config_path)
    monitor_model(config)
