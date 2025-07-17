from typing import Any, Dict, List

import torch

from amp_searcher.featurizers.featurizer_factory import FeaturizerFactory
from amp_searcher.models.model_factory import ModelFactory
from amp_searcher.models.lightning_module_factory import LightningModuleFactory
from amp_searcher.training.trainer import AmpTrainer  # Import AmpTrainer


class ScreeningPipeline:
    """
    A pipeline for performing virtual screening of antimicrobial peptides.

    This class handles loading a trained screening model and featurizing
    new sequences for prediction.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        model_checkpoint_path: str | None = None,
        featurizer_config: Dict[str, Any] | None = None,
        mlflow_model_uri: str | None = None,
    ):
        """
        Initializes the screening pipeline.

        Args:
            model_config: Configuration dictionary for the model architecture.
            model_checkpoint_path: Path to the trained PyTorch Lightning model checkpoint (optional).
            featurizer_config: Configuration dictionary for the featurizer.
            mlflow_model_uri: MLflow URI for the trained model (e.g., "models:/<model_name>/<version_or_stage>") (optional).
        """
        if featurizer_config is None:
            raise ValueError("featurizer_config cannot be None.")
        self.featurizer = self._init_featurizer(featurizer_config)
        self.model = self._load_model(
            model_config, model_checkpoint_path, mlflow_model_uri
        )
        self.model.eval()  # Set model to evaluation mode

    def _init_featurizer(self, featurizer_config: Dict[str, Any]):
        name = str(featurizer_config.get("name"))
        params = featurizer_config.get("params", {})
        return FeaturizerFactory.build_featurizer(name, **params)

    def _load_model(
        self,
        model_config: Dict[str, Any],
        model_checkpoint_path: str | None,
        mlflow_model_uri: str | None,
    ):
        if mlflow_model_uri:
            # Load model from MLflow
            return AmpTrainer.load_model(mlflow_model_uri)
        elif model_checkpoint_path:
            # Fallback to loading from local checkpoint
            model_architecture_name = model_config.get("architecture", {}).get("name")
            model_architecture_params = model_config.get("architecture", {}).get("params", {})
            task_type = model_config.get("lightning_module_params", {}).get("task_type")
            lightning_module_name = str(
                model_config.get("lightning_module_name", "ScreeningLightningModule")
            )

            # Build the underlying model architecture using the factory
            model_architecture = ModelFactory.build_model(model_architecture_name, **model_architecture_params)

            # Initialize the LightningModule with the built architecture using the factory
            lightning_model = LightningModuleFactory.build_lightning_module(
                lightning_module_name,
                model_architecture=model_architecture,
                task_type=task_type,
                optimizer_params=model_config.get("optimizer_params"),
                scheduler_params=model_config.get("scheduler_params"),
            )
            # Load state_dict from checkpoint
            checkpoint = torch.load(
                model_checkpoint_path,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
            lightning_model.load_state_dict(checkpoint["state_dict"])
            return lightning_model
        else:
            raise ValueError(
                "Either model_checkpoint_path or mlflow_model_uri must be provided."
            )

    def predict(self, sequences: List[str]) -> torch.Tensor:
        """
        Predicts the class probabilities for a list of protein sequences.

        Args:
            sequences: A list of protein sequences.

        Returns:
            A torch.Tensor containing the predicted probabilities (for classification)
            or values (for regression).
        """
        if not sequences:
            return torch.tensor([])

        import numpy as np

        features = [self.featurizer.featurize(seq) for seq in sequences]
        X = torch.tensor(np.array(features), dtype=torch.float32)

        with torch.no_grad():
            logits: torch.Tensor = self.model(X)
            if self.model.task_type == "classification":
                probabilities = torch.sigmoid(logits)
                return probabilities
            else:  # regression
                return logits
