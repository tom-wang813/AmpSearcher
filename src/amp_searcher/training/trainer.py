from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader  # Add Dataset, DataLoader
import pandas as pd  # Add pandas
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger, Logger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

import mlflow

from amp_searcher.featurizers.featurizer_factory import FeaturizerFactory
from amp_searcher.models.model_factory import ModelFactory
from amp_searcher.models.lightning_module_factory import LightningModuleFactory
from amp_searcher.training.callbacks import GradientMonitor


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[Any], featurizer: Any):
        self.sequences = sequences
        self.labels = labels
        self.featurizer = featurizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        features = self.featurizer.featurize(sequence)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )


class ContrastiveDataset(Dataset):
    def __init__(self, sequences: List[str], featurizer: Any):
        self.sequences = sequences
        self.featurizer = featurizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # For contrastive learning, we create two "views" of the same sequence.
        # In this simple case, we featurize it twice. In a real-world scenario,
        # you might apply different augmentations to the sequence string itself.
        x_i = self.featurizer.featurize(sequence)
        x_j = self.featurizer.featurize(sequence) # Simple second view
        return torch.tensor(x_i, dtype=torch.float32), torch.tensor(x_j, dtype=torch.float32)


class AmpTrainer:
    """
    A unified trainer class for Antimicrobial Peptide (AMP) models.
    This class handles model instantiation, data preparation (featurization),
    and the training loop using PyTorch Lightning.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mlflow_config = config.get("mlflow", {})
        if self.mlflow_config:
            mlflow.set_tracking_uri(self.mlflow_config.get("tracking_uri", "mlruns"))
            mlflow.set_experiment(
                self.mlflow_config.get("experiment_name", "AmpSearcher_Experiment")
            )

        self.featurizer = self._init_featurizer(config.get("featurizer", {}))
        self.model = self._init_model(
            config.get("model", {}), self.featurizer.feature_dim
        )  # Initialize model here
        self.trainer = self._init_lightning_trainer(config.get("trainer", {}))

    def _init_featurizer(self, featurizer_config: Dict[str, Any]):
        name = str(featurizer_config.get("name"))
        params = featurizer_config.get("params", {})
        featurizer = FeaturizerFactory.build_featurizer(name, **params)
        # Featurizer should now have a feature_dim attribute
        if not hasattr(featurizer, "feature_dim"):
            raise AttributeError(
                f"Featurizer {name} does not have a 'feature_dim' attribute. Please ensure it's defined in the featurizer's __init__."
            )
        return featurizer

    def _init_model(self, model_config: Dict[str, Any], input_dim: int):
        lightning_module_name = str(
            model_config.get("lightning_module_name", model_config.get("type"))
        )
        model_architecture_name = model_config.get("architecture", {}).get("name")
        model_architecture_params = model_config.get("architecture", {}).get(
            "params", {}
        )
        lightning_module_params = model_config.get("lightning_module_params", {})

        # Dynamically set the input dimension
        model_architecture_params["input_dim"] = input_dim

        if (
            "latent_dim" not in lightning_module_params
            and "latent_dim" in model_architecture_params
        ):
            lightning_module_params["latent_dim"] = model_architecture_params[
                "latent_dim"
            ]

        if not model_architecture_name:
            raise ValueError(
                "Model architecture name is required in model configuration."
            )

        try:
            core_model = ModelFactory.build_model(
                model_architecture_name, **model_architecture_params
            )
        except ValueError as e:
            raise ValueError(f"Error building model architecture: {e}") from e

        try:
            if lightning_module_name == "ContrastiveLightningModule":
                return LightningModuleFactory.build_lightning_module(
                    lightning_module_name,
                    model_architecture=core_model,
                    config=lightning_module_params, # Pass as config dict
                )
            else:
                return LightningModuleFactory.build_lightning_module(
                    lightning_module_name,
                    model_architecture=core_model,
                    **lightning_module_params,
                )
        except ValueError as e:
            raise ValueError(f"Error building LightningModule: {e}") from e

    def _init_lightning_trainer(self, trainer_config: Dict[str, Any]):
        logger_name = trainer_config.get("logger_name", "amp_logs")
        log_dir = trainer_config.get("log_dir", "./lightning_logs")

        callbacks: List[Callback] = []
        loggers: List[Logger] = []

        # Add TensorBoard Logger
        loggers.append(TensorBoardLogger(save_dir=log_dir, name=logger_name))

        # Add MLflow Logger if configured
        if self.mlflow_config:
            loggers.append(
                MLFlowLogger(
                    experiment_name=self.mlflow_config.get(
                        "experiment_name", "AmpSearcher_Experiment"
                    ),
                    tracking_uri=self.mlflow_config.get("tracking_uri", "mlruns"),
                    log_model=False,  # <-- 修改這裡，禁用自動記錄模型
                )
            )

        # Add Gradient Monitor Callback if enabled in config
        if trainer_config.get("monitor_gradients", False):
            callbacks.append(GradientMonitor())

        # Handle other callbacks, specifically ModelCheckpoint
        if "callbacks" in trainer_config:
            for callback_config in trainer_config["callbacks"]:
                if callback_config.get("_target_") == "pytorch_lightning.callbacks.ModelCheckpoint":
                    # Extract ModelCheckpoint specific parameters
                    checkpoint_params = {k: v for k, v in callback_config.items() if k != "_target_"}
                    callbacks.append(ModelCheckpoint(**checkpoint_params))
                # Add other callback types here if needed

        # Extract other trainer parameters
        trainer_params = {
            k: v
            for k, v in trainer_config.items()
            if k
            not in [
                "logger_name",
                "log_dir",
                "monitor_gradients",
                "batch_size",
                "callbacks", # Exclude 'callbacks' key from trainer_params
            ]
        }

        return Trainer(logger=loggers, callbacks=callbacks, **trainer_params)

    def _init_dataloader(self, data_config: Dict[str, Any]) -> DataLoader:
        data_path = data_config.get("path")
        sequence_col = data_config.get("sequence_col", "sequence")
        label_col = data_config.get("label_col", "label")
        batch_size = data_config.get("batch_size", 32)

        if not data_path:
            raise ValueError("Data path is required in data configuration.")

        df = pd.read_csv(data_path)
        sequences = df[sequence_col].tolist()
        if self.config.get("model", {}).get("type") == "contrastive":
            dataset = ContrastiveDataset(sequences, self.featurizer)
        else:
            labels = df[label_col].tolist() if label_col in df.columns else [0] * len(sequences)
            dataset = SequenceDataset(sequences, labels, self.featurizer)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, data_config: Dict[str, Any]):
        """
        Starts the training process.

        Args:
            data_config: Configuration for data loading (path, sequence_col, label_col, batch_size).
        """
        dataloader = self._init_dataloader(data_config)

        if self.mlflow_config:
            with mlflow.start_run(run_name=self.mlflow_config.get("run_name")):
                # Log parameters from the config
                mlflow.log_params(self.config.get("featurizer", {}))
                mlflow.log_params(self.config.get("model", {}))
                mlflow.log_params(self.config.get("trainer", {}))
                mlflow.log_params(self.config.get("data", {}))  # Log data config

                self.trainer.fit(self.model, dataloader)
                # Log the model after training
                mlflow.pytorch.log_model(
                    pytorch_model=self.model,
                    artifact_path="model",
                    registered_model_name=self.mlflow_config.get(
                        "registered_model_name", None
                    ),
                )
        else:
            self.trainer.fit(self.model, dataloader)

    def featurize_data(self, sequences: list[str]) -> torch.Tensor:
        """
        Featurizes a list of sequences using the initialized featurizer.

        Args:
            sequences: A list of protein sequences.

        Returns:
            A torch.Tensor containing the featurized data.
        """
        import numpy as np

        features = [self.featurizer.featurize(seq) for seq in sequences]
        return torch.tensor(np.array(features), dtype=torch.float32)

    @classmethod
    def load_model(cls, model_uri: str):
        """
        Loads a trained model from MLflow Model Registry.

        Args:
            model_uri: The MLflow model URI (e.g., "models:/<model_name>/<version_or_stage>").

        Returns:
            An instance of the loaded LightningModule.
        """
        return mlflow.pytorch.load_model(model_uri)
