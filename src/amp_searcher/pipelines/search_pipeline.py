from typing import Any, Dict, List, Tuple

import torch

from amp_searcher.featurizers.featurizer_factory import FeaturizerFactory
from amp_searcher.models.model_factory import ModelFactory
from amp_searcher.models.lightning_module_factory import LightningModuleFactory
from amp_searcher.training.trainer import AmpTrainer  # Import AmpTrainer

from amp_searcher.utils.sequence_decoder_factory import SequenceDecoderFactory


class SearchPipeline:
    """
    A pipeline for searching and generating novel antimicrobial peptides (AMPs).

    This pipeline combines a generative model (e.g., VAE) with a screening model
    to iteratively generate and evaluate peptide sequences.
    """

    def __init__(
        self,
        generative_model_config: Dict[str, Any],
        screening_model_config: Dict[str, Any],
        featurizer_config: Dict[str, Any],
        sequence_decoder_config: Dict[str, Any],
        search_config: Dict[str, Any],
        generative_model_checkpoint_path: str | None = None,
        generative_mlflow_model_uri: str | None = None,
        screening_model_checkpoint_path: str | None = None,
        screening_mlflow_model_uri: str | None = None,
    ):
        """
        Initializes the search pipeline.

        Args:
            generative_model_config: Configuration dictionary for the generative model architecture.
            generative_model_checkpoint_path: Path to the trained generative model checkpoint (optional).
            generative_mlflow_model_uri: MLflow URI for the generative model (optional).
            screening_model_config: Configuration dictionary for the screening model architecture.
            screening_model_checkpoint_path: Path to the trained screening model checkpoint (optional).
            screening_mlflow_model_uri: MLflow URI for the screening model (optional).
            featurizer_config: Configuration dictionary for the featurizer.
            sequence_decoder_config: Configuration dictionary for the sequence decoder.
        """
        self.featurizer = self._init_featurizer(featurizer_config)
        self.generative_model = self._load_generative_model(
            generative_model_config,
            generative_model_checkpoint_path,
            generative_mlflow_model_uri,
        )
        self.screening_model = self._load_screening_model(
            screening_model_config,
            screening_model_checkpoint_path,
            screening_mlflow_model_uri,
        )
        self.sequence_decoder = self._init_sequence_decoder(sequence_decoder_config)
        self.search_config = search_config

        self.generative_model.eval()  # Set to evaluation mode
        self.screening_model.eval()  # Set to evaluation mode

    def _init_featurizer(self, featurizer_config: Dict[str, Any]):
        name = str(featurizer_config.get("name"))
        params = featurizer_config.get("params", {})
        return FeaturizerFactory.build_featurizer(name, **params)

    def _init_sequence_decoder(self, decoder_config: Dict[str, Any]):
        name = str(decoder_config.get("name"))
        params = decoder_config.get("params", {})
        return SequenceDecoderFactory.build_decoder(name, **params)

    def _load_generative_model(
        self,
        model_config: Dict[str, Any],
        model_checkpoint_path: str | None,
        mlflow_model_uri: str | None,
    ):
        if mlflow_model_uri:
            return AmpTrainer.load_model(mlflow_model_uri)
        elif model_checkpoint_path:
            model_architecture_name = str(model_config["architecture"]["name"])
            model_architecture_params = model_config["architecture"].get("params", {})
            lightning_module_name = str(
                model_config.get("lightning_module_name", "GenerativeLightningModule")
            )

            model_architecture = ModelFactory.build_model(model_architecture_name, **model_architecture_params)

            lightning_model = LightningModuleFactory.build_lightning_module(
                lightning_module_name,
                model_architecture=model_architecture,
                latent_dim=model_config["architecture"]["params"]["latent_dim"],
                kl_weight=model_config.get("kl_weight", 0.001),
                optimizer_params=model_config.get("optimizer_params"),
                scheduler_params=model_config.get("scheduler_params"),
            )
            checkpoint = torch.load(
                model_checkpoint_path,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
            lightning_model.load_state_dict(checkpoint["state_dict"])
            return lightning_model
        else:
            raise ValueError(
                "Either generative_model_checkpoint_path or generative_mlflow_model_uri must be provided."
            )

    def _load_screening_model(
        self,
        model_config: Dict[str, Any],
        model_checkpoint_path: str | None,
        mlflow_model_uri: str | None,
    ):
        if mlflow_model_uri:
            return AmpTrainer.load_model(mlflow_model_uri)
        elif model_checkpoint_path:
            model_architecture_name = str(model_config["architecture"]["name"])
            model_architecture_params = model_config["architecture"].get("params", {})
            task_type = model_config.get("lightning_module_params", {}).get("task_type")
            lightning_module_name = str(
                model_config.get("lightning_module_name", "ScreeningLightningModule")
            )

            model_architecture = ModelFactory.build_model(model_architecture_name, **model_architecture_params)

            lightning_model = LightningModuleFactory.build_lightning_module(
                lightning_module_name,
                model_architecture=model_architecture,
                task_type=task_type,
                optimizer_params=model_config.get("optimizer_params"),
                scheduler_params=model_config.get("scheduler_params"),
            )
            checkpoint = torch.load(
                model_checkpoint_path,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
            lightning_model.load_state_dict(checkpoint["state_dict"])
            return lightning_model
        else:
            raise ValueError(
                "Either screening_model_checkpoint_path or screening_mlflow_model_uri must be provided."
            )

    def search(self) -> List[Tuple[str, float]]:
        """
        Generates and screens peptides to find top_k candidates.

        Returns:
            A list of tuples, where each tuple contains (peptide_sequence, predicted_score).
        """
        num_generations = self.search_config.get("num_generations", 100)
        top_k = self.search_config.get("top_k", 10)

        with torch.no_grad():
            generated_features_tensor = self.generative_model.generate(num_generations)

            screening_scores = self.screening_model(generated_features_tensor)

            if self.screening_model.task_type == "classification":
                probabilities = torch.sigmoid(screening_scores)
                screening_scores = probabilities

            top_scores, top_indices = torch.topk(screening_scores.squeeze(), top_k)

            results = []
            for i in range(top_k):
                feature_vector = generated_features_tensor[top_indices[i]].cpu().numpy()
                score = top_scores[i].item()
                peptide_sequence = self.sequence_decoder.decode(feature_vector)
                results.append((peptide_sequence, score))

        return results
