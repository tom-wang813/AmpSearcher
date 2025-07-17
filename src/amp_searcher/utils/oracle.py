import torch
from typing import List, Dict, Any

class Oracle:
    """A wrapper for a trained model to be used as an oracle for optimization."""

    def __init__(
        self, model_config: Dict[str, Any], model_checkpoint_path: str, featurizer_config: Dict[str, Any]
    ):
        from amp_searcher.featurizers import FeaturizerFactory
        
        featurizer_name = featurizer_config.get("name")
        featurizer_params = featurizer_config.get("params", {})
        self.featurizer = FeaturizerFactory.build_featurizer(
            featurizer_name, **featurizer_params
        )
        self.model = self._load_model(model_config, model_checkpoint_path)
        self.model.eval()

    def _load_model(self, model_config: Dict[str, Any], model_checkpoint_path: str):
        from amp_searcher.models import ModelFactory, LightningModuleFactory
        
        model_name = model_config.get("name")
        architecture_params = model_config["architecture"].get("params", {})
        model_architecture = ModelFactory.build_model(model_name, **architecture_params)
        lightning_module_name = model_config["lightning_module"].get("name")
        lightning_module_params = model_config["lightning_module"].get("params", {})

        lightning_model = LightningModuleFactory.build_lightning_module(
            lightning_module_name,
            model_architecture=model_architecture,
            **lightning_module_params,
        )
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        lightning_model.load_state_dict(checkpoint["state_dict"])
        return lightning_model

    def predict(self, sequences: List[str]) -> List[float]:
        if not sequences:
            return []

        features = [self.featurizer.featurize(seq) for seq in sequences]
        X = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(X)
            # Assuming regression or a score where higher is better
            scores = logits.squeeze().tolist()
            if isinstance(scores, float):  # Handle single prediction
                return [scores]
            return scores
