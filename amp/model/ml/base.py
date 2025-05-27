from abc import ABC, abstractmethod

class ModelModule(ABC):
    """
    Abstract base class for all ML models.
    Subclasses must implement build(), train(), predict() and evaluate().
    """

    def __init__(self, **params):
        """
        :param params: hyperparameters or keyword args for the model.
        """
        self.params = params
        self.model = None

    @classmethod
    def from_cfg(cls, cfg: dict):
        """
        Alternative constructor to initialize from an OmegaConf dict.
        Usage:
            svm = SVM.from_cfg(cfg.model.svm)
        """
        return cls(**dict(cfg))

    @classmethod
    def from_yaml(cls, path: str):
        """Instantiate the model from a YAML configuration file.

        The YAML file should contain key-value pairs that match the
        constructor arguments of the concrete model class.

        Parameters
        ----------
        path : str
            Path to the YAML configuration file.
        """
        from amp.utils import load_yaml_config

        cfg = load_yaml_config(path)
        return cls(**cfg)

    @abstractmethod
    def build(self):
        """
        Instantiate and return the underlying model (e.g. sklearn or torch model).
        Should set self.model.
        """
        ...

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train self.model on the provided data.
        """
        ...

    @abstractmethod
    def predict(self, X):
        """
        Run inference on X and return predicted labels or scores.
        """
        ...

    @abstractmethod
    def evaluate(self, X, y):
        """
        Compute and return performance metrics (e.g. accuracy, ROC AUC).
        """
        ...