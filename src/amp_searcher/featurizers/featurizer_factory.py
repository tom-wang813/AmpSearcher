from typing import Any, Callable, Dict, TypeVar

T = TypeVar("T", bound=Callable[..., Any])


class FeaturizerFactory:
    _featurizers: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[T], T]:
        """
        A decorator to register a featurizer class with the FeaturizerFactory.

        Args:
            name: The name under which the featurizer will be registered.
        """

        def decorator(featurizer_class: T) -> T:
            if name in cls._featurizers:
                raise ValueError(f"Featurizer with name '{name}' already registered.")
            cls._featurizers[name] = featurizer_class
            return featurizer_class

        return decorator

    @classmethod
    def build_featurizer(cls, name: str, **kwargs: Any) -> Any:
        """
        Builds and returns an instance of the registered featurizer.

        Args:
            name: The name of the featurizer to build.
            **kwargs: Keyword arguments to pass to the featurizer's constructor.

        Returns:
            An instance of the requested featurizer.

        Raises:
            ValueError: If no featurizer with the given name is registered.
        """
        featurizer_builder = cls._featurizers.get(name)
        if not featurizer_builder:
            raise ValueError(
                f"No featurizer registered with name '{name}'. "
                f"Available featurizers: {list(cls._featurizers.keys())}"
            )
        return featurizer_builder(**kwargs)
