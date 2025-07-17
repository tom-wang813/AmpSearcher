from typing import Any, Callable, Dict, TypeVar

T = TypeVar("T", bound=Callable[..., Any])


class ModelFactory:
    _models: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[T], T]:
        """
        A decorator to register a model class or a model builder function
        with the ModelFactory.

        Args:
            name: The name under which the model will be registered.
        """

        def decorator(model_class_or_builder: T) -> T:
            if name in cls._models:
                raise ValueError(f"Model with name '{name}' already registered.")
            cls._models[name] = model_class_or_builder
            return model_class_or_builder

        return decorator

    @classmethod
    def build_model(cls, name: str, **kwargs: Any) -> Any:
        """
        Builds and returns an instance of the registered model.

        Args:
            name: The name of the model to build.
            **kwargs: Keyword arguments to pass to the model's constructor or builder function.

        Returns:
            An instance of the requested model.

        Raises:
            ValueError: If no model with the given name is registered.
        """
        model_builder = cls._models.get(name)
        if not model_builder:
            raise ValueError(
                f"No model registered with name '{name}'. "
                f"Available models: {list(cls._models.keys())}"
            )
        return model_builder(**kwargs)
