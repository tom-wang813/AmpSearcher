from typing import Any, Callable, Dict, TypeVar

T = TypeVar("T", bound=Callable[..., Any])


class OptimizerFactory:
    _optimizers: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[T], T]:
        """
        A decorator to register an optimizer class with the OptimizerFactory.

        Args:
            name: The name under which the optimizer will be registered.
        """

        def decorator(optimizer_class: T) -> T:
            if name in cls._optimizers:
                raise ValueError(f"Optimizer with name '{name}' already registered.")
            cls._optimizers[name] = optimizer_class
            return optimizer_class

        return decorator

    @classmethod
    def build_optimizer(cls, name: str, **kwargs: Any) -> Any:
        """
        Builds and returns an instance of the registered optimizer.

        Args:
            name: The name of the optimizer to build.
            **kwargs: Keyword arguments to pass to the optimizer's constructor.

        Returns:
            An instance of the requested optimizer.

        Raises:
            ValueError: If no optimizer with the given name is registered.
        """
        optimizer_builder = cls._optimizers.get(name)
        if not optimizer_builder:
            raise ValueError(
                f"No optimizer registered with name '{name}'. "
                f"Available optimizers: {list(cls._optimizers.keys())}"
            )
        return optimizer_builder(**kwargs)
