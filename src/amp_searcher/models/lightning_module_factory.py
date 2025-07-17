from typing import Any, Callable, Dict, TypeVar

T = TypeVar("T", bound=Callable[..., Any])


class LightningModuleFactory:
    _lightning_modules: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[T], T]:
        """
        A decorator to register a LightningModule class with the LightningModuleFactory.

        Args:
            name: The name under which the LightningModule will be registered.
        """

        def decorator(lightning_module_class: T) -> T:
            if name in cls._lightning_modules:
                raise ValueError(
                    f"LightningModule with name '{name}' already registered."
                )
            cls._lightning_modules[name] = lightning_module_class
            return lightning_module_class

        return decorator

    @classmethod
    def build_lightning_module(cls, name: str, **kwargs: Any) -> Any:
        """
        Builds and returns an instance of the registered LightningModule.

        Args:
            name: The name of the LightningModule to build.
            **kwargs: Keyword arguments to pass to the LightningModule's constructor.

        Returns:
            An instance of the requested LightningModule.

        Raises:
            ValueError: If no LightningModule with the given name is registered.
        """
        lightning_module_builder = cls._lightning_modules.get(name)
        if not lightning_module_builder:
            raise ValueError(
                f"No LightningModule registered with name '{name}'. "
                f"Available LightningModules: {list(cls._lightning_modules.keys())}"
            )
        return lightning_module_builder(**kwargs)
