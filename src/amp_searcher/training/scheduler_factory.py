from typing import Any, Callable, Dict, TypeVar, Union, cast

import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

T = TypeVar("T", bound=Callable[..., Any])


class SchedulerFactory:
    _schedulers: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[T], T]:
        """
        A decorator to register a learning rate scheduler class with the SchedulerFactory.

        Args:
            name: The name under which the scheduler will be registered.
        """

        def decorator(scheduler_class: T) -> T:
            if name in cls._schedulers:
                raise ValueError(f"Scheduler with name '{name}' already registered.")
            cls._schedulers[name] = scheduler_class
            return scheduler_class

        return decorator

    @classmethod
    def build_scheduler(
        cls, optimizer: optim.Optimizer, name: str, **kwargs: Any
    ) -> Union[_LRScheduler, Dict[str, Any]]:
        """
        Builds and returns an instance of the registered learning rate scheduler.

        Args:
            optimizer: The optimizer to which the scheduler will be attached.
            name: The name of the scheduler to build.
            **kwargs: Keyword arguments to pass to the scheduler's constructor.

        Returns:
            An instance of the requested scheduler, or a dictionary for PyTorch Lightning's
            ReduceLROnPlateau configuration.

        Raises:
            ValueError: If no scheduler with the given name is registered.
        """
        scheduler_builder = cls._schedulers.get(name)
        if not scheduler_builder:
            raise ValueError(
                f"No scheduler registered with name '{name}'. "
                f"Available schedulers: {list(cls._schedulers.keys())}"
            )

        # Special handling for ReduceLROnPlateau as it requires a monitor key in Lightning
        if name == "ReduceLROnPlateau":
            return {
                "scheduler": cast(_LRScheduler, scheduler_builder(optimizer, **kwargs)),
                "monitor": "val_loss",
            }
        else:
            return cast(_LRScheduler, scheduler_builder(optimizer, **kwargs))


# Register common PyTorch schedulers
@SchedulerFactory.register("StepLR")
class StepLR(optim.lr_scheduler.StepLR):
    pass


@SchedulerFactory.register("ReduceLROnPlateau")
class ReduceLROnPlateau(optim.lr_scheduler.ReduceLROnPlateau):
    pass


# Add more schedulers as needed, e.g., CosineAnnealingLR, ExponentialLR
