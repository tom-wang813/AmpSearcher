from typing import Dict, Type

from amp_searcher.data.processors.base_processor import BaseProcessor


class ProcessorFactory:
    _processors: Dict[str, Type[BaseProcessor]] = {}

    @classmethod
    def clear_registry(cls):
        cls._processors = {}

    @classmethod
    def register(cls, name: str):
        def decorator(processor_class: Type[BaseProcessor]):
            if not issubclass(processor_class, BaseProcessor):
                raise ValueError("Processor class must inherit from BaseProcessor")
            if name in cls._processors:
                raise ValueError(f"Processor with name '{name}' already registered.")
            cls._processors[name] = processor_class
            return processor_class

        return decorator

    @classmethod
    def build_processor(cls, name: str, **kwargs) -> BaseProcessor:
        processor_class = cls._processors.get(name)
        if not processor_class:
            raise ValueError(f"Unknown processor: {name}")
        return processor_class(**kwargs)
