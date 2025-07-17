from typing import Any, Callable, Dict, TypeVar

T = TypeVar("T", bound=Callable[..., Any])


class SequenceDecoderFactory:
    _decoders: Dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[T], T]:
        """
        A decorator to register a sequence decoder class with the SequenceDecoderFactory.

        Args:
            name: The name under which the decoder will be registered.
        """

        def decorator(decoder_class: T) -> T:
            if name in cls._decoders:
                raise ValueError(
                    f"Sequence decoder with name '{name}' already registered."
                )
            cls._decoders[name] = decoder_class
            return decoder_class

        return decorator

    @classmethod
    def build_decoder(cls, name: str, **kwargs: Any) -> Any:
        """
        Builds and returns an instance of the registered sequence decoder.

        Args:
            name: The name of the decoder to build.
            **kwargs: Keyword arguments to pass to the decoder's constructor.

        Returns:
            An instance of the requested decoder.

        Raises:
            ValueError: If no decoder with the given name is registered.
        """
        decoder_builder = cls._decoders.get(name)
        if not decoder_builder:
            raise ValueError(
                f"No sequence decoder registered with name '{name}'. "
                f"Available decoders: {list(cls._decoders.keys())}"
            )
        return decoder_builder(**kwargs)
