import pytest
from abc import ABC, abstractmethod
from amp_searcher.data.processors.base_processor import BaseProcessor

def test_base_processor_abstract_methods():
    """
    Test that BaseProcessor is an abstract class and cannot be instantiated directly.
    Also, ensure that concrete implementations must implement the 'process' method.
    """
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseProcessor"):
        BaseProcessor()

    class ConcreteProcessor(BaseProcessor):
        def process(self, sequences):
            return [s.upper() for s in sequences]
        def process_single(self, sequence):
            return sequence.upper()

    # Should not raise an error
    processor = ConcreteProcessor()
    assert isinstance(processor, BaseProcessor)

    # Test the implemented method
    result = processor.process(["seq1", "seq2"])
    assert result == ["SEQ1", "SEQ2"]

    class IncompleteProcessor(BaseProcessor):
        # Missing 'process' method
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class IncompleteProcessor"):
        IncompleteProcessor()
