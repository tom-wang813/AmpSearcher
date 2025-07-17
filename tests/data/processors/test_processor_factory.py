import pytest
from amp_searcher.data.processors.processor_factory import ProcessorFactory
from amp_searcher.data.processors.base_processor import BaseProcessor

# Define a dummy processor for testing registration
class DummyProcessor(BaseProcessor):
    def __init__(self, param1=None):
        self.param1 = param1

    def process(self, sequences):
        return [s + "_processed" for s in sequences]

    def process_single(self, sequence):
        return sequence + "_processed"


class AnotherDummyProcessor(BaseProcessor):
    def process(self, sequences):
        return [s + "_another" for s in sequences]

    def process_single(self, sequence):
        return sequence + "_another"


def test_processor_registration():
    """Test that a processor can be registered and built."""
    ProcessorFactory._processors = {}

    @ProcessorFactory.register("dummy_processor")
    class TempDummyProcessor(DummyProcessor):
        pass

    assert "dummy_processor" in ProcessorFactory._processors
    processor = ProcessorFactory.build_processor("dummy_processor")
    assert isinstance(processor, TempDummyProcessor)
    assert processor.process(["test"]) == ["test_processed"]


def test_processor_registration_with_params():
    """Test that a processor can be registered and built with parameters."""
    ProcessorFactory._processors = {}

    @ProcessorFactory.register("dummy_processor_with_params")
    class TempDummyProcessorWithParams(DummyProcessor):
        pass

    processor = ProcessorFactory.build_processor("dummy_processor_with_params", param1="value1")
    assert isinstance(processor, TempDummyProcessorWithParams)
    assert processor.param1 == "value1"


def test_processor_registration_duplicate_name():
    """Test that registering a processor with a duplicate name raises an error."""
    ProcessorFactory._processors = {}

    @ProcessorFactory.register("duplicate_name")
    class FirstProcessor(DummyProcessor):
        pass

    with pytest.raises(ValueError, match="Processor with name 'duplicate_name' already registered."):
        @ProcessorFactory.register("duplicate_name")
        class SecondProcessor(DummyProcessor):
            pass


def test_build_non_existent_processor():
    """Test that building a non-existent processor raises an error."""
    ProcessorFactory._processors = {}
    with pytest.raises(ValueError, match="Unknown processor: non_existent_processor"):
        ProcessorFactory.build_processor("non_existent_processor")


def test_processor_registration_not_subclass_of_base_processor():
    """Test that registering a class not inheriting from BaseProcessor raises an error."""
    ProcessorFactory._processors = {}

    with pytest.raises(ValueError, match="Processor class must inherit from BaseProcessor"):
        @ProcessorFactory.register("invalid_processor")
        class InvalidProcessor:
            pass


def test_processor_factory_multiple_registrations():
    """Test that multiple different processors can be registered."""
    ProcessorFactory._processors = {}

    @ProcessorFactory.register("dummy1")
    class Dummy1(DummyProcessor):
        pass

    @ProcessorFactory.register("dummy2")
    class Dummy2(AnotherDummyProcessor):
        pass

    assert "dummy1" in ProcessorFactory._processors
    assert "dummy2" in ProcessorFactory._processors

    proc1 = ProcessorFactory.build_processor("dummy1")
    proc2 = ProcessorFactory.build_processor("dummy2")

    assert isinstance(proc1, Dummy1)
    assert isinstance(proc2, Dummy2)
    assert proc1.process(["a"]) == ["a_processed"]
    assert proc2.process(["b"]) == ["b_another"]
