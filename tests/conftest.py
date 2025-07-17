import pytest
from amp_searcher.data.processors.processor_factory import ProcessorFactory
from amp_searcher.data.processors.sequence_processor import SequenceProcessor

@pytest.fixture(autouse=True)
def clear_processor_factory_registry():
    ProcessorFactory.clear_registry()
    # Re-register SequenceProcessor as it's used in many tests
    ProcessorFactory.register("sequence_processor")(SequenceProcessor)
