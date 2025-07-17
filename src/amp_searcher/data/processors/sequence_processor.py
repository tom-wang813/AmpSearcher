from typing import List, Any, Dict

from amp_searcher.data.processors.base_processor import BaseProcessor
from amp_searcher.data.validators.base_validator import BaseValidator
from amp_searcher.data.validators.validator_factory import ValidatorFactory
from amp_searcher.featurizers.base import BaseFeaturizer
from amp_searcher.featurizers.featurizer_factory import FeaturizerFactory
from amp_searcher.data.processors.processor_factory import ProcessorFactory


@ProcessorFactory.register("sequence_processor")
class SequenceProcessor(BaseProcessor):
    def __init__(
        self,
        featurizer_config: Dict[str, Any],
        validator_configs: List[Dict[str, Any]] | None = None,
        **kwargs # Add kwargs to capture any extra parameters
    ):
        self.featurizer: BaseFeaturizer = FeaturizerFactory.build_featurizer(
            str(featurizer_config.get("name")), **featurizer_config.get("params", {})
        )
        self.validators: List[BaseValidator] = []
        if validator_configs:
            for config in validator_configs:
                validator_name = str(config.get("name"))
                validator_params = config.get("params", {})
                self.validators.append(
                    ValidatorFactory.build_validator(validator_name, **validator_params)
                )

    def process(self, sequences: List[str]) -> List[Any]:
        # Apply validators first
        if self.validators:
            for validator in self.validators:
                is_valid, errors = validator.validate(sequences)
                if not is_valid:
                    raise ValueError(f"Data validation failed: {', '.join(errors)}")

        # Then featurize
        return [self.featurizer.featurize(seq) for seq in sequences]

    def process_single(self, sequence: str) -> Any:
        # Apply validators first for a single sequence (wrap in list for validation)
        if self.validators:
            for validator in self.validators:
                is_valid, errors = validator.validate([sequence])
                if not is_valid:
                    raise ValueError(
                        f"Data validation failed for single sequence: {', '.join(errors)}"
                    )

        # Then featurize
        return self.featurizer.featurize(sequence)
