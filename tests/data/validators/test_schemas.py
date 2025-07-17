import pytest
from pydantic import ValidationError, ConfigDict
from amp_searcher.data.validators.schemas import RawDataSchema

def test_raw_data_schema_valid_data():
    valid_data = {"sequence": "ACGT", "label": 0.5}
    schema = RawDataSchema(**valid_data)
    assert schema.sequence == "ACGT"
    assert schema.label == 0.5

def test_raw_data_schema_missing_sequence():
    invalid_data = {"label": 0.5}
    with pytest.raises(ValidationError) as excinfo:
        RawDataSchema(**invalid_data)
    assert "Field required" in str(excinfo.value)
    assert "sequence" in str(excinfo.value)

def test_raw_data_schema_missing_label():
    # label is optional, so this should pass
    valid_data = {"sequence": "ACGT"}
    schema = RawDataSchema(**valid_data)
    assert schema.sequence == "ACGT"
    assert schema.label is None

def test_raw_data_schema_invalid_sequence_type():
    invalid_data = {"sequence": 123, "label": 0.5}
    with pytest.raises(ValidationError) as excinfo:
        RawDataSchema(**invalid_data)
    assert "Input should be a valid string" in str(excinfo.value)

def test_raw_data_schema_invalid_label_type():
    invalid_data = {"sequence": "ACGT", "label": "not_a_float"}
    with pytest.raises(ValidationError) as excinfo:
        RawDataSchema(**invalid_data)
    assert "Input should be a valid number" in str(excinfo.value)

def test_raw_data_schema_extra_fields_not_allowed():
    # Pydantic V2 by default does not allow extra fields
    invalid_data = {"sequence": "ACGT", "label": 0.5, "extra_field": "value"}
    with pytest.raises(ValidationError) as excinfo:
        RawDataSchema(**invalid_data)
    assert "Extra inputs are not permitted" in str(excinfo.value)
    assert "extra_field" in str(excinfo.value)
