from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Type


class RawDataSchema(BaseModel):
    sequence: str = Field(..., min_length=1)
    label: Optional[float] = None

    class Config:
        extra = "forbid"  # Forbid extra fields not defined in the schema


def validate_dataframe(df, schema: Type[BaseModel]):
    """
    Validates a pandas DataFrame against a Pydantic schema.
    """
    errors = []
    for i, row in df.iterrows():
        try:
            schema.model_validate(row.to_dict())
        except ValidationError as e:
            errors.append(f"Row {i}: {e.errors()}")
    if errors:
        raise ValueError(
            f"Data validation failed with {len(errors)} errors:\n" + "\n".join(errors)
        )
    return True
