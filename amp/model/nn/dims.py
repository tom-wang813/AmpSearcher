from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Type, Union

class DimStrategy(ABC):
    @abstractmethod
    def calculate(self,
                  input_dim: int,
                  output_dim: int,
                  num_layers: int,
                  **kwargs) -> List[int]:
        """计算每层维度。"""

class DimCalculator:
    _strategies: Dict[str, DimStrategy] = {}

    @classmethod
    def register_method(cls, name: str):
        """
        用作 decorator：
            @DimCalculator.register_method('linear')
            class LinearStrategy(DimStrategy): ...
        """
        def decorator(strategy_cls: Type[DimStrategy]):
            cls._strategies[name.lower()] = strategy_cls()
            return strategy_cls
        return decorator

    @classmethod
    def calculate(cls,
                  input_dim: int,
                  output_dim: int,
                  num_layers: int,
                  method: Union[str, DimStrategy, Type[DimStrategy]] = 'linear',
                  initial_hidden_dim: Optional[int] = None,
                  ratios: Optional[List[float]] = None,
                  custom_dims: Optional[List[int]] = None
                  ) -> List[int]:
        if isinstance(method, DimStrategy):
            strategy = method
        elif isinstance(method, type) and issubclass(method, DimStrategy):
            strategy = method()
        elif isinstance(method, str):
            method = method.lower()
            if method not in cls._strategies:
                raise ValueError(f"Unknown method '{method}'. Available methods: {list(cls._strategies.keys())}")
            strategy = cls._strategies[method]
        else:
            raise TypeError("method must be a string or an instance of DimStrategy.")
        return strategy.calculate(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,          
            initial_hidden_dim=initial_hidden_dim,
            ratios=ratios,
            custom_dims=custom_dims
        )

@DimCalculator.register_method('linear')
class LinearStrategy(DimStrategy):
    def calculate(self,
                  input_dim: int,
                  output_dim: int,
                  num_layers: int,
                  initial_hidden_dim: Optional[int] = None,
                  **kwargs) -> List[int]:
        h0 = input_dim if initial_hidden_dim is None else initial_hidden_dim
        steps = num_layers if h0 == input_dim else num_layers - 1
        step = (h0 - output_dim) // steps
        if h0 == input_dim:
            dims = [input_dim - i * step for i in range(num_layers + 1)]
        else:
            dims = [h0 - i * step for i in range(num_layers)]
            dims.insert(0, input_dim)
        dims[-1] = output_dim
        return dims

@DimCalculator.register_method('ratio')
class RatioStrategy(DimStrategy):
    def calculate(self,
                  input_dim: int,
                  output_dim: int,
                  num_layers: int,
                  initial_hidden_dim: int,
                  ratios: Optional[List[float]] = None,
                  **kwargs) -> List[int]:
        if ratios is None or len(ratios) != num_layers - 1:
            raise ValueError(f"'ratios' must be a list of {num_layers - 1} floats.")
        dims = [input_dim]
        dims.append(initial_hidden_dim)  # Start from initial_hidden_dim
        current_dim = initial_hidden_dim
        for r in ratios:
            current_dim = max(1, int(current_dim * r))
            dims.append(current_dim)
        dims[-1] = output_dim  # Ensure the last dimension is exactly output_dim
        return dims

@DimCalculator.register_method('custom')
class CustomStrategy(DimStrategy):
    def calculate(self,
                  input_dim: int,
                  output_dim: int,
                  num_layers: int,
                  custom_dims: Optional[List[int]] = None,
                  **kwargs) -> List[int]:
        if custom_dims is None or len(custom_dims) != num_layers + 1:
            raise ValueError(f"'custom_dims' must be a list of {num_layers+1} ints.")
        if custom_dims[0] != input_dim or custom_dims[-1] != output_dim:
            raise ValueError("First/last entries of custom_dims must match input/output.")
        return custom_dims