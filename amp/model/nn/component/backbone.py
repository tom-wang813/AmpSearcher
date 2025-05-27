import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Union, List

from ..dims import DimCalculator, DimStrategy


class Backbone(nn.Module):
    def __init__(self,
                 model: Optional[nn.Module] = None,
                 *,
                 input_dim: Optional[int] = None,
                 output_dim: Optional[int] = None,
                 num_layers: Optional[int] = None,
                 method: Union[str, DimStrategy, type[DimStrategy]] = 'linear',
                 initial_hidden_dim: Optional[int] = None,
                 ratios: Optional[List[float]] = None,
                 custom_dims: Optional[List[int]] = None,
                 activation: Union[str, nn.Module] = 'relu',
                 use_residual: bool = False,
                 **kwargs: Any):
        super().__init__()
        self.use_residual = use_residual
        self.provided_model = model is not None
        self.model = model
        if not self.provided_model:
            if input_dim is None or output_dim is None or num_layers is None:
                raise ValueError("input_dim, output_dim, and num_layers must be provided if model is not specified.")
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.num_layers = num_layers
            self.method = method
            self.initial_hidden_dim = initial_hidden_dim
            self.ratios = ratios
            self.custom_dims = custom_dims
            # compute hidden dims
            self.hidden_dims = DimCalculator.calculate(
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                method=method,
                initial_hidden_dim=initial_hidden_dim,
                ratios=ratios,
                custom_dims=custom_dims
            )
            # process activation
            if isinstance(activation, str):
                act = activation.lower()
                if act == 'relu':
                    self.activation = nn.ReLU()
                elif act == 'tanh':
                    self.activation = nn.Tanh()
                elif act == 'gelu':
                    self.activation = nn.GELU()
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
            elif isinstance(activation, nn.Module):
                self.activation = activation
            else:
                raise TypeError("activation must be a str or nn.Module.")
            # build layers
            layers = []
            for i in range(num_layers):
                in_dim = self.hidden_dims[i]
                out_dim = self.hidden_dims[i+1] if i+1 < len(self.hidden_dims) else output_dim
                layers.append(nn.Linear(in_dim, out_dim))
                if i < num_layers - 1:
                    layers.append(self.activation)
            self.network = nn.Sequential(*layers)
            # residual skip layer
            if use_residual:
                if input_dim == output_dim:
                    self.skip = nn.Identity()
                else:
                    self.skip = nn.Linear(input_dim, output_dim)
            self.initialize_weights()

    def initialize_weights(self):
        for layer in getattr(self, 'network', []):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        if self.use_residual and hasattr(self, 'skip') and isinstance(self.skip, nn.Linear):
            nn.init.xavier_uniform_(self.skip.weight)
            if self.skip.bias is not None:
                nn.init.zeros_(self.skip.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.provided_model and self.model is not None:
            y = self.model(x)
        else:
            y = self.network(x)
        if self.use_residual and not self.provided_model:
            return y + self.skip(x)
        return y

    def get_config(self) -> Dict[str, Any]:
        return {
            'input_dim': getattr(self, 'input_dim', None),
            'output_dim': getattr(self, 'output_dim', None),
            'num_layers': getattr(self, 'num_layers', None),
            'method': getattr(self, 'method', None),
            'initial_hidden_dim': getattr(self, 'initial_hidden_dim', None),
            'ratios': getattr(self, 'ratios', None),
            'custom_dims': getattr(self, 'custom_dims', None),
            'activation': self.activation.__class__.__name__ if hasattr(self, 'activation') else None,
            'use_residual': self.use_residual
        }

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(input_dim={getattr(self,'input_dim',None)}, output_dim={getattr(self,'output_dim',None)}, "
                f"num_layers={getattr(self,'num_layers',None)}, method={getattr(self,'method',None)}, "
                f"activation={self.activation}, use_residual={self.use_residual})")
