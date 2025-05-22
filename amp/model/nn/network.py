import torch 
import torch.nn as nn
from typing import Optional, List

from amp.model.nn.dims import DimCalculator

class Network(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_layers: int,
                 method: str = 'linear',
                 initial_hidden_dim: Optional[int] = None,
                 ratios: Optional[List[float]] = None,
                 custom_dims: Optional[List[int]] = None):
        
        super(Network, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.method = method
        self.initial_hidden_dim = initial_hidden_dim
        self.ratios = ratios
        self.custom_dims = custom_dims

        # 计算每层的维度
        self.hidden_dims = DimCalculator.calculate(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            method=method,
            initial_hidden_dim=initial_hidden_dim,
            ratios=ratios,
            custom_dims=custom_dims
        )

        # 创建网络层
        layers = []
        for i in range(num_layers):
            in_dim = self.hidden_dims[i]
            out_dim = self.hidden_dims[i + 1] if i + 1 < len(self.hidden_dims) else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.network(x)
        return x
    
    def get_hidden_feature(self, 
                           x, 
                           layer_index: Optional[int] = None):
        """
        获取隐藏层特征
        """
        if layer_index is None:
            return self.network(x)
        if layer_index < 0 or layer_index >= len(self.hidden_dims):
            raise ValueError(f"Invalid layer index: {layer_index}. "
                             f"Must be between 0 and {len(self.hidden_dims) - 1}.")
        for i in range(layer_index):
            x = self.network[i](x)
        return x