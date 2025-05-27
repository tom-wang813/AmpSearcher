from typing import Any, Dict

import torch
import torch.nn as nn

from .component import InputBackbone, SharedBackbone, OutputBackbone


class MultiIONetwork(nn.Module):
    """
    Multi-input, multi-output network composed of separate InputBackbone, SharedBackbone, and OutputBackbone modules.
    """
    def __init__(
        self,
        input_configs: Dict[str, Dict[str, Any]],
        shared_config: Dict[str, Any],
        output_configs: Dict[str, Dict[str, Any]]
    ):
        super(MultiIONetwork, self).__init__()
        # instantiate input heads
        self.input_backbones = nn.ModuleDict({
            name: InputBackbone(**cfg) for name, cfg in input_configs.items()
        })
        # instantiate shared backbone
        self.shared_backbone = SharedBackbone(**shared_config)
        # instantiate output heads
        self.output_backbones = nn.ModuleDict({
            name: OutputBackbone(**cfg) for name, cfg in output_configs.items()
        })

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # process each input through its backbone
        feats = []
        for name, backbone in self.input_backbones.items():
            if name not in inputs:
                raise KeyError(f"Missing input for head '{name}'")
            feats.append(backbone(inputs[name]))
        # concatenate features from all inputs
        h = torch.cat(feats, dim=-1)
        # pass through shared network
        h_shared = self.shared_backbone(h)
        # distribute to each output head
        outputs = {
            name: backbone(h_shared)
            for name, backbone in self.output_backbones.items()
        }
        return outputs
