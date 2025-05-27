"""
AmpSearcher Model Module

This module provides various machine learning and neural network models for peptide analysis.
"""

from amp.model.ml import RF
from amp.model.nn import network, dims, multi
from amp.model.nn.component import backbone, inputs, outputs, shared

# Machine Learning Models
from amp.model.ml.RF import RF
from amp.model.ml.svm import SVM

# Neural Network Models
from amp.model.nn.network import Network
from amp.model.nn.multi import MultiIONetwork

# Neural Network Components
from amp.model.nn.component import (
    Backbone,
    InputBackbone,
    OutputBackbone,
    SharedBackbone
)

__all__ = [
    # ML Models
    'RF',
    'SVM',
    
    # Neural Networks
    'Network',
    'MultiIONetwork',
    
    # NN Components
    'Backbone',
    'InputBackbone',
    'OutputBackbone',
    'SharedBackbone',
    
    # Submodules
    'network',
    'dims',
    'multi',
    'backbone',
    'inputs',
    'outputs',
    'shared'
]