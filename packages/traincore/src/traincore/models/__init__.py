"""Pytorch models and model components for audio."""

from traincore.models.resnet import SpectrogramResNet
from traincore.models.simplecnn import SimpleCNNBase
from traincore.models.simplenn import SimpleNN

__all__ = ["SpectrogramResNet", "SimpleNN", "SimpleCNNBase"]
