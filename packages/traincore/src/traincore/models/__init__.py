"""Pytorch models and model components for audio."""


__all__ = ["CQTResNet", "SpectrogramResNet"]  # , "SimpleCNNBase", "SimpleNN"]

from .resnet import CQTResNet, SpectrogramResNet
# from .simplecnn import SimpleCNNBase
# from .simplenn import SimpleNN