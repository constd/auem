"""A simple, configurable Transformer encoder-decoder model."""
from typing import Tuple

import torch
import torch.nn as nn

from .base import AuemClassifierBase

__all__ = [
    "AudioTransformerEncDec",
    "AudioTransformerEncoder",
    "SimpleTransformerEncDecClassifier",
    "SimpleTransformerEncoderClassifier",
]


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()

    def forward(self, x: torch.tensor) -> torch.tensor:
        # the assumption is that the frames at this point
        # have the shape (batch_num_frames, time_step, num_frequency_bins)
        posencs = torch.arange(start=0, end=x.shape[1])
        x[:, :, 0] = posencs
        return x


class ATRegressionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.linear(x[0])
        # x =  self.layer_norm(x)
        return x


class ATClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.linear(x[0])
        # x =  self.layer_norm(x)
        return x


class AudioTransformerEncDec(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        # TODO: add more positional encodings
        self.posenc = AbsolutePositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

    def forward(
        self,
        src: torch.tensor = None,
        tgt: torch.tensor = None,
        src_mask: torch.tensor = None,
        tgt_mask: torch.tensor = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        # input to the transformer is (sequence, batch, embedding)
        # in our case is (num_frames, batch, num_frequencies)
        # TODO: figure out why we get an extra dimension and have to use .squeeze()
        src_enc = self.posenc(src).squeeze(1).permute(2, 0, 1)

        if tgt is None:
            tgt = src
            tgt_mask = src_mask
        tgt_enc = self.posenc(tgt).squeeze(1).permute(2, 0, 1)
        out = self.transformer(
            src=src_enc, tgt=tgt_enc, src_mask=src_mask, tgt_mask=tgt_mask
        )
        return out


class AudioTransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 2,
        num_encoder_layers: int = 1,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        activation: str = "gelu",
    ):
        super().__init__()
        # TODO: add more positional encodings
        print("d_model")
        self.posenc = AbsolutePositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

    def forward(
        self,
        src: torch.tensor = None,
        src_mask: torch.tensor = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        # input to the transformer is (sequence, batch, embedding)
        # in our case is (num_frames, batch, num_frequencies)
        # TODO: figure out why we get an extra dimension and have to use .squeeze()
        src_enc = self.posenc(src).squeeze(1)
        out = self.transformer(src_enc.permute(1, 0, 2), src_mask)
        return out.permute(1, 0, 2)


class SimpleTransformerEncDecClassifier(AuemClassifierBase):
    def __init__(
        self,
        child_embedding_size,
        num_classes,
        out_nonlinearity,
        d_model=256,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=512,
        dropout=0.2,
        activation="gelu",
        **kwargs
    ):
        super().__init__(
            child_embedding_size=child_embedding_size,
            num_classes=10,
            out_nonlinearity="log_softmax",
        )
        self.transformer = AudioTransformerEncDec(
            d_model=child_embedding_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

    def get_embedding(self, src, tgt=None, src_mask=None, tgt_mask=None):
        # expects (sequence_length, batch_size, embedding_size)
        out = self.transformer(src, tgt, src_mask, tgt_mask)
        return out

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):
        out = self.get_embedding(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        out = self.class_layer(out)
        return out


class SimpleTransformerEncoderClassifier(AuemClassifierBase):
    def __init__(
        self,
        child_embedding_size,
        num_classes,
        out_nonlinearity,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.2,
        activation="gelu",
        **kwargs
    ):
        super().__init__(
            child_embedding_size=child_embedding_size,
            num_classes=num_classes,
            out_nonlinearity=out_nonlinearity,
        )
        self.transformer = AudioTransformerEncoder(
            d_model=child_embedding_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

    def get_embedding(self, src, src_mask=None):
        # expects and outputs (sequence_length, batch_size, embedding_size)
        out = self.transformer(src, src_mask)
        return out

    def forward(self, src, src_mask=None):
        src = torch.cat(
            [torch.ones(src.size(0), 1, src.size(2)).to(src.device), src], dim=1
        )
        out = self.get_embedding(src=src, src_mask=src_mask)
        out = self.class_layer(out[:, 0])
        return out
