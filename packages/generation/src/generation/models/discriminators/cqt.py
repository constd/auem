import logging

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm
from torchaudio.transforms import Resample
from traincore.config_stores.models import model_store

# from traincore.models.decoders.protocol import DecoderProtocol
# from traincore.models.encoders.protocol import EncoderProtocol
from generation.models.discriminators.protocol import DiscriminatorReturnType

logger = logging.getLogger(__name__)
__all__ = ["CQTDiscriminator"]


# @model_store(name="cqt", group="model/discriminator")
# class CQTDiscriminator(nn.Module):
#     def __init__(
#         self,
#         num_filters: int = 32,
#         max_filters: int = 1024,
#         scale: float = 1.0,
#         in_channels: int = 1,
#         out_channels: int = 1,
#         hop_length: int = 512,
#         n_octaves: int,
#         bins_per_octave: int,
#         kernel_size: tuple(int, int) = (3, 9),
#         dilations: tuple(int, int) = (1, 1),
#         stride: tuple(int, int) = (1, 2),
#         encoder: EncoderProtocol | None = None,
#         decoder: DecoderProtocol | None = None,
#         sample_rate: float = 44100.0,
#         num_samples: int = -1,
#     ):
#         super().__init__()
#         self.mtype = "a2e"
#         self.encoder = encoder
#         self.decoder = decoder

#         self.sample_rate = sample_rate
#         self.num_samples = num_samples

#     def forward(self, x: Tensor) -> DiscriminatorReturnType:
#         fmap: list[Float[Tensor, "..."]] = []

#         x = None

#         return {"estimate": x, "feature_map": fmap}


# self.cfg["cqtd_hop_lengths"] = self.cfg.get("cqtd_hop_lengths", [512, 256, 256])
# self.cfg["cqtd_n_octaves"] = self.cfg.get("cqtd_n_octaves", [9, 9, 9])
# self.cfg["cqtd_bins_per_octaves"] = self.cfg.get(
#     "cqtd_bins_per_octaves", [24, 36, 48]
# )


# Yoinked straight from https://github.com/NVIDIA/BigVGAN/blob/main/discriminators.py
# to ensure it matches the original implementation.
# Adapted from https://github.com/open-mmlab/Amphion/blob/main/models/vocoders/gan/discriminator/mssbcqtd.py under the MIT license.
#   LICENSE is in incl_licenses directory.
@model_store(name="cqt", group="model/discriminator")
class CQTDiscriminator(nn.Module):
    def __init__(
        self,
        hop_length: int,
        n_octaves: int,
        bins_per_octave: int,
        num_filters: int = 32,
        num_channels: int = 1,
        filters_scale: float = 1.0,
        dilations: list[int] = [1, 2, 4],
        sample_rate: int | float = 44100.0,
        max_filters: int = 1024,
        normalize_volume: bool = False,
    ):
        super().__init__()
        self.mtype = "a2e"

        self.num_channels = num_channels
        self.sr = sample_rate
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        self.num_filters = num_filters
        self.max_filters = max_filters
        self.filters_scale = filters_scale
        self.kernel_size = (3, 9)
        self.dilations = dilations
        self.stride = (1, 2)

        # Lazy-load
        from nnAudio import features

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=int(self.sr * 2),
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for _ in range(self.n_octaves):
            self.conv_pres.append(
                nn.Conv2d(
                    self.num_channels * 2,
                    self.num_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=self.get_2d_padding(self.kernel_size),
                )
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Conv2d(
                self.num_channels * 2,
                self.num_filters,
                kernel_size=self.kernel_size,
                padding=self.get_2d_padding(self.kernel_size),
            )
        )

        in_chs = int(min(self.filters_scale * self.num_filters, self.max_filters))
        for i, dilation in enumerate(self.dilations):
            out_chs = int(
                min(
                    (self.filters_scale ** (i + 1)) * self.num_filters, self.max_filters
                )
            )
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        in_chs,
                        out_chs,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=(dilation, 1),
                        padding=self.get_2d_padding(self.kernel_size, (dilation, 1)),
                    )
                )
            )
            in_chs = out_chs
        out_chs = int(
            min(
                (self.filters_scale ** (len(self.dilations) + 1)) * self.num_filters,
                self.max_filters,
            )
        )
        self.convs.append(
            weight_norm(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                    padding=self.get_2d_padding((
                        self.kernel_size[0],
                        self.kernel_size[0],
                    )),
                )
            )
        )

        self.conv_post = weight_norm(
            nn.Conv2d(
                out_chs,
                self.num_channels,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            )
        )

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = Resample(orig_freq=self.sr, new_freq=self.sr * 2)

        self.cqtd_normalize_volume = normalize_volume
        if self.cqtd_normalize_volume:
            logger.info(
                "[INFO] cqtd_normalize_volume set to True. Will apply DC offset removal & peak volume normalization in CQTD!"
            )

    def get_2d_padding(
        self,
        kernel_size: tuple[int, int],
        dilation: tuple[int, int] = (1, 1),
    ):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(self, x: Tensor) -> DiscriminatorReturnType:
        fmap: list[Float[Tensor, "..."]] = []

        if self.cqtd_normalize_volume:
            # Remove DC offset
            x = x - x.mean(dim=-1, keepdims=True)
            # Peak normalize the volume of input audio
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        x = self.resample(x)

        z = self.cqt_transform(x)

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, W, T] -> [B, C, T, W]

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                        :,
                        :,
                        :,
                        i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for i, layer in enumerate(self.convs):
            latent_z = layer(latent_z)

            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)

        return {
            "estimate": latent_z,
            "feature_map": fmap,
        }
