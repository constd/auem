# """A simple, configurable CNN model."""

# from collections.abc import Iterable

# import numpy as np
# from torch import nn

# from .base import AuemClassifierBase

# # from torch.nn import functional as F


# __all__ = ["SimpleCNNBase"]


# DEFAULT_CONV_LAYER_DEF = [
#     (8, 7, (2,)),
#     (16, 5, (1, 2)),
#     (32, 3, (1, 2)),
#     (32, 4, (2, 3)),
# ]

# DEFAULT_DENSE_LAYER_DEF = [128]


# class SimpleCNNBase(AuemClassifierBase):
#     """Configurable CNN Model."""

#     def __init__(
#         self,
#         input_shape,
#         num_classes=10,
#         conv_layer_def=DEFAULT_CONV_LAYER_DEF,
#         dense_layer_def=DEFAULT_DENSE_LAYER_DEF,
#         out_nonlinearity="softmax",
#     ):
#         super(SimpleCNNBase, self).__init__(
#             dense_layer_def[-1], num_classes, out_nonlinearity=out_nonlinearity
#         )

#         conv_layers = []
#         dense_layers = []

#         prev_layer_def = (1, 1, (1,))
#         last_out_shape = (None,) + input_shape[1:]
#         for layer_def in conv_layer_def:
#             conv_layers.append(
#                 nn.Conv2d(
#                     prev_layer_def[0],  # previous n channels
#                     layer_def[0],  # current n channels
#                     kernel_size=layer_def[1],
#                     stride=tuple(layer_def[2])
#                     if isinstance(layer_def[2], Iterable)
#                     else layer_def[2],
#                 )
#             )
#             conv_layers.append(nn.BatchNorm2d(layer_def[0]))
#             conv_layers.append(nn.ReLU())

#             prev_layer_def = layer_def
#             x_stride = layer_def[2][0]
#             y_stride = layer_def[2][1] if len(layer_def[2]) > 1 else layer_def[2][0]
#             x_kernel_size = layer_def[1]
#             y_kernel_size = layer_def[1]
#             last_out_shape = (
#                 None,
#                 layer_def[0],
#                 ((last_out_shape[2] - x_kernel_size) // x_stride) + 1,
#                 ((last_out_shape[3] - y_kernel_size) // y_stride) + 1,
#             )

#         last_out_shape = (None, np.prod(last_out_shape[1:]))
#         for layer_def in dense_layer_def:
#             dense_layers.append(nn.Linear(last_out_shape[1], layer_def))
#             last_out_shape = (None, layer_def)
#             dense_layers.append(nn.ReLU())

#         self.convs = nn.Sequential(*conv_layers)
#         self.dense = nn.Sequential(*dense_layers)
#         # self.embedding_model = nn.Sequential(
#         #     *conv_layers,
#         #     *dense_layers
#         # )

#     def get_embedding(self, x):
#         """Calculate this model's outputs."""
#         out = self.convs(x)
#         out = out.view(out.size(0), -1)
#         out = self.dense(out)
#         # out = self.embedding_model(x)
#         # out = x
#         # for i in range(len(self.conv_layers)):
#         #     out = F.relu(self.conv_bns[i](self.conv_layers[i](out)))
#         # out = out.view(out.size(0), -1)
#         # for dense_layer in self.dense_layers:
#         #     out = F.relu(dense_layer(out))
#         return out
