import warnings

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
from compressai.registry import register_model

import numpy as np
import onnxruntime

from .base import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    get_scale_table,
)
from .utils import conv, deconv

__all__ = [
    "CompressionModel",
    "FactorizedPrior",
    "FactorizedPriorReLU",
    "ScaleHyperprior",
    "MeanScaleHyperprior",
    "JointAutoregressiveHierarchicalPriors",
    "get_scale_table",
    "SCALES_MIN",
    "SCALES_MAX",
    "SCALES_LEVELS",
]

@register_model("bmshj2018-hyperprior")
class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, **kwargs):
        super().__init__(**kwargs)

        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    def _convert(self, model, model_name, input_names, input_shapes, output_names, output_path):
        print(f"converting: {model_name}")
        dummy_inputs = torch.randn((1, *input_shapes))
        model_path = os.path.join(output_path, f"{model_name}.onnx")
        print(model_path)
        dynamic_axes = {name: {0: "batch", 2: "height", 3: "width"} for name in input_names}
        dynamic_axes.update({name: {0: "batch", 1: "height", 2: "width"} for name in output_names})
        torch.onnx.export(
            model=model, args=tuple(dummy_inputs), f=model_path,
            input_names=input_names, dynamic_axes=dynamic_axes,
            output_names=output_names, opset_version=11)
        print(' -------------------------------- ')

    def _convert_y_encoder(self, model_name, input_shapes, output_path):
        class Encoder(nn.Module):
            def __init__(self, g_a):
                super(Encoder, self).__init__()
                self.g_a = g_a

            def forward(self, y):
                y = self.g_a(y)
                return torch.round(y)

        new_mode = Encoder(self.g_a)
        self._convert(new_mode, model_name, ['x_input'], input_shapes, ['y_output'], output_path)

    def _convert_z_encoder(self, model_name, input_shapes, output_path):
        class Encoder(nn.Module):
            def __init__(self, h_a):
                super(Encoder, self).__init__()
                self.h_a = h_a
                root_url = "https://compressai.s3.amazonaws.com/models/v1"
                model_url = f"{root_url}/bmshj2018-hyperprior-1-7eb97409.pth.tar"
                state_dict = torch.hub.load_state_dict_from_url(model_url)
                self.quantiles = state_dict['entropy_bottleneck.quantiles']

            def forward(self, z):
                z = self.h_a(z)
                z_hat = self.dequantize(z, self._get_medians())
                return z_hat
            
            def dequantize(self, inputs, means):
                mode = "dequantize"
                outputs = inputs.clone()
                if means is not None:
                    outputs -= means
                outputs = torch.round(outputs)
                if means is not None:
                    outputs += means
                return outputs

            def _get_medians(self):
                medians = self.quantiles[:, :, 1:2]
                return medians

        new_mode = Encoder(self.h_a)
        self._convert(new_mode, model_name, ['y_input'], input_shapes, ['z_hat_output'], output_path)

    def _check_onnx_model_accuracy(self, model_name, input_name, input_data, output_data, check_path, onnx_path):
        print(f"checking: {model_name}")
        if not os.path.exists(check_path):
            os.makedirs(check_path)
        input_check_path = os.path.join(check_path, f'{model_name}_input.pt')
        output_check_path = os.path.join(check_path, f'{model_name}_output.pt')
        onnx_path = os.path.join(onnx_path, f'{model_name}.onnx')

        torch.save(input_data, input_check_path) 
        torch.save(output_data, output_check_path) 

        sess = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_onnx = input_data.numpy()
        output_onnx = sess.run(None, {input_name: input_onnx})[0]

        # print("output_data", output_data)
        # print("output_onnx", output_onnx)

        diff = np.abs(output_data.numpy() - output_onnx)
        print(f' == Mean difference: {np.mean(diff)}')
        print(f' == Max difference: {np.max(diff)}')
        print(' -------------------------------- ')

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        print('forward')
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        # torch.save(y_hat, output_check_path) #
        x_hat = self.g_s(y_hat)

        onnx_path = 'D:\_AIR\compressai\onnx'
        check_path = 'D:\_AIR\compressai\check'
        # g_a: x -> y -> y_hat
        # self._convert(self.g_a, "g_a", ['x_input'], x.shape, ['y_output'], onnx_path)
        # self._convert_y_encoder("g_a", x.shape, onnx_path); 
        self._check_onnx_model_accuracy('g_a', 'x_input', x, y_hat, check_path, onnx_path)
        # h_a: abs(y) -> z -> z_hat
        # self._convert(self.h_a, "h_a", ['y_input'], torch.abs(y).shape, ['z_output'], onnx_path)
        # self._convert_z_encoder("h_a", torch.abs(y).shape, onnx_path)
        self._check_onnx_model_accuracy('h_a', 'y_input', torch.abs(y), z_hat, check_path, onnx_path)
        # h_s: z_hat -> scales_hat 
        # self._convert(self.h_s, "h_s", ['z_hat_input'], z_hat.shape, ['scales_hat_output'], onnx_path) 
        self._check_onnx_model_accuracy('h_s', 'z_hat_input', z_hat, scales_hat, check_path, onnx_path)
        # g_s: y_hat -> x_hat
        # self._convert(self.g_s, "g_s", ['y_hat_input'], y_hat.shape, ['x_hat_output'], onnx_path)
        self._check_onnx_model_accuracy('g_s', 'y_hat_input', y_hat, x_hat, check_path, onnx_path) 

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

