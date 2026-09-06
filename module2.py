"""
SPQ - Module 2
Uniform Quantization Module

Purpose:
    Convert FP32 activation tensors into INT8
    using scale and zero-point parameters.

Quantization:
    q = clip(floor(x / S) + Z, q_min, q_max)

Dequantization:
    x_hat = S * (q - Z)
"""

import torch


class UniformQuantizer:

    def __init__(
        self,
        q_min=-128,
        q_max=127
    ):
        self.q_min = q_min
        self.q_max = q_max

    def calculate_parameters(self, x):
        """
        Calculate scale (S) and zero-point (Z).
        """

        x_min = x.min()
        x_max = x.max()

        # Symmetric INT8 quantization
        max_abs = torch.maximum(
            torch.abs(x_min),
            torch.abs(x_max)
        )

        scale = max_abs / 127.0

        # Avoid division by zero
        scale = torch.clamp(
            scale,
            min=torch.finfo(
                torch.float32
            ).eps
        )

        zero_point = torch.tensor(
            0.0,
            dtype=torch.float32
        )

        return scale, zero_point

    def quantize(
        self,
        x,
        scale,
        zero_point
    ):
        """
        FP32 → INT8

        q = clip(
                floor(x/S) + Z,
                q_min,
                q_max
            )
        """

        q = torch.floor(
            x / scale
        ) + zero_point

        q = torch.clamp(
            q,
            self.q_min,
            self.q_max
        )

        return q.to(torch.int8)

    def dequantize(
        self,
        q,
        scale,
        zero_point
    ):
        """
        INT8 → simulated FP32

        x_hat = S * (q - Z)
        """

        return scale * (
            q.float() - zero_point
        )
