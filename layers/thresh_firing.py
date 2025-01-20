# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch
import torch.nn as nn

from .gradient_approx import ThreshActRectangleGrad

spike_func = ThreshActRectangleGrad.apply

class ThreshFiring(nn.Module):
    """
    Threshold firing module that applies a spiking activation function to the input tensor.

    Args:
        mode (str): The firing mode, either 'fixed' or 'randn'.
        noise_scale (float): The scale factor for noise addition in 'randn' mode.
    """
    def __init__(self, mode: str = 'fixed', noise_scale: float = 1e-9):
        super(ThreshFiring, self).__init__()
        if mode not in ['fixed', 'randn']:
            raise ValueError(f"Unsupported firing mode: {mode}. Supported modes are 'fixed' and 'randn'.")
        self.mode = mode
        self.noise_scale = noise_scale

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the spiking activation function to the input tensor.

        Args:
            xi (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the spiking activation function.
        """
        if self.mode == 'fixed':
            xo = spike_func(xi, 0.)
        elif self.mode == 'randn':
            if self.training:
                noise_ = torch.randn(xi.shape, dtype=xi.dtype, device=xi.device)
                scale = torch.std(xi, dim=[i for i in range(2, len(xi.shape))], keepdim=True) * self.noise_scale
                noise = scale * noise_
            else:
                noise = 0.
            xo = spike_func(xi, noise)
        else:
            raise NotImplementedError
        return xo