# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import torch
 
class Aggregation(torch.nn.Module):
    """
    Aggregation module that can perform sum, mean, or pick operations on input tensors.
 
    Args:
        mode (str): The aggregation mode, either 'sum', 'mean', or 'pick'.
        dim (int): The dimension along which to apply the aggregation.
        idx (int): The index to pick when mode is 'pick'.
    """
    def __init__(self, mode: str = 'mean', dim: int = 1, idx: int = -1):
        super(Aggregation, self).__init__()
        if mode not in ['sum', 'mean', 'pick']:
            raise ValueError(f"Unsupported aggregation mode: {mode}. Supported modes are 'sum', 'mean', and 'pick'.")
        self.mode = mode
        self.dim = dim
        self.idx = idx
 
    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies the aggregation operation to the input tensor.
 
        Args:
            xi (torch.Tensor): The input tensor.
 
        Returns:
            torch.Tensor: The aggregated output tensor.
        """
        if self.mode == 'sum':
            return torch.sum(xi, dim=self.dim)
        elif self.mode == 'mean':
            return torch.mean(xi, dim=self.dim)
        elif self.mode == 'pick':
            return torch.index_select(xi, dim=self.dim, index=torch.tensor([self.idx]))
        else:
            raise NotImplementedError