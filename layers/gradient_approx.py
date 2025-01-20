# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.


import torch

class ThreshActRectangleGrad_INFER(torch.autograd.Function):
    """
    Threshold activation function with rectangular gradient for inference.
    """
    @staticmethod
    def forward(ctx, input, thresh=0.0):
        """
        Forward pass of the threshold activation function.
        
        Args:
            input (torch.Tensor): The input tensor.
            thresh (float): The threshold value.
            
        Returns:
            torch.Tensor: The binarized output tensor.
        """
        bina = torch.gt(input, thresh).to(input.dtype)
        return bina


class ThreshActRectangleGrad(torch.autograd.Function):
    """
    Threshold activation function with rectangular gradient for training.
    """
    @staticmethod
    def forward(ctx, input, thresh=0.0, approx_region=0.5):
        """
        Forward pass of the threshold activation function.
        
        Args:
            input (torch.Tensor): The input tensor.
            thresh (float): The threshold value.
            approx_region (float): The approximation region for the gradient.
            
        Returns:
            torch.Tensor: The binarized output tensor.
        """
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        ctx.approx_region = approx_region
        bina = torch.gt(input, thresh).to(input.dtype)
        return bina

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the threshold activation function.
        
        Args:
            grad_output (torch.Tensor): The gradient of the loss w.r.t. the output.
            
        Returns:
            tuple: The gradient of the loss w.r.t. the input and the parameters.
        """
        input, = ctx.saved_tensors
        thresh = ctx.thresh
        approx_region = ctx.approx_region
        temp = torch.abs(input - thresh) < approx_region
        grad_input = grad_output * temp.to(grad_output.dtype) / (2. * approx_region)
        return grad_input, None, None


class Threshold_act_IRNet_grad(torch.autograd.Function):
    """
    Threshold activation function with IRNet gradient.
    """
    @staticmethod
    def forward(ctx, input, k, t):
        """
        Forward pass of the threshold activation function.
        
        Args:
            input (torch.Tensor): The input tensor.
            k (float): A scaling factor.
            t (float): A temperature parameter.
            
        Returns:
            torch.Tensor: The sign of the input tensor.
        """
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the threshold activation function.
        
        Args:
            grad_output (torch.Tensor): The gradient of the loss w.r.t. the output.
            
        Returns:
            tuple: The gradient of the loss w.r.t. the input and the parameters.
        """
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


class Threshold_act_BiDet_grad(torch.autograd.Function):
    """
    Threshold activation function with Bi-Det gradient (placeholder).
    Note: The backward method should be implemented based on actual requirements.
    """
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass of the threshold activation function.
        
        Args:
            input (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The sign of the input tensor.
        """
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        return None,


class Threshold_act_ReActNet_grad(torch.autograd.Function):
    """
    Threshold activation function with ReActNet gradient.
    """
    @staticmethod
    def forward(ctx, input):
        """
        Forward pass of the threshold activation function.
        
        Args:
            input (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The sign of the input tensor.
        """
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the threshold activation function.
        
        Args:
            grad_output (torch.Tensor): The gradient of the loss w.r.t. the output.
            
        Returns:
            tuple: The gradient of the loss w.r.t. the input.
        """
        input, = ctx.saved_tensors
        grad = grad_output.clone()
        temp = (input <= 1).float() * (input >= -1).float()
        return grad * temp,