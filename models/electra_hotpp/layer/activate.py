# TODO: There are now in-place operations in TensorTanh and TensorRelu, why?
import torch
import torch.nn.functional as F
from .base import TensorActivateLayer
from ..utils import expand_to


class TensorTanh(TensorActivateLayer):

    def activate(self,
                 input_tensor: torch.Tensor,
                 ) -> torch.Tensor:
        return torch.tanh(input_tensor)

    def tensor_activate(self, input_tensor: torch.Tensor, way: int) -> torch.Tensor:
        #norm = self.weights * torch.sum(input_tensor ** 2, dim=tuple(range(2, 2 + way))) + self.bias
        #nonzero_norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        #factor = torch.tanh(norm) / nonzero_norm
        #return expand_to(factor, 2 + way) * input_tensor
        norm = torch.sum(input_tensor ** 2, dim=tuple(dim for dim in range(2, 2 + way)))
        norm = self.weights * norm + self.bias
        norm = torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
        eps = self._stability_eps.to(norm.dtype)
        safe_sign = torch.sign(norm)
        safe_sign = torch.where(safe_sign == 0, torch.ones_like(safe_sign), safe_sign)
        safe_norm = torch.where(norm.abs() < eps, safe_sign * eps, norm)
        factor = torch.tanh(norm) / safe_norm
        factor = torch.nan_to_num(factor, nan=0.0, posinf=0.0, neginf=0.0)
        output_tensor = input_tensor * expand_to(factor, 2 + way)
        return torch.nan_to_num(output_tensor, nan=0.0, posinf=0.0, neginf=0.0)


class TensorRelu(TensorActivateLayer):

    def activate(self,
                 input_tensor: torch.Tensor,
                 ) -> torch.Tensor:
        return F.relu(input_tensor)
    
    def tensor_activate(self, input_tensor: torch.Tensor, way: int) -> torch.Tensor:
        input_tensor_ = input_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1], -1)
        norm = self.weights * torch.sum(input_tensor_ ** 2, dim=2) + self.bias
        norm = torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
        factor = torch.heaviside(norm, torch.zeros_like(norm))
        output = expand_to(factor, 2 + way) * input_tensor
        return torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


class TensorSilu(TensorActivateLayer):
    """TensorSilu
    silu(x) = x * sigmoid(x), so we the factor should be F.sigmoid(norm)
    """

    def activate(self,
                 input_tensor: torch.Tensor,
                 ) -> torch.Tensor:
        return F.silu(input_tensor)

    def tensor_activate(self, input_tensor: torch.Tensor, way: int) -> torch.Tensor:
        input_tensor_ = input_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1], -1)
        norm = self.weights * torch.sum(input_tensor_ ** 2, dim=2) + self.bias
        norm = torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
        factor = torch.sigmoid(norm)
        output = expand_to(factor, 2 + way) * input_tensor
        return torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


class TensorJilu(TensorActivateLayer):
    """TensorJilu
    Similar to TensorSilu, but use use tanh(x) as factor so the factor could be negative
    """
    def activate(self,
                 input_tensor: torch.Tensor,
                 ) -> torch.Tensor:
        return F.silu(input_tensor)

    def tensor_activate(self, input_tensor: torch.Tensor, way: int) -> torch.Tensor:
        input_tensor_ = input_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1], -1)
        norm = self.weights * torch.sum(input_tensor_ ** 2, dim=2) + self.bias
        norm = torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
        factor = torch.tanh(norm)
        output = expand_to(factor, 2 + way) * input_tensor
        return torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    

class TensorIdentity(TensorActivateLayer):
    """TensorIdentity
    """
    def activate(self,
                 input_tensor: torch.Tensor,
                 ) -> torch.Tensor:
        return input_tensor

    def tensor_activate(self, input_tensor: torch.Tensor, way: int) -> torch.Tensor:
        return input_tensor


TensorActivateDict = {
    "tanh": TensorTanh,
    "relu": TensorRelu,
    "silu": TensorSilu,
    "jilu": TensorJilu,
    "none": TensorIdentity,
}
