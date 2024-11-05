import torch
from torch import nn
from typing import List, Dict, Optional
from ase import Atoms
from ..utils import _scatter_add, find_distances, add_scaling
import lightning as L

class AtomicModule(L.LightningModule, nn.Module):

    def forward(self,
                batch_data   : Dict[str, torch.Tensor],
                properties   : Optional[List[str]]=None,
                create_graph : bool=True
                ) -> Dict[str, torch.Tensor]:
        # Use properties=None instead of properties=['energy'] because
        # Mutable default parameters are not supported because
        # Python binds them to the function and they persist across function calls.
        required_derivatives = torch.jit.annotate(List[str], [])

        output_tensors = self.calculate(batch_data)

        return output_tensors

    def calculate(self):
        raise NotImplementedError(f"{self.__class__.__name__} must have 'calculate'!")


# Only support energy model now!
class MultiAtomicModule(AtomicModule):
    def __init__(self, models: Dict[str, AtomicModule]) -> None:
        super().__init__()
        self.models = nn.ModuleDict(models)

    def calculate(self,
                  batch_data : Dict[str, torch.Tensor],
                  ) -> Dict[str, torch.Tensor]:
        output_tensor = {'site_energy': torch.zeros_like(batch_data['atomic_number'], dtype=batch_data['coordinate'].dtype)}
        for name, model in self.models.items():
            output_tensor['site_energy'] += model.calculate(batch_data)['site_energy']
        return output_tensor
