from abc import ABC, abstractmethod
from huggingface_hub import PyTorchModelHubMixin

import torch
from torch import nn

'''
Abstract Neural Translator
''' 
class AbsNTranslator(nn.Module, ABC, PyTorchModelHubMixin):
    def __init__(
        self,
        encoder_dims: dict[str, int],
        d_adapter: int,
        depth: int = 3,
    ):
        super().__init__()
        if d_adapter is None:
            d_adapter = d_model
        
        self.n = len(encoder_dims)

        ### d_adapter stands for the dimension of adapters.
        ### Normally, input adapter projects the input to d_adapter dimension.
        ### Output adapter projects the output from d_adapter dimension to the original dimension.
        
        self.d_adapter = d_adapter
        self.depth = depth
        self.in_adapters = nn.ModuleDict()
        self.out_adapters = nn.ModuleDict()
        self.transform = None

    @abstractmethod
    def _make_adapters(self):
        pass

    ### str stands for the name of the encoder while torch.Tensor stands for embeddings.
    ### This translators consist of multiple adapters like A1, A2, T, B1, B2.
    ###
    ### Combining those, we can make translator function F1, F2 and reconstruct function R1, R2.
    @abstractmethod
    def forward(self, ins: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pass