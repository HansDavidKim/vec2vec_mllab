### Diffuser is used for processing (originally) audio data with UNet1DModel.
### In this case, however, it is used to process vector data.
import diffusers
import torch

from translators.transforms.AbsTransform import AbsTransform

'''
UNet1dTransform is a custom transform that uses a UNet1DModel
to process input data of a specified source dimension and project it to a target dimension.
 
Those transforms are units that can be used in a pipeline to transform data of Vex2Vec.
'''
class UNet1dTransform(AbsTransform):
    def __init__(self, src_dim: int, target_dim: int):
        super().__init__()

        ### UNet 1D Model is designed for 1D data, such as audio or time series.
        ### Here, it is adapted to work with vector data.
        ### 
        ### UNet is a type of encoder-decoder architecture that is 
        ### commonly used in image segmentation tasks.
        ###
        ### In this case, it is one of transform units that can be 
        ### used in a pipeline to transform data of Vec2Vec.

        self.base = diffusers.UNet1DModel(
            sample_size=src_dim,  
            in_channels=1, 
            out_channels=1,  
            layers_per_block=2,  
            block_out_channels=(128, 128, 256, 256, 512, 512),  
            down_block_types=(
                "DownBlock1D",  
                "DownBlock1D",
                "DownBlock1D",
                "DownBlock1D",
                "AttnDownBlock1D",  
                "DownBlock1D",
            ),
            up_block_types=(
                "UpBlock1D",  
                "AttnUpBlock1D",  
                "UpBlock1D",
                "UpBlock1D",
                "UpBlock1D",
                "UpBlock1D",
            ),
        )
        self.internal_dim = src_dim
        self.project = torch.nn.Linear(src_dim, target_dim)


    ### Forward propagation method for the UNet1dTransform.
    ### 
    ### It takes a tensor `x` as input, reshapes it, and passes it
    ### through the UNet1DModel to obtain a transformed output.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.internal_dim), f"invalid shapes: {x.shape} should be {(batch_size, self.internal_dim)}"
        
        x_r = x.view(batch_size, 1, self.internal_dim).contiguous()
        output = self.base(x_r, timestep=0)

        z = output.sample.view(batch_size, self.internal_dim).contiguous()
        return self.project(z)