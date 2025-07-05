import torch
import torch.nn as nn

'''
Discriminator Class for GAN-like Vec2Vec
unsupervised embedding translation

@param latent_dim: Dimension of the input latent space.
@param discriminator_dim: Dimension of the hidden layers in the discriminator.
@param depth: Number of hidden layers in the discriminator.
@param weight_init: Type of weight initialization to use ('kaiming', 'xavier', 'orthogonal').
'''
class Discriminator(nn.Module):
    def __init__(self, latent_dim, discriminator_dim: int = 1024, depth: int = 3, weight_init: str = 'kaiming'):
        super().__init__()

        self.latent_dim = latent_dim

        assert depth >= 1, "Depth must be at least 1"
        self.layers = nn.ModuleList()
        if depth >= 2:
            layers = []
            layers.append(nn.Linear(latent_dim, discriminator_dim))

            ### They originally used Dropout but discarded it.
            ### It might decreased the performance of the model.

            layers.append(nn.Dropout(0.0))
            for _ in range(depth - 2):
                layers.append(nn.SiLU())
                layers.append(nn.Linear(discriminator_dim, discriminator_dim))
                layers.append(nn.LayerNorm(discriminator_dim))
                layers.append(nn.Dropout(0.0))
            layers.append(nn.SiLU())
            layers.append(nn.Linear(discriminator_dim, 1))
            self.layers.append(nn.Sequential(*layers))
        else:
            self.layers.append(nn.Linear(latent_dim, 1))
        self.initialize_weights(weight_init)
    
    def initialize_weights(self, weight_init: str):
        ### Initialize weights of the model based on the specified method.
        ### Default is 'kaiming' initialization.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if weight_init == 'kaiming':
                    torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
                elif weight_init == 'xavier':
                    torch.nn.init.xavier_normal_(module.weight)
                elif weight_init == 'orthogonal':
                    torch.nn.init.orthogonal_(module.weight)
                else:
                    raise ValueError(f"Unknown weight initialization: {weight_init}")
                module.bias.data.fill_(0)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.constant_(module.bias, 0)
                torch.nn.init.constant_(module.weight, 1.0)

    ### This returns logits, which is the output of the last layer.
    ### Therefore, return value itself is not a probability.
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x