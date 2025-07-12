# Deals with the action-space, including defining the available actions
# Parameterizing them from layers
# Defining the proper alignment mechanism for calculating conditional rates, etc.

import torch.nn as nn

class Action(nn.Module):
    """
    Defines prediction heads for each action and its parameter space.

    Args:
        d_feature: dimension of the input/output feature vectors
        n_heads: number of attention heads per layer
        n_layers: number of layers in the model
    """
    def __init__(
        self,
        d_feature: int,
        n_heads: int,
        n_layers: int,
        tokenizer, # TODO interface for this type??

        # might need the mapping function from latent to actual
    ):
        super(EFM, self).__init__()
        self.embed = nn.Embedding(tokenizer.vocab_size, d_feature)

    def forward(self, x):
        return self.embed(x)
