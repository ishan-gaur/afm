import torch.nn as nn

class EFM(nn.Module):
    """
    Edit Flow Model

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
    ):
        super(EFM, self).__init__()
        self.embed = nn.Embedding(tokenizer.vocab_size, d_feature)


    def forward(self, x):
        return self.embed(x)
