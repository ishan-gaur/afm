import torch
from esm.models.esmc import ESMC, ESMCOutput
from esm.layers.regression_head import RegressionHead
from typing import cast


class ESMCEmbedding(torch.nn.Module):
    """
    Wrapper around ESMC for getting protein sequence embeddings.
    Pretrained weights are bfloat16
    Model weights are multiples of 8/16 so when used with FlashAttention this should use fused kernels on tensor cores

    Tensor Indices:
        B: Batch Dimension
        L: Hidden Layer (0 is output of first transformer block, -1 are the traditional embeddings)
        S: Sequence Position
        H: ESMC Hidden Dimension
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_BSH: torch.Tensor, hidden_layer: int = -1, apply_layer_norm: bool = True):
        if apply_layer_norm and hidden_layer != -1:
            raise ValueError("apply_layer_norm is not supported for hidden_layer != -1, must use your own layer norm for activations before the last layer.")

        # move input to device automatically, but don't move back so that if the user is
        # moving things around unnecessarily they can fix that
        if x_BSH.device != self.device:
            x_BSH = x_BSH.to(self.device)

        # recent research somewhere that second to last or earlier embeddings can be better right? Le Cun tweet or smthg?
        assert hidden_layer < 0, "hidden_layer uses list-like indexing for the n-th-to-last set of model activations. -1 is the 'embeddings'"
        esmc_output: ESMCOutput = self.model(x_BSH)
        hiddens_LSH = cast(torch.Tensor, esmc_output.hidden_states) # complains that the hidden_states could technically be None
        hidden_SH = hiddens_LSH[hidden_layer + 1]

        # Layer norm is addditionally applied on the embeddings! Must apply layer norm yourself before the output layer
        # if you want to stay consistent with the actual ESMC architecture and using hidden states other than the last one
        # see in ESM code https://github.com/evolutionaryscale/esm/blob/e0669a4a05a05395dde08d4ff2eb0219bcc2f79e/esm/layers/transformer_stack.py#L94
        # embeddings = cast(torch.Tensor, esmc_output.embeddings)
        # assert torch.equal(self.model.transformer.norm(hiddens_LSH[-1]), embeddings), "Embeddings do not match with the last hidden state"

        return hidden_SH

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype


class ESMCOutputHead(torch.nn.Module):
    """
    Wrapper around ESMC for getting output logits from the model.
    Pretrained weights are bfloat16
    Model weights are multiples of 8/16 so when used with FlashAttention this should use fused kernels on tensor cores

    Tensor Indices:
        B: Batch Dimension
        L: Hidden Layer (0 is output of first transformer block, -1 are the traditional embeddings)
        S: Sequence Position
        H: ESMC Hidden Dimension
    """
    def __init__(self, model: ESMC):
        super().__init__()
        self.model = model
        d_model = self.model.sequence_head.get_submodule("0").get_parameter("weight").shape[1] # since in ml it is output x input dim
        output_dim = self.model.sequence_head.get_submodule("3").get_parameter("weight").shape[0]
        reinit_sequence_head = RegressionHead(d_model, output_dim).to(self.device).to(self.dtype)
        self.model.sequence_head = reinit_sequence_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.model.sequence_head(hidden_states)
        return output

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return next(self.model.sequence_head.parameters()).dtype


def test_plm_wrapper():
    from esm.models.esmc import ESMC
    from esm.utils.constants.models import ESMC_300M
    from afm.utils.tokenizers import tokenize_protein_batch

    model = ESMC.from_pretrained(ESMC_300M)
    print(model)
    embed = ESMCEmbedding(model)
    print(embed.dtype)
    output_head = ESMCOutputHead(model)
    sequence_batch = [
        "<cls>LAGVSERTIDPKQN<unk>FYMHWCXBUZO.-|<mask><eos><pad>"
        "<cls>LAGVSERTIDPKQN<unk>.-|<mask><eos>"
    ]
    batch = tokenize_protein_batch(sequence_batch)
    embeddings_BSH = embed(batch)
    print("EMBEDDINGS")
    print(embeddings_BSH.dtype)
    print(embeddings_BSH.shape)
    output = output_head(embeddings_BSH)
    print("OUTPUT")
    print(output.dtype)
    print(output.shape)

if __name__ == "__main__":
    test_plm_wrapper()
