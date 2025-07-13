# Lol, we'll just use ESM to tokenize and then train the extra output logits of the model
# then we'll also just use ESM-C to get the embeddings and logits, only retraining the output layer
# I'll just need to extra the decoder of the ouputs to use the last 31 logits for the 25 AA tokens, ., -, |, pad, eos, mask
# don't need cls because only right insertions are allowed, I also don't know what you might want unk for
# for now anyways only the AA and - will show up in the data, the rest will also manually be masked out, but for general actions,
# they might be useful

import torch
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from typing import List, cast
from functools import cache

@cache
def _get_esm_tokenizer() -> EsmSequenceTokenizer:
    return EsmSequenceTokenizer()

def tokenize_protein_batch(batch: List[str]) -> torch.Tensor:
    tokenizer = _get_esm_tokenizer()
    tokenized = tokenizer.batch_encode_plus(
        batch,
        padding=True,
        truncation=False,
        return_tensors='pt'
    )
    tokenized_batch_BSH = cast(torch.Tensor, tokenized['input_ids']) # otherwise type EncodingFast
    return tokenized_batch_BSH
