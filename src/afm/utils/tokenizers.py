# Lol, we'll just use ESM to tokenize and then train the extra output logits of the model
# then we'll also just use ESM-C to get the embeddings and logits, only retraining the output layer
# I'll just need to extra the decoder of the ouputs to use the last 31 logits for the 25 AA tokens, ., -, |, pad, eos, mask
# don't need cls because only right insertions are allowed, I also don't know what you might want unk for
# for now anyways only the AA and - will show up in the data, the rest will also manually be masked out, but for general actions,
# they might be useful

from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer as ProteinTokenizer
