import torch


def sequence_mask_by_len(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
             .type_as(lengths)
             .unsqueeze(0)
             .expand(batch_size, max_len)
             .lt(lengths.unsqueeze(1)))


def random_mask_by_pos(embeddings, word_dropout=0.2):
    batch, length, size = embeddings.size()
    mask = embeddings.new_empty(batch, length)
    mask = mask.bernoulli_(1 - word_dropout)
    embeddings = embeddings * mask.unsqueeze(-1).expand_as(embeddings).float()
    return embeddings, mask