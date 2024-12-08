import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from lib.modules import Classifier, Module

"""From d2l"""


def masked_softmax(X, valid_lens):
    """Perform softmax along each row, masking elements on the last axis."""

    # X: 3D tensor (batch*num_queries, num_keys), valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = (
            torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
            < valid_len[:, None]
        )  # broadcast column < pad[j] along rows
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # flatten( [pad_i * num_queries] for i in batch )
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # First pad_ij tokens are allowed for query i*batch_size+j. i.e. [1, 2 ... num_queries]
            valid_lens = valid_lens.reshape(-1)
        # -infty exponentiates to 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """Scaled dot product attention."""

    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # (batch_size, num_queries / num_kv , d/value_d) -> (batch_size, num_queries, num_kv)
    # Shape of valid_lens: (batch_size,) or (batch_size, num_queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(Module):
    """Multi-head attention."""

    def __init__(self, hidden_size, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(hidden_size, bias=bias)
        self.W_k = nn.LazyLinear(hidden_size, bias=bias)
        self.W_v = nn.LazyLinear(hidden_size, bias=bias)
        self.W_o = nn.LazyLinear(hidden_size, bias=bias)
        self.s

    def forward(self, queries, keys, values, valid_lens):
        # Transforms the last dimension of each input: (batch_size, num_queries/num_keys = num_values, query/key/value size -> hidden_size)
        # For self-attention, q, k, v = x
        # Transpose rotates qkv to (batch_size * num_heads, num_queries, hidden_size / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        # Shape of valid_lens: (batch_size,) or (batch_size, num_queries)
        # each sequence in the batch can have a different valid length, we can use
        # valid_lens to obscure padding, as well as future info for self-attention
        # Adapt to interleaving by copy the first item (vector or to be broadcasted scalar,
        # representing to keys to keep for each query in a batch) for num_heads
        # times, then copy the next item, and so on, see transpose_qkv
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        # Shape of output: (batch_size * num_heads, num_queries,
        # hidden_size / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Shape is unchanged: (batch_size, num_queries, hidden_size)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads.

        Defined in :numref:`sec_multihead-attention`"""
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        # Shape of output: (batch_size * num_heads, num_queries, hidden_size / num_heads)
        # each head attends to a different section (every num_heads values) of the input feature space, so that [0:num_heads, :] gives the first section of each head, or the first contiguous block of X
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv.

        Defined in :numref:`sec_multihead-attention`"""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)
