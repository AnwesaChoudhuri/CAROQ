"""
---
Relative Multi-Headed Attention
Borrowed from:
https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/xl/relative_mha.py
"""

import torch
from torch import nn
import pdb
import math
from typing import Optional, List


def shift_right(x: torch.Tensor):
    """
    This method shifts $i^{th}$ row of a matrix by $i$ columns.
    If the input is `[[1, 2 ,3], [4, 5 ,6], [7, 8, 9]]`, the shifted
    result would be `[[1, 2 ,3], [0, 4, 5], [9, 0, 7]]`.
    *Ideally we should mask out the lower triangle but it's ok for our purpose*.
    """

    # Concatenate a column of zeros
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    x_padded = torch.cat([x, zero_pad], dim=1)

    # Reshape and remove excess elements from the end
    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])
    x = x_padded[:-1].view_as(x)

    #
    return x


class PrepareForMultiHeadAttention(nn.Module):
    """
    <a id="PrepareMHA"></a>
    ## Prepare for multi-head attention
    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """

    def __init__(self, d_model: int, heads: int, d_k: int):
        super().__init__()
        # Linear layer for linear transform
        self.d_model = d_model

        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def linear(self, input, weight=None, bias=None):
        return torch.matmul(input, weight.to(input.device)) + bias.to(input.device)

    def forward(self, x: torch.Tensor, weight=None, bias=None):

        # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the heads.
        head_shape = x.shape[:-1]

        # Linear transform
        x = self.linear(x, weight=weight.transpose(0, 1), bias=bias)

        # Split last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, d_model]`
        return x


class MultiHeadAttention(nn.Module):
    r"""
    <a id="MHA"></a>
    ## Multi-Head Attention Module
    This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.
    $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
    In simple terms, it finds keys that matches the query, and gets the values of
     those keys.
    It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
    This is done to avoid large dot-product values causing softmax to
    give very small gradients when $d_k$ is large.
    Softmax is calculated along the axis of of the sequence (or time).
    """

    def __init__(
        self, heads: int, d_model: int, dropout: float = 0.0, bias: bool = True
    ):
        """
        * `heads` is the number of heads.
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        """

        super().__init__()

        # Number of features per head
        self.d_k = d_model // heads
        self.d_model = d_model
        # Number of heads
        self.heads = heads
        self.modes = {"query": 0, "key": 1, "value": 2}

        self.in_proj_weight = nn.Parameter(
            torch.empty(self.d_model * 3, self.d_model), requires_grad=True
        )
        self.in_proj_bias = nn.Parameter(
            torch.empty(self.d_model * 3), requires_grad=True
        )
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k)

        # Softmax for attention along the time dimension of `key`
        self.softmax = nn.Softmax(dim=1)

        # Output layer
        self.out_proj = nn.Linear(self.d_model, self.heads * self.d_k)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)

    def get_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_pos=None,
        query_pos=None,
        pos_embeddings=None,
        rel_scale=0,
    ):
        """
        ### Calculate scores between queries and keys
        This method can be overridden for other variations like relative attention.
        """

        # Calculate $Q K^\top$ or $S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}$
        return torch.einsum("ibhd,jbhd->ijbh", query, key).float()

    def prepare_mask(
        self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]
    ):
        """
        `mask` has shape `[seq_len_q, seq_len_k, batch_size]`, where first dimension is the query dimension.
        If the query dimension is equal to $1$ it will be broadcasted.
        """
        if mask.dim() == 4:
            mask = mask.permute(2, 3, 0, 1)
        elif mask.dim() == 3:
            mask = mask.permute(1, 2, 0)

        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        if mask.dim() == 3:
            # Same mask applied to all heads.
            mask = mask.unsqueeze(-1)

        # resulting mask has shape `[seq_len_q, seq_len_k, batch_size, heads]`
        return mask

    def forward(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_pos=None,
        query_pos=None,
        pos_embeddings=None,
        rel_scale=0
    ):
        """
        `query`, `key` and `value` are the tensors that store
        collection of *query*, *key* and *value* vectors.
        They have shape `[seq_len, batch_size, d_model]`.
        `mask` has shape `[seq_len, seq_len, batch_size]` and
        `mask[i, j, b]` indicates whether for batch `b`,
        query at position `i` has access to key-value at position `j`.
        """

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias

        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[seq_len, batch_size, heads, d_k]`.

        query = self.query(
            query,
            weight=in_proj_weight[
                self.modes["query"]
                * self.d_model : (self.modes["query"] + 1)
                * self.d_model
            ],
            bias=in_proj_bias[
                self.modes["query"]
                * self.d_model : (self.modes["query"] + 1)
                * self.d_model
            ],
        )

        key = self.key(
            key,
            weight=in_proj_weight[
                self.modes["key"]
                * self.d_model : (self.modes["key"] + 1)
                * self.d_model
            ],
            bias=in_proj_bias[
                self.modes["key"]
                * self.d_model : (self.modes["key"] + 1)
                * self.d_model
            ],
        )

        value = self.value(
            value,
            weight=in_proj_weight[
                self.modes["value"]
                * self.d_model : (self.modes["value"] + 1)
                * self.d_model
            ],
            bias=in_proj_bias[
                self.modes["value"]
                * self.d_model : (self.modes["value"] + 1)
                * self.d_model
            ],
        )

        del in_proj_bias, in_proj_weight

        scores = self.get_scores(
            query,
            key,
            key_pos=key_pos,
            query_pos=query_pos,
            pos_embeddings=pos_embeddings,
            rel_scale=rel_scale,
        )

        scores *= self.scale

        del key, query

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 1, -1e9)

        attn = self.softmax(scores)

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        del attn, value

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        return self.out_proj(x)


class RelativeMultiHeadAttention(MultiHeadAttention):
    """
    ## Relative Multi-Head Attention Module
    We override [Multi-Head Attention](mha.html) module so we only need to
    write the `get_scores` method.
    """

    def __init__(self, heads: int, d_model: int, dropout: float = 0.1, P1=100, P2=100):
        # The linear transformations do not need a bias since we
        # explicitly include it when calculating scores.
        # However having a bias for `value` might make sense.
        super().__init__(heads, d_model, dropout, bias=False)

        # Number of relative positions
        self.P1 = P1
        self.P2 = P2

        # self.Wr=Wr

        self.key_pos_embeddings = nn.Parameter(
            torch.empty(self.P1 + self.P2, heads, self.d_k), requires_grad=True
        )
        nn.init.xavier_normal_(self.key_pos_embeddings)

    def get_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_pos=None,
        query_pos=None,
        pos_embeddings=None,
        rel_scale=0,
    ):

        a = torch.einsum("ibhd,jbhd->ijbh", query, key)
        if pos_embeddings == None:
            key_pos_emb = self.key_pos_embeddings[
                self.P1 - key.shape[0] : self.P1 + query.shape[0]
            ]
        else:
            key_pos_emb = pos_embeddings.weight.view(
                pos_embeddings.weight.shape[0], self.heads, self.d_k
            )[self.P - key.shape[0] : self.P + query.shape[0]]
        b = torch.einsum("ibhd,jhd->ijbh", query, key_pos_emb.to(query.device))

        b = shift_right(b)
        b = b[:, -key.shape[0] :]

        del key_pos_emb

        return (a + b + rel_scale * b * b).float()
