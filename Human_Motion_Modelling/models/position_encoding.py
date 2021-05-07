# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

class PositionEmbeddingSine_1D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):

        # mask shape: N * L 
        '''
        not_mask = ~mask
        position = not_mask.cumsum(1, dtype=torch.float32)
        '''
        N, L = mask.shape
        position = torch.arange(0, L, dtype=torch.float32, device=mask.device).unsqueeze(0)
        position = position.repeat(N, 1)

        if self.normalize:
            eps = 1e-6
            position = position / (position[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # shape: N * L * 1 / 1 * C -> N * L * C

        pe = torch.zeros(mask.shape[0], mask.shape[1], self.num_pos_feats*2)
        pe[:, :, 0::2] = torch.sin(position[:, :, None] / dim_t)
        pe[:, :, 1::2] = torch.cos(position[:, :, None] / dim_t)

        # shape: N * L * C -> L * N * C

        pe = pe.permute(1, 0, 2)

        return pe

class PositionEmbeddingLearned_1D(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.embed = nn.Embedding(160, num_pos_feats*2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, mask):

        # N * L
        l = mask.shape[-1]

        position = torch.arange(l, device=mask.device)
        emb = self.embed(position)

        # N * L * C -> L * N * C
        pe = emb.unsqueeze(0).repeat(mask.shape[0], 1, 1).permute(1, 0, 2)

        return pe


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine_1D(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned_1D(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
