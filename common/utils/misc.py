# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from Keypoint Transformer
# ------------------------------------------------------------------------------

import torch
from main.config import cfg


def get_mano_tgt_mask():
    tgt_mask = torch.zeros(
        (cfg.mano_num_queries, cfg.mano_num_queries), dtype=torch.bool
    )
    # global rot
    tgt_mask[0, :] = True
    tgt_mask[0, 0] = False

    # fingers
    for i in range(5):
        # right hand
        s = 3 * i + 1
        e = 3 * i + 4
        tgt_mask[s:e, :] = True
        tgt_mask[s:e, s:e] = False

    # trans and shape
    tgt_mask[cfg.mano_shape_indx, :] = True
    tgt_mask[cfg.mano_shape_indx, cfg.mano_shape_indx] = False

    return tgt_mask


def get_manoshape_memory_mask():
    memory_mask = torch.zeros(
        (1, cfg.num_samp_hand + cfg.num_samp_obj), dtype=torch.bool
    )
    memory_mask[:, cfg.num_samp_hand :] = True
    return memory_mask


def get_mano_memory_mask():
    memory_mask = torch.zeros(
        (cfg.mano_num_queries, cfg.num_samp_hand + cfg.num_samp_obj), dtype=torch.bool
    )
    memory_mask[:, cfg.num_samp_hand :] = True
    return memory_mask
