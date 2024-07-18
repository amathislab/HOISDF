# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from AlignSDF
# ------------------------------------------------------------------------------

import os

import numpy as np
import torch


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def filter_invalid_sdf(tensor, lab_tensor, dist):
    keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
    return tensor[keep, :], lab_tensor[keep, :]


def filter_invalid_sdf_or(tensor, lab_tensor, dist):
    keep = (torch.abs(tensor[:, 3]) < abs(dist)) | (torch.abs(tensor[:, 4]) < abs(dist))
    return tensor[keep, :], lab_tensor[keep, :]


def filter_invalid_sdf_type(tensor, lab_tensor, dist, hand):
    if hand:
        keep = torch.abs(tensor[:, 3]) < abs(dist)
    else:
        keep = torch.abs(tensor[:, 4]) < abs(dist)
    return tensor[keep, :], lab_tensor[keep, :]


def unpack_sdf_samples(
    data_source, key, hand=True, clamp=None, filter_dist=False, dist=2.0
):
    if hand:
        npz_path = os.path.join(data_source, "sdf_hand", key + ".npz")
    else:
        npz_path = os.path.join(data_source, "sdf_obj", key + ".npz")

    npz = np.load(npz_path)

    try:
        pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
        neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
        pos_sdf_other = torch.from_numpy(npz["pos_other"])
        neg_sdf_other = torch.from_numpy(npz["neg_other"])
        if hand:
            lab_pos_tensor = torch.from_numpy(npz["lab_pos"])
            lab_neg_tensor = torch.from_numpy(npz["lab_neg"])
        else:
            lab_pos_tensor = torch.from_numpy(npz["lab_pos_other"])
            lab_neg_tensor = torch.from_numpy(npz["lab_neg_other"])
    except Exception as e:
        print("fail to load {}, {}".format(key, e))

    if hand:
        pos_tensor = torch.cat([pos_tensor, pos_sdf_other], 1)
        neg_tensor = torch.cat([neg_tensor, neg_sdf_other], 1)
    else:
        xyz_pos = pos_tensor[:, :3]
        sdf_pos = pos_tensor[:, 3].unsqueeze(1)
        pos_tensor = torch.cat([xyz_pos, pos_sdf_other, sdf_pos], 1)

        xyz_neg = neg_tensor[:, :3]
        sdf_neg = neg_tensor[:, 3].unsqueeze(1)
        neg_tensor = torch.cat([xyz_neg, neg_sdf_other, sdf_neg], 1)

    if filter_dist:
        pos_tensor, lab_pos_tensor = filter_invalid_sdf(
            pos_tensor, lab_pos_tensor, dist
        )
        neg_tensor, lab_neg_tensor = filter_invalid_sdf(
            neg_tensor, lab_neg_tensor, dist
        )

    hand_part_pos = lab_pos_tensor[:, 0]
    hand_part_neg = lab_neg_tensor[:, 0]
    samples = torch.cat([pos_tensor, neg_tensor], 0)

    labels = torch.cat([hand_part_pos, hand_part_neg], 0)

    if clamp:
        labels[samples[:, 3] < -clamp] = -1
        labels[samples[:, 3] > clamp] = -1

    if not hand:
        labels[:] = -1

    return samples, labels


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_nerf_embedder(multires):
    embed_kwargs = {
        "include_input": False,
        "input_dims": 3,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim
