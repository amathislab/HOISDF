# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from AlignSDF
# ------------------------------------------------------------------------------

import os

import numpy as np
import torch
from tqdm import tqdm


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def filter_invalid_sdf(tensor, lab_tensor, dist):
    keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
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

    # split the sample into half
    if filter_dist:
        pos_tensor, lab_pos_tensor = filter_invalid_sdf(
            pos_tensor, lab_pos_tensor, dist
        )
        neg_tensor, lab_neg_tensor = filter_invalid_sdf(
            neg_tensor, lab_neg_tensor, dist
        )

    hand_part_pos = lab_pos_tensor[:, 1]
    hand_part_neg = lab_neg_tensor[:, 1]
    samples = torch.cat([pos_tensor, neg_tensor], 0)
    labels = torch.cat([hand_part_pos, hand_part_neg], 0)

    if clamp:
        labels[samples[:, 3] < -clamp] = -1
        labels[samples[:, 3] > clamp] = -1

    if not hand:
        labels[:] = -1

    return samples, labels


sdf_path = "/media/data/haozhe/implicit_shape/HO3D_v2/First_sdf"
clamp = 0.05
dist = 2.0
filter_dist = True

output_dir = "sdf_processed"

sdf_dir_list = ["train", "test"]
for sdf_dir in sdf_dir_list:
    filelist = [
        filename.split(".")[0]
        for filename in os.listdir(os.path.join(sdf_path, sdf_dir, "sdf_hand"))
    ]
    filelist.sort()

    index_list = []
    for fname in tqdm(filelist):
        seq_name = fname.split("_")[0]
        frame_idx = fname.split("_")[1]

        sdf_norm = np.load(os.path.join(sdf_path, sdf_dir, "norm", fname + ".npz"))

        # sdf load
        hand_samples, hand_labels = unpack_sdf_samples(
            os.path.join(sdf_path, sdf_dir),
            seq_name + "_" + frame_idx,
            hand=True,
            clamp=clamp,
            filter_dist=filter_dist,
            dist=dist,
        )
        obj_samples, obj_labels = unpack_sdf_samples(
            os.path.join(sdf_path, sdf_dir),
            seq_name + "_" + frame_idx,
            hand=False,
            clamp=clamp,
            filter_dist=filter_dist,
            dist=dist,
        )

        # transform points into the camera coordinate system
        hand_samples[:, 0:3] = (
            hand_samples[:, 0:3] / sdf_norm["scale"] - sdf_norm["offset"]
        )
        obj_samples[:, 0:3] = (
            obj_samples[:, 0:3] / sdf_norm["scale"] - sdf_norm["offset"]
        )

        hand_samples = hand_samples.detach().numpy()
        obj_samples = obj_samples.detach().numpy()
        hand_labels = hand_labels.detach().numpy()
        obj_labels = obj_labels.detach().numpy()

        hand_samples[:, 3:] = hand_samples[:, 3:] / sdf_norm["scale"]
        obj_samples[:, 3:] = obj_samples[:, 3:] / sdf_norm["scale"]

        hand_data = np.concatenate((hand_samples, hand_labels[:, None]), axis=-1)
        obj_data = np.concatenate((obj_samples, obj_labels[:, None]), axis=-1)
        index_list.append([hand_data.shape[0], obj_data.shape[0]])
        sdf_data = np.concatenate((hand_data, obj_data), axis=0)
        np.save(
            os.path.join(sdf_path, sdf_dir, output_dir, fname + ".npy"),
            sdf_data.astype(np.float32),
        )
    np.save(os.path.join(sdf_path, sdf_dir, "sdf_index.npy"), np.array(index_list))
