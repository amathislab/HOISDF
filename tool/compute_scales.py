# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from AlignSDF
# ------------------------------------------------------------------------------

import os
import pickle

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm


matplotlib.use("Agg")


def data_analysis(dataset, mode):
    data_dir = f"{dataset}/{mode}/"
    norm_dir = data_dir + "norm/"
    meta_dir = data_dir + "meta/"
    hand_dir = data_dir + "sdf_hand/"
    obj_dir = data_dir + "sdf_obj/"
    rgb_dir = data_dir + "rgb/"

    # if 'obman' in dataset or 'ho3d' in dataset:
    cam_extr = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    # else:
    #     cam_extr = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # dist_hand_points = []
    # dist_obj_points = []
    sample_idx = []
    scale_list = []
    hand_scale_list = []
    obj_scale_list = []
    filenames = os.listdir(hand_dir)
    for filename in tqdm(filenames[131226:]):
        sample_idx.append(filename.split(".")[0])
        scale = np.load(os.path.join(norm_dir, filename))["scale"]
        offset = np.load(os.path.join(norm_dir, filename))["offset"]

        img = cv2.imread(os.path.join(rgb_dir, filename.split(".")[0] + ".jpg"))

        hand_data = np.load(os.path.join(hand_dir, filename))
        hand_pos_xyz = hand_data["pos"][:, :3]
        hand_neg_xyz = hand_data["neg"][:, :3]

        obj_data = np.load(os.path.join(obj_dir, filename))
        obj_pos_xyz = obj_data["pos"][:, :3]
        obj_neg_xyz = obj_data["neg"][:, :3]

        # transform all points into camera space
        hand_pos_xyz_cam = hand_pos_xyz / scale - offset
        hand_neg_xyz_cam = hand_neg_xyz / scale - offset
        obj_pos_xyz_cam = obj_pos_xyz / scale - offset
        obj_neg_xyz_cam = obj_neg_xyz / scale - offset

        with open(os.path.join(meta_dir, filename.replace("npz", "pkl")), "rb") as f:
            meta_data = pickle.load(f)

        if "obman" in dataset:
            cam_joints = np.dot(
                cam_extr, meta_data["coords_3d"].transpose(1, 0)
            ).transpose(1, 0)
        else:
            cam_joints = np.dot(
                cam_extr, meta_data["handJoints3D"].transpose(1, 0)
            ).transpose(1, 0)
        hand_neg_dist_wrist = np.linalg.norm(hand_neg_xyz_cam - cam_joints[0], axis=1)
        obj_neg_dist_wrist = np.linalg.norm(obj_neg_xyz_cam - cam_joints[0], axis=1)

        # dist_hand_points.append(np.max(hand_neg_dist_wrist))
        # dist_obj_points.append(np.max(obj_neg_dist_wrist))
        hand_scale_list.append(1 / np.max(hand_neg_dist_wrist))
        obj_scale_list.append(1 / np.max(obj_neg_dist_wrist))
        scale_list.append(scale)
    np.save(data_dir + "hand_scale_list.npy", np.array(hand_scale_list))
    np.save(data_dir + "obj_scale_list.npy", np.array(obj_scale_list))
    np.save(data_dir + "scale_list.npy", np.array(scale_list))

    # np.savez(os.path.join(norm_dir, filename), scale=scale, offset=offset, scale_hand = 1 / np.max(hand_neg_dist_wrist))


if __name__ == "__main__":
    dataset_path = "/media/data1/Dataset/HO3D_v2/First_sdf"
    # print('compute for ho3d train')
    # data_analysis(dataset_path, 'train')

    # dataset_path = '/media/data1/Dataset/HO3D_v2/First_sdf'
    # print('compute for ho3d test')
    # data_analysis(dataset_path, 'test')

    # dataset_path = '/media/data1/implicit_shape/AlignSDF/data/obman'
    # print('compute for obman train')
    # data_analysis(dataset_path, 'train')

    data_dir = dataset_path + "/train/"
    hand_scale_list = np.load(data_dir + "hand_scale_list.npy")
    obj_scale_list = np.load(data_dir + "obj_scale_list.npy")
    scale_list = np.load(data_dir + "scale_list.npy")
    aa = 1
