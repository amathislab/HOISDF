# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from HFL-Net
# ------------------------------------------------------------------------------

import os

import numpy as np

_YCB_CLASSES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}


def load_objects_dex_ycb(dataset_root):
    import trimesh

    model_dir = os.path.join(dataset_root, "models")
    obj_files = {
        k: np.array(trimesh.load(os.path.join(model_dir, v, "points.xyz")).vertices)
        for k, v in _YCB_CLASSES.items()
    }
    return obj_files


def projectPoints(xyz, K, rt=None):
    if rt is not None:
        cam_3D_points = (np.matmul(rt[:3, :3], xyz.T) + rt[:3, 3].reshape(-1, 1)).T
        uv = np.matmul(K, np.matmul(rt[:3, :3], xyz.T) + rt[:3, 3].reshape(-1, 1)).T
    else:
        uv = np.matmul(K, xyz.T).T
        cam_3D_points = None
    return cam_3D_points, uv[:, :2] / uv[:, -1:]


def get_bbox(joint_img, joint_valid, expansion_factor=1.0):

    x_img, y_img = joint_img[:, 0], joint_img[:, 1]
    x_img = x_img[joint_valid == 1]
    y_img = y_img[joint_valid == 1]
    xmin = min(x_img)
    ymin = min(y_img)
    xmax = max(x_img)
    ymax = max(y_img)

    x_center = (xmin + xmax) / 2.0
    width = (xmax - xmin) * expansion_factor
    xmin = x_center - 0.5 * width
    xmax = x_center + 0.5 * width

    y_center = (ymin + ymax) / 2.0
    height = (ymax - ymin) * expansion_factor
    ymin = y_center - 0.5 * height
    ymax = y_center + 0.5 * height

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox


def process_bbox(bbox, img_width, img_height, expansion_factor=1.25):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None
    return bbox


def pose_from_initial_martrix(Matrix):
    pose = np.zeros((4, 4))
    pose[:3, :] = Matrix
    pose[3, 3] = 1
    return pose
