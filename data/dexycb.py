# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from HFL-Net
# ------------------------------------------------------------------------------

import random

import numpy as np
from PIL import Image, ImageFile, ImageFilter
from torch.utils import data

ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import os

import cv2
import torch
import torchvision.transforms as transforms

from tqdm import tqdm

from data import dataset_util, dex_ycb_util
from main.config import cfg
from manopth.manopth.manolayer import ManoLayer


class Dataset(data.Dataset):
    def __init__(
        self,
        mode="evaluation",
        max_rot=np.pi,
        scale_jittering=0.2,
        center_jittering=0.1,
        hue=0.15,
        saturation=0.5,
        contrast=0.5,
        brightness=0.5,
        blur_radius=0.5,
    ) -> None:
        # super(dex_ycb,self).__init__()
        self.root = cfg.dexycb_data_dir
        self.mode = mode
        self.joint_root_id = 0
        self.jointsMapManoToSimple = [
            0,
            13,
            14,
            15,
            16,
            1,
            2,
            3,
            17,
            4,
            5,
            6,
            18,
            10,
            11,
            12,
            19,
            7,
            8,
            9,
            20,
        ]  # 注意
        self.jointsMapSimpleToMano = np.argsort(self.jointsMapManoToSimple)
        self.coord_change_mat = np.array(
            [[1.0, 0.0, 0.0], [0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32
        )
        # object information
        self.obj_mesh = dex_ycb_util.load_objects_dex_ycb(self.root)
        self.obj_bbox3d = dataset_util.get_bbox21_3d_from_dict(self.obj_mesh)
        self.obj_diameters = dataset_util.get_diameter(self.obj_mesh)

        self.obj_depth_mean_value = cfg.obj_depth_mean_value
        self.inp_res = cfg.input_img_shape[0]
        self.heatmap_res = cfg.output_hm_shape[0]
        self.joint_num = 21
        self.transform = transforms.ToTensor()

        self.num_samp_hand = cfg.num_samp_hand
        self.num_samp_obj = cfg.num_samp_obj
        self.dist = cfg.points_filter_dist
        self.coarse_cls = cfg.num_class == 6
        self.hand_sdf_scale = cfg.hand_sdf_scale
        self.obj_sdf_scale = cfg.obj_sdf_scale
        self.sdf_fast_path = cfg.fast_data_dir
        self.image_fast_path = cfg.image_fast_path

        mano_layer = ManoLayer(
            flat_hand_mean=False,
            side="right",
            mano_root="tool/mano_models",
            use_pca=False,
        )
        mano_layerl = ManoLayer(
            flat_hand_mean=False,
            side="left",
            mano_root="tool/mano_models",
            use_pca=False,
        )
        self.mano_handcomponent_right = np.array(
            mano_layer.smpl_data["hands_components"]
        )
        self.mano_handcomponent_left = np.array(
            mano_layerl.smpl_data["hands_components"]
        )
        self.mano_handmean = np.array(mano_layer.smpl_data["hands_mean"])

        if self.mode == "train":
            self.hue = hue
            self.contrast = contrast
            self.brightness = brightness
            self.saturation = saturation
            self.blur_radius = blur_radius
            self.scale_jittering = scale_jittering
            self.center_jittering = center_jittering
            self.max_rot = max_rot
            if cfg.small_dexycb:
                train_annotation_path = os.path.join(
                    cfg.annotation_dir, "dex_ycb_s0_train_data_cut.json"
                )
                sdf_split = "train"
            else:
                train_annotation_path = os.path.join(
                    cfg.annotation_dir, "dex_ycb_s0_train_data.json"
                )
                sdf_split = "full_train"

            with open(train_annotation_path, "r", encoding="utf-8") as f:
                self.sample_dict = json.load(f)
        else:
            if cfg.small_dexycb:
                test_annotation_path = os.path.join(
                    cfg.annotation_dir, "dex_ycb_s0_test_data_cut.json"
                )
                sdf_split = "test"
            else:
                test_annotation_path = os.path.join(
                    cfg.annotation_dir, "dex_ycb_s0_test_data.json"
                )
                sdf_split = "full_test"

            with open(test_annotation_path, "r", encoding="utf-8") as f:
                self.sample_dict = json.load(f)

        sdf_list = [
            filename.split(".")[0]
            for filename in os.listdir(
                os.path.join(self.sdf_fast_path, sdf_split, "sdf_processed")
            )
        ]
        sdf_list.sort()
        raw_sdf_index = np.load(
            os.path.join(self.sdf_fast_path, sdf_split, "sdf_index.npy")
        )

        self.sample_list = sorted(self.sample_dict.keys(), key=lambda x: int(x[3:]))
        self.sample_list_processed = []
        if cfg.small_dexycb:
            self.sample_list_processed = self.sample_list
        else:
            for sample in self.sample_list:
                joint_2d = np.array(
                    self.sample_dict[sample]["joint_2d"], dtype=np.float32
                ).squeeze()
                hand_bbox = dex_ycb_util.get_bbox(
                    joint_2d, np.ones_like(joint_2d[:, 0]), expansion_factor=1.5
                )
                hand_bbox = dex_ycb_util.process_bbox(
                    hand_bbox, 640, 480, expansion_factor=1.0
                )
                if hand_bbox is None:
                    continue
                else:
                    self.sample_list_processed.append(sample)

        # self.sample_list_processed = self.sample_list_processed[:2000]

        self.hand_segs = []
        self.obj_segs = []
        self.sdf_path_list = []
        self.sdf_index_list = []
        for sample in tqdm(self.sample_list_processed):
            label = np.load(
                os.path.join(self.root, self.sample_dict[sample]["label_file"])
            )
            hand_seg = label["seg"] == 255
            obj_seg = (
                label["seg"]
                == self.sample_dict[sample]["ycb_ids"][
                    self.sample_dict[sample]["ycb_grasp_ind"]
                ]
            )
            self.hand_segs.append(np.packbits(hand_seg))
            self.obj_segs.append(np.packbits(obj_seg))
            if cfg.small_dexycb:
                sdf_file_name = (
                    self.sample_dict[sample]["color_file"]
                    .split("-")[-1]
                    .split(".")[0]
                    .replace("/", "_")
                )
                sdf_file_name = sdf_file_name[:-12] + sdf_file_name[-2:]
            else:
                sdf_file_name = sample
            self.sdf_path_list.append(
                os.path.join(
                    self.sdf_fast_path,
                    sdf_split,
                    "sdf_processed",
                    sdf_file_name + ".npy",
                )
            )
            self.sdf_index_list.append(raw_sdf_index[sdf_list.index(sdf_file_name)])

    def data_aug(
        self,
        img,
        mano_param,
        joints_uv,
        K,
        hand_seg,
        obj_seg,
        p2d,
        sdf_points,
        joints_3d,
        p3d,
        obj_rot,
        obj_trans,
    ):
        # Copy to prevent data corruption
        img = img.copy()
        mano_param = mano_param.copy()
        joints_uv = joints_uv.copy()
        K = K.copy()
        # gray = gray.copy()
        hand_seg = hand_seg.copy()
        obj_seg = obj_seg.copy()
        p2d = p2d.copy()
        sdf_points = sdf_points.copy()
        joints_3d = joints_3d.copy()
        p3d = p3d.copy()
        obj_rot = obj_rot.copy()
        obj_trans = obj_trans.copy()

        crop_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)

        # Randomly jitter center
        center_offsets = (
            self.center_jittering * scale * np.random.uniform(low=-1, high=1, size=2)
        )
        center = center + center_offsets

        # Scale jittering
        scale_jittering = self.scale_jittering * np.random.randn() + 1
        scale_jittering = np.clip(
            scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering
        )
        scale = scale * scale_jittering

        # rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
        rot_factor = 30
        rot = (
            np.clip(np.random.randn(), -2.0, 2.0) * rot_factor
            if random.random() <= 0.6
            else 0
        )
        rot = rot * self.max_rot / 180

        affinetrans, post_rot_trans, rot_mat = dataset_util.get_affine_transform(
            center, scale, [self.inp_res, self.inp_res], rot=rot, K=K
        )
        # Change mano from openGL coordinates to normal coordinates
        mano_param[:3] = dataset_util.rotation_angle(
            mano_param[:3], rot_mat, coord_change_mat=np.eye(3)
        )

        joints_uv = dataset_util.transform_coords(
            joints_uv, affinetrans
        )  # hand landmark trans

        # rotate for 3D points
        sdf_points[:, :3] = sdf_points[:, :3].dot(rot_mat.T)
        joints_3d = joints_3d.dot(rot_mat.T)
        p3d = p3d.dot(rot_mat.T)

        obj_rot = cv2.Rodrigues(rot_mat.dot(cv2.Rodrigues(obj_rot)[0]))[0].squeeze()
        obj_trans = rot_mat.dot(obj_trans)

        K = post_rot_trans.dot(K)
        p2d = dataset_util.transform_coords(p2d, affinetrans)  # obj landmark trans

        # get hand bbox and normalize landmarks to [0,1]
        bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.1)
        # joints_uv = dataset_util.normalize_joints(joints_uv, bbox_hand)
        joints_uv = joints_uv / self.inp_res * self.heatmap_res

        # get obj bbox and normalize landmarks to [0,1]
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        p2d = dataset_util.normalize_joints(p2d, bbox_obj)

        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))

        # Img blurring and color jitter
        blur_radius = random.random() * self.blur_radius
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        img = dataset_util.color_jitter(
            img,
            brightness=self.brightness,
            saturation=self.saturation,
            hue=self.hue,
            contrast=self.contrast,
        )

        # Trasform hand seg and obj seg
        hand_seg = dataset_util.transform_img(
            hand_seg, affinetrans, [self.inp_res, self.inp_res]
        )
        hand_seg = hand_seg.crop((0, 0, self.inp_res, self.inp_res))
        hand_seg = np.asarray(
            hand_seg.resize((self.heatmap_res, self.heatmap_res), Image.NEAREST)
        )
        obj_seg = dataset_util.transform_img(
            obj_seg, affinetrans, [self.inp_res, self.inp_res]
        )
        obj_seg = obj_seg.crop((0, 0, self.inp_res, self.inp_res))
        obj_seg = np.asarray(
            obj_seg.resize((self.heatmap_res, self.heatmap_res), Image.NEAREST)
        )

        return (
            img,
            mano_param,
            K,
            hand_seg,
            obj_seg,
            p2d,
            joints_uv,
            bbox_hand,
            bbox_obj,
            sdf_points,
            joints_3d,
            p3d,
            obj_rot,
            obj_trans,
        )

    def data_crop(self, img, K, joints_uv, p2d, hand_seg, obj_seg):
        # Copy to prevent data corruption
        img = img.copy()
        K = K.copy()
        joints_uv = joints_uv.copy()
        p2d = p2d.copy()
        hand_seg = hand_seg.copy()
        obj_seg = obj_seg.copy()

        crop_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.1)
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)
        affinetrans, post_rot_trans, _ = dataset_util.get_affine_transform(
            center, scale, [self.inp_res, self.inp_res], K=K
        )
        bbox_hand = dataset_util.transform_coords(
            bbox_hand.reshape(2, 2), affinetrans
        ).flatten()
        bbox_obj = dataset_util.transform_coords(
            bbox_obj.reshape(2, 2), affinetrans
        ).flatten()
        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))

        joints_uv = dataset_util.transform_coords(joints_uv, affinetrans)
        joints_uv = joints_uv / self.inp_res * self.heatmap_res
        K = post_rot_trans.dot(K)
        p2d = dataset_util.transform_coords(p2d, affinetrans)  # obj landmark trans
        p2d = dataset_util.normalize_joints(p2d, bbox_obj)

        # Trasform hand seg and obj seg
        hand_seg = dataset_util.transform_img(
            hand_seg, affinetrans, [self.inp_res, self.inp_res]
        )
        hand_seg = hand_seg.crop((0, 0, self.inp_res, self.inp_res))
        hand_seg = np.asarray(
            hand_seg.resize((self.heatmap_res, self.heatmap_res), Image.NEAREST)
        )
        obj_seg = dataset_util.transform_img(
            obj_seg, affinetrans, [self.inp_res, self.inp_res]
        )
        obj_seg = obj_seg.crop((0, 0, self.inp_res, self.inp_res))
        obj_seg = np.asarray(
            obj_seg.resize((self.heatmap_res, self.heatmap_res), Image.NEAREST)
        )

        return img, bbox_hand, bbox_obj, K, joints_uv, p2d, hand_seg, obj_seg

    def __len__(self):
        return len(self.sample_list_processed)

    def __getitem__(self, idx):
        # sample = {}
        sample_info = self.sample_dict[self.sample_list_processed[idx]].copy()
        do_flip = sample_info["mano_side"] == "left"
        img = Image.open(
            os.path.join(self.image_fast_path, sample_info["color_file"])
        ).convert("RGB")
        # camintr
        fx = sample_info["intrinsics"]["fx"]
        fy = sample_info["intrinsics"]["fy"]
        cx = sample_info["intrinsics"]["ppx"]
        cy = sample_info["intrinsics"]["ppy"]
        K = np.zeros((3, 3))
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        K[2, 2] = 1
        if do_flip:
            img = np.array(img, np.uint8, copy=True)
            img = img[:, ::-1, :]
            img = Image.fromarray(np.uint8(img))

        # hand information
        mano_pose_pca_mean = np.array(sample_info["pose_m"], dtype=np.float32).squeeze()
        mano_betas = np.array(sample_info["mano_betas"], dtype=np.float32)
        joints_3d = np.array(sample_info["joint_3d"], dtype=np.float32).squeeze()
        joints_uv = np.array(sample_info["joint_2d"], dtype=np.float32).squeeze()

        mano_pose_aa_mean = np.concatenate(
            (
                mano_pose_pca_mean[0:3],
                np.matmul(
                    mano_pose_pca_mean[3:48], self.mano_handcomponent_right.copy()
                ),
                mano_pose_pca_mean[48:],
            ),
            axis=0,
        )
        if do_flip:
            mano_pose_aa_mean = np.concatenate(
                (
                    mano_pose_pca_mean[0:3],
                    np.matmul(
                        mano_pose_pca_mean[3:48], self.mano_handcomponent_left.copy()
                    ),
                    mano_pose_pca_mean[48:],
                ),
                axis=0,
            )
            mano_pose_aa_mean_wo_trans = mano_pose_aa_mean[:48].reshape(-1, 3)
            mano_pose_aa_mean_wo_trans[:, 1:] *= -1
            mano_pose_aa_mean[0:48] = mano_pose_aa_mean_wo_trans.reshape(-1)
            joints_3d[:, 0] *= -1
            joints_uv[:, 0] = (
                np.array(img.size[0], dtype=np.float32) - joints_uv[:, 0] - 1
            )
        mano_pose_aa_flat = np.concatenate(
            (
                mano_pose_aa_mean[:3],
                mano_pose_aa_mean[3:48] + self.mano_handmean.copy(),
            ),
            axis=0,
        )
        mano_param = np.concatenate((mano_pose_aa_flat, mano_betas))

        bit_hand_seg = self.hand_segs[idx].copy()
        hand_seg = np.unpackbits(bit_hand_seg).reshape((480, 640))
        bit_obj_seg = self.obj_segs[idx].copy()
        obj_seg = np.unpackbits(bit_obj_seg).reshape((480, 640))
        if do_flip:
            hand_seg = hand_seg[:, ::-1]
            obj_seg = obj_seg[:, ::-1]
        hand_seg = Image.fromarray(hand_seg)
        obj_seg = Image.fromarray(obj_seg)

        # object_information
        grasp_object_pose = np.array(
            sample_info["pose_y"][sample_info["ycb_grasp_ind"]], dtype=np.float32
        )
        p3d, p2d = dex_ycb_util.projectPoints(
            self.obj_bbox3d[
                sample_info["ycb_ids"][sample_info["ycb_grasp_ind"]]
            ].copy(),
            K,
            rt=grasp_object_pose,
        )

        obj_rot = cv2.Rodrigues(grasp_object_pose[:, :3])[0].squeeze()
        obj_trans = grasp_object_pose[:, 3]

        if do_flip:
            K[0, 2] = img.size[0] - K[0, 2] - 1
            obj_trans[0] *= -1
            obj_rot[1:] *= -1
            p3d, p2d = dex_ycb_util.projectPoints(
                self.obj_bbox3d[
                    sample_info["ycb_ids"][sample_info["ycb_grasp_ind"]]
                ].copy(),
                K,
                rt=np.concatenate(
                    [cv2.Rodrigues(obj_rot)[0], obj_trans[:, None]], axis=1
                ),
            )

        # load sdf points
        sdf_data = np.load(self.sdf_path_list[idx])
        sdf_idx = self.sdf_index_list[idx].copy()
        assert sdf_data.shape[0] == sdf_idx[0] + sdf_idx[1]

        hand_sdf_idx = np.random.choice(
            list(range(sdf_idx[0])), size=int(self.num_samp_hand), replace=False
        )
        obj_sdf_idx = np.random.choice(
            list(range(sdf_idx[0], sdf_data.shape[0])),
            size=int(self.num_samp_obj),
            replace=False,
        )

        if self.mode == "train":
            hand_pre_idx = np.random.choice(
                np.where(np.abs(sdf_data[: sdf_idx[0], 3]) < self.dist)[0],
                size=self.num_samp_hand,
                replace=False,
            )
            obj_pre_idx = np.random.choice(
                np.where(np.abs(sdf_data[sdf_idx[0] :, 4]) < self.dist)[0] + sdf_idx[0],
                size=self.num_samp_obj,
                replace=False,
            )
            all_idx = np.concatenate(
                (hand_sdf_idx, obj_sdf_idx, hand_pre_idx, obj_pre_idx)
            )
        else:
            all_idx = np.concatenate((hand_sdf_idx, obj_sdf_idx))

        sdf_data = sdf_data[all_idx]
        sdf_points, sdf_raw_label = sdf_data[:, :5], sdf_data[:, 5]

        if do_flip:
            sdf_points[:, 0] *= -1

        if self.mode == "train":
            # data augumentation
            (
                img,
                mano_param,
                K,
                hand_seg,
                obj_seg,
                p2d,
                joints_uv,
                bbox_hand,
                bbox_obj,
                sdf_points,
                joints_3d,
                p3d,
                obj_rot,
                obj_trans,
            ) = self.data_aug(
                img,
                mano_param,
                joints_uv,
                K,
                hand_seg,
                obj_seg,
                p2d,
                sdf_points,
                joints_3d,
                p3d,
                obj_rot,
                obj_trans,
            )
        else:
            # crop
            img, bbox_hand, bbox_obj, K, joints_uv, p2d, hand_seg, obj_seg = (
                self.data_crop(img, K, joints_uv, p2d, hand_seg, obj_seg)
            )

        # obtain normalized points
        hand_root = joints_3d[0].copy()
        joints_3d = joints_3d - hand_root[None]
        obj_center_cam = dataset_util.get_center_cam(bbox_obj, hand_root[-1], K).astype(
            np.float32
        )
        p3d = p3d - obj_center_cam[None]

        hand_sdf_points = sdf_points[: int(self.num_samp_hand)]
        obj_sdf_points = sdf_points[
            int(self.num_samp_hand) : int(self.num_samp_hand + self.num_samp_obj)
        ]
        hand_sdf_points[:, :3] = hand_sdf_points[:, :3] - hand_root[None]
        hand_sdf_points = hand_sdf_points * self.hand_sdf_scale
        obj_sdf_points[:, :3] = obj_sdf_points[:, :3] - obj_center_cam[None]
        obj_sdf_points = obj_sdf_points * self.obj_sdf_scale

        if self.mode == "train":
            hand_pre_points = sdf_points[
                int(self.num_samp_hand + self.num_samp_obj) : int(
                    self.num_samp_hand * 2 + self.num_samp_obj
                )
            ]
            obj_pre_points = sdf_points[
                int(self.num_samp_hand * 2 + self.num_samp_obj) :
            ]
            hand_pre_points = hand_pre_points[:, :3] - hand_root[None]
            hand_pre_points = hand_pre_points * self.hand_sdf_scale
            obj_pre_points = obj_pre_points[:, :3] - obj_center_cam[None]
            obj_pre_points = obj_pre_points * self.obj_sdf_scale
        else:
            hand_pre_points = False
            obj_pre_points = False

        img = self.transform(np.asarray(img).astype(np.float32)) / 255.0
        hand_seg = torch.from_numpy(hand_seg.astype(np.float32))
        obj_seg = torch.from_numpy(obj_seg.astype(np.float32))
        obj_trans = obj_trans.astype(np.float32) - obj_center_cam

        inputs = {
            "img": img,
            "hand_sdf_points": hand_sdf_points[:, :3],
            "obj_sdf_points": obj_sdf_points[:, :3],
            "hand_pre_points": hand_pre_points,
            "obj_pre_points": obj_pre_points,
        }

        targets = {
            "joint_coord": joints_uv.astype(np.float32),
            "joint_cam_no_trans": joints_3d * 1000,
            "obj_rot": obj_rot,
            "rel_obj_trans": obj_trans.astype(np.float32),
            "obj_seg": obj_seg,
            "hand_seg": hand_seg,
            #    'sdf_label': sdf_label,
            "hand_sdf": hand_sdf_points[:, 3],
            "obj_sdf": obj_sdf_points[:, 4],
            "mano_param": mano_param.astype(np.float32),
        }

        meta_info = {
            "cam_intr": K.astype(np.float32),
            "mano_root": hand_root,
            "obj_cls": sample_info["ycb_ids"][sample_info["ycb_grasp_ind"]],
            "obj_center_cam": obj_center_cam,
            "bbox_hand": bbox_hand,
            "bbox_obj": bbox_obj,
        }

        return inputs, targets, meta_info
