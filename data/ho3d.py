# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from HFL-Net
# ------------------------------------------------------------------------------

import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))
import random

import numpy as np
from PIL import Image, ImageFile, ImageFilter
from torch.utils import data
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
import json

import cv2
import torch
import torchvision.transforms as transforms

from data import dataset_util, ho3d_util
from data.dataset_util import convert_pose_to_opencv, load_img
from main.config import cfg


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
    ):
        # Dataset attributes
        self.root = cfg.ho3d_data_dir
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
        ]
        self.jointsMapSimpleToMano = np.argsort(self.jointsMapManoToSimple)
        self.coord_change_mat = np.array(
            [[1.0, 0.0, 0.0], [0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32
        )

        # object informations
        self.obj_mesh = ho3d_util.load_objects_HO3D(cfg.object_models_dir)
        self.obj_bbox3d = dataset_util.get_bbox21_3d_from_dict(self.obj_mesh)
        self.obj_diameters = dataset_util.get_diameter(self.obj_mesh)

        self.obj_depth_mean_value = cfg.obj_depth_mean_value
        self.inp_res = cfg.input_img_shape[0]
        self.heatmap_res = cfg.output_hm_shape[0]
        self.joint_num = 21
        self.transform = transforms.ToTensor()

        if self.mode == "train":
            self.hue = hue
            self.contrast = contrast
            self.brightness = brightness
            self.saturation = saturation
            self.blur_radius = blur_radius
            self.scale_jittering = scale_jittering
            self.center_jittering = center_jittering
            self.max_rot = max_rot

            self.num_samp_hand = cfg.num_samp_hand
            self.num_samp_obj = cfg.num_samp_obj
            self.dist = cfg.points_filter_dist
            self.coarse_cls = cfg.num_class == 6
            self.hand_sdf_scale = cfg.hand_sdf_scale
            self.obj_sdf_scale = cfg.obj_sdf_scale
            self.sdf_fast_path = cfg.fast_data_dir

            sdf_index = np.load(
                os.path.join(self.sdf_fast_path, "full", "sdf_index.npy"),
                allow_pickle=True,
            ).tolist()

            self.mano_params = []
            self.joints_3d = []
            self.joints_uv = []
            self.obj_p3ds = []
            self.obj_p2ds = []
            self.K = []
            self.set_list = []
            self.image_paths = []
            self.sdf_paths = []
            self.sdf_indexes = []
            self.hand_segs = []
            self.obj_segs = []
            self.obj_rot_list = []
            self.obj_trans_list = []
            self.obj_cls_list = []
            # data_ho3d = []
            with open(
                os.path.join(cfg.annotation_dir, "ho3d_train_data.json"), "r"
            ) as f:
                data_ho3d = json.load(f)

            for data in tqdm(data_ho3d):
                # for data in tqdm(data_ho3d[:100]):
                sdf_path = os.path.join(
                    self.sdf_fast_path,
                    "train",
                    "sdf_processed",
                    data["seqName_id"].replace("/", "_") + ".npy",
                )
                if not os.path.exists(sdf_path):
                    continue
                # self.image_paths.append(os.path.join(self.sdf_fast_path, 'train', 'rgb', data['seqName_id'].replace('/', '_') + '.png'))
                self.image_paths.append(
                    os.path.join(
                        self.root,
                        "train",
                        data["seqName_id"].split("/")[0],
                        "rgb",
                        data["seqName_id"].split("/")[1] + ".png",
                    )
                )
                self.sdf_paths.append(sdf_path)
                self.sdf_indexes.append(sdf_index[data["seqName_id"].replace("/", "_")])
                seg = load_img(
                    os.path.join(
                        self.root,
                        self.mode,
                        data["seqName_id"].split("/")[0],
                        "seg",
                        data["seqName_id"].split("/")[1] + ".jpg",
                    )
                )
                seg = cv2.resize(
                    seg.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST
                )
                self.hand_segs.append(np.packbits(seg[:, :, 0] > 200))
                self.obj_segs.append(np.packbits(seg[:, :, 2] > 200))

                self.set_list.append(data["seqName_id"])
                K = np.array(data["K"], dtype=np.float32)
                self.K.append(K)
                self.joints_3d.append(np.array(data["joints_3d"], dtype=np.float32))
                self.joints_uv.append(
                    ho3d_util.projectPoints(
                        np.array(data["joints_3d"], dtype=np.float32), K
                    )
                )
                self.mano_params.append(np.array(data["mano_params"], dtype=np.float32))
                self.obj_p3ds.append(np.array(data["obj_p3ds"], dtype=np.float32))
                self.obj_p2ds.append(np.array(data["obj_p2ds"], dtype=np.float32))
                annotations = np.load(
                    os.path.join(
                        os.path.join(self.root, self.mode),
                        data["seqName_id"].split("/")[0],
                        "meta",
                        data["seqName_id"].split("/")[1] + ".pkl",
                    ),
                    allow_pickle=True,
                )
                obj_rot, obj_trans = convert_pose_to_opencv(
                    annotations["objRot"].squeeze(), annotations["objTrans"]
                )
                self.obj_rot_list.append(obj_rot)
                self.obj_trans_list.append(obj_trans.astype(np.float32))
                self.obj_cls_list.append(annotations["objName"])

            if cfg.add_render:
                render_filelist = [
                    filename.split(".")[0]
                    for filename in os.listdir(
                        os.path.join(self.sdf_fast_path, "render", "sdf_processed")
                    )
                ]
                render_filelist.sort()
                # render_filelist = render_filelist[:1000]
                render_sdf_index = np.load(
                    os.path.join(self.sdf_fast_path, "render", "sdf_index.npy")
                )
                print("process addtional render data")
                for idx, fname in enumerate(tqdm(render_filelist)):
                    self.image_paths.append(
                        os.path.join(
                            self.sdf_fast_path, "render", "rgb", fname + ".png"
                        )
                    )
                    self.sdf_paths.append(
                        os.path.join(
                            self.sdf_fast_path,
                            "render",
                            "sdf_processed",
                            fname + ".npy",
                        )
                    )
                    self.sdf_indexes.append(render_sdf_index[idx])
                    anno_path = os.path.join(
                        self.sdf_fast_path, "render", "anno", fname + ".json"
                    )
                    seg = load_img(
                        os.path.join(
                            self.sdf_fast_path, "render", "seg", fname + ".png"
                        )
                    )
                    self.hand_segs.append(np.packbits(seg[:, :, 0] > 200))
                    self.obj_segs.append(np.packbits(seg[:, :, 2] > 200))
                    with open(anno_path, "r") as f:
                        anno = json.load(f)
                    for key in anno.keys():
                        if type(anno[key]) == list:
                            anno[key] = np.array(anno[key])
                    self.set_list.append(fname)
                    K = anno["camMat"].copy().astype(np.float32)
                    self.K.append(anno["camMat"].copy().astype(np.float32))
                    self.joints_3d.append(
                        np.array(anno["handJoints3D"], dtype=np.float32)
                    )
                    self.joints_uv.append(
                        ho3d_util.projectPoints(
                            np.array(anno["handJoints3D"].copy(), dtype=np.float32), K
                        )
                    )
                    self.mano_params.append(np.zeros(58, dtype=np.float32))
                    obj_rot_mat = np.array(anno["objRot"], dtype=np.float32)
                    obj_trans = np.array(anno["objTrans"], dtype=np.float32)
                    obj_p3d = (
                        np.dot(
                            self.obj_bbox3d[anno["objName"]].copy().astype(np.float32),
                            obj_rot_mat.T,
                        )
                        + obj_trans[None]
                    )
                    self.obj_p3ds.append(obj_p3d)
                    self.obj_p2ds.append(ho3d_util.projectPoints(obj_p3d.copy(), K))
                    self.obj_rot_list.append(cv2.Rodrigues(obj_rot_mat)[0].squeeze())
                    self.obj_trans_list.append(obj_trans)
                    self.obj_cls_list.append(anno["objName"])

        else:
            self.set_list = ho3d_util.load_names(
                os.path.join(self.root, "evaluation.txt")
            )

        # self.set_list = self.set_list[:100]

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

        rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
        affinetrans, post_rot_trans, rot_mat = dataset_util.get_affine_transform(
            center, scale, [self.inp_res, self.inp_res], rot=rot, K=K
        )
        # Change mano from openGL coordinates to normal coordinates
        mano_param[:3] = dataset_util.rotation_angle(
            mano_param[:3], rot_mat, coord_change_mat=self.coord_change_mat
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
        bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.2)
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

    def data_crop(self, img, K, bbox_hand, p2d):
        # Copy to prevent data corruption
        img = img.copy()
        K = K.copy()
        bbox_hand = bbox_hand.copy()

        crop_hand = dataset_util.get_bbox_joints(
            bbox_hand.reshape(2, 2), bbox_factor=1.5
        )
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        bbox_hand = dataset_util.get_bbox_joints(
            bbox_hand.reshape(2, 2), bbox_factor=1.2
        )
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_obj, img.size)
        affinetrans, _ = dataset_util.get_affine_transform(
            center, scale, [self.inp_res, self.inp_res]
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
        K = affinetrans.dot(K)
        return img, K, bbox_hand, bbox_obj

    def __len__(self):
        return len(self.set_list)

    def __getitem__(self, idx):
        if self.mode == "train":
            img_path = self.image_paths[idx]
            # img = load_img(img_path)
            img = Image.open(img_path).convert("RGB")
            K = self.K[idx].copy()
            # hand information
            joints_uv = self.joints_uv[idx].copy()
            mano_param = self.mano_params[idx].copy()
            joints_3d = self.joints_3d[idx].copy()

            # segmentation information
            bit_hand_seg = self.hand_segs[idx].copy()
            hand_seg = np.unpackbits(bit_hand_seg).reshape((480, 640))
            hand_seg = Image.fromarray(hand_seg)
            bit_obj_seg = self.obj_segs[idx].copy()
            obj_seg = np.unpackbits(bit_obj_seg).reshape((480, 640))
            obj_seg = Image.fromarray(obj_seg)

            # object information
            p2d = self.obj_p2ds[idx].copy()
            p3d = self.obj_p3ds[idx].copy()
            obj_rot = self.obj_rot_list[idx].copy()
            obj_trans = self.obj_trans_list[idx].copy()

            # load sdf points
            sdf_data = np.load(self.sdf_paths[idx])
            sdf_idx = self.sdf_indexes[idx].copy()
            assert sdf_data.shape[0] == sdf_idx[0] + sdf_idx[1]

            hand_sdf_idx = np.random.choice(
                list(range(sdf_idx[0])), size=int(self.num_samp_hand), replace=False
            )
            obj_sdf_idx = np.random.choice(
                list(range(sdf_idx[0], sdf_data.shape[0])),
                size=int(self.num_samp_obj),
                replace=False,
            )

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
            sdf_data = sdf_data[all_idx]

            sdf_points, sdf_raw_label = sdf_data[:, :5], sdf_data[:, 5]

            # data augmentation
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

            # obtain normalized points
            hand_root = joints_3d[0].copy()
            joints_3d = joints_3d - hand_root[None]
            obj_center_cam = dataset_util.get_center_cam(
                bbox_obj, self.obj_depth_mean_value, K
            ).astype(np.float32)
            p3d = p3d - obj_center_cam[None]

            hand_sdf_points = sdf_points[: int(self.num_samp_hand)]
            obj_sdf_points = sdf_points[
                int(self.num_samp_hand) : int(self.num_samp_hand + self.num_samp_obj)
            ]
            hand_sdf_points[:, :3] = hand_sdf_points[:, :3] - hand_root[None]
            hand_sdf_points = hand_sdf_points * self.hand_sdf_scale
            obj_sdf_points[:, :3] = obj_sdf_points[:, :3] - obj_center_cam[None]
            obj_sdf_points = obj_sdf_points * self.obj_sdf_scale

            hand_pre_points = sdf_points[
                int(self.num_samp_hand + self.num_samp_obj) : int(
                    self.num_samp_hand * 2 + self.num_samp_obj
                )
            ]
            obj_pre_points = sdf_points[
                int(self.num_samp_hand * 2 + self.num_samp_obj) :
            ]
            hand_pre_points[:, :3] = hand_pre_points[:, :3] - hand_root[None]
            hand_pre_points = hand_pre_points * self.hand_sdf_scale
            obj_pre_points[:, :3] = obj_pre_points[:, :3] - obj_center_cam[None]
            obj_pre_points = obj_pre_points * self.obj_sdf_scale

            img = self.transform(np.asarray(img).astype(np.float32)) / 255.0
            hand_seg = torch.from_numpy(hand_seg.astype(np.float32))
            obj_seg = torch.from_numpy(obj_seg.astype(np.float32))
            obj_trans = obj_trans.astype(np.float32) - obj_center_cam

            obj_mask = (
                self.obj_cls_list[idx] == "021_bleach_cleanser"
                or self.obj_cls_list[idx] == "006_mustard_bottle"
                or self.obj_cls_list[idx] == "010_potted_meat_can"
            )

            inputs = {
                "img": img,
                "hand_sdf_points": hand_sdf_points[:, :3],
                "obj_sdf_points": obj_sdf_points[:, :3],
                "hand_pre_points": hand_pre_points[:, :3],
                "obj_pre_points": obj_pre_points[:, :3],
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
                "mano_param": mano_param,
            }

            meta_info = {
                "cam_intr": K,
                "mano_root": hand_root,
                "obj_mask": obj_mask,  # 'obj_cls': self.obj_cls_list[idx],
                "obj_center_cam": obj_center_cam,
                "bbox_hand": bbox_hand,
                "bbox_obj": bbox_obj,
            }

        else:
            seqName, frame_id = self.set_list[idx].split("/")
            img_path = os.path.join(
                self.root, self.mode, seqName, "rgb", frame_id + ".png"
            )
            img = Image.open(img_path).convert("RGB")
            annotations = np.load(
                os.path.join(
                    os.path.join(self.root, self.mode),
                    seqName,
                    "meta",
                    frame_id + ".pkl",
                ),
                allow_pickle=True,
            )
            K = np.array(annotations["camMat"], dtype=np.float32)
            obj_bbox3d = self.obj_bbox3d[annotations["objName"]]
            obj_pose = ho3d_util.pose_from_RT(
                annotations["objRot"].reshape((3,)), annotations["objTrans"]
            )
            p2d = ho3d_util.projectPoints(obj_bbox3d, K, rt=obj_pose)

            # hand
            bbox_hand = np.array(annotations["handBoundingBox"], dtype=np.float32)
            root_joint = np.array(annotations["handJoints3D"], dtype=np.float32)
            root_joint = root_joint.dot(self.coord_change_mat.T)

            img, K, bbox_hand, bbox_obj = self.data_crop(img, K, bbox_hand, p2d)

            obj_center_cam = dataset_util.get_center_cam(
                bbox_obj, self.obj_depth_mean_value, K
            ).astype(np.float32)

            img = self.transform(np.asarray(img).astype(np.float32)) / 255.0
            obj_rot, obj_trans = convert_pose_to_opencv(
                annotations["objRot"].squeeze(), annotations["objTrans"]
            )
            obj_trans = obj_trans.astype(np.float32) - obj_center_cam

            obj_mask = (
                annotations["objName"] == "021_bleach_cleanser"
                or annotations["objName"] == "006_mustard_bottle"
                or annotations["objName"] == "010_potted_meat_can"
            )

            inputs = {"img": img}
            targets = {
                "obj_rot": obj_rot,
                "rel_obj_trans": obj_trans.astype(np.float32),
            }

            meta_info = {
                "cam_intr": K,
                "mano_root": root_joint,
                "hand_type": "right",
                "obj_cls": annotations["objName"],
                "obj_mask": obj_mask,
                "obj_center_cam": obj_center_cam,
                "bbox_hand": bbox_hand.astype(np.float32),
                "bbox_obj": bbox_obj.astype(np.float32),
            }

        return inputs, targets, meta_info
