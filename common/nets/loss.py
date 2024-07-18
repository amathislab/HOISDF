# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from HFL-Net
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F

from main.config import cfg


class JointHeatmapLoss(nn.Module):
    def __ini__(self):
        super(JointHeatmapLoss, self).__init__()

    def forward(self, joint_out, joint_gt):
        loss = (joint_out - joint_gt) ** 2
        return loss


class JointvoteLoss(nn.Module):
    def __init__(self):
        super(JointvoteLoss, self).__init__()
        self.cls_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.reg_loss = nn.SmoothL1Loss(reduction="none")
        self.points_softmax = nn.Softmax(dim=1)
        self.sum_reg_loss = nn.SmoothL1Loss(reduction="mean")

    def forward(self, hand_points, hand_off, hand_cls, joint_gt):
        l, p, b, j = hand_cls.shape
        hand_vote = (
            hand_points.unsqueeze(2).unsqueeze(0)
            + hand_off.reshape((l, p, b, j, 3)).permute(0, 2, 1, 3, 4).contiguous()
        )
        hand_cls_gt = (
            torch.norm(hand_points.unsqueeze(2) - joint_gt.unsqueeze(1) / 1000, dim=-1)
            < cfg.hand_cls_dist
        ).float()
        loss_joint_3d = self.reg_loss(
            hand_vote * 1000, joint_gt.unsqueeze(1).unsqueeze(0).expand(l, b, p, j, 3)
        )
        loss_joint_3d = loss_joint_3d * hand_cls_gt.unsqueeze(-1).unsqueeze(0).expand(
            l, b, p, j, 3
        )
        loss_joint_3d = loss_joint_3d.sum((1, 2, 3)) / hand_cls_gt.sum()
        loss_joint_cls = self.cls_loss(
            hand_cls.permute(0, 2, 1, 3).contiguous(),
            hand_cls_gt.unsqueeze(0).expand(l, b, p, j),
        )

        hand_weights = (
            self.points_softmax(hand_cls).permute(0, 2, 1, 3).unsqueeze(-1).contiguous()
        )
        hand_joints = torch.sum(hand_vote * hand_weights, dim=2)
        loss_all_joint_3d = self.sum_reg_loss(
            hand_joints * 1000, joint_gt.unsqueeze(0).expand(l, b, j, 3)
        )

        return loss_joint_3d.mean(), loss_joint_cls, loss_all_joint_3d, hand_joints


class SepSDFLoss(nn.Module):
    def __init__(
        self,
    ):
        super(SepSDFLoss, self).__init__()

        self.loss_l1 = torch.nn.L1Loss(reduction="mean")

    def forward(self, hand_sdf, obj_sdf, hand_sdf_gt, obj_sdf_gt):

        loss_hand = self.loss_l1(hand_sdf, hand_sdf_gt.unsqueeze(-1))

        loss_obj = self.loss_l1(obj_sdf, obj_sdf_gt.unsqueeze(-1))

        return loss_hand, loss_obj


class ManoLoss(nn.Module):
    def __init__(
        self,
        lambda_verts3d=None,
        lambda_joints3d=None,
        lambda_manopose=None,
        lambda_manoshape=None,
        lambda_regulshape=None,
        lambda_regulpose=None,
    ):
        super(ManoLoss, self).__init__()
        self.lambda_verts3d = lambda_verts3d
        self.lambda_joints3d = lambda_joints3d
        self.lambda_manopose = lambda_manopose
        self.lambda_manoshape = lambda_manoshape
        self.lambda_regulshape = lambda_regulshape
        self.lambda_regulpose = lambda_regulpose

    def forward(self, preds, gts):

        if self.lambda_verts3d is not None and "verts3d" in gts:
            mesh3d_loss = self.lambda_verts3d * F.mse_loss(
                preds["verts3d"],
                gts["verts3d"].unsqueeze(0).expand(preds["verts3d"].shape),
            )

        if self.lambda_joints3d is not None and "joints3d" in gts:
            joints3d_loss = self.lambda_joints3d * F.mse_loss(
                preds["joints3d"],
                gts["joints3d"].unsqueeze(0).expand(preds["joints3d"].shape),
            )

        if self.lambda_manopose is not None and "mano_pose" in gts:
            pose_param_loss = self.lambda_manopose * F.mse_loss(
                preds["mano_pose"],
                gts["mano_pose"].unsqueeze(0).expand(preds["mano_pose"].shape),
            )

        if self.lambda_manoshape is not None and "mano_shape" in gts:
            shape_param_loss = self.lambda_manoshape * F.mse_loss(
                preds["mano_shape"],
                gts["mano_shape"].unsqueeze(0).expand(preds["mano_shape"].shape),
            )

        if self.lambda_regulshape:
            shape_regul_loss = self.lambda_regulshape * F.mse_loss(
                preds["mano_shape"], torch.zeros_like(preds["mano_shape"])
            )

        if self.lambda_regulpose:
            pose_regul_loss = self.lambda_regulpose * F.mse_loss(
                preds["mano_pose"][:, 3:], torch.zeros_like(preds["mano_pose"][:, 3:])
            )

        if self.lambda_regulshape:
            return (
                mesh3d_loss,
                joints3d_loss,
                pose_param_loss,
                shape_param_loss,
                shape_regul_loss,
                pose_regul_loss,
            )
        else:
            return (
                mesh3d_loss,
                joints3d_loss,
                pose_param_loss,
                shape_param_loss,
                None,
                None,
            )


class ManoShapeLoss(nn.Module):
    def __init__(self, lambda_manoshape=None, lambda_regulshape=None):
        super(ManoShapeLoss, self).__init__()
        self.lambda_manoshape = lambda_manoshape
        self.lambda_regulshape = lambda_regulshape

    def forward(self, pred_shape, gt_shape):

        shape_param_loss = self.lambda_manoshape * F.mse_loss(
            pred_shape, gt_shape.unsqueeze(0).expand(pred_shape.shape)
        )

        shape_regul_loss = self.lambda_regulshape * F.mse_loss(
            pred_shape, torch.zeros_like(pred_shape)
        )

        return shape_param_loss, shape_regul_loss
