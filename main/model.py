# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from Keypoint Transformer, HFL-Net, DenseMutualAttention, and AlignSDF
# ------------------------------------------------------------------------------

import random

import torch.nn as nn

from common.nets.layer import MLP
from common.nets.loss import (
    JointHeatmapLoss,
    JointvoteLoss,
    ManoLoss,
    ManoShapeLoss,
    SepSDFLoss,
)
from common.nets.mano_head import ManoHead
from common.nets.module import BackboneNet, DecoderNet, DecoderNet_big
from common.nets.sdf_net import SDFDecoder
from common.nets.transformer import Transformer, VoteTransformer
from common.utils.misc import *
from common.utils.sdf_utils import get_nerf_embedder
from manopth.manopth.manolayer import ManoLayer


class Model(nn.Module):
    def __init__(
        self,
        backbone_net,
        decoder_net,
        hand_sdf_decoder,
        obj_sdf_decoder,
        hand_transformer,
        obj_transformer,
        mano_layer,
    ):
        super(Model, self).__init__()

        # modules
        self.backbone_net = backbone_net
        self.decoder_net = decoder_net
        self.hand_sdf_decoder = hand_sdf_decoder
        self.obj_sdf_decoder = obj_sdf_decoder
        self.hand_transformer = hand_transformer
        self.obj_transformer = obj_transformer

        self.hand_sigmoid_beta = nn.Parameter(0.1 * torch.ones(1))
        self.obj_sigmoid_beta = nn.Parameter(0.1 * torch.ones(1))

        output_dim = cfg.hidden_dim - cfg.PointFeatSize

        # MLP for converting concatenated image features to 256-D features
        self.norm1 = nn.LayerNorm(cfg.mutliscale_dim)
        self.linear_transformerin = MLP(
            input_dim=cfg.mutliscale_dim,
            hidden_dim=[1024, 512, 256],
            output_dim=output_dim,
            num_layers=4,
            is_activation_last=True,
        )
        self.linear_sdfin = MLP(
            input_dim=cfg.mutliscale_dim,
            hidden_dim=[512],
            output_dim=int(cfg.hidden_dim),
            num_layers=2,
            is_activation_last=True,
        )
        self.activation = nn.functional.relu

        coord_change_mat = torch.tensor(
            [[1.0, 0.0, 0.0], [0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=torch.float32
        )
        if cfg.use_inverse_kinematics:
            self.mano_query_embed = nn.Embedding(1, cfg.hidden_dim)
        else:
            self.mano_query_embed = nn.Embedding(cfg.mano_num_queries, cfg.hidden_dim)
            self.mano_head = ManoHead(mano_layer, coord_change_mat=coord_change_mat)
            pose_fan_out = 6
            self.linear_pose = MLP(cfg.hidden_dim, cfg.hidden_dim, pose_fan_out, 3)
        self.linear_shape = MLP(cfg.hidden_dim, cfg.hidden_dim, 10, 3)

        self.linear_handvote = MLP(cfg.hidden_dim, cfg.hidden_dim, 20 * 3, 4)
        self.linear_handcls = MLP(cfg.hidden_dim, cfg.hidden_dim, 20, 3)
        self.linear_objvote = MLP(cfg.hidden_dim, cfg.hidden_dim, 8 * 3, 4)
        self.linear_objcls = MLP(cfg.hidden_dim, cfg.hidden_dim, 8, 3)

        self.linear_obj_rel_trans = MLP(cfg.hidden_dim, cfg.hidden_dim, 3, 3)
        self.linear_obj_rot = MLP(cfg.hidden_dim, cfg.hidden_dim, 3, 3)

        # loss functions
        self.joint_heatmap_loss = JointHeatmapLoss()
        self.obj_seg_loss = torch.nn.BCELoss(reduction="none")
        self.hand_seg_loss = torch.nn.BCELoss(reduction="none")
        self.joints_vote_loss = JointvoteLoss()

        self.obj_rot_loss = nn.SmoothL1Loss(reduction="mean")
        self.obj_trans_loss = nn.SmoothL1Loss(reduction="mean")
        if cfg.use_inverse_kinematics:
            self.mano_shape_loss = ManoShapeLoss(
                lambda_manoshape=cfg.lambda_manoshape,
                lambda_regulshape=cfg.mano_lambda_regulshape,
            )
        else:
            self.mano_loss = ManoLoss(
                lambda_verts3d=cfg.lambda_verts3d,
                lambda_joints3d=cfg.lambda_joints3d,
                lambda_manopose=cfg.lambda_manopose,
                lambda_manoshape=cfg.lambda_manoshape,
            )

        self.sdf_loss = SepSDFLoss()

        # Freeze batch norm layers
        self.freeze_stages()

    def freeze_stages(self):
        for name, param in self.backbone_net.named_parameters():
            if "bn" in name:
                param.requires_grad = False

    def sdf_activation(self, input, beta):
        beta.data.copy_(max(torch.zeros_like(beta.data) + 2e-3, beta.data))
        sigma = torch.sigmoid(input / beta) / beta
        return sigma

    def render_gaussian_heatmap(self, joint_coord):
        x = torch.arange(cfg.output_hm_shape[2])
        y = torch.arange(cfg.output_hm_shape[1])
        yy, xx = torch.meshgrid(y, x)
        xx = xx[None, None, :, :].cuda().float()
        yy = yy[None, None, :, :].cuda().float()

        x = joint_coord[:, :, 0, None, None]
        y = joint_coord[:, :, 1, None, None]
        heatmap = torch.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2
        )
        heatmap = torch.sum(heatmap, 1)
        heatmap = heatmap * 255

        return heatmap

    def get_input_transformer(
        self, feature_pyramid, sdf_points, center_joint, cam_intr, sdf_scale
    ):
        cam_sdf_points = (sdf_points / sdf_scale) + center_joint[:, None, :]
        sdf_2D = torch.bmm(cam_sdf_points, cam_intr.transpose(1, 2))
        sdf_2D = sdf_2D[:, :, :2] / sdf_2D[:, :, [2]]

        normalizer = (
            torch.tensor([cfg.input_img_shape[1] - 1, cfg.input_img_shape[0] - 1]) / 2
        )
        normalizer = normalizer.to(sdf_2D.device)

        grids = (sdf_2D.detach().clone() - normalizer) / normalizer
        # Sample the CNN features
        multiscale_features = []
        grids_tensor = grids.unsqueeze(1).to(
            feature_pyramid[cfg.mutliscale_layers[0]].device
        )

        for layer_name in cfg.mutliscale_layers:
            multiscale_features.append(
                torch.nn.functional.grid_sample(
                    feature_pyramid[layer_name],
                    grids_tensor,
                    padding_mode="border",
                    align_corners=True,
                )
            )

        multiscale_features = torch.cat(multiscale_features, dim=1).squeeze(2)
        multiscale_features = multiscale_features.permute(0, 2, 1).contiguous()

        transformer_latent = self.linear_transformerin(multiscale_features)

        return transformer_latent, cam_sdf_points

    def sdf_forward(
        self,
        feature_pyramid,
        sdf_points,
        center_joint,
        cam_intr,
        sdf_scale,
        type="hand",
    ):
        cam_sdf_points = (sdf_points / sdf_scale) + center_joint[:, None, :]
        sdf_2D = torch.bmm(cam_sdf_points, cam_intr.transpose(1, 2))
        sdf_2D = sdf_2D[:, :, :2] / sdf_2D[:, :, [2]]

        normalizer = (
            torch.tensor([cfg.input_img_shape[1] - 1, cfg.input_img_shape[0] - 1]) / 2
        )
        normalizer = normalizer.to(sdf_2D.device)
        grids = (sdf_2D.detach().clone() - normalizer) / normalizer
        grids_tensor = grids.unsqueeze(1).to(
            feature_pyramid[cfg.mutliscale_layers[0]].device
        )

        multiscale_features = []
        for layer_name in cfg.mutliscale_layers:
            multiscale_features.append(
                torch.nn.functional.grid_sample(
                    feature_pyramid[layer_name],
                    grids_tensor,
                    padding_mode="border",
                    align_corners=True,
                )
            )
        multiscale_features = torch.cat(multiscale_features, dim=1).squeeze(2)
        multiscale_features = multiscale_features.permute(0, 2, 1).contiguous()

        points_fea = self.linear_sdfin(multiscale_features)

        nerf_embedding, _ = get_nerf_embedder((cfg.PointFeatSize - 3) // 6)
        pos_enc3d = nerf_embedding(sdf_points.reshape(-1, 3))

        decoder_inputs = torch.cat(
            [
                points_fea.reshape((-1, points_fea.shape[-1])),
                pos_enc3d,
                sdf_points.reshape(-1, 3),
            ],
            1,
        ).contiguous()

        if type == "hand":
            pred_sdf, pred_class = self.hand_sdf_decoder(decoder_inputs)
        elif type == "obj":
            pred_sdf, pred_class = self.obj_sdf_decoder(decoder_inputs)

        pred_sdf = pred_sdf.reshape((sdf_points.shape[0], sdf_points.shape[1], 1))
        if cfg.ClassifierBranch:
            pred_class = pred_class.reshape(
                (sdf_points.shape[0], sdf_points.shape[1], 6)
            )

        pred_sdf = torch.clamp(pred_sdf, -cfg.ClampingDistance, cfg.ClampingDistance)
        pos_enc3d = pos_enc3d.reshape((sdf_points.shape[0], sdf_points.shape[1], -1))

        return pred_sdf, pred_class, pos_enc3d

    def sdf_infer(
        self,
        feature_pyramid,
        center_joint,
        cam_intr,
        bbox,
        sdf_scale,
        num_points,
        type="hand",
    ):
        batch_size = center_joint.shape[0]
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (cfg.bins_n - 1)

        overall_index = torch.arange(0, cfg.bins_n**3, 1, out=torch.LongTensor())
        samples = torch.zeros(cfg.bins_n**3, 3)

        # transform first 3 columns
        # to be the x, y, z index
        samples[:, 2] = overall_index % cfg.bins_n
        samples[:, 1] = (overall_index.long() / cfg.bins_n) % cfg.bins_n
        samples[:, 0] = ((overall_index.long() / cfg.bins_n) / cfg.bins_n) % cfg.bins_n

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        pose_points = torch.zeros((batch_size, num_points, 3)).float().cuda()
        pose_sdf = torch.zeros((batch_size, num_points, 1)).float().cuda()
        pose_posenc3d = (
            torch.zeros((batch_size, num_points, cfg.PointFeatSize - 3)).float().cuda()
        )
        if cfg.ClassifierBranch:
            pose_class = torch.zeros((batch_size, num_points, 6)).float().cuda()
        else:
            pose_class = None

        for bth_i in range(batch_size):
            b_cam_xyz_points = (samples.clone() / sdf_scale) + center_joint[
                bth_i
            ].unsqueeze(0).cpu()
            b_bbox = bbox[bth_i].cpu()
            b_sdf_2D = torch.mm(b_cam_xyz_points, cam_intr[bth_i].transpose(0, 1).cpu())
            b_sdf_2D = b_sdf_2D[:, :2] / b_sdf_2D[:, [2]]

            filtered_id = torch.logical_and(
                torch.logical_and(
                    b_sdf_2D[:, 0] > b_bbox[0], b_sdf_2D[:, 0] < b_bbox[2]
                ),
                torch.logical_and(
                    b_sdf_2D[:, 1] > b_bbox[1], b_sdf_2D[:, 1] < b_bbox[3]
                ),
            )
            b_sdf_2D = b_sdf_2D[filtered_id].unsqueeze(0).cuda()
            b_samples = samples[filtered_id].detach().clone().cuda()

            normalizer = (
                torch.tensor([cfg.input_img_shape[1] - 1, cfg.input_img_shape[0] - 1])
                / 2
            )
            normalizer = normalizer.to(b_sdf_2D.device)

            grids = (b_sdf_2D.detach().clone() - normalizer) / normalizer

            # Sample the CNN features
            grids_tensor = grids.unsqueeze(1).to(
                feature_pyramid[cfg.mutliscale_layers[0]].device
            )
            multiscale_features = []
            for layer_name in cfg.mutliscale_layers:
                multiscale_features.append(
                    torch.nn.functional.grid_sample(
                        feature_pyramid[layer_name][bth_i].unsqueeze(0),
                        grids_tensor,
                        padding_mode="border",
                        align_corners=True,
                    )
                )

            multiscale_features = torch.cat(multiscale_features, dim=1).squeeze(2)
            multiscale_features = multiscale_features.permute(0, 2, 1)

            b_points_fea = self.linear_sdfin(multiscale_features)

            nerf_embedding, _ = get_nerf_embedder((cfg.PointFeatSize - 3) // 6)
            b_pos_enc3d = nerf_embedding(b_samples)
            decoder_inputs = torch.cat(
                [b_points_fea.squeeze(0), b_pos_enc3d, b_samples], 1
            ).contiguous()

            if type == "hand":
                b_pred_sdf, b_pred_class = self.hand_sdf_decoder(decoder_inputs)
            elif type == "obj":
                b_pred_sdf, b_pred_class = self.obj_sdf_decoder(decoder_inputs)

            b_pred_sdf = b_pred_sdf.squeeze(1)

            _, value_indices = torch.sort(b_pred_sdf.abs().detach())
            value_indices = value_indices[:num_points]

            pose_points[bth_i] = b_samples[value_indices].detach().clone()
            pose_sdf[bth_i] = b_pred_sdf[value_indices].unsqueeze(-1).clone()
            pose_posenc3d[bth_i] = b_pos_enc3d[value_indices].detach().clone()
            if cfg.ClassifierBranch:
                pose_class[bth_i] = b_pred_class[value_indices].detach().clone()

        pose_sdf = torch.clamp(pose_sdf, -cfg.ClampingDistance, cfg.ClampingDistance)
        return pose_points, pose_sdf, pose_posenc3d, pose_class

    def forward(self, inputs, targets, meta_info, mode, epoch_cnt=1e8, batch_ratio=0):
        input_img = inputs["img"]

        loss = {}
        out = {}

        mano_root = meta_info["mano_root"]
        obj_center_cam = meta_info["obj_center_cam"]
        cam_intr = meta_info["cam_intr"]

        img_feat, enc_skip_conn_layers = self.backbone_net(input_img)
        feature_pyramid, decoder_out = self.decoder_net(img_feat, enc_skip_conn_layers)

        if mode == "train" or cfg.dataset == "dexycb":
            hand_sdf_points = inputs["hand_sdf_points"]
            obj_sdf_points = inputs["obj_sdf_points"]
            hand_sdf_gt = targets["hand_sdf"]
            obj_sdf_gt = targets["obj_sdf"]

            hand_sdf_sample, _, _ = self.sdf_forward(
                feature_pyramid,
                hand_sdf_points,
                mano_root,
                cam_intr,
                cfg.hand_sdf_scale,
                type="hand",
            )
            obj_sdf_sample, _, _ = self.sdf_forward(
                feature_pyramid,
                obj_sdf_points,
                obj_center_cam,
                cam_intr,
                cfg.obj_sdf_scale,
                type="obj",
            )

            hand_sdf_gt = torch.clamp(
                hand_sdf_gt, -cfg.ClampingDistance, cfg.ClampingDistance
            )
            obj_sdf_gt = torch.clamp(
                obj_sdf_gt, -cfg.ClampingDistance, cfg.ClampingDistance
            )

            loss["sdfhand_loss"], loss["sdfobj_loss"] = self.sdf_loss(
                hand_sdf_sample, obj_sdf_sample, hand_sdf_gt, obj_sdf_gt
            )

            joint_heatmap_out = decoder_out[:, 0]
            hand_seg_out = decoder_out[:, 1]
            hand_seg_gt = targets["hand_seg"]
            obj_seg_out = decoder_out[:, 2]
            obj_seg_gt = targets["obj_seg"]

            out["joint_heatmap_out"] = joint_heatmap_out
            out["hand_seg_gt_out"] = hand_seg_gt
            out["hand_seg_pred_out"] = hand_seg_out
            out["obj_seg_gt_out"] = obj_seg_gt
            out["obj_seg_pred_out"] = obj_seg_out

            target_joint_heatmap = self.render_gaussian_heatmap(targets["joint_coord"])
            loss["joint_heatmap"] = self.joint_heatmap_loss(
                joint_heatmap_out, target_joint_heatmap
            )

            loss["obj_seg"] = self.obj_seg_loss(obj_seg_out, obj_seg_gt)
            loss["hand_seg"] = self.hand_seg_loss(hand_seg_out, hand_seg_gt)

        bbox_hand = meta_info["bbox_hand"]
        bbox_obj = meta_info["bbox_obj"]
        p = random.uniform(0, 1)
        if (p < 0.4 or epoch_cnt < cfg.point_sampling_epoch) and mode == "train":
            hand_pre_points = inputs["hand_pre_points"]
            obj_pre_points = inputs["obj_pre_points"]
            dist_range = cfg.random_move_dist[
                len([aa for aa in cfg.random_ratio if batch_ratio > aa])
            ]
            hand_points = (
                hand_pre_points
                + torch.empty_like(hand_pre_points)
                .uniform_(-dist_range, dist_range)
                .cuda()
            )
            obj_points = (
                obj_pre_points
                + torch.empty_like(obj_pre_points)
                .uniform_(-dist_range, dist_range)
                .cuda()
            )
            hand_sdf, _, hand_posenc3d = self.sdf_forward(
                feature_pyramid,
                hand_points,
                mano_root,
                cam_intr,
                cfg.hand_sdf_scale,
                type="hand",
            )
            obj_sdf, _, obj_posenc3d = self.sdf_forward(
                feature_pyramid,
                obj_points,
                obj_center_cam,
                cam_intr,
                cfg.obj_sdf_scale,
                type="obj",
            )

        else:
            with torch.no_grad():
                hand_points, hand_sdf, hand_posenc3d, hand_class = self.sdf_infer(
                    feature_pyramid,
                    mano_root,
                    cam_intr,
                    bbox_hand,
                    cfg.hand_sdf_scale,
                    cfg.num_samp_hand,
                    type="hand",
                )
                obj_points, obj_sdf, obj_posenc3d, _ = self.sdf_infer(
                    feature_pyramid,
                    obj_center_cam,
                    cam_intr,
                    bbox_obj,
                    cfg.obj_sdf_scale,
                    cfg.num_samp_obj,
                    type="obj",
                )

        sigma_hand = self.sdf_activation(hand_sdf.detach(), self.hand_sigmoid_beta)
        sigma_obj = self.sdf_activation(obj_sdf.detach(), self.obj_sigmoid_beta)

        hand_fea, hand_points_cam = self.get_input_transformer(
            feature_pyramid, hand_points, mano_root, cam_intr, cfg.hand_sdf_scale
        )
        hand_points_notrans = hand_points_cam - mano_root[:, None, :]
        obj_fea, obj_points_cam = self.get_input_transformer(
            feature_pyramid, obj_points, obj_center_cam, cam_intr, cfg.obj_sdf_scale
        )
        obj_points_notrans = obj_points_cam - obj_center_cam[:, None, :]

        hand_o_points = (
            hand_points_cam - obj_center_cam[:, None, :]
        ) * cfg.obj_sdf_scale
        hand_o_points_notrans = hand_points_cam - obj_center_cam[:, None, :]  # bug
        hand_o_sdf, _, hand_o_posenc3d = self.sdf_forward(
            feature_pyramid,
            hand_o_points,
            obj_center_cam,
            cam_intr,
            cfg.obj_sdf_scale,
            type="obj",
        )
        obj_h_points = (obj_points_cam - mano_root[:, None, :]) * cfg.hand_sdf_scale
        obj_h_points_notrans = obj_points_cam - mano_root[:, None, :]  # bug
        obj_h_sdf, _, obj_h_posenc3d = self.sdf_forward(
            feature_pyramid,
            obj_h_points,
            mano_root,
            cam_intr,
            cfg.hand_sdf_scale,
            type="hand",
        )
        sigma_hand_o = self.sdf_activation(hand_o_sdf.detach(), self.obj_sigmoid_beta)
        sigma_obj_h = self.sdf_activation(obj_h_sdf.detach(), self.hand_sigmoid_beta)

        hand_transformer_in = (
            torch.cat(
                [
                    hand_points_notrans,
                    hand_posenc3d,
                    hand_fea * sigma_hand,
                ],
                dim=2,
            )
            .permute(1, 0, 2)
            .contiguous()
        )
        obj_h_transformer_in = (
            torch.cat(
                [obj_h_points_notrans, obj_h_posenc3d, obj_fea * sigma_obj_h], dim=2
            )
            .permute(1, 0, 2)
            .contiguous()
        )
        hand_transformer_in = torch.cat(
            [hand_transformer_in, obj_h_transformer_in.detach()], dim=0
        )
        hand_positions = torch.zeros_like(hand_transformer_in).to(
            hand_transformer_in.device
        )
        obj_transformer_in = (
            torch.cat([obj_points_notrans, obj_posenc3d, obj_fea * sigma_obj], dim=2)
            .permute(1, 0, 2)
            .contiguous()
        )
        hand_o_transformer_in = (
            torch.cat(
                [hand_o_points_notrans, hand_o_posenc3d, hand_fea * sigma_hand_o], dim=2
            )
            .permute(1, 0, 2)
            .contiguous()
        )
        obj_transformer_in = torch.cat(
            [obj_transformer_in, hand_o_transformer_in.detach()], dim=0
        )
        obj_positions = torch.zeros_like(obj_transformer_in).to(
            obj_transformer_in.device
        )

        if cfg.use_inverse_kinematics:
            tgt_mask = None
            memory_mask = get_manoshape_memory_mask().to(hand_transformer_in.device)
        else:
            tgt_mask = get_mano_tgt_mask().to(hand_transformer_in.device)
            memory_mask = get_mano_memory_mask().to(hand_transformer_in.device)

        hand_transformer_out, memory, hand_encoder_out, attn_wts = (
            self.hand_transformer(
                src=hand_transformer_in,
                mask=None,
                pos_embed=hand_positions,
                src_mask=None,
                query_embed=self.mano_query_embed.weight,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
        )
        obj_memory, obj_encoder_out = self.obj_transformer(
            src=obj_transformer_in, mask=None, pos_embed=obj_positions, src_mask=None
        )

        # Make all the predictions
        hand_off = self.linear_handvote(hand_encoder_out[:, : cfg.num_samp_hand])
        hand_cls = self.linear_handcls(hand_encoder_out[:, : cfg.num_samp_hand])

        obj_rot = self.linear_obj_rot(
            obj_encoder_out[:, : cfg.num_samp_obj]
        )  # 6 x N x 3
        obj_trans = self.linear_obj_rel_trans(obj_encoder_out[:, : cfg.num_samp_obj])

        if cfg.use_inverse_kinematics:
            mano_shape = self.linear_shape(hand_transformer_out[:, 0])
            out["mano_shape_out"] = mano_shape[-1]
        else:
            mano_pose6d = self.linear_pose(
                hand_transformer_out[:, : cfg.mano_shape_indx]
            )  # 6 x 16 x N x 3(9)
            mano_shape = self.linear_shape(
                hand_transformer_out[:, cfg.mano_shape_indx]
            )  # 6 x N x 10

            mano_params = (
                targets["mano_param"]
                if mode == "train" or cfg.dataset == "dexycb"
                else None
            )
            pred_mano_results, gt_mano_results = self.mano_head(
                mano_pose6d, mano_shape, mano_params=mano_params
            )
            # verify = torch.norm(gt_mano_results['joints3d'][0] - targets['joint_cam_no_trans'][0]/1000, dim=-1)

            out["mano_mesh_out"] = pred_mano_results["verts3d"][-1]
            out["mano_joints_out"] = pred_mano_results["joints3d"][-1]
            if cfg.dataset == "dexycb":
                out["mano_joints_gt_out"] = gt_mano_results["joints3d"]
                out["mano_mesh_gt_out"] = gt_mano_results["verts3d"]

        if mode != "train":
            out["obj_rot_out"] = obj_rot[-1].permute(1, 0, 2).contiguous()
            out["obj_trans_out"] = obj_trans[-1].permute(1, 0, 2).contiguous()

        if mode == "train" or cfg.dataset == "dexycb":
            joints3d_gt = targets["joint_cam_no_trans"][:, 1:]
        else:
            joints3d_gt = torch.zeros((mano_root.shape[0], 20, 3)).cuda()

        # Get all the losses for all the predictions
        (
            loss["loss_joint_3d"],
            loss["loss_joint_cls"],
            loss["loss_all_joint_3d"],
            hand_joints,
        ) = self.joints_vote_loss(hand_points_notrans, hand_off, hand_cls, joints3d_gt)
        out["hand_joints_out"] = hand_joints[-1]

        if mode == "train" or cfg.dataset == "dexycb":
            if cfg.use_inverse_kinematics:
                mano_shape_gt = targets["mano_param"][:, -10:]
                loss["shape_param_loss"], loss["shape_reg_loss"] = self.mano_shape_loss(
                    mano_shape, mano_shape_gt
                )
            else:
                (
                    loss["mano_mesh_loss"],
                    loss["mano_joint_loss"],
                    loss["pose_param_loss"],
                    loss["shape_param_loss"],
                    _,
                    _,
                ) = self.mano_loss(pred_mano_results, gt_mano_results)

        loss["obj_rot"] = self.obj_rot_loss(
            obj_rot, targets["obj_rot"].unsqueeze(0).unsqueeze(0).expand_as(obj_rot)
        )
        loss["obj_trans"] = self.obj_trans_loss(
            obj_trans,
            targets["rel_obj_trans"].unsqueeze(0).unsqueeze(0).expand_as(obj_trans),
        )

        out1 = {**loss, **out}
        return out1


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


def get_model(mode):
    backbone_net = BackboneNet()

    if cfg.use_big_decoder:
        decoder_net = DecoderNet_big()
    else:
        decoder_net = DecoderNet()

    hand_sdf_decoder = SDFDecoder(
        latent_size=cfg.hidden_dim,
        point_feat_size=cfg.PointFeatSize,
        use_classifier=cfg.ClassifierBranch,
    )
    obj_sdf_decoder = SDFDecoder(
        latent_size=cfg.hidden_dim,
        point_feat_size=cfg.PointFeatSize,
        use_classifier=cfg.ClassifierBranch,
    )

    hand_transformer = Transformer(
        d_model=cfg.hidden_dim,
        dropout=cfg.dropout,
        nhead=cfg.nheads,
        dim_feedforward=cfg.dim_feedforward,
        num_encoder_layers=cfg.enc_layers,
        num_decoder_layers=cfg.dec_layers,
        normalize_before=cfg.pre_norm,
        return_intermediate_dec=True,
    )

    obj_transformer = VoteTransformer(
        d_model=cfg.hidden_dim,
        dropout=cfg.dropout,
        nhead=cfg.nheads,
        dim_feedforward=cfg.dim_feedforward,
        num_encoder_layers=cfg.enc_layers // 2,
        normalize_before=cfg.pre_norm,
        return_intermediate_dec=True,
    )

    print(
        "BackboneNet No. of Params = %d"
        % (sum(p.numel() for p in backbone_net.parameters() if p.requires_grad))
    )
    print(
        "decoder_net No. of Params = %d"
        % (sum(p.numel() for p in decoder_net.parameters() if p.requires_grad))
    )
    print(
        "hand transformer No. of Params = %d"
        % (sum(p.numel() for p in hand_transformer.parameters() if p.requires_grad))
    )

    mano_layer = ManoLayer(
        ncomps=45,
        center_idx=0,
        flat_hand_mean=True,
        side="right",
        mano_root="tool/mano_models",
        use_pca=False,
    )

    if mode == "train":
        backbone_net.init_weights()
        decoder_net.apply(init_weights)
        hand_sdf_decoder.apply(init_weights)
        obj_sdf_decoder.apply(init_weights)
        hand_transformer.apply(init_weights)
        obj_transformer.apply(init_weights)

    model = Model(
        backbone_net,
        decoder_net,
        hand_sdf_decoder,
        obj_sdf_decoder,
        hand_transformer,
        obj_transformer,
        mano_layer,
    )
    print(
        "Total No. of Params = %d"
        % (sum(p.numel() for p in model.parameters() if p.requires_grad))
    )

    return model
