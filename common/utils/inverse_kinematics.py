#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File        :inverse_kinematics.py
# @Date        :2022/06/29 09:55:39
# @Author      :zerui chen
# @Contact     :zerui.chen@inria.fr

import torch
from kornia.geometry.conversions import rotation_matrix_to_axis_angle

from manopth.manopth.manolayer import ManoLayer
from manopth.manopth.rodrigues_layer import batch_rodrigues


def ik_solver_mano(mano_shape, pred_joints):
    mano_layer = ManoLayer(
        flat_hand_mean=True,
        side="right",
        mano_root="tool/mano_models",
        use_pca=False,
        center_idx=0,
    )
    mano_layer = mano_layer.cuda()
    batch_size = pred_joints.shape[0]

    mano_pose = torch.eye(3).repeat(batch_size, 16, 1, 1).to(pred_joints.device)
    mano_axisang = torch.zeros(
        (batch_size, 16, 3), dtype=torch.float32, device=pred_joints.device
    )
    target_joints = (pred_joints[:, :21] - pred_joints[:, [0]]).clone().detach()
    if mano_shape is None:
        target_shape = torch.zeros(
            (batch_size, 10), dtype=torch.float32, device=pred_joints.device
        )
    else:
        target_shape = mano_shape.clone().detach()
    _, template_joints = mano_layer(
        mano_axisang.reshape((batch_size, -1)), target_shape
    )
    template_joints = template_joints / 1000

    P_0 = torch.cat(
        [
            target_joints[:, [1]] - target_joints[:, [0]],
            target_joints[:, [5]] - target_joints[:, [0]],
            target_joints[:, [9]] - target_joints[:, [0]],
            target_joints[:, [13]] - target_joints[:, [0]],
            target_joints[:, [17]] - target_joints[:, [0]],
        ],
        axis=1,
    ).transpose(1, 2)
    T_0 = torch.cat(
        [
            template_joints[:, [1]] - template_joints[:, [0]],
            template_joints[:, [5]] - template_joints[:, [0]],
            template_joints[:, [9]] - template_joints[:, [0]],
            template_joints[:, [13]] - template_joints[:, [0]],
            template_joints[:, [17]] - template_joints[:, [0]],
        ],
        axis=1,
    ).transpose(1, 2)
    H = torch.matmul(T_0, P_0.transpose(1, 2))
    U, S, V_T = torch.linalg.svd(H)
    V = V_T.transpose(1, 2)
    R = torch.matmul(V, U.transpose(1, 2)).to(pred_joints.device)

    det0 = torch.linalg.det(R)
    valid_idx = (abs(det0 + 1) > 1e-6).unsqueeze(-1).long()
    batch_id = torch.where(abs(det0 + 1) > 1e-6)[0]
    mano_axisang[batch_id, 0] = rotation_matrix_to_axis_angle(R)[batch_id]
    mano_pose[batch_id, 0] = R[batch_id]

    finger_list = [
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 17, 18, 19, 20],
        [0, 13, 14, 15, 16],
        [0, 1, 2, 3, 4],
    ]
    for group_idx, group in enumerate(finger_list):
        recon_joints = torch.zeros(
            (batch_size, 5, 3), dtype=torch.float32, device=pred_joints.device
        )
        for joint_idx, joint in enumerate(group):
            if joint_idx < 2:
                continue

            vec_template = (
                template_joints[:, group[joint_idx]]
                - template_joints[:, group[joint_idx - 1]]
            )

            R_pa = R.clone()
            for i in range(joint_idx - 2):
                R_pa = torch.matmul(R_pa, mano_pose[:, group_idx * 3 + i + 1])

            recon_joints[:, joint_idx - 1] = (
                torch.matmul(
                    R_pa,
                    (
                        template_joints[:, group[joint_idx - 1]]
                        - template_joints[:, group[joint_idx - 2]]
                    ).unsqueeze(-1),
                ).squeeze(-1)
                + recon_joints[:, joint_idx - 2]
            )

            vec_target = torch.matmul(
                R_pa.transpose(1, 2),
                (
                    target_joints[:, group[joint_idx]] - recon_joints[:, joint_idx - 1]
                ).unsqueeze(-1),
            ).squeeze(-1)

            temp_axis = torch.cross(vec_template, vec_target)
            temp_axis = temp_axis / (torch.norm(temp_axis, dim=-1, keepdim=True) + 1e-7)
            overall_angle = torch.acos(
                torch.clamp(
                    torch.einsum("bk, bk->b", vec_template, vec_target).unsqueeze(-1)
                    / (torch.norm(vec_template, dim=-1, keepdim=True) + 1e-7)
                    / (torch.norm(vec_target, dim=-1, keepdim=True) + 1e-7),
                    -1 + 1e-7,
                    1 - 1e-7,
                )
            )
            mano_axisang[batch_id, group_idx * 3 + joint_idx - 2 + 1] = (
                overall_angle * temp_axis
            )[batch_id]
            local_R = batch_rodrigues(overall_angle * temp_axis).reshape(
                batch_size, 3, 3
            )
            mano_pose[batch_id, group_idx * 3 + joint_idx - 2 + 1] = local_R[batch_id]

    verts, joints = mano_layer(mano_axisang.reshape((batch_size, -1)), target_shape)
    verts = verts / 1000
    joints = joints / 1000
    verts += pred_joints[:, [0]]
    joints += pred_joints[:, [0]]

    mano_axisang = mano_axisang.reshape((batch_size, -1))

    results = {
        "verts": verts,
        "joints": joints,
        "shape": target_shape,
        "pose": mano_axisang,
        "vis": valid_idx,
    }

    return results
