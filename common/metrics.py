# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from HFL-Net, AlignSDF and DenseMutualAttention
# ------------------------------------------------------------------------------

import cv2
import numpy as np
import torch

from manopth.manopth.rodrigues_layer import batch_rodrigues


def compute_ADD_s_error(pred_pose, gt_pose, obj_mesh):
    N = obj_mesh.shape[0]
    add_gt = np.matmul(gt_pose[:3, 0:3], obj_mesh.T) + gt_pose[:3, 3].reshape(
        -1, 1
    )  # (3,N)
    add_gt = torch.from_numpy(add_gt.T).cuda()
    add_gt = add_gt.unsqueeze(0).repeat(N, 1, 1)

    add_pred = np.matmul(pred_pose[:3, 0:3], obj_mesh.T) + pred_pose[:3, 3].reshape(
        -1, 1
    )
    add_pred = torch.from_numpy(add_pred.T).cuda()
    add_pred = add_pred.unsqueeze(1).repeat(1, N, 1)

    dis = torch.norm(add_gt - add_pred, dim=2)
    add_bias = torch.mean(torch.min(dis, dim=1)[0])
    add_bias = add_bias.detach().cpu().numpy()
    return add_bias


def eval_batch_obj_direct(
    out, targets, meta_info, mesh_dict, ADD_res_dict, imgs=None, bboxs_dict=None
):
    # bestCnt: choose best N count for fusion
    ADD_res_dict = ADD_res_dict.copy()
    bs = targets["obj_rot"].shape[0]
    obj_rots = out["obj_rot"].cpu().numpy().mean(0)
    obj_trans = out["obj_trans"].cpu().numpy().mean(0)
    obj_rots_gt = targets["obj_rot"].cpu().numpy()
    obj_trans_gt = targets["rel_obj_trans"].cpu().numpy()
    intrinsics = meta_info["cam_intr"].cpu().numpy()
    obj_clses = meta_info["obj_cls"]
    for i in range(bs):
        pred_pose = np.concatenate(
            (cv2.Rodrigues(obj_rots[i, :3])[0], obj_trans[i][:, None]), 1
        )
        obj_pose = np.concatenate(
            (cv2.Rodrigues(obj_rots_gt[i, :3])[0], obj_trans_gt[i][:, None]), 1
        )
        # calculate REP and ADD error
        mesh = mesh_dict[obj_clses[i]].copy()

        ADD_error = compute_ADD_s_error(pred_pose, obj_pose, mesh)
        ADD_res_dict[obj_clses[i]].append(ADD_error)

    return ADD_res_dict


def compute_obj_metrics_dexycb(pred_meshes, target_meshes):
    B, N, _ = pred_meshes.shape
    add_gt = target_meshes.unsqueeze(1).repeat(1, N, 1, 1)
    add_pred = pred_meshes.unsqueeze(2).repeat(1, 1, N, 1)
    dis = torch.norm(add_gt - add_pred, dim=-1)
    add_bias = torch.mean(torch.min(dis, dim=2)[0], dim=1)
    add_bias = add_bias.detach().cpu()

    corner_indexes = torch.tensor(
        [[0, 1, 0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 1, 0, 1], [0, 0, 0, 1, 0, 1, 1, 1]]
    ).cuda()
    target_mm = torch.stack(
        [torch.min(target_meshes, dim=1)[0], torch.max(target_meshes, dim=1)[0]], dim=2
    )
    target_bboxes = torch.stack(
        [
            target_mm[:, 0, corner_indexes[0]],
            target_mm[:, 1, corner_indexes[1]],
            target_mm[:, 2, corner_indexes[2]],
        ],
        dim=2,
    )
    pred_mm = torch.stack(
        [torch.min(pred_meshes, dim=1)[0], torch.max(pred_meshes, dim=1)[0]], dim=2
    )
    pred_bboxes = torch.stack(
        [
            pred_mm[:, 0, corner_indexes[0]],
            pred_mm[:, 1, corner_indexes[1]],
            pred_mm[:, 2, corner_indexes[2]],
        ],
        dim=2,
    )

    MCE_error = (
        (pred_bboxes - target_bboxes.float()).norm(2, -1).mean(-1).detach().cpu()
    )

    return add_bias, MCE_error


def compute_obj_metrics_ho3d(pred_meshes, target_meshes):
    B, N, _ = pred_meshes.shape
    add_gt = target_meshes.unsqueeze(1).repeat(1, N, 1, 1)
    add_pred = pred_meshes.unsqueeze(2).repeat(1, 1, N, 1)
    dis = torch.norm(add_gt - add_pred, dim=-1)
    add_bias = torch.mean(torch.min(dis, dim=2)[0], dim=1)
    add_bias = add_bias.detach().cpu()

    MME_error = (target_meshes - pred_meshes).norm(2, -1).mean(-1).detach().cpu()

    return add_bias, MME_error


def eval_batched_obj_direct(
    out, targets, meta_info, templates, radius, obj_names, imgs=None, bboxs_dict=None
):
    # bestCnt: choose best N count for fusion
    bs = targets["obj_rot"].shape[0]
    obj_rots = out["obj_rot"].detach().mean(1)
    obj_trans = out["obj_trans"].detach().mean(1)
    obj_rots_gt = targets["obj_rot"].cuda()
    obj_trans_gt = targets["rel_obj_trans"].cuda()
    intrinsics = meta_info["cam_intr"].cuda()
    obj_clses = meta_info["obj_cls"]
    ho3d_eval = False if torch.is_tensor(obj_clses[0]) else True

    sample_nums = 0

    if ho3d_eval:
        used_obj = np.array([obj_cls != "019_pitcher_base" for obj_cls in obj_clses])
        sample_nums = used_obj.sum()
        obj_ids = torch.tensor(
            [
                list(obj_names.values()).index(obj_cls)
                for obj_cls in np.array(obj_clses)[used_obj]
            ]
        ).cuda()
        obj_rots = obj_rots[used_obj]
        obj_trans = obj_trans[used_obj]
        obj_rots_gt = obj_rots_gt[used_obj]
        obj_trans_gt = obj_trans_gt[used_obj]
    else:
        sample_nums = bs
        obj_ids = (obj_clses - 1).cuda()

    if sample_nums == 0:
        return 0, None, None, 0, sample_nums

    template_meshes = torch.stack(
        [templates[obj_id]["verts"].clone().detach() for obj_id in obj_ids]
    ).cuda()

    target_meshes = (
        torch.bmm(
            template_meshes,
            batch_rodrigues(obj_rots_gt).reshape(sample_nums, 3, 3).permute(0, 2, 1),
        )
        + obj_trans_gt[:, None, :]
    )
    pred_meshes = (
        torch.bmm(
            template_meshes,
            batch_rodrigues(obj_rots).reshape(sample_nums, 3, 3).permute(0, 2, 1),
        )
        + obj_trans[:, None, :]
    )

    if ho3d_eval:
        ADDS_error, MME_error = compute_obj_metrics_ho3d(pred_meshes, target_meshes)
        OCE_error = torch.norm(obj_trans - obj_trans_gt, dim=-1).detach().cpu()
        _, MCE_error = compute_obj_metrics_dexycb(pred_meshes, target_meshes)
        MCE_error = OCE_error = None
        ADDS_error = ADDS_error.mean().item()
        MME_error = MME_error.mean().item()
    else:
        ADDS_error, MCE_error = compute_obj_metrics_dexycb(pred_meshes, target_meshes)
        OCE_error = torch.norm(obj_trans - obj_trans_gt, dim=-1).detach().cpu()
        MME_error = None
        ADDS_error = ADDS_error.mean().item()
        MCE_error = MCE_error.mean().item()
        OCE_error = OCE_error.mean().item()

    return ADDS_error, MCE_error, OCE_error, MME_error, sample_nums


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)

    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
    return A2


def eval_hand_joint(preds_joint, gts_joints_coord_cam):
    sample_num = len(preds_joint)
    MJE_list = []
    PAMJE_list = []
    for n in range(sample_num):
        pred_joint = preds_joint[n].detach().cpu().numpy()
        gt_joints_coord_cam = gts_joints_coord_cam[n].detach().cpu().numpy()

        # GT and rigid align
        joints_out_aligned = rigid_align(pred_joint, gt_joints_coord_cam)

        # [mpjpe_list, pa-mpjpe_list]
        MJE_list.append(
            np.sqrt(np.sum((pred_joint - gt_joints_coord_cam) ** 2, 1)).mean()
        )
        PAMJE_list.append(
            np.sqrt(np.sum((joints_out_aligned - gt_joints_coord_cam) ** 2, 1)).mean()
        )

    return np.mean(MJE_list), np.mean(PAMJE_list)
