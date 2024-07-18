# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from Keypoint Transformer, HFL-Net, DenseMutualAttention, and AlignSDF
# ------------------------------------------------------------------------------

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dir + "/..")
import configargparse
import numpy as np
import tqdm

from common.base import Tester
from common.eval_util import EvalUtil, calculate_fscore
from common.metrics import eval_batched_obj_direct, eval_hand_joint, rigid_align
from common.utils.inverse_kinematics import ik_solver_mano
from common.utils.misc import *
from data.dataset_util import get_radius, prepare_model_template
from data.ho3d_util import dump

jointsMapManoToDefault = [
    16,
    15,
    14,
    13,
    17,
    3,
    2,
    1,
    18,
    6,
    5,
    4,
    19,
    12,
    11,
    10,
    20,
    9,
    8,
    7,
    0,
]


def parse_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, dest="gpu_ids", default="0")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        dest="ckpt_path",
        help="Full path to the checkpoint file",
    )
    args = parser.parse_args()
    args.seq_name = None

    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if "-" in args.gpu_ids:
        gpus = args.gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():

    args = parse_args()
    cfg.set_args(args.gpu_ids, args.ckpt_path)
    cfg.create_log_dir()
    tester = Tester(None)
    tester._make_batch_generator()

    templates, obj_names = prepare_model_template(cfg.simple_object_models_dir)
    radius = torch.from_numpy(np.array(get_radius(templates))).float()

    tester.ckpt_path = args.ckpt_path
    tester._make_model()
    log_dir = os.path.dirname(args.ckpt_path)

    log_file = open(os.path.join(log_dir, "results.txt"), "w+")

    indices_order = tester.jointsMapSimpleToMano

    results = {}

    results["ADDS_error"] = 0.0
    total_samples = 0

    if cfg.dataset == "dexycb":
        results["mano_mje"] = 0.0
        results["mano_pamje"] = 0.0
        results["OCE_error"] = 0.0
        results["MCE_error"] = 0.0

        eval_mesh_err, eval_mesh_err_aligned = EvalUtil(num_kp=778), EvalUtil(
            num_kp=778
        )
        f_score, f_score_aligned = list(), list()
        f_threshs = [0.005, 0.015]
    else:
        coord_change_mat = np.array(
            [[1.0, 0.0, 0.0], [0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32
        )
        mano_joint_list, mano_mesh_list = list(), list()
        results["MME_error"] = 0.0

    with torch.no_grad():
        for itr, (inputs_data, targets, meta_info) in enumerate(
            tqdm.tqdm(tester.batch_generator)
        ):

            batch_samples = meta_info["mano_root"].size(0)

            model_out = tester.model(inputs_data, targets, meta_info, "eval")
            out = {k[:-4]: model_out[k] for k in model_out.keys() if "_out" in k}
            loss = {k: model_out[k] for k in model_out.keys() if "_out" not in k}
            loss = {k: loss[k].mean() for k in loss}

            ADDS_error, MCE_error, OCE_error, MME_error, sample_nums = (
                eval_batched_obj_direct(
                    out, targets, meta_info, templates, radius, obj_names
                )
            )
            total_samples += sample_nums

            if cfg.dataset == "ho3d":
                hand_joints = torch.cat(
                    [torch.zeros_like(out["hand_joints"][:, :1]), out["hand_joints"]],
                    dim=1,
                )

                if cfg.use_inverse_kinematics:
                    hand_pose_results = ik_solver_mano(out["mano_shape"], hand_joints)
                    mano_joints = (
                        hand_pose_results["joints"].detach().cpu().numpy()
                        + meta_info["mano_root"].detach().cpu().numpy()[:, None, :]
                    )
                    mano_joints = np.matmul(mano_joints, coord_change_mat)
                    mano_mesh = (
                        hand_pose_results["verts"].detach().cpu().numpy()
                        + meta_info["mano_root"].detach().cpu().numpy()[:, None, :]
                    )
                    mano_mesh = np.matmul(mano_mesh, coord_change_mat)
                else:
                    mano_joints = out["mano_joints"].detach().cpu().numpy()
                    mano_joints = (
                        mano_joints + meta_info["mano_root"].cpu().numpy()[:, None, :]
                    )
                    mano_joints = np.matmul(mano_joints, coord_change_mat)
                    mano_mesh = out["mano_mesh"].detach().cpu().numpy()
                    mano_mesh = (
                        mano_mesh + meta_info["mano_root"].cpu().numpy()[:, None, :]
                    )
                    mano_mesh = np.matmul(mano_mesh, coord_change_mat)

                results["MME_error"] += MME_error * sample_nums * 100
                for mano_xyz, mano_verts in zip(mano_joints, mano_mesh):
                    if indices_order is not None:
                        mano_xyz = mano_xyz[indices_order]

                    mano_joint_list.append(mano_xyz)
                    mano_mesh_list.append(mano_verts)

            elif cfg.dataset == "dexycb":
                if cfg.use_inverse_kinematics:
                    hand_joints = torch.cat(
                        [
                            torch.zeros_like(out["hand_joints"][:, :1]),
                            out["hand_joints"],
                        ],
                        dim=1,
                    )
                    hand_pose_results = ik_solver_mano(out["mano_shape"], hand_joints)
                    mano_mje, mano_pamje = eval_hand_joint(
                        hand_pose_results["joints"],
                        targets["joint_cam_no_trans"] / 1000,
                    )
                else:
                    mano_mje, mano_pamje = eval_hand_joint(
                        out["mano_joints"], out["mano_joints_gt"]
                    )
                results["mano_mje"] += mano_mje * batch_samples * 100
                results["mano_pamje"] += mano_pamje * batch_samples * 100

                if cfg.eval_mesh:
                    pred_verts = out["mano_mesh"].detach().cpu().numpy().copy()
                    gts_verts = out["mano_mesh_gt"].detach().cpu().numpy().copy()
                    sample_num = len(pred_verts)
                    for n in range(sample_num):
                        preds_vert = pred_verts[n].copy()
                        gts_vert = gts_verts[n]

                        # GT and rigid align
                        verts_out_aligned = rigid_align(preds_vert, gts_vert)
                        eval_mesh_err.feed(
                            gts_vert, np.ones_like(gts_vert[:, 0]), preds_vert
                        )
                        eval_mesh_err_aligned.feed(
                            gts_vert, np.ones_like(gts_vert[:, 0]), verts_out_aligned
                        )

                        l, la = list(), list()
                        for t in f_threshs:
                            # for each threshold calculate the f score and the f score of the aligned vertices
                            f, _, _ = calculate_fscore(gts_vert, preds_vert, t)
                            l.append(f)
                            f, _, _ = calculate_fscore(gts_vert, verts_out_aligned, t)
                            la.append(f)
                        f_score.append(l)
                        f_score_aligned.append(la)

                results["OCE_error"] += OCE_error * sample_nums * 100
                results["MCE_error"] += MCE_error * sample_nums * 100

            results["ADDS_error"] += ADDS_error * sample_nums * 100

    for k in results.keys():
        print(k, ": ", results[k] / total_samples, file=log_file)

    if cfg.dataset == "dexycb" and cfg.eval_mesh:
        mesh_mean3d, _, mesh_auc3d, pck_mesh, thresh_mesh = eval_mesh_err.get_measures(
            0.0, 0.05, 100
        )
        print("Evaluation 3D MESH results:", file=log_file)
        print(
            "auc=%.3f, mean_vert3d_avg=%.2f cm" % (mesh_auc3d, mesh_mean3d * 100.0),
            file=log_file,
        )

        mesh_al_mean3d, _, mesh_al_auc3d, pck_mesh_al, thresh_mesh_al = (
            eval_mesh_err_aligned.get_measures(0.0, 0.05, 100)
        )
        print("Evaluation 3D MESH ALIGNED results:", file=log_file)
        print(
            "auc=%.3f, mean_vert3d_avg=%.2f cm\n"
            % (mesh_al_auc3d, mesh_al_mean3d * 100.0),
            file=log_file,
        )

        print("F-scores", file=log_file)
        f_score, f_score_aligned = np.array(f_score).T, np.array(f_score_aligned).T
        for f, fa, t in zip(f_score, f_score_aligned, f_threshs):
            print(
                "F@%.1fmm = %.3f" % (t * 1000, f.mean()),
                "\tF_aligned@%.1fmm = %.3f" % (t * 1000, fa.mean()),
                file=log_file,
            )

    log_file.close()

    if cfg.dataset == "ho3d":
        mano_out_path = os.path.join(log_dir, "pred_mano.json")
        dump(mano_out_path, mano_joint_list, mano_mesh_list)


if __name__ == "__main__":
    main()
