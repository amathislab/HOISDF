# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from Keypoint Transformer
# ------------------------------------------------------------------------------

import os
import sys

file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dir + "/..")
import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.tensorboard import SummaryWriter

from common.base import Evaler, Trainer, adjust_learning_rate
from common.metrics import eval_hand_joint
from main.config import cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, dest="gpu_ids", default="0")
    parser.add_argument("--continue", dest="continue_train", action="store_true")
    parser.add_argument(
        "--run_dir_name",
        dest="run_dir_name",
        type=str,
        default="train",
        help="Name of the Run",
    )
    parser.add_argument("--end_epoch", type=int, default=70)
    parser.add_argument("--point_sampling_epoch", type=int, default=40)
    parser.add_argument("--lr_drop", type=int, default=9)
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if "-" in args.gpu_ids:
        gpus = args.gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():

    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.run_dir_name, args.continue_train)
    cfg.create_run_dirs()
    cfg.end_epoch = args.end_epoch
    cfg.point_sampling_epoch = args.point_sampling_epoch
    cfg.lr_drop = args.lr_drop
    cfg.calc_mutliscale_dim(cfg.use_big_decoder, cfg.resnet_type)
    cudnn.benchmark = True

    members = [
        attr
        for attr in dir(cfg)
        if not callable(getattr(cfg, attr)) and not attr.startswith("__")
    ]
    cfg_dict = {}
    for m in members:
        cfg_dict[m] = cfg.__getattribute__(m)
    f = os.path.join(cfg.model_dir, "cfg.txt")
    with open(f, "w") as file:
        for arg in cfg_dict.keys():
            file.write("{} = {}\n".format(arg, cfg_dict[arg]))

    f = os.path.join(cfg.model_dir, "args.txt")
    with open(f, "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write("{} = {}\n".format(arg, attr))

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
    if cfg.continue_train:
        trainer.load_model()

    evaler = Evaler()
    evaler._make_batch_generator()
    evaler._make_model()

    writer = SummaryWriter(cfg.tensorboard_dir)

    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        adjust_learning_rate(trainer.lr_scheduler, trainer.optimizer)

        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()
            batch_ratio = itr / len(trainer.batch_generator)

            # forward
            trainer.optimizer.zero_grad()
            model_out = trainer.model(
                inputs, targets, meta_info, "train", epoch, batch_ratio
            )
            out = {k[:-4]: model_out[k] for k in model_out.keys() if "_out" in k}
            loss = {k: model_out[k] for k in model_out.keys() if "_out" not in k}
            loss = {k: loss[k].mean() for k in loss}

            loss["sdfhand_loss"] *= cfg.sdf_hand_weight
            loss["sdfobj_loss"] *= cfg.sdf_obj_weight

            joint_heatmap_out = out["joint_heatmap"]
            loss["joint_heatmap"] *= cfg.hm_weight
            loss["obj_seg"] *= cfg.obj_hm_weight
            loss["hand_seg"] *= cfg.obj_hm_weight
            loss["obj_rot"] *= cfg.obj_rot_weight
            loss["obj_trans"] *= cfg.obj_trans_weight

            loss["loss_joint_3d"] *= cfg.joint_weight
            loss["loss_joint_cls"] *= cfg.cls_weight
            loss["loss_all_joint_3d"] *= cfg.joint_weight

            if itr % 400 == 0:
                for k in loss:
                    writer.add_scalar(
                        "train_" + k,
                        loss[k],
                        epoch * len(trainer.batch_generator) + itr,
                    )

            # backward
            sum(loss[k] for k in loss).backward()

            trainer.optimizer.step()
            trainer.gpu_timer.toc()
            screen = [
                "Epoch %d/%d itr %d/%d:"
                % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                "lr: %g" % (trainer.lr_scheduler.get_last_lr()[-1]),
                "speed: %.2f(%.2fs r%.2f)s/itr"
                % (
                    trainer.tot_timer.average_time,
                    trainer.gpu_timer.average_time,
                    trainer.read_timer.average_time,
                ),
                "%.2fh/epoch"
                % (trainer.tot_timer.average_time / 3600.0 * trainer.itr_per_epoch),
            ]
            screen += ["%s: %.4f" % ("loss_" + k, v.detach()) for k, v in loss.items()]
            trainer.logger.info(" ".join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()

        trainer.lr_scheduler.step()

        if epoch >= cfg.point_sampling_epoch:
            save_gap = 1
        else:
            save_gap = 5

        if epoch % save_gap == 0:
            with torch.no_grad():
                evaler._load_state(trainer.model)
                epoch_loss = {}
                epoch_loss["total_loss"] = 0.0

                if cfg.dataset == "dexycb":
                    epoch_loss["sdfhand_loss"] = 0.0
                    epoch_loss["sdfobj_loss"] = 0.0
                    epoch_loss["loss_joint_3d"] = 0.0
                    epoch_loss["loss_joint_cls"] = 0.0
                    epoch_loss["loss_all_joint_3d"] = 0.0

                    if cfg.use_inverse_kinematics:
                        epoch_loss["shape_param_loss"] = 0.0
                        epoch_loss["shape_reg_loss"] = 0.0
                    else:
                        epoch_loss["mano_mesh_loss"] = 0.0
                        epoch_loss["mano_joint_loss"] = 0.0
                        epoch_loss["pose_param_loss"] = 0.0
                        epoch_loss["shape_param_loss"] = 0.0

                    epoch_loss["out_mje"] = 0.0
                    epoch_loss["out_pamje"] = 0.0
                    epoch_loss["OCE_error"] = 0.0
                    epoch_loss["MCE_error"] = 0.0
                else:
                    epoch_loss["MME_error"] = 0.0

                epoch_loss["obj_rot"] = 0.0
                epoch_loss["obj_trans"] = 0.0
                epoch_loss["DMA_adds"] = 0.0

                for itr, (inputs, targets, meta_info) in enumerate(
                    evaler.batch_generator
                ):
                    evaler.read_timer.toc()
                    evaler.gpu_timer.tic()
                    batch_ratio = itr / len(evaler.batch_generator)

                    batch_samples = meta_info["mano_root"].size(0)

                    # forward
                    model_out = evaler.model(
                        inputs, targets, meta_info, "eval", epoch, batch_ratio
                    )
                    out = {
                        k[:-4]: model_out[k] for k in model_out.keys() if "_out" in k
                    }
                    loss = {
                        k: model_out[k] for k in model_out.keys() if "_out" not in k
                    }
                    loss = {k: loss[k].mean() for k in loss}

                    if cfg.dataset == "dexycb":
                        loss["sdfhand_loss"] *= cfg.sdf_hand_weight
                        loss["sdfobj_loss"] *= cfg.sdf_obj_weight
                        joint_heatmap_out = out["joint_heatmap"]
                        loss["joint_heatmap"] *= cfg.hm_weight
                        loss["obj_seg"] *= cfg.obj_hm_weight
                        loss["hand_seg"] *= cfg.obj_hm_weight
                        loss["loss_joint_3d"] *= cfg.joint_weight
                        loss["loss_joint_cls"] *= cfg.cls_weight
                        loss["loss_all_joint_3d"] *= cfg.joint_weight

                        hand_joints = torch.cat(
                            [
                                torch.zeros_like(out["hand_joints"][:, :1]),
                                out["hand_joints"],
                            ],
                            dim=1,
                        )
                        mje, pamje = eval_hand_joint(
                            hand_joints, targets["joint_cam_no_trans"] / 1000
                        )
                        epoch_loss["out_mje"] += mje * batch_samples * 100
                        epoch_loss["out_pamje"] += pamje * batch_samples * 100

                        epoch_loss["sdfhand_loss"] += (
                            loss["sdfhand_loss"].detach().cpu().numpy() * batch_samples
                        )
                        epoch_loss["sdfobj_loss"] += (
                            loss["sdfobj_loss"].detach().cpu().numpy() * batch_samples
                        )
                        epoch_loss["loss_joint_3d"] += (
                            loss["loss_joint_3d"].detach().cpu().numpy() * batch_samples
                        )
                        epoch_loss["loss_joint_cls"] += (
                            loss["loss_joint_cls"].detach().cpu().numpy()
                            * batch_samples
                        )
                        epoch_loss["loss_all_joint_3d"] += (
                            loss["loss_all_joint_3d"].detach().cpu().numpy()
                            * batch_samples
                        )

                        if cfg.use_inverse_kinematics:
                            epoch_loss["shape_param_loss"] += (
                                loss["shape_param_loss"].detach().cpu().numpy()
                                * batch_samples
                            )
                            epoch_loss["shape_reg_loss"] += (
                                loss["shape_reg_loss"].detach().cpu().numpy()
                                * batch_samples
                            )
                        else:
                            epoch_loss["mano_mesh_loss"] += (
                                loss["mano_mesh_loss"].detach().cpu().numpy()
                                * batch_samples
                            )
                            epoch_loss["mano_joint_loss"] += (
                                loss["mano_joint_loss"].detach().cpu().numpy()
                                * batch_samples
                            )
                            epoch_loss["pose_param_loss"] += (
                                loss["pose_param_loss"].detach().cpu().numpy()
                                * batch_samples
                            )
                            epoch_loss["shape_param_loss"] += (
                                loss["shape_param_loss"].detach().cpu().numpy()
                                * batch_samples
                            )

                    loss["obj_rot"] *= cfg.obj_rot_weight
                    loss["obj_trans"] *= cfg.obj_trans_weight

                    epoch_loss["obj_rot"] += (
                        loss["obj_rot"].detach().cpu().numpy() * batch_samples
                    )
                    epoch_loss["obj_trans"] += (
                        loss["obj_trans"].detach().cpu().numpy() * batch_samples
                    )

                    if itr % 400 == 0:
                        if cfg.dataset == "dexycb":
                            writer.add_scalar(
                                "eval sdfhand_loss",
                                loss["sdfhand_loss"],
                                epoch * len(trainer.batch_generator) + itr,
                            )
                            writer.add_scalar(
                                "eval sdfobj_loss",
                                loss["sdfobj_loss"],
                                epoch * len(trainer.batch_generator) + itr,
                            )
                            hm_grid = torchvision.utils.make_grid(
                                joint_heatmap_out[:4].unsqueeze(1), normalize=True
                            )
                            writer.add_image(
                                "eval heatmap",
                                hm_grid,
                                epoch * len(evaler.batch_generator) + itr,
                            )
                            seg_grid_gt = torchvision.utils.make_grid(
                                out["obj_seg_gt"][:4].unsqueeze(1), normalize=True
                            )
                            seg_grid_pred = torchvision.utils.make_grid(
                                out["obj_seg_pred"][:4].unsqueeze(1), normalize=True
                            )
                            writer.add_image(
                                "eval obj seg gt patches",
                                seg_grid_gt,
                                epoch * len(evaler.batch_generator) + itr,
                            )
                            writer.add_image(
                                "eval obj seg pred patches",
                                seg_grid_pred,
                                epoch * len(evaler.batch_generator) + itr,
                            )
                            hseg_grid_gt = torchvision.utils.make_grid(
                                out["hand_seg_gt"][:4].unsqueeze(1), normalize=True
                            )
                            hseg_grid_pred = torchvision.utils.make_grid(
                                out["hand_seg_pred"][:4].unsqueeze(1), normalize=True
                            )
                            writer.add_image(
                                "eval hand seg gt patches",
                                hseg_grid_gt,
                                epoch * len(trainer.batch_generator) + itr,
                            )
                            writer.add_image(
                                "eval hand seg pred patches",
                                hseg_grid_pred,
                                epoch * len(trainer.batch_generator) + itr,
                            )
                            writer.add_scalar(
                                "eval joint_3d loss",
                                loss["loss_joint_3d"],
                                epoch * len(evaler.batch_generator) + itr,
                            )
                            writer.add_scalar(
                                "eval joints cls loss",
                                loss["loss_joint_cls"],
                                epoch * len(evaler.batch_generator) + itr,
                            )
                            writer.add_scalar(
                                "eval sum joints 3d loss",
                                loss["loss_all_joint_3d"],
                                epoch * len(evaler.batch_generator) + itr,
                            )

                            if cfg.use_inverse_kinematics:
                                writer.add_scalar(
                                    "eval shape_param_loss loss",
                                    loss["shape_param_loss"],
                                    epoch * len(evaler.batch_generator) + itr,
                                )
                                writer.add_scalar(
                                    "eval shape_reg_loss loss",
                                    loss["shape_reg_loss"],
                                    epoch * len(evaler.batch_generator) + itr,
                                )
                            else:
                                writer.add_scalar(
                                    "eval mano_mesh_loss loss",
                                    loss["mano_mesh_loss"],
                                    epoch * len(evaler.batch_generator) + itr,
                                )
                                writer.add_scalar(
                                    "eval mano_joint_loss loss",
                                    loss["mano_joint_loss"],
                                    epoch * len(evaler.batch_generator) + itr,
                                )
                                writer.add_scalar(
                                    "eval pose_param_loss loss",
                                    loss["pose_param_loss"],
                                    epoch * len(evaler.batch_generator) + itr,
                                )
                                writer.add_scalar(
                                    "eval shape_param_loss loss",
                                    loss["shape_param_loss"],
                                    epoch * len(evaler.batch_generator) + itr,
                                )

                            writer.add_scalar(
                                "eval obj_seg loss",
                                loss["obj_seg"],
                                epoch * len(evaler.batch_generator) + itr,
                            )
                            writer.add_scalar(
                                "eval hand_seg loss",
                                loss["hand_seg"],
                                epoch * len(evaler.batch_generator) + itr,
                            )
                            writer.add_scalar(
                                "eval heatmap loss",
                                loss["joint_heatmap"],
                                epoch * len(evaler.batch_generator) + itr,
                            )

                        # dump the outputs
                        img_grid = torchvision.utils.make_grid(inputs["img"][:4])
                        writer.add_image(
                            "eval input patches",
                            img_grid,
                            epoch * len(evaler.batch_generator) + itr,
                        )
                        writer.add_scalar(
                            "eval obj_rot loss",
                            loss["obj_rot"],
                            epoch * len(evaler.batch_generator) + itr,
                        )
                        writer.add_scalar(
                            "eval obj_trans loss",
                            loss["obj_trans"],
                            epoch * len(evaler.batch_generator) + itr,
                        )
                        writer.add_scalar(
                            "eval total loss",
                            sum(loss[k] for k in loss),
                            epoch * len(evaler.batch_generator) + itr,
                        )

                    evaler.gpu_timer.toc()
                    screen = [
                        "eval_Epoch %d/%d itr %d/%d:"
                        % (epoch, cfg.end_epoch, itr, evaler.itr_per_epoch),
                        "speed: %.2f(%.2fs r%.2f)s/itr"
                        % (
                            evaler.tot_timer.average_time,
                            evaler.gpu_timer.average_time,
                            evaler.read_timer.average_time,
                        ),
                        "%.2fh/eval_epoch"
                        % (
                            evaler.tot_timer.average_time
                            / 3600.0
                            * evaler.itr_per_epoch
                        ),
                    ]
                    screen += [
                        "%s: %.4f" % ("eval_loss_" + k, v.detach())
                        for k, v in loss.items()
                    ]
                    evaler.logger.info(" ".join(screen))

                    evaler.tot_timer.toc()
                    evaler.tot_timer.tic()
                    evaler.read_timer.tic()

                if cfg.dataset == "dexycb":
                    writer.add_scalar(
                        "epoch sdfhand_loss",
                        epoch_loss["sdfhand_loss"] / evaler.total_sample,
                        epoch,
                    )
                    writer.add_scalar(
                        "epoch sdfobj_loss",
                        epoch_loss["sdfobj_loss"] / evaler.total_sample,
                        epoch,
                    )
                    writer.add_scalar(
                        "epoch joint 3d loss",
                        epoch_loss["loss_joint_3d"] / evaler.total_sample,
                        epoch,
                    )
                    writer.add_scalar(
                        "epoch joint cls loss",
                        epoch_loss["loss_joint_cls"] / evaler.total_sample,
                        epoch,
                    )
                    writer.add_scalar(
                        "epoch sum joint 3d loss",
                        epoch_loss["loss_all_joint_3d"] / evaler.total_sample,
                        epoch,
                    )

                    if cfg.use_inverse_kinematics:
                        writer.add_scalar(
                            "epoch shape_param_loss loss",
                            epoch_loss["shape_param_loss"] / evaler.total_sample,
                            epoch,
                        )
                        writer.add_scalar(
                            "epoch shape_reg_loss loss",
                            epoch_loss["shape_reg_loss"] / evaler.total_sample,
                            epoch,
                        )
                    else:
                        writer.add_scalar(
                            "epoch mano_mesh_loss loss",
                            epoch_loss["mano_mesh_loss"] / evaler.total_sample,
                            epoch,
                        )
                        writer.add_scalar(
                            "epoch mano_joint_loss loss",
                            epoch_loss["mano_joint_loss"] / evaler.total_sample,
                            epoch,
                        )
                        writer.add_scalar(
                            "epoch pose_param_loss loss",
                            epoch_loss["pose_param_loss"] / evaler.total_sample,
                            epoch,
                        )
                        writer.add_scalar(
                            "epoch shape_param_loss loss",
                            epoch_loss["shape_param_loss"] / evaler.total_sample,
                            epoch,
                        )

                    writer.add_scalar(
                        "epoch hand mje",
                        epoch_loss["out_mje"] / evaler.total_sample,
                        epoch,
                    )
                    writer.add_scalar(
                        "epoch hand pamje",
                        epoch_loss["out_pamje"] / evaler.total_sample,
                        epoch,
                    )

                epoch_loss["total_loss"] += (
                    sum(loss[k] for k in loss).detach().cpu().numpy() * batch_samples
                )
                writer.add_scalar(
                    "epoch total loss",
                    epoch_loss["total_loss"] / evaler.total_sample,
                    epoch,
                )
                writer.add_scalar(
                    "epoch obj_rot loss",
                    epoch_loss["obj_rot"] / evaler.total_sample,
                    epoch,
                )
                writer.add_scalar(
                    "epoch obj_trans loss",
                    epoch_loss["obj_trans"] / evaler.total_sample,
                    epoch,
                )

                trainer.save_model(
                    {
                        "epoch": epoch,
                        "network": trainer.model.state_dict(),
                        "optimizer": trainer.optimizer.state_dict(),
                        "lr_scheduler": trainer.lr_scheduler.state_dict(),
                    },
                    epoch,
                    iter=itr,
                )


if __name__ == "__main__":
    main()
