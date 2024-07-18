# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from Keypoint Transformer, HFL-Net, DenseMutualAttention, and AlignSDF
# ------------------------------------------------------------------------------

import os
import os.path as osp
import random
import sys

import numpy as np
import torch


def fix_seeds(random_seed):

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


class Config:
    setting = "ho3d"  # ho3d, ho3d_render, dexycb, dexycb_full
    # ~~~~~~~~~~~~~~~~~~~~~~Dataset~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if "ho3d" in setting:
        dataset = "ho3d"
    else:
        dataset = "dexycb"

    # please set the path here according to the readme#
    object_models_dir = None
    output_dir = "outputs"
    annotation_dir = None
    simple_object_models_dir = None
    if dataset == "ho3d":
        ho3d_data_dir = None
        fast_data_dir = None
    elif dataset == "dexycb":
        dexycb_data_dir = None
        fast_data_dir = None
        image_fast_path = None


    train_batch_size = 22 
    test_batch_size = 22 
    eval_batch_size = 22 

    num_samp_hand = 600
    num_samp_obj = 200
    points_filter_dist = 0.05
    test_seg_thresh = 0.1
    random_ratio = [0.3, 0.7]
    random_move_dist = [0.03, 0.05, 0.07]
    if dataset == "ho3d":
        add_render = True if "render" in setting else False
        sdf_config_path = "./sdf_nets/ho3d_first.json"
        obj_depth_mean_value = 0.5244322
        hand_sdf_scale = 3.1
        obj_sdf_scale = 3.1
        hand_cls_dist = 0.04
        obj_cls_dist = 0.05
    if dataset == "dexycb":
        small_dexycb = False if "full" in setting else True
        sdf_config_path = "./sdf_nets/dexycb_first.json"
        obj_depth_mean_value = None
        hand_sdf_scale = 3.1
        obj_sdf_scale = 3.1
        hand_cls_dist = 0.04
        obj_cls_dist = 0.05

    # SDF config
    bins_n = 64
    num_class = 6
    PointFeatSize = 33
    ClassifierBranch = False
    ClampingDistance = 0.15

    # ~~~~~~~~~~~~~~~~~~~~~~Training Setup~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ## model
    use_big_decoder = True if setting == "ho3d" else False
    use_inverse_kinematics = True if setting == "ho3d_render" else False
    resnet_type = 50  # 18, 34, 50, 101, 152
    mutliscale_layers = ["stride2", "stride4", "stride8", "stride16", "stride32"]

    def calc_mutliscale_dim(self, use_big_decoder_l, resnet_type_l):
        if use_big_decoder_l:
            self.mutliscale_dim = 128 + 256 + 512 + 1024 + 2048
        else:
            if resnet_type_l >= 50:
                self.mutliscale_dim = 32 + 64 + 128 + 256 + 512
            else:
                self.mutliscale_dim = 32 + 64 + 128 + 256 + 512

    ## input, output
    input_img_shape = (256, 256)
    output_hm_shape = (128, 128, 128)  # (depth, height, width)
    sigma = 2.5 / 2

    # Some more hyper-parameters
    hidden_dim = 256
    dropout = 0.1
    nheads = 4
    dim_feedforward = 1024
    enc_layers = 6
    dec_layers = 4
    pre_norm = False

    # Queries config
    mano_num_queries = 15 + 1 + 1
    mano_shape_indx = 16

    ## optimization config
    end_epoch = 70
    point_sampling_epoch = 40
    lr = 1e-4
    lr_decay_gamma = 0.7
    lr_drop = 9

    ## weights
    sdf_hand_weight = 50
    sdf_obj_weight = 25
    sdf_cls_weight = 10
    hm_weight = 100 / 100000
    joint_weight = 1 / 10
    cls_weight = 1 / 1

    obj_hm_weight = 1
    obj_rot_weight = 0.7
    obj_trans_weight = 100 / 1

    lambda_verts3d = 1e4
    lambda_joints3d = 1e4
    lambda_manopose = 10
    lambda_manoshape = 0.1
    mano_lambda_regulshape = 0.000001

    # test config
    eval_mesh = True if setting == "dexycb_full" else False

    ## directory setup
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.dirname(cur_dir)
    data_dir = osp.join(root_dir, "data")
    base_log_dir = osp.join(output_dir, "log")
    result_dir = osp.join(output_dir, "result")

    def setup_out_dirs(self, model_dir_name):
        self.log_dir = osp.join(self.output_dir, "log", model_dir_name)
        self.model_dir = osp.join(self.output_dir, "model_dump", model_dir_name)
        self.tensorboard_dir = osp.join(self.output_dir, "tensorboard", model_dir_name)

    ## others
    num_thread = 15
    gpu_ids = "0"
    num_gpus = 1
    continue_train = True

    def set_args(self, gpu_ids, model_dir_name, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(","))
        self.continue_train = continue_train
        self.model_dir_name = model_dir_name
        self.setup_out_dirs(model_dir_name)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print(">>> Using GPU: {}".format(self.gpu_ids))

    def create_run_dirs(self):
        make_folder(cfg.log_dir)
        make_folder(cfg.model_dir)
        make_folder(cfg.tensorboard_dir)

    def create_log_dir(self):
        make_folder(cfg.log_dir)


cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, "common"))
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.dataset))
make_folder(cfg.base_log_dir)
