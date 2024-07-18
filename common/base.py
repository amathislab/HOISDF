# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from Keypoint Transformer
# ------------------------------------------------------------------------------

import abc
import glob
import math
import os
import os.path as osp

import torch.optim
from torch.utils.data import DataLoader

from main.config import cfg

if cfg.dataset == "ho3d":
    from data.ho3d import Dataset
elif cfg.dataset == "dexycb":
    from data.dexycb import Dataset

from torch.nn.parallel.data_parallel import DataParallel

from common.logger import colorlogger
from common.timer import Timer
from main.model import get_model


def adjust_learning_rate(lr_schedules, optimizer):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = max(lr_schedules._last_lr[i], 0.00001)


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name="logs.txt"):

        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):

    def __init__(self):
        super(Trainer, self).__init__(log_name="train_logs.txt")

    def get_optimizer(self, model):
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        param_dicts = [{"params": [p for n, p in model.named_parameters()]}]

        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, cfg.lr_drop, gamma=cfg.lr_decay_gamma
        )

        return optimizer, lr_scheduler

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")
        trainset_loader = Dataset("train")
        batch_generator = DataLoader(
            dataset=trainset_loader,
            batch_size=cfg.num_gpus * cfg.train_batch_size,
            shuffle=True,
            num_workers=cfg.num_thread,
            pin_memory=True,
            drop_last=True,
        )

        self.joint_num = trainset_loader.joint_num
        self.itr_per_epoch = math.ceil(
            trainset_loader.__len__() / cfg.num_gpus / cfg.train_batch_size
        )
        self.batch_generator = batch_generator

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model("train")
        print("Number of trainable parameters = %d" % (self.count_parameters(model)))
        model = model.cuda()
        model = DataParallel(model)

        optimizer, lr_scheduler = self.get_optimizer(model)
        model.train()

        self.start_epoch = 0
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def save_model(self, state, epoch, iter=0):
        file_path = osp.join(
            cfg.model_dir, "snapshot_%s_%s.pth.tar" % (str(epoch), str(iter))
        )
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self):
        model_file_list = glob.glob(osp.join(cfg.model_dir, "*.pth.tar"))
        model_file_list = [
            file_name[file_name.find("snapshot_") :] for file_name in model_file_list
        ]
        cur_epoch = max(
            [
                int(file_name.split("_")[1].split(".")[0])
                for file_name in model_file_list
                if "snapshot" in file_name
            ]
        )
        cur_iter = max(
            [
                int(file_name.split("_")[2].split(".")[0])
                for file_name in model_file_list
                if "snapshot_%d" % cur_epoch in file_name
            ]
        )
        model_path = osp.join(
            cfg.model_dir,
            "snapshot_" + str(cur_epoch) + "_" + str(cur_iter) + ".pth.tar",
        )

        self.logger.info("Load checkpoint from {}".format(model_path))
        ckpt = torch.load(model_path)
        self.start_epoch = ckpt["epoch"] + 1

        self.model.load_state_dict(ckpt["network"], strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])


class Tester(Base):

    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        super(Tester, self).__init__(log_name="test_logs.txt")

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating test dataset...")
        testset_loader = Dataset("evaluation")
        batch_generator = DataLoader(
            dataset=testset_loader,
            batch_size=cfg.num_gpus * cfg.test_batch_size,
            shuffle=False,
            num_workers=cfg.num_thread,
            pin_memory=True,
        )

        self.joint_num = testset_loader.joint_num
        self.jointsMapSimpleToMano = testset_loader.jointsMapSimpleToMano
        self.mesh_dict = testset_loader.obj_mesh
        self.diameter_dict = testset_loader.obj_diameters
        self.batch_generator = batch_generator
        self.testset = testset_loader
        self.total_sample = len(testset_loader)

    def _make_model(self):
        model_path = self.ckpt_path
        assert os.path.exists(model_path), "Cannot find model at " + model_path
        self.logger.info("Load checkpoint from {}".format(model_path))

        # prepare network
        self.logger.info("Creating graph...")
        model = get_model("test")
        model = model.cuda()
        model = DataParallel(model)
        ckpt = torch.load(model_path)

        model.load_state_dict(ckpt["network"], strict=True)
        model.eval()
        self.model = model


class Evaler(Base):

    def __init__(self):
        super(Evaler, self).__init__(log_name="eval_logs.txt")

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating eval dataset...")
        evalset_loader = Dataset("evaluation")
        batch_generator = DataLoader(
            dataset=evalset_loader,
            batch_size=cfg.num_gpus * cfg.eval_batch_size,
            shuffle=False,
            num_workers=cfg.num_thread,
            pin_memory=True,
        )

        self.joint_num = evalset_loader.joint_num
        self.jointsMapSimpleToMano = evalset_loader.jointsMapSimpleToMano
        self.mesh_dict = evalset_loader.obj_mesh
        self.diameter_dict = evalset_loader.obj_diameters
        self.batch_generator = batch_generator
        self.itr_per_epoch = math.ceil(
            evalset_loader.__len__() / cfg.num_gpus / cfg.eval_batch_size
        )
        self.total_sample = len(evalset_loader)
        # self.evalset = evalset_loader

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model("eval")
        model = model.cuda()
        model = DataParallel(model)

        model.eval()

        self.model = model

    def _load_state(self, model_train):
        train_state = model_train.state_dict()
        test_state = self.model.state_dict()
        pretrained_dict = {k: v for k, v in train_state.items() if k in test_state}
        test_state.update(pretrained_dict)

        self.model.load_state_dict(test_state)
