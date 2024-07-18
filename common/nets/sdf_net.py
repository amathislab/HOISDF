# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Modified from AlignSDF
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDFDecoder(nn.Module):
    def __init__(
        self,
        latent_size,
        point_feat_size,
        dims=[512, 512, 512, 512],
        num_class=6,
        dropout=[0, 1, 2, 3],
        dropout_prob=0.2,
        norm_layers=[0, 1, 2, 3],
        latent_in=[2],
        weight_norm=True,
        xyz_in_all=False,
        use_tanh=False,
        latent_dropout=False,
        use_classifier=False,
    ):
        super(SDFDecoder, self).__init__()

        def make_sequence():
            return []

        self.point_feat_size = point_feat_size

        dims_hand = [latent_size + point_feat_size] + dims + [1]

        self.latent_size = latent_size
        self.num_hand_layers = len(dims_hand)
        self.num_class = num_class
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm
        self.use_classifier = use_classifier

        for layer in range(0, self.num_hand_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims_hand[layer + 1] - dims_hand[0]
            else:
                out_dim = dims_hand[layer + 1]

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "linh" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims_hand[layer], out_dim)),
                )
            else:
                setattr(self, "linh" + str(layer), nn.Linear(dims_hand[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bnh" + str(layer), nn.LayerNorm(out_dim))

            # classifier
            if self.use_classifier and layer == self.num_hand_layers - 2:
                self.classifier_head = nn.Linear(dims_hand[layer], self.num_class)

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xh = input
        input_hand = xh

        for layer in range(0, self.num_hand_layers - 1):
            # classify
            if self.use_classifier and layer == self.num_hand_layers - 2:
                predicted_class = self.classifier_head(xh)

            lin = getattr(self, "linh" + str(layer))
            if layer in self.latent_in:
                xh = torch.cat([xh, input_hand], 1)
            xh = lin(xh)
            # last layer Tanh
            if layer == self.num_hand_layers - 2 and self.use_tanh:
                xh = self.tanh(xh)
            if layer < self.num_hand_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bnh" + str(layer))
                    xh = bn(xh)
                xh = self.relu(xh)
                if self.dropout is not None and layer in self.dropout:
                    xh = F.dropout(xh, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            xh = self.th(xh)

        # hand, object, class label
        if self.use_classifier:
            return xh[:, 0].unsqueeze(1), predicted_class
        else:
            return xh[:, 0].unsqueeze(1), torch.Tensor([0]).cuda()
