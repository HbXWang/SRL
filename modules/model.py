import torch
import torch.nn as nn
from torch.nn.functional import normalize


class PrototypeLayer(nn.Module):
    def __init__(self, feature_dim=10, cluster_num=10):
        super(PrototypeLayer, self).__init__()
        self.n_clusters = cluster_num
        self.feature_dim = feature_dim
        self.prototype_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.n_clusters, bias=False),
            nn.Softmax(dim=1)
        )
        self.prototypes = self.prototype_projector[0].weight.data

    def forward(self, input_x):
        c = self.prototype_projector(input_x)
        return c


class SRL_Network(nn.Module):
    def __init__(self, resnet, feature_dim, class_num, k_crop):
        super(SRL_Network, self).__init__()
        self.k_crop = k_crop
        self.resnet = resnet
        self.class_num = class_num
        self.feature_dim = feature_dim

        self.instance_projector = nn.Sequential(
            nn.Linear(self.resnet.rep_dim, self.resnet.rep_dim),
            nn.BatchNorm1d(self.resnet.rep_dim),
            nn.ReLU(),
            nn.Linear(self.resnet.rep_dim, self.feature_dim),
        )
        self.cluster_projector = PrototypeLayer(self.feature_dim, self.class_num)

    def forward(self, aug_imgs):

        z_list = []
        for aug_imgs_ in aug_imgs:
            h = self.resnet(aug_imgs_)
            z_ = normalize(self.instance_projector(h), dim=1)
            z_list.append(z_)

        z_i = z_list[0]
        c_i = self.cluster_projector(z_i)

        z_means = sum(z_list[1:])/len(z_list[1:])
        c_j = self.cluster_projector(z_means)

        with torch.no_grad():
            # caculate sigma
            z_j = torch.stack(z_list[1:])
            # Sample * Batch_size * Feature_dim
            value_minus_mean = z_j - z_means.unsqueeze(0).detach()
            # batch_size * sample * feature_dim
            value_minus_mean_part_1 = value_minus_mean.permute(1, 0, 2)
            # batch_size * feature_dim * sample
            value_minus_mean_part_2 = value_minus_mean.permute(1, 2, 0)
            # batch_size * feature_dim * feature_dim
            z_sigma = torch.bmm(value_minus_mean_part_2, value_minus_mean_part_1) / self.k_crop

        return z_i, z_means, z_sigma, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        z = normalize(self.instance_projector(h), dim=1)
        c = self.cluster_projector(z)
        return z, c
