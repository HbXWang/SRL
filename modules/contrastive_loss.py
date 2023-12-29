import math
import torch
import torch.nn as nn


class Local_structure_clustering(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(Local_structure_clustering, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * self.batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_means, z_sigma, c_i, c_j, prior_p, ratio):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_means), dim=0)

        sim = torch.matmul(z, z.t()) / self.temperature
        sim_pos = torch.cat((torch.diag(sim, self.batch_size), torch.diag(sim, -self.batch_size)), dim=0)

        pos = torch.exp(sim_pos)
        neg = torch.exp(sim[self.mask].reshape(N, -1))

        z_sigma = z_sigma * ratio / self.temperature
        sim_sigma = torch.mul(z_i, 0.5 * torch.bmm(z_sigma, z_i.unsqueeze(dim=-1)).squeeze(dim=-1)).sum(-1)
        sim_sigma = torch.cat((sim_sigma, sim_sigma), dim=0) / self.temperature
        l_pos = torch.exp(sim_pos + sim_sigma)

        with torch.no_grad():
            label_z = torch.argmax(torch.cat((c_i, c_j), dim=0).detach(), dim=1)
            prior_z = torch.tensor([prior_p[i] for i in label_z]).to(self.device)

        # distribution
        Ng = (- prior_z * (N - 2) * l_pos + neg.sum(dim=-1)) / (1 - prior_z)
        # constrain
        Ng = torch.clamp(Ng, min=(N - 2) * math.e ** (-1 / self.temperature))

        loss = (- torch.log(pos / (l_pos + Ng))).mean()

        return loss


class Global_structure_clustering(nn.Module):
    def __init__(self, class_num, temperature):
        super(Global_structure_clustering, self).__init__()
        self.class_num = class_num
        self.temperature = temperature

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * self.class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(self.class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss
