import math

import torch
from sklearn.cluster import KMeans
from torch import nn
import torch.nn.functional as F


class DenseBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            torch.nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class AutoEncoder(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = [DenseBlock(cfg.input_dim, cfg.ae.encoder[0])]
        for i in range(len(cfg.ae.encoder)):  # 1024, 512, 256, 10
            if i == len(cfg.ae.encoder) - 1:
                self.encoder.append(nn.Linear(cfg.ae.encoder[i], cfg.n_clusters))
            else:
                self.encoder.append(DenseBlock(cfg.ae.encoder[i], cfg.ae.encoder[i + 1]))
        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = [DenseBlock(cfg.n_clusters, cfg.ae.decoder[0])]
        for i in range(len(cfg.ae.decoder)):  # 10, 256, 512, 1024
            if i == len(cfg.ae.decoder) - 1:
                self.decoder.append(nn.Linear(cfg.ae.decoder[i], cfg.input_dim))
            else:
                self.decoder.append(DenseBlock(cfg.ae.decoder[i], cfg.ae.decoder[i + 1]))
        self.decoder = nn.Sequential(*self.decoder)
        print(self)

    def forward(self, x):
        z = self.encoder(x)
        o = self.decoder(z)
        return z, o


class RBFLayer(nn.Module):
    "Clustering layer using Normalised Radial Basis Functions"

    def __init__(self, cfg):
        super().__init__()
        self.n_clusters = cfg.n_clusters
        self.in_features = cfg.n_clusters
        self.out_features = cfg.n_clusters
        self._centroids = nn.Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad=True)
        self.sigmas = nn.Parameter(torch.Tensor(self.out_features), requires_grad=True)
        self.reset_parameters()
        self._initialized = False

    def reset_parameters(self):
        nn.init.normal_(self._centroids, 0, 1)
        nn.init.constant_(self.sigmas, 1)

    def forward(self, x):
        " the ouput is log probability of instance x_j to be assigned to cluster c_i"
        size = (x.size(0), self.out_features, self.in_features)
        x = x.unsqueeze(1).expand(size)
        c = self._centroids.unsqueeze(0).expand(size)
        distances = - (x - c).pow(2).sum(-1) / self.sigmas.unsqueeze(0).pow(2)
        return torch.log_softmax(distances, dim=-1)

    @property
    def centroids(self):
        return self._centroids

    @property
    def initialized(self):
        return self._initialized

    def init_centroids(self, embeddings):
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=100, random_state=0).fit(embeddings.detach().cpu().numpy())
        self._centroids.__init__(torch.tensor(kmeans.cluster_centers_.T, device=embeddings.device), requires_grad=True)
        self._initialized = True


class STGLayer(torch.nn.Module):
    " Feature selection layer as in the paper: https://arxiv.org/abs/2106.06468"

    def __init__(self, cfg):
        super(STGLayer, self).__init__()
        self._sqrt_2 = math.sqrt(2)
        self.sigma = cfg.stg.sigma
        self.reg_lamba = cfg.stg.reg_lamba
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.input_dim, cfg.stg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.stg.hidden_dim, cfg.input_dim),
            nn.Tanh())

    def forward(self, prev_x, train=False):
        noise = torch.randn_like(prev_x, device=prev_x.device)
        h = self.net(prev_x)
        z = h + self.sigma * noise * train
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x, h

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularization(self, h):
        return self.reg_lamba * torch.mean(0.5 - 0.5 * torch.erf((-1 / 2 - h) / (self.sigma * self._sqrt_2)))


class Gumble_Softmax(nn.Module):
    def __init__(self, tau, straight_through=False):
        super().__init__()
        self.tau = tau
        self.straight_through = straight_through

    def forward(self, logps):
        gumble = torch.rand_like(logps).log().mul(-1).log().mul(-1)
        logits = logps + gumble
        out = (logits / self.tau).softmax(dim=1)
        if not self.straight_through:
            return out
        else:
            out_binary = (logits * 1e8).softmax(dim=1).detach()
            out_diff = (out_binary - out).detach()
            return out_diff + out


class SubspaceClusterNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # the same encoder as in ae
        self.backbone = [DenseBlock(cfg.input_dim, cfg.ae.encoder[0])]
        for i in range(len(cfg.ae.encoder)-1):
            self.backbone.append(DenseBlock(cfg.ae.encoder[i], cfg.ae.encoder[i + 1]))
        self.backbone = nn.Sequential(*self.backbone)
        self.cluster = nn.Linear(cfg.ae.encoder[-1], cfg.n_clusters)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.cluster(z)
        z = F.normalize(z, p=2)
        return z, logits

