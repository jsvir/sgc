import math

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn


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
                self.encoder.append(nn.Linear(cfg.ae.encoder[i], cfg.ae.latent_dim))
            else:
                self.encoder.append(DenseBlock(cfg.ae.encoder[i], cfg.ae.encoder[i + 1]))
        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = [DenseBlock(cfg.ae.latent_dim, cfg.ae.decoder[0])]
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
        self.in_features = cfg.ckm.centroid_dim
        self.out_features = cfg.n_clusters
        self._centroids = nn.Parameter(torch.randn(self.out_features, self.in_features), requires_grad=True)
        self.sigma = nn.Parameter(torch.ones(self.out_features), requires_grad=True)
        self._initialized = False

    def forward(self, x, tau=None):
        " the ouput is log probability of instance x_j to be assigned to cluster c_i"
        x = x.unsqueeze(1).repeat(1, self.out_features, 1)
        c = self._centroids.unsqueeze(0).repeat(x.size(0), 1, 1)
        distances = - (x - c).pow(2).sum(-1) * self.sigma.pow(-2).unsqueeze(0)
        if tau is not None: distances /= tau
        return torch.log_softmax(distances, dim=-1)

    @property
    def centroids(self):
        return self._centroids

    @property
    def initialized(self):
        return self._initialized

    def init_centroids(self, embeddings):
        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=100, random_state=0).fit(embeddings.detach().cpu().numpy())
        self._centroids.__init__(torch.tensor(kmeans.cluster_centers_, device=embeddings.device), requires_grad=True)
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

    def num_open_gates(self, x, reduction='mean'):
        with torch.no_grad():
            gates = self.hard_sigmoid(self.net(x) + self.sigma)
        if reduction == 'mean':
            return gates.mean()
        elif reduction == 'sum':
            return gates.sum()


class SelfAttention(torch.nn.Module):
    def __init__(self, cfg):
        super(SelfAttention, self).__init__()
        self.emb_dim = 512
        self.heads = 2
        self.multihead_attn = nn.MultiheadAttention(self.emb_dim, self.heads, batch_first=True)
        self.norm_layer = nn.LayerNorm(self.emb_dim)
        self.embed = nn.Linear(1, self.emb_dim)
        self.output = nn.Linear(self.emb_dim, 1)

    def forward(self, x):
        v = self.embed(x.reshape(-1, 1)).reshape(-1, x.size(1), self.emb_dim)
        v = self.norm_layer(v)
        attn_output, attn_output_weights = self.multihead_attn(query=v, key=v, value=v)
        return attn_output.squeeze(-1)


class STGLayerExt(torch.nn.Module):
    " Feature selection layer as in the paper: https://arxiv.org/abs/2106.06468"

    def __init__(self, cfg):
        super(STGLayerExt, self).__init__()
        self.cfg = cfg
        self._sqrt_2 = math.sqrt(2)
        self.sigma = cfg.stg.sigma
        self.reg_lamba = cfg.stg.reg_lamba
        activation = nn.ReLU() if cfg.activation == 'relu' else nn.Tanh()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.input_dim, cfg.stg.hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.stg.hidden_dim, cfg.input_dim),
            nn.Tanh()
        )
        self.embedding_layer = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.mcrr.hidden_dim),
            nn.BatchNorm1d(cfg.mcrr.hidden_dim),
            activation,
            nn.Linear(cfg.mcrr.hidden_dim, cfg.mcrr.hidden_dim),
        )

        self.clustering_layer = nn.Sequential(
            nn.Linear(cfg.mcrr.hidden_dim, cfg.mcrr.hidden_dim),
            nn.BatchNorm1d(cfg.mcrr.hidden_dim),
            activation,
            nn.Linear(cfg.mcrr.hidden_dim, cfg.n_clusters),
        )
        #TODO: checking if it's useful
        if cfg.init_weights:
            self.net.apply(self.init_weights)
            self.embedding_layer.apply(self.init_weights)
            self.clustering_layer.apply(self.init_weights)

        #TODO: define an attention layer and for each cluster choose the emb that has maximal att score for the cluster
        # the attention matrix is calculated for each cluster separately

        # self.att = SelfAttention()

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.1)
            m.bias.data.fill_(0.0)

    def forward(self, x, pretrain=False, gated_x=False, epoch=None):
        if pretrain or (not self.cfg.stg.enabled):
            c = self.embedding_layer(x)
            logits = self.clustering_layer(c)
            return c, logits
        else:
            # if epoch is not None:
            #     std = 0.1 + 0.9 * min(epoch/10, 1)
            # else: std = 1
            noise = torch.normal(mean=0, std=1., size=x.size(), device=x.device)
            h = self.net(x)
            z = h + self.sigma * noise * self.training
            stochastic_gate = self.hard_sigmoid(z)
            new_x = x * stochastic_gate
            c = self.embedding_layer(new_x)
            logits = self.clustering_layer(c)
            if gated_x: return h, c, logits, new_x
            else: return h, c, logits

    def hard_sigmoid(self, x):
        return torch.clamp(x + self.sigma, 0.0, 1.0)

    def regularization(self, h):
        return self.reg_lamba * torch.mean(0.5 - 0.5 * torch.erf((-1 / 2 - h) / (self.sigma * self._sqrt_2)))

    def get_gates(self, x):
        with torch.no_grad():
            gates = self.hard_sigmoid(self.net(x))
            gates[x==x.min()] = 0
        return gates

    def num_open_gates(self, x, reduction='mean'):
        if reduction == 'mean':
            return self.get_gates(x).mean()
        elif reduction == 'sum':
            return self.get_gates(x).sum()


class STGLayerAE(torch.nn.Module):
    " Feature selection layer as in the paper: https://arxiv.org/abs/2106.06468"

    def __init__(self, cfg):
        super(STGLayerAE, self).__init__()
        self._sqrt_2 = math.sqrt(2)
        self.sigma = cfg.stg.sigma
        self.reg_lamba = cfg.stg.reg_lamba
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cfg.input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )
        self.out_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, cfg.input_dim),
            nn.Tanh())

        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, cfg.input_dim),
        )

    def forward(self, x, train=False):
        noise = torch.randn_like(x, device=x.device)
        h1 = self.net(x)
        h = self.out_layer(h1)
        recon_x = self.decoder(h1)
        z = h + self.sigma * noise * train
        stochastic_gate = self.hard_sigmoid(z)
        gated_x = x * stochastic_gate
        return gated_x, h, recon_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularization(self, h):
        return self.reg_lamba * torch.mean(0.5 - 0.5 * torch.erf((-1 / 2 - h) / (self.sigma * self._sqrt_2)))

    def num_open_gates(self, x, reduction='mean'):
        gates = self.hard_sigmoid(self.net(x) + self.sigma)
        if reduction == 'mean':
            return gates.mean()
        elif reduction == 'sum':
            return gates.sum()


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
        for i in range(len(cfg.ae.encoder) - 1):
            self.backbone.append(DenseBlock(cfg.ae.encoder[i], cfg.ae.encoder[i + 1]))
        self.backbone = nn.Sequential(*self.backbone)
        self.cluster = nn.Linear(cfg.ae.encoder[-1], cfg.n_clusters)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.cluster(z)
        z = F.normalize(z, p=2)
        return z, logits


class ClusterLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            DenseBlock(cfg.cluster.input_dim, cfg.cluster.layers[0]),
            nn.Linear(cfg.cluster.layers[0], cfg.n_clusters)
        )

    def forward(self, z):
        logits = self.layers(z)
        z = F.normalize(z, p=2)
        return z, logits


class SelectLayer(nn.Module):
    """ from github.com/jsvir/lscae"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_features = self.cfg.input_dim
        self.output_features = self.cfg.selector.k_selected
        self.num_epochs = self.cfg.trainer.max_epochs
        self.start_temp = self.cfg.selector.start_temp
        self.min_temp = self.cfg.selector.min_temp
        self.logits = torch.nn.Parameter(torch.zeros(self.input_features, self.output_features), requires_grad=True)

    def current_temp(self, epoch, sched_type='exponential'):
        schedules = {
            'exponential': max(self.min_temp, self.start_temp * ((self.min_temp / self.start_temp) ** (epoch / self.num_epochs))),
            'linear': max(self.min_temp, self.start_temp - (self.start_temp - self.min_temp) * (epoch / self.num_epochs)),
            'cosine': self.min_temp + 0.5 * (self.start_temp - self.min_temp) * (1. + math.cos(epoch * math.pi / self.num_epochs))
        }
        return schedules[sched_type]

    def forward(self, x, epoch=None):
        from torch.distributions.uniform import Uniform
        uniform_pdfs = Uniform(low=1e-6, high=1.).sample(self.logits.size()).to(x.device)
        gumbel = -torch.log(-torch.log(uniform_pdfs))

        if self.training:
            temp = self.current_temp(epoch)
            noisy_logits = (self.logits + gumbel) / temp
            weights = F.softmax(noisy_logits / temp, dim=0)
            x = x @ weights
        else:
            weights = F.one_hot(torch.argmax(self.logits, dim=0), self.input_features).float()
            x = x @ weights.T
        return x, weights

    def get_weights(self, epoch):
        temp = self.current_temp(epoch)
        return F.softmax(self.logits / temp, dim=0)

    def get_selected_feats(self):
        feats = torch.argmax(self.logits, dim=0)
        return feats


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_features = cfg.selector.k_selected
        self.output_features = cfg.input_dim
        self.hidden_dim = cfg.selector.hidden_dim

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(self.input_features, self.hidden_dim, bias=False),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.LeakyReLU(.2, True),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.LeakyReLU(.2, True),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.output_features, bias=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
