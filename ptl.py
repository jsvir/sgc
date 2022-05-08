import math
import os
import platform
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST

from func import cluster_match, cluster_merge_match, normalized_mutual_info_score, adjusted_rand_score
from loss import MaximalCodingRateReduction, TotalCodingRate
from model import AutoEncoder, RBFLayer, Gumble_Softmax, STGLayer, SubspaceClusterNetwork, ClusterLayer, STGLayerAE, DimReduction, SelectLayer, Decoder

__all__ = [
    "CKMMNIST", "CKMFSMNIST", "MCRRMNIST", "CKMShallowMNIST"
]

class BaseModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

    def save_image(self, array, name):
        out_dir = f"outputs/epoch_{self.current_epoch}"
        os.makedirs(out_dir, exist_ok=True)
        image = array.reshape((28, 28))
        plt.clf()
        fig, ax = plt.subplots()
        ax.set_title('Gates, white is open')
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        plt.imsave(f"{out_dir}/{name}", image, cmap='gray')
        plt.close()

    def on_validation_epoch_start(self):
        self.val_cluster_list = []
        self.val_label_list = []

    def on_validation_epoch_end(self):
        """ Based on https://github.com/zengyi-li/NMCE-release/blob/main/NMCE/func.py"""
        cluster_mtx = torch.cat(self.val_cluster_list, dim=0)
        label_mtx = torch.cat(self.val_label_list, dim=0)
        _, _, acc_single = cluster_match(cluster_mtx, label_mtx, n_classes=label_mtx.max() + 1, print_result=False)
        _, _, acc_merge = cluster_merge_match(cluster_mtx, label_mtx, print_result=False)
        NMI = normalized_mutual_info_score(label_mtx.numpy(), cluster_mtx.numpy())
        ARI = adjusted_rand_score(label_mtx.numpy(), cluster_mtx.numpy())
        self.log('val/acc_single', acc_single)  # this is ACC
        self.log('val/acc_merge', acc_merge)
        self.log('val/NMI', NMI)
        self.log('val/ARI', ARI)

    def tau(self):
        self.tau_max = self.cfg.tau_max
        self.tau_min = self.cfg.tau_min
        self.num_epochs = self.cfg.trainer.max_epochs - self.cfg.ae_pretrain_epochs
        epoch = self.current_epoch - self.cfg.ae_pretrain_epochs
        schedules = {
            'exponential': max(self.tau_min, self.tau_max * ((self.tau_min / self.tau_max) ** (epoch / self.num_epochs))),
            'linear': max(self.tau_min, self.tau_max - (self.tau_max - self.tau_min) * (epoch / self.num_epochs)),
            'cosine': self.tau_min + 0.5 * (self.tau_max - self.tau_min) * (1. + np.cos(epoch * math.pi / self.num_epochs))
        }
        return schedules[self.cfg.tau_sched]


class MNISTModule(BaseModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        MNIST(self.cfg.data_dir, train=True, download=True)
        MNIST(self.cfg.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.dataset = ConcatDataset([
            MNIST(self.cfg.data_dir, train=True, transform=self.transform),
            MNIST(self.cfg.data_dir, train=False, transform=self.transform)
        ])

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.cfg.batch_size, drop_last=True, shuffle=True, num_workers=8 if platform.system() == 'Linux' else 0)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.cfg.batch_size, drop_last=True, shuffle=False, num_workers=8 if platform.system() == 'Linux' else 0)


class CMRRModule(BaseModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ae = AutoEncoder(cfg)
        self.cluster_net = ClusterLayer(cfg)
        self.mcrr_loss = MaximalCodingRateReduction(eps=cfg.mcrr.eps, gamma=cfg.mcrr.gamma)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.reshape(x.size(0), -1)
        z, recon_x = self.ae(x)
        z, logits = self.cluster_net(z)
        if self.current_epoch < self.cfg.ae_pretrain_epochs:
            loss = F.l1_loss(x, recon_x)
            self.log('ae/recon_loss', loss.item())
            return loss
        else:
            ae_loss = F.l1_loss(recon_x, x)
            prob = Gumble_Softmax(self.tau())(logits)
            class_discrimn_loss, class_compress_loss = self.mcrr_loss(z, prob, num_classes=self.cfg.n_clusters)
            self.log('train/cmrr_discrim_loss', -class_discrimn_loss.item())
            self.log('train/cmrr_compress_loss', class_compress_loss.item())
            self.log('train/recon_loss', ae_loss.item())
            loss = - class_discrimn_loss + self.cfg.mcrr.reg_lamba * class_compress_loss + self.cfg.recon_lamba * ae_loss
            return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        z, recon_x = self.ae(x)
        z, logits = self.cluster_net(z)
        y_hat = torch.argmax(logits, dim=1)
        self.val_cluster_list.append(y_hat.cpu())
        self.val_label_list.append(y.cpu())

    def configure_optimizers(self):
        opt = torch.optim.Adam(chain(self.ae.parameters(), self.cluster_net.parameters()), lr=self.cfg.lr)
        # sch = lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.trainer.max_epochs, eta_min=1e-5, last_epoch=-1)
        return [opt] #, [sch]


class MCRRModule(BaseModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ae = AutoEncoder(cfg)
        self.cluster_net = ClusterLayer(cfg)
        self.mcrr_loss = MaximalCodingRateReduction(eps=cfg.mcrr.eps, gamma=cfg.mcrr.gamma)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.reshape(x.size(0), -1)
        z, recon_x = self.ae(x)
        z, logits = self.cluster_net(z)
        if self.current_epoch < self.cfg.ae_pretrain_epochs:
            loss = F.l1_loss(x, recon_x)
            self.log('ae/recon_loss', loss.item())
            return loss
        else:
            ae_loss = F.l1_loss(recon_x, x)
            prob = Gumble_Softmax(self.tau())(logits)
            class_discrimn_loss, class_compress_loss = self.mcrr_loss(z, prob, num_classes=self.cfg.n_clusters)
            self.log('train/cmrr_discrim_loss', -class_discrimn_loss.item())
            self.log('train/cmrr_compress_loss', class_compress_loss.item())
            self.log('train/recon_loss', ae_loss.item())
            loss = - class_discrimn_loss + self.cfg.mcrr.reg_lamba * class_compress_loss + self.cfg.recon_lamba * ae_loss
            return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        z, recon_x = self.ae(x)
        z, logits = self.cluster_net(z)
        y_hat = torch.argmax(logits, dim=1)
        self.val_cluster_list.append(y_hat.cpu())
        self.val_label_list.append(y.cpu())

    def configure_optimizers(self):
        opt = torch.optim.Adam(chain(self.ae.parameters(), self.cluster_net.parameters()), lr=self.cfg.lr)
        # sch = lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.trainer.max_epochs, eta_min=1e-5, last_epoch=-1)
        return [opt] #, [sch]


class CKMModule(BaseModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ae = AutoEncoder(cfg)
        self.clustering_layer = RBFLayer(cfg)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.reshape(x.size(0), -1)
        embs, recon_x = self.ae(x)
        if self.current_epoch < self.cfg.ae_pretrain_epochs:
            loss = F.l1_loss(x, recon_x)
            self.log('ae/recon_loss', loss.item())
            return loss
        else:
            ae_loss = F.l1_loss(x, recon_x)
            if not self.clustering_layer.initialized:
                self.clustering_layer.init_centroids(embs)
            probs = self.clustering_layer(embs)
            gumble_probs = Gumble_Softmax(self.tau(), straight_through=True)(probs)
            ckm_loss = F.l1_loss(gumble_probs @ self.clustering_layer.centroids, embs)
            loss = self.cfg.ckm.reg_lamba * ckm_loss + ae_loss
            self.log('train/loss', loss.item())
            self.log('train/ckm_loss', ckm_loss.item())
            self.log('train/recon_loss', ae_loss.item())
            return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        z, _ = self.ae(x)
        probs = self.clustering_layer(z).exp()
        y_hat = torch.argmax(probs, dim=1)
        self.val_cluster_list.append(y_hat.cpu())
        self.val_label_list.append(y.cpu())

    def configure_optimizers(self):
        opt = torch.optim.Adam(chain(self.ae.parameters(), self.clustering_layer.parameters()), lr=self.cfg.lr)
        sch = lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.trainer.max_epochs, eta_min=1e-5, last_epoch=-1)
        return [opt], [sch]


class CKMShallowModule(BaseModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.clustering_layer = RBFLayer(cfg)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.reshape(x.size(0), -1)
        if not self.clustering_layer.initialized:
            self.clustering_layer.init_centroids(x)
        probs = self.clustering_layer(x)
        gumble_probs = Gumble_Softmax(self.tau(), straight_through=True)(probs)
        ckm_loss = F.l1_loss(gumble_probs @ self.clustering_layer.centroids, x)
        loss = self.cfg.ckm.reg_lamba * ckm_loss
        self.log('train/loss', loss.item())
        self.log('train/ckm_loss', ckm_loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        probs = self.clustering_layer(x).exp()
        y_hat = torch.argmax(probs, dim=1)
        self.val_cluster_list.append(y_hat.cpu())
        self.val_label_list.append(y.cpu())

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.clustering_layer.parameters(), lr=self.cfg.lr)
        sch = lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.trainer.max_epochs, eta_min=1e-5, last_epoch=-1)
        return [opt], [sch]


class CKMFS(CKMShallowModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.stg = STGLayer(cfg)
        self.mcrr_loss = MaximalCodingRateReduction(eps=0.1, gamma=1)
        # self.dim_reduction = DimReduction(cfg)
        self.discrimin_loss = TotalCodingRate()
        self.feats_selector = SelectLayer(cfg)
        self.feats_decoder = Decoder(cfg)


    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.reshape(x.size(0), -1)
        gated_x, h = self.stg(x, train=True)
        selected_feats, weights = self.feats_selector(gated_x, self.current_epoch)
        # embs = self.dim_reduction(gated_x)   # linear transform
        if not self.clustering_layer.initialized:
            self.clustering_layer.init_centroids(selected_feats)
        # reg_loss = self.stg.regularization(h)
        log_probs = self.clustering_layer(selected_feats)
        # probs = Gumble_Softmax(self.tau())(log_probs)
        # gumble_probs = Gumble_Softmax(self.tau(), straight_through=True)(log_probs)
        # gated_x = F.normalize(gated_x, p=2)
        # assigments = F.normalize(gumble_probs @ self.clustering_layer.centroids, p=2)
        # ckm_loss = - F.cosine_similarity(assigments, gated_x).mean()
        # ckm_loss = - torch.cosine_similarity(F.normalize(gumble_probs @ self.clustering_layer.centroids), F.normalize(embs)).mean()
        # ckm_loss = F.mse_loss(gumble_probs @ self.clustering_layer.centroids, selected_feats)
        discrimn_loss, compress_loss = self.mcrr_loss(selected_feats, log_probs.exp(), num_classes=self.cfg.n_clusters)
        selected_feats, weights = self.feats_selector(gated_x, self.current_epoch)
        gated_recon = self.feats_decoder(selected_feats)
        gted_recon_loss = F.mse_loss(gated_recon, gated_x)

        # if batch_idx % 2 == 0:
        #     ckm_loss = F.l1_loss(gumble_probs.detach() @ self.clustering_layer.centroids, embs.detach())
        # else:
        #     ckm_loss = F.l1_loss(gumble_probs @ self.clustering_layer.centroids.detach(), embs)

        # discrimn_loss, compress_loss = self.mcrr_loss(embs, probs, num_classes=self.cfg.n_clusters)
        self.log('train/discrim_loss', -discrimn_loss.item())
        self.log('train/compress_loss', compress_loss.item())

        # self.log('train/centoroids_batch_norm', self.clustering_layer.centroids.norm(dim=1).mean())
        # loss = self.cfg.ckm.reg_lamba * ckm_loss + (class_compress_loss - class_discrimn_loss) / self.cfg.batch_size
        # loss = ckm_loss + (0.1 * class_compress_loss - class_discrimn_loss)
        loss = 10*gted_recon_loss + 0.1 * (- discrimn_loss + 0.1 * compress_loss)
        # if self.current_epoch > 20:
        #     loss += 0.01 * ckm_loss #+ reg_loss #+ 0.01 * compress_loss  - 0.1 * discrimn_loss
        # self.log('train/ckm_loss', ckm_loss.item())
        self.log('train/open_gates', self.stg.num_open_gates(x))
        # self.log('train/reg_loss', reg_loss.item())
        self.log('train/gted_recon_loss', gted_recon_loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        gated_x, h= self.stg(x, train=False)
        selected_feats, _ = self.feats_selector(gated_x)
        # embs = self.dim_reduction(gated_x)
        probs = self.clustering_layer(selected_feats).exp()
        y_hat = torch.argmax(probs, dim=1)
        self.val_cluster_list.append(y_hat.cpu())
        self.val_label_list.append(y.cpu())

    def configure_optimizers(self):
        # opt1 = torch.optim.Adam(chain(self.ae.parameters(), self.clustering_layer.parameters()), lr=self.cfg.lr)
        # opt2 = torch.optim.Adam(chain(self.stg.parameters()), lr=self.cfg.stg.lr)
        # sch1 = lr_scheduler.CosineAnnealingLR(opt1, T_max=self.cfg.trainer.max_epochs, eta_min=1e-5, last_epoch=-1)
        # sch2 = lr_scheduler.CosineAnnealingLR(opt2, T_max=self.cfg.trainer.max_epochs, eta_min=1e-5, last_epoch=-1)
        # return [opt1, opt2], [sch1, sch2]
        opt = torch.optim.Adam(chain(self.clustering_layer.parameters(), self.stg.parameters(), self.feats_decoder.parameters(), self.feats_selector.parameters()), lr=self.cfg.lr)
        # sch = lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.trainer.max_epochs, eta_min=1e-5, last_epoch=-1)
        return [opt] #, [sch]


class CKMMNIST(CKMModule, MNISTModule):
    def __init__(self, cfg):
        super().__init__(cfg)


class CKMFSMNIST(CKMFS, MNISTModule):
    def __init__(self, cfg):
        super().__init__(cfg)


class MCRRMNIST(MCRRModule, MNISTModule):
    def __init__(self, cfg):
        super().__init__(cfg)


class CKMShallowMNIST(CKMShallowModule, MNISTModule):
    def __init__(self, cfg):
        super().__init__(cfg)