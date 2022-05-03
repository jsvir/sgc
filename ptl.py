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
from loss import MaximalCodingRateReduction
from model import AutoEncoder, RBFLayer, Gumble_Softmax, STGLayer, SubspaceClusterNetwork

__all__ = [
    "CKMMNIST", "CKMFSMNIST", "CMRRMNIST"
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
        self.cluster_net = SubspaceClusterNetwork(cfg)
        self.mcrr_loss = MaximalCodingRateReduction(eps=cfg.mcrr.eps, gamma=cfg.mcrr.gamma)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.reshape(x.size(0), -1)
        z, logits = self.cluster_net(x)
        prob = Gumble_Softmax(self.tau())(logits)
        class_discrimn_loss, class_compress_loss = self.mcrr_loss(z, prob, num_classes=self.cfg.n_clusters)
        self.log('train/cmrr_discrim_loss', -class_discrimn_loss.item())
        self.log('train/cmrr_compress_loss', class_compress_loss.item())
        loss = -class_discrimn_loss + self.cfg.mcrr.reg_lamba * class_compress_loss
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        z, logits = self.cluster_net(x)
        y_hat = torch.argmax(logits, dim=1)
        self.val_cluster_list.append(y_hat.cpu())
        self.val_label_list.append(y.cpu())

    def configure_optimizers(self):
        opt = torch.optim.Adam(chain(self.cluster_net.parameters()), lr=self.cfg.lr)
        sch = lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.trainer.max_epochs, eta_min=1e-5, last_epoch=-1)
        return [opt], [sch]


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


class CKMFS(CKMModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.stg = STGLayer(cfg)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.reshape(x.size(0), -1)
        if self.current_epoch < self.cfg.ae_pretrain_epochs:
            embs, recon_x = self.ae(x)
            loss = F.l1_loss(x, recon_x)
            self.log('ae/recon_loss', loss.item())
            return loss
        else:
            if self.current_epoch > self.cfg.stg.start_reg_after_epoch:
                gated_x, h = self.stg(x, train=True)
                embs, recon_x = self.ae(gated_x)
                ae_loss = F.l1_loss(gated_x, recon_x)
                reg_loss = self.stg.regularization(h)
                loss = reg_loss + ae_loss
                self.log('train/reg_loss', reg_loss.item())
            else:
                embs, recon_x = self.ae(x)
                ae_loss = F.l1_loss(x, recon_x)
                loss = ae_loss

            if not self.clustering_layer.initialized:
                self.clustering_layer.init_centroids(embs)
            probs = self.clustering_layer(embs)
            gumble_probs = Gumble_Softmax(self.tau(), straight_through=True)(probs)
            ckm_loss = F.l1_loss(gumble_probs @ self.clustering_layer.centroids, embs)
            loss += self.cfg.ckm.reg_lamba * ckm_loss
            self.log('train/ckm_loss', ckm_loss.item())
            self.log('train/recon_loss', ae_loss.item())

            return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        if self.current_epoch > self.cfg.stg.start_reg_after_epoch:
            gated_x, h = self.stg(x, train=False)
            z, _ = self.ae(gated_x)
        else:
            z, _ = self.ae(x)
        probs = self.clustering_layer(z).exp()
        y_hat = torch.argmax(probs, dim=1)
        self.val_cluster_list.append(y_hat.cpu())
        self.val_label_list.append(y.cpu())

    def configure_optimizers(self):
        # opt1 = torch.optim.Adam(chain(self.ae.parameters(), self.clustering_layer.parameters()), lr=self.cfg.lr)
        # opt2 = torch.optim.Adam(chain(self.stg.parameters()), lr=self.cfg.stg.lr)
        # sch1 = lr_scheduler.CosineAnnealingLR(opt1, T_max=self.cfg.trainer.max_epochs, eta_min=1e-5, last_epoch=-1)
        # sch2 = lr_scheduler.CosineAnnealingLR(opt2, T_max=self.cfg.trainer.max_epochs, eta_min=1e-5, last_epoch=-1)
        # return [opt1, opt2], [sch1, sch2]
        opt = torch.optim.Adam(chain(self.ae.parameters(), self.clustering_layer.parameters(), self.stg.parameters()), lr=self.cfg.lr)
        sch = lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.trainer.max_epochs, eta_min=1e-5, last_epoch=-1)
        return [opt], [sch]


class CKMMNIST(CKMModule, MNISTModule):
    def __init__(self, cfg):
        super().__init__(cfg)


class CKMFSMNIST(CKMFS, MNISTModule):
    def __init__(self, cfg):
        super().__init__(cfg)


class CMRRMNIST(CMRRModule, MNISTModule):
    def __init__(self, cfg):
        super().__init__(cfg)
