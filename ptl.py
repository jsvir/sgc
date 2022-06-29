import math
import os
import platform
from itertools import chain
from pytorch_lightning import seed_everything
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning import LightningModule
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST

from func import cluster_match, cluster_merge_match, normalized_mutual_info_score, adjusted_rand_score
from loss import MaximalCodingRateReduction
from model import AutoEncoder, RBFLayer, Gumble_Softmax, ClusterLayer, STGLayerExt

__all__ = [
    "CKMMNIST", "CKMFSMNIST", "CKMFS2MNIST", "MCRRMNIST", "CKMShallowMNIST", "MCRRFSMNIST", "CCMNIST"
]


class BaseModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

    def save_image(self, array, name):
        out_dir = f"{self.logger.log_dir}/outputs/epoch_{self.current_epoch}"
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
        self.val_cluster_list_gated = []
        self.val_label_list = []
        if self.current_epoch == 9:
            self.val_x_list = []

    def on_validation_epoch_end(self):
        """ Based on https://github.com/zengyi-li/NMCE-release/blob/main/NMCE/func.py"""
        # if self.current_epoch == 9:
        #     x = torch.cat(self.val_x_list, dim=0)
        #     kmeans = KMeans(n_clusters=self.cfg.n_clusters, init='k-means++', n_init=100, random_state=0).fit(x.numpy())
        #     y_hat = torch.tensor(kmeans.labels_)
        #     # y_corr = []
        #     # x_corr = []
        #     # y_gt = []
        #     # threshold = 0.5
        #     # for y_i in y_hat.unique():
        #     #     xy = x[y_hat == y_i]
        #     #     y_gt_y = torch.cat(self.val_label_list,dim=0)[y_hat == y_i]
        #     #     mean_xy = xy.mean(dim=0)
        #     #     # dists = []
        #     #     for xy_i, y_gt_y_i in zip(xy, y_gt_y):
        #     #         if F.l1_loss(xy_i, mean_xy) >= threshold:
        #     #             # x_corr.append(xy_i.reshape(1, -1))
        #     #             y_corr.append(y_i.reshape(1))
        #     #             y_gt.append(y_gt_y_i.reshape(1))
        #     #         # dists.append(-torch.cosine_similarity(cy_i, mean_cy, dim=0).item())
        #     #     # print(f"np.max(dists)={np.max(dists)}, np.min(dists)={np.min(dists)}")
        #     # ratio = len(x_corr) / len(x)
        #     # self.log('val/labels_ratio', ratio)
        #     # self.log('val/threshold', threshold)
        #     # if len(y_corr) > 0:
        #     #     y_gt = torch.cat(y_gt, dim=0)
        #     #     y_hat = torch.cat(y_corr, dim=0)
        #     cluster_mtx = torch.tensor(y_hat)
        #     label_mtx = torch.cat(self.val_label_list, dim=0) #y_gt
        # else:
        cluster_mtx = torch.cat(self.val_cluster_list, dim=0)
        label_mtx = torch.cat(self.val_label_list, dim=0)
        if self.cfg.n_clusters == 2:
            label_mtx[label_mtx == 3] = 0
            label_mtx[label_mtx == 8] = 1
        _, _, acc_single = cluster_match(cluster_mtx, label_mtx, n_classes=label_mtx.max() + 1, print_result=False)
        _, _, acc_merge = cluster_merge_match(cluster_mtx, label_mtx, print_result=False)
        NMI = normalized_mutual_info_score(label_mtx.numpy(), cluster_mtx.numpy())
        ARI = adjusted_rand_score(label_mtx.numpy(), cluster_mtx.numpy())
        format_str = ''  # '_kmeans' if self.current_epoch == 9 else ''
        self.log(f'val/acc_single{format_str}', acc_single)  # this is ACC
        self.log(f'val/acc_merge{format_str}', acc_merge)
        self.log(f'val/NMI{format_str}', NMI)
        self.log(f'val/ARI{format_str}', ARI)

        # cluster_mtx = torch.cat(self.val_cluster_list_gated, dim=0)
        # label_mtx = torch.cat(self.val_label_list, dim=0)
        # if self.cfg.n_clusters == 2:
        #     label_mtx[label_mtx == 3] = 0
        #     label_mtx[label_mtx == 8] = 1
        # _, _, acc_single = cluster_match(cluster_mtx, label_mtx, n_classes=label_mtx.max() + 1, print_result=False)
        # _, _, acc_merge = cluster_merge_match(cluster_mtx, label_mtx, print_result=False)
        # NMI = normalized_mutual_info_score(label_mtx.numpy(), cluster_mtx.numpy())
        # ARI = adjusted_rand_score(label_mtx.numpy(), cluster_mtx.numpy())
        # format_str = ''  # '_kmeans' if self.current_epoch == 9 else ''
        # self.log(f'val/gated_acc_single{format_str}', acc_single)  # this is ACC
        # self.log(f'val/gated_acc_merge{format_str}', acc_merge)
        # self.log(f'val/gated_NMI{format_str}', NMI)
        # self.log(f'val/gated_ARI{format_str}', ARI)

    def tau(self):
        self.tau_max = self.cfg.tau_max
        self.tau_min = self.cfg.tau_min
        self.num_epochs = self.cfg.trainer.max_epochs - self.cfg.ae_pretrain_epochs if 'ae_pretrain_epochs' in self.cfg.keys() else self.cfg.trainer.max_epochs
        epoch = self.current_epoch - self.cfg.ae_pretrain_epochs if 'ae_pretrain_epochs' in self.cfg.keys() else self.current_epoch
        schedules = {
            'exponential': max(self.tau_min, self.tau_max * ((self.tau_min / self.tau_max) ** (epoch / self.num_epochs))),
            'linear': max(self.tau_min, self.tau_max - (self.tau_max - self.tau_min) * (epoch / self.num_epochs)),
            'cosine': self.tau_min + 0.5 * (self.tau_max - self.tau_min) * (1. + np.cos(epoch * math.pi / self.num_epochs))
        }
        tau = schedules[self.cfg.tau_sched]
        self.log('train/tau', tau)
        return tau


class MNISTModule(BaseModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,)),
                # transforms.Normalize((0.14608,), (0.32195,))
                transforms.Normalize((0.5,), (0.5,))

            ]
        )

    def prepare_data(self):
        MNIST(self.cfg.data_dir, train=True, download=True)
        MNIST(self.cfg.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if self.cfg.n_clusters == 2:
            train = MNIST(self.cfg.data_dir, train=True, transform=self.transform)
            test = MNIST(self.cfg.data_dir, train=False, transform=self.transform)
            train_indices = [i for i, x in enumerate((train.targets == 3) | (train.targets == 8)) if x]
            test_indices = [i for i, x in enumerate((test.targets == 3) | (test.targets == 8)) if x]
            self.dataset = ConcatDataset([
                torch.utils.data.Subset(train, train_indices),
                # torch.utils.data.Subset(test, test_indices)
            ])
        else:
            self.dataset = ConcatDataset([
                MNIST(self.cfg.data_dir, train=True, transform=self.transform),
                MNIST(self.cfg.data_dir, train=False, transform=self.transform)
            ])
        print(f"Dataset length: {self.dataset.__len__()}")

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.cfg.batch_size, drop_last=True, shuffle=True, num_workers=8 if platform.system() == 'Linux' else 0)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.cfg.batch_size, drop_last=True, shuffle=False, num_workers=8 if platform.system() == 'Linux' else 0)


class CCModule(BaseModule):
    """ we train it by iteratively switching between fs + classification and clustering. The labels are produced by clustering"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.stg = STGLayerExt(cfg)
        self.ae = AutoEncoder(cfg)
        self.mcrr_loss = MaximalCodingRateReduction(eps=cfg.mcrr.eps, gamma=cfg.mcrr.gamma)
        if os.path.exists(self.cfg.ae.model_path):
            loaded_states = torch.load(self.cfg.ae.model_path)
            self.ae.load_state_dict(loaded_states, strict=True)
            self.ae_train = False
        else:
            self.ae_train = True
        self.automatic_optimization = False

    def get_labels(self, x, gated_x, z):
        kmeans = KMeans(n_clusters=self.cfg.n_clusters, init='k-means++', n_init=100, random_state=0).fit(z.detach().cpu().numpy())
        y_hat = torch.tensor(kmeans.labels_, device=z.device).long()
        y_selected = []
        gx_selected = []
        x_selected = []
        z_selected = []
        for y_i in y_hat.unique():
            zy = z[y_hat == y_i]
            gxy = gated_x[y_hat == y_i]
            xy = x[y_hat == y_i]
            mean_cy = zy.mean(dim=0)
            for zy_i, gxy_i,xy_i  in zip(zy, gxy, xy):
                if -torch.cosine_similarity(zy_i, mean_cy, dim=0).item() < - self.cfg.pseudo_label_thresh:
                    gx_selected.append(gxy_i.reshape(1, -1))
                    x_selected.append(xy_i.reshape(1,-1))
                    y_selected.append(y_i.reshape(1))
                    z_selected.append(zy_i.reshape(1,-1))
        ratio = len(gx_selected) / len(x)
        self.log('train/labels_ratio', ratio)
        if len(gx_selected) > 0:
            x_hat = torch.cat(x_selected, dim=0)
            gx_hat = torch.cat(gx_selected, dim=0)
            y_hat = torch.cat(y_selected, dim=0)
            z_hat = torch.cat(z_selected, dim=0)
            return x_hat, gx_hat, y_hat, z_hat
        else:
            return None, None, None, None

    def on_train_epoch_end(self):
        if self.ae_train and self.current_epoch == self.cfg.ae_pretrain_epochs:
            torch.save(self.ae.state_dict(), self.cfg.ae.model_path)

    def mcrr(self, c, logits):
        logprobs = torch.log_softmax(logits, dim=-1)
        prob = Gumble_Softmax(self.tau())(logprobs)
        discrimn_loss, compress_loss = self.mcrr_loss(F.normalize(c), prob, num_classes=self.cfg.n_clusters)
        discrimn_loss /= c.size(1)
        compress_loss /= c.size(1)
        return 0.1 * compress_loss - discrimn_loss

    def up_scale(self, min_lamba=0.01, max_lamba=0.1, epochs=None):
        epochs = self.cfg.trainer.max_epochs if epochs is None else epochs
        return min(max_lamba, min_lamba + (max_lamba - min_lamba) * min(1, self.current_epoch / epochs))

    def down_scale(self, min_lamba=0.01, max_lamba=0.1, epochs=None):
        epochs = self.cfg.trainer.max_epochs if epochs is None else epochs
        return max(min_lamba, max_lamba - (max_lamba - min_lamba) * min(1, self.current_epoch / epochs))

    def ae_step(self, x):
        # TRAIN AE + gates
        h, gated_x = self.stg(x)
        z, recon = self.ae(gated_x)
        recon_loss = F.l1_loss(recon, x)
        self.log('train/recon_x', recon_loss.item())
        reg_loss = self.stg.regularization(h)
        self.log('train/reg_loss', reg_loss.item())
        loss = recon_loss + 0.1 * reg_loss
        return loss

    def clustering_step(self, x):
        h, gated_x = self.stg(x)
        z, recon = self.ae(gated_x)
        c = self.stg.embedding_layer(z.detach())
        logits = self.stg.clustering_layer(c)
        mcrr_loss = self.mcrr(c, logits)
        self.log('train/mcrr_loss_gated_x', mcrr_loss.item())
        return mcrr_loss

    def classification_step(self, x):
        # get labels from clustering:
        with torch.no_grad():
            h, gated_x = self.stg(x)
            z, recon = self.ae(gated_x)
            c = self.stg.embedding_layer(z)
            logits = self.stg.clustering_layer(c)
            y_hat = torch.argmax(logits, dim=1)

        # now optimize classification on these labels:
        h, gated_x = self.stg(x)
        z, recon = self.ae(gated_x)
        c = self.stg.embedding_layer2(z)
        logits = self.stg.classifier(c)
        ce_loss = F.cross_entropy(logits, y_hat)
        self.log('train/ce_loss', ce_loss.item())
        reg_loss = self.stg.regularization(h)
        self.log('train/reg_loss', reg_loss.item())
        lamba = self.up_scale(1, 100) if self.current_epoch > 100 else .1
        loss = ce_loss + lamba * reg_loss
        return loss

    def training_step(self, batch, batch_idx):
        ae_opt, clu_opt, class_opt = self.optimizers()
        x, _ = batch
        x = x.reshape(x.size(0), -1)
        if self.ae_train and self.current_epoch < self.cfg.ae_pretrain_epochs:
            ae_opt.zero_grad()
            loss = self.ae_step(x)
            self.manual_backward(loss)
            ae_opt.step()
        else:
            # ae regularization step:
            ae_opt.zero_grad()
            loss = self.ae_step(x)
            self.manual_backward(loss)
            ae_opt.step()

            clu_opt.zero_grad()
            loss = self.clustering_step(x)
            self.manual_backward(loss)
            clu_opt.step()

            class_opt.zero_grad()
            loss = self.classification_step(x)
            self.manual_backward(loss)
            class_opt.step()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        _, gated_x = self.stg(x)
        z, recon = self.ae(gated_x)
        c = self.stg.embedding_layer(z)
        logits = self.stg.clustering_layer(c)
        gates = self.stg.get_gates(x)
        for i, (gx, label, xx, gated_xx) in enumerate(zip(gates, y, x, gated_x)):
            label = label.cpu().numpy()
            img = gx.reshape(28, 28).cpu().numpy()
            orig_img = xx.reshape(28, 28).cpu().numpy()
            gated_img = gated_xx.reshape(28, 28).cpu().numpy()
            self.save_image(img, f'img_{i}_{label}.png')
            self.save_image(orig_img, f'orig_img_{i}_{label}.png')
            self.save_image(gated_img, f'gated_img_{i}_{label}.png')
            if i == 2: break
        self.val_cluster_list.append(torch.argmax(logits, dim=1).cpu())
        self.val_label_list.append(y.cpu())

    def configure_optimizers(self):
        ae_opt = torch.optim.Adam(chain(self.stg.net.parameters(), self.ae.parameters()), lr=0.001, betas=(0.8, 0.99))
        clu_opt = torch.optim.Adam(chain(
            self.stg.clustering_layer.parameters(),
            self.stg.embedding_layer.parameters()), lr=0.001, betas=(0.8, 0.99))
        class_opt = torch.optim.Adam(chain(
            self.stg.net.parameters(),
            self.stg.classifier.parameters(),
            self.stg.embedding_layer2.parameters()), lr=0.001, betas=(0.8, 0.99))
        return [ae_opt, clu_opt, class_opt]


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
        return [opt]  # , [sch]


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
            logprobs = torch.log_softmax(logits, dim=-1)
            prob = Gumble_Softmax(self.tau())(logprobs)
            discrimn_loss, compress_loss = self.mcrr_loss(z, prob, num_classes=self.cfg.n_clusters)
            discrimn_loss /= z.size(1)
            compress_loss /= z.size(1)
            self.log('train/discrim_loss', -discrimn_loss.item())
            self.log('train/compress_loss', compress_loss.item())
            self.log('train/recon_loss', ae_loss.item())
            loss = - discrimn_loss + self.cfg.mcrr.reg_lamba * compress_loss + self.cfg.recon_lamba * ae_loss
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
        return [opt]  # , [sch]


# TODO: try to train two models with different seeds, then produce the labels from the agreements on labels

class MCRRFS(BaseModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cluster_net = ClusterLayer(cfg)
        self.mcrr_loss = MaximalCodingRateReduction(eps=cfg.mcrr.eps, gamma=cfg.mcrr.gamma)
        self.stg = STGLayerExt(cfg)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.reshape(x.size(0), -1)
        gated_x, h, _ = self.stg(x, train=True)
        _, logits = self.cluster_net(gated_x)
        logprobs = torch.log_softmax(logits, dim=-1)
        reg_loss = self.stg.regularization(h)
        prob = Gumble_Softmax(self.tau())(logprobs)
        discrimn_loss, compress_loss = self.mcrr_loss(F.normalize(gated_x), prob, num_classes=self.cfg.n_clusters)
        discrimn_loss /= gated_x.size(1)
        compress_loss /= gated_x.size(1)
        self.log('train/discrim_loss', -discrimn_loss.item())
        self.log('train/compress_loss', compress_loss.item())
        self.log('train/open_gates', self.stg.num_open_gates(x))
        self.log('train/reg_loss', reg_loss.item())
        loss = - discrimn_loss + compress_loss + reg_loss
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        gated_x, _, _ = self.stg(x, train=False)
        z, logits = self.cluster_net(gated_x)
        gates = self.stg.get_gates(x)
        for i, (gx, label) in enumerate(zip(gates, y)):
            label = label.cpu().numpy()
            img = gx.reshape(28, 28).cpu().numpy()
            self.save_image(img, f'img_{i}_{label}.png')
            if i == 2: break
        y_hat = torch.argmax(logits, dim=1)
        self.val_cluster_list.append(y_hat.cpu())
        self.val_label_list.append(y.cpu())

    def configure_optimizers(self):
        opt = torch.optim.Adam(chain(self.stg.parameters(), self.cluster_net.parameters()), lr=self.cfg.lr)
        # sch = lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.trainer.max_epochs, eta_min=1e-5, last_epoch=-1)
        return [opt]  # , [sch]


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
            logprobs = self.clustering_layer(embs)
            gumble_probs = Gumble_Softmax(self.tau(), straight_through=True)(logprobs)
            ckm_loss = F.l1_loss(gumble_probs @ self.clustering_layer.centroids, embs)
            loss = self.cfg.ckm.reg_lamba * ckm_loss + ae_loss
            # loss = ckm_loss + ae_loss
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
        if self.current_epoch == 9:
            self.val_x_list.append(x.cpu())
            self.val_label_list.append(y.cpu())
        else:
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
        self.stg = STGLayerExt(cfg)
        # self.automatic_optimization = False
        # self.mcrr_loss = MaximalCodingRateReduction(eps=0.001, gamma=1)

    def get_labels(self, x):
        h, c, logits = self.stg(x)
        kmeans = KMeans(n_clusters=self.cfg.n_clusters, init='k-means++', n_init=10, random_state=0).fit(c.detach().cpu().numpy())
        y_hat = torch.tensor(kmeans.labels_, device=c.device).long()
        y_conf = []
        x_conf = []
        threshold = 0.7  # 0.2 + 0.5 * (1 - self.current_epoch / self.cfg.trainer.max_epochs)
        c = F.normalize(c)
        for y_i in y_hat.unique():
            cy = c[y_hat == y_i]
            xy = x[y_hat == y_i]
            mean_cy = cy.mean(dim=0)
            # dists = []
            for cy_i, xy_i in zip(cy, xy):
                if -torch.cosine_similarity(cy_i, mean_cy, dim=0).item() < - threshold:
                    x_conf.append(xy_i.reshape(1, -1))
                    y_conf.append(y_i.reshape(1))
                # dists.append(-torch.cosine_similarity(cy_i, mean_cy, dim=0).item())
            # print(f"np.max(dists)={np.max(dists)}, np.min(dists)={np.min(dists)}")
        ratio = len(x_conf) / len(x)
        self.log('train/labels_ratio', ratio)
        self.log('train/threshold', threshold)
        if len(x_conf) > 0:
            x_hat = torch.cat(x_conf, dim=0)
            y_hat = torch.cat(y_conf, dim=0)
            return x_hat.detach(), y_hat.detach()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.reshape(x.size(0), -1)
        if self.current_epoch < 100:
            # train on raw data:
            batch_x = x
        else:
            h, c, logits, gated_x = self.stg(x, gated_x=True)
            batch_x = torch.cat([x, gated_x], dim=0)
            reg_loss = self.stg.regularization(h)
            self.log('train/open_gates', self.stg.num_open_gates(x))
            self.log('train/reg_loss', reg_loss.item())
        if not self.clustering_layer.initialized or (self.current_epoch == 100 and batch_idx == 0):
            self.clustering_layer.init_centroids(batch_x)
        logprobs = self.clustering_layer(batch_x)
        # logprobs = torch.log_softmax(logits, dim=-1)
        gumble_probs = Gumble_Softmax(self.tau(), straight_through=True)(logprobs)
        ckm_loss = F.l1_loss(gumble_probs @ self.clustering_layer.centroids, batch_x)
        # ckm_loss = -torch.cosine_similarity(gumble_probs @ self.clustering_layer.centroids, c, dim=1).mean()
        self.log('train/ckm_loss', ckm_loss.item())
        return ckm_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)

        # if self.current_epoch == 9:
        #     self.val_x_list.append(x.cpu())
        #     self.val_label_list.append(y.cpu())
        # else:
        _, c, logits, gated_x = self.stg(x, gated_x=True)
        # c = F.normalize(c)
        gates = self.stg.get_gates(x)
        if self.current_epoch < 100:
            gated_x = x
        probs = self.clustering_layer(gated_x).exp()
        for i, (gx, label) in enumerate(zip(gates, y)):
            label = label.cpu().numpy()
            img = gx.reshape(28, 28).cpu().numpy()
            self.save_image(img, f'img_{i}_{label}.png')
            if i == 2: break
        # y_hat = torch.argmax(logits, dim=1)
        y_hat = torch.argmax(probs, dim=1)

        self.val_cluster_list.append(y_hat.cpu())
        self.val_label_list.append(y.cpu())

    def configure_optimizers(self):
        opt = torch.optim.Adam(chain(
            self.clustering_layer.parameters(),
            self.stg.parameters(),
        ), lr=self.cfg.lr)
        sch = lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.trainer.max_epochs, eta_min=1e-5, last_epoch=-1)
        return [opt], [sch]


class CKMFS2(CKMModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.stg = STGLayerExt(cfg)
        self.mcrr_loss = MaximalCodingRateReduction(eps=0.001, gamma=1)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.reshape(x.size(0), -1)

        gated_x, h, z = self.stg(x, train=True)

        embs, recon_x = self.ae(gated_x)
        if self.current_epoch < self.cfg.ae_pretrain_epochs:
            loss = F.l1_loss(gated_x, recon_x)
            self.log('ae/recon_loss', loss.item())
            return loss
        else:
            ae_loss = F.l1_loss(gated_x, recon_x)
            if not self.clustering_layer.initialized:
                self.clustering_layer.init_centroids(embs)
            reg_loss = self.stg.regularization(h)
            log_probs = self.clustering_layer(embs)
            gumble_probs = Gumble_Softmax(self.tau(), straight_through=True)(log_probs)
            ckm_loss = F.l1_loss(gumble_probs @ self.clustering_layer.centroids, embs)
            loss = 0.1 * ckm_loss + ae_loss
            self.log('train/ckm_loss', ckm_loss.item())
            self.log('train/open_gates', self.stg.num_open_gates(x))
            self.log('train/reg_loss', reg_loss.item())
            self.log('train/ae_loss', ae_loss.item())

        return loss

    def ce_step(self):
        # generate model outputs from different initializations on the same input
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        gated_x, _, _ = self.stg(x, train=False)
        gates = self.stg.get_gates(x)
        embs, recon_x = self.ae(gated_x)
        probs = self.clustering_layer(embs).exp()
        for i, (gx, label) in enumerate(zip(gates, y)):
            label = label.cpu().numpy()
            img = gx.reshape(28, 28).cpu().numpy()
            self.save_image(img, f'img_{i}_{label}.png')
            if i == 2: break
        y_hat = torch.argmax(probs, dim=1)
        self.val_cluster_list.append(y_hat.cpu())
        self.val_label_list.append(y.cpu())

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            chain(
                self.clustering_layer.parameters(),
                self.stg.parameters(),
            ),
            lr=self.cfg.lr, weight_decay=self.cfg.wd)
        return [opt]  # , [sch]


class CKMMNIST(CKMModule, MNISTModule):
    def __init__(self, cfg):
        super().__init__(cfg)


class CKMFSMNIST(CKMFS, MNISTModule):
    def __init__(self, cfg):
        super().__init__(cfg)


class CKMFS2MNIST(CKMFS2, MNISTModule):
    def __init__(self, cfg):
        super().__init__(cfg)


class MCRRMNIST(MCRRModule, MNISTModule):
    def __init__(self, cfg):
        super().__init__(cfg)


class MCRRFSMNIST(MCRRFS, MNISTModule):
    def __init__(self, cfg):
        super().__init__(cfg)


class CKMShallowMNIST(CKMShallowModule, MNISTModule):
    def __init__(self, cfg):
        super().__init__(cfg)


class CCMNIST(CCModule, MNISTModule):
    def __init__(self, cfg):
        super().__init__(cfg)
