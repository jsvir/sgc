import argparse
import os
import torch

from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything

import ptl

# after last Windows update I need this flag
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)
    cfg.update(OmegaConf.load(cfg.base_cfg))
    # for init_weights in [True, False]:
    #     cfg.init_weights = init_weights
    #     for activation in ['relu', 'tanh']:
    #         cfg.activation = activation
    #         for tau in [1000, 100, 50, 20, 10]:
    #             cfg.tau_min = tau
    #             cfg.tau_max = tau
    #             for eps in [0.1, 0.001, 0.0001]:
    #                 cfg.mcrr.eps = eps
    #                 for gamma in [0.01, 0.1, 1, 10]:
    #                     cfg.mcrr.gamma = gamma
    for seed in range(cfg.repitions):
        print(cfg.items())
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seed_everything(seed)
        model = getattr(ptl, f'{cfg.clustering}{cfg.dataset}')(cfg)
        trainer = Trainer(**cfg.trainer)
        trainer.fit(model)
