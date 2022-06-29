import argparse
import os
import torch

from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
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
    # add py files into yaml:
    with open("model.py") as r:
        cfg.model_py = r.read()
    with open("ptl.py") as r:
        cfg.ptl_py = r.read()
    for seed in range(cfg.repitions):
        print(cfg.items())
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seed_everything(seed)
        model = getattr(ptl, f'{cfg.clustering}{cfg.dataset}')(cfg)
        trainer = Trainer(**cfg.trainer, callbacks=LearningRateMonitor(logging_interval='step'))
        trainer.fit(model)
