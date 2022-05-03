import argparse
import os

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
    for seed in range(cfg.repitions):
        seed_everything(seed)
        model = getattr(ptl, f'{cfg.clustering}{cfg.dataset}')(cfg)
        trainer = Trainer(**cfg.trainer)
        trainer.fit(model)
