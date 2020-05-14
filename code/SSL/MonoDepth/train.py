# Digging into Self-Supervised Monocular Depth Prediction (https://github.com/nianticlabs/monodepth2 code modified for our purpose


from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
