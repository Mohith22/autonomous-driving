#Utils File

# -- Imports -- #

import random
import torch
import numpy as np

#ThreatScore Per Sample - Determines Model Performance - Challenge Metric
def ThreatScore(true, pred):
    tp = (true * pred).sum()
    return (tp * 1.0 / (true.sum() + pred.sum() - tp)).item()

#ThreatScore Per Batch- Determines Model Performance - Challenge Metric
def BatchThreatScore(true, pred):
    batch_size = true.size(0)
    true = true.reshape(batch_size, -1)
    pred = pred.reshape(batch_size, -1)
    tp = (true * pred).sum(1)
    return (tp * 1.0 / (true.sum(1) + pred.sum(1) - tp)).sum().item()

#Dice Loss
def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))
#Set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

