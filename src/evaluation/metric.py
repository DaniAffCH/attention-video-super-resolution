from __future__ import absolute_import

from torch import nn
from kornia.color import rgb_to_ycbcr
import torch
import math

def mse(O, Ohat):
    squaredErr = (O-Ohat)**2
    return squaredErr.mean(2).mean(1)

def psnr(O, Ohat):
    O = rgb_to_ycbcr(O)
    O = O[:,0,:,:]
    Ohat = rgb_to_ycbcr(Ohat)
    Ohat = Ohat[:,0,:,:]

    val,_ = torch.max(Ohat.flatten(1), 1) 

    ratio = val/mse(O, Ohat)

    ratio = float(ratio.mean(0))

    return 10*math.log10(ratio)
