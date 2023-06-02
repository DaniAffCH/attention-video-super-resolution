from __future__ import absolute_import

from torch import nn
from kornia.color import rgb_to_ycbcr
import torch
import cv2

def mse(O, Ohat):
    squaredErr = (O-Ohat)**2
    return squaredErr.mean(2).mean(1).mean(0)

def psnr(O, Ohat):
    O = rgb_to_ycbcr(O)
    O = O[:,0,:,:]
    Ohat = rgb_to_ycbcr(Ohat)
    Ohat = Ohat[:,0,:,:]

    val,_ = torch.max(Ohat.flatten(1), 1) 
    val = float(val)

    return 10*torch.log10(val/mse(O, Ohat))
