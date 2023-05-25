from __future__ import absolute_import
import torch

def sanitizeInput(input, device):
    x = torch.stack(input,dim=0)
    return x.permute(1,0,4,2,3).to(device) 

def sanitizeGT(gt, device):
    return gt.permute(0,3,1,2).to(device)
    