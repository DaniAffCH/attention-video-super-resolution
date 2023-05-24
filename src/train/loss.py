from __future__ import absolute_import
from torch import nn
import torch
    
class Loss(nn.Module):
    def __init__(self, eps = 1e-3) -> None:
        self.eps = eps
        super().__init__()

    def CharbonnierLoss(self, Ohat, O):
        return torch.sqrt(torch.norm(Ohat - O) ** 2 + self.eps ** 2)
    
    def forward(self, Ohat, OTrue):
        return self.CharbonnierLoss(Ohat, OTrue)
        
        
