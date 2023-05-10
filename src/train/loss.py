from torch import nn
import torch

class CharbonnierLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, Ohat, O, eps = 1e-3):
        return torch.sqrt(torch.norm(Ohat - O) ** 2 + eps ** 2)