from __future__ import absolute_import
import torch

def bilinear_upsample(input, scale_factor):
    # Input:
    # B x C x H x W
    # Output:
    # B x C x H_up x W_up
    
    B, C, H, W = input.shape
    

    H_up = H * scale_factor
    W_up = W * scale_factor
    

    y = torch.linspace(-1, 1, H_up)
    x = torch.linspace(-1, 1, W_up)
    yy, xx = torch.meshgrid(y, x)
    grid = torch.stack((yy, xx), dim=2).unsqueeze(0).repeat(B, 1, 1, 1)
    
    # Normalizza le coordinate per il tensore originale
    grid = grid * torch.tensor([H-1, W-1], dtype=torch.float32).to(input.device)
    grid = grid / torch.tensor([H_up-1, W_up-1], dtype=torch.float32).to(input.device)
    grid = grid * 2 - 1
    
    # Applica l'interpolazione bilineare
    output = torch.nn.functional.grid_sample(input, grid, align_corners=True)
    
    return output