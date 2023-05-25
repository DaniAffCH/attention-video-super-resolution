from __future__ import absolute_import
import torch
import tqdm
from sr_utils.sr_utils import sanitizeInput, sanitizeGT

@torch.no_grad()
def validate(model, dl, loss, device):
    model.eval()
    losses = []
    for batch in tqdm.tqdm(dl):
        x = sanitizeInput(batch["x"], device)     
        Ohat = model(x)
        O = sanitizeGT(batch["y"], device)
        l = loss(Ohat, O)
        losses.append(l)

    return sum(losses)/len(losses)


