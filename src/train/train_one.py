from __future__ import absolute_import

import tqdm
import torch
from sr_utils.sr_utils import sanitizeInput, sanitizeGT

def trainOne(model, dataloader, optimizer, device, loss, conf, isTest=False):
    model.train()

    update = conf["TRAINING"].getint("update_rate")
    losses = []
    totLoss = 0
    totUpdate = 0

    for n, batch in tqdm.tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        x = sanitizeInput(batch["x"], device)     

        Ohat = model(x)

        O = sanitizeGT(batch["y"], device)

        l = loss(Ohat, O)

        l.backward()

        losses.append(float(l))

        optimizer.step()

        if(isTest):
            return .0

        with torch.no_grad():
            if(n%update == 0):
                lavg = sum(losses)/len(losses)
                totUpdate += 1
                totLoss += lavg
                print(f"[UPDATE] batch {n} avg loss {lavg}")
                losses = []

    return totLoss/totUpdate

