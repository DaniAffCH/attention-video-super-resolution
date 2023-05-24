from __future__ import absolute_import

import tqdm
import torch

def trainOne(model, dataloader, optimizer, device, loss, isTest=False):
    model.train()

    update = 100
    losses = []

    for n, batch in tqdm.tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        x = torch.stack(batch["x"],dim=0)
        x = x.permute(1,0,4,2,3).to(device)        

        Ohat = model(x)
        O = batch["y"].permute(0,3,1,2).to(device)

        l = loss(Ohat, O)

        l.backward()

        losses.append(float(l))

        optimizer.step()

        if(isTest):
            return

        if(n%100 == 0):
            print(sum(losses)/len(losses))
            losses = []

    # ritorna le loss varie 
    return l

