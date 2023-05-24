from __future__ import absolute_import

import tqdm
import torch

def trainOne_generator(generator, dataloader, optimizer, device, loss):
    generator.train()
    for batch in tqdm.tqdm(dataloader):
        optimizer.zero_grad()

        x = batch.stack(batch["x"],dim=0)
        x = x.permute(1,0,4,2,3).to(device)        

        Ohat = generator(x)

        l = loss(Ohat, batch["y"])

        l.backward()

        optimizer.step()

        # ritorna le loss varie 

        return l

