from __future__ import absolute_import

import tqdm
import torch

def trainOne_generator(generator, discriminator, dataloader, optimizer, device, loss):
    generator.train()
    for batch in tqdm.tqdm(dataloader):
        optimizer.zero_grad()

        x = batch.stack(batch["x"],dim=0)
        x = x.permute(1,0,4,2,3).to(device)        

        Ohat = generator(x)

        discHat = discriminator(Ohat)
        discTrue = torch.zeros(batch.shape[0])

        l = loss(Ohat, batch["y"], discTrue, discHat)

        l.backward()

        optimizer.step()

        # ritorna le loss varie 

def trainOne_discriminator():
    pass