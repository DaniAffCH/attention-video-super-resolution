from __future__ import absolute_import

import tqdm

def trainOne_generator(model, dataloader):
    model.train()
    for i, batch in tqdm.tqdm(dataloader):
        pass
def trainOne_discriminator():
    pass