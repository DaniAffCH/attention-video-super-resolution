from __future__ import absolute_import

import torch
import tqdm

from data.REDS_loader import getDataLoader
from model.generator import Generator
from evaluation.metric import psnr
from sr_utils.sr_utils import sanitizeInput, sanitizeGT

@torch.no_grad()
def evaluate(conf, path, test=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    dlVal = getDataLoader(conf, "val", True)

    gen = Generator(conf).to(device)
    
    if(not test):
        gen.load_state_dict(torch.load(path))

    psnr_list = list()

    for batch in tqdm.tqdm(dlVal):
        x = sanitizeInput(batch["x"], device)

        Ohat = gen(x)
        O = sanitizeGT(batch["y"], device)
        res = psnr(O ,Ohat)
        psnr_list.append(float(res))

        del O
        del Ohat

        if(test):
            return

    avg_psnr = sum(psnr_list)/len(psnr_list)
    print(f"Evaluation of {path} completed. PSNR score = {avg_psnr}")

