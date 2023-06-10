from __future__ import absolute_import
import torch
import cv2
from sr_utils.sr_utils import sanitizeInput, sanitizeGT
import os
from data.REDS_loader import REDS_loader
import albumentations as A
from model.generator import Generator
from data.REDS_loader import getDataLoader  
import numpy
import torchvision.transforms.functional
import tqdm

def inference(conf, testing = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    num_vid = conf["INFERENCE"].getint("video_number")
    path = conf["INFERENCE"]["save_path"]
    model = Generator(conf).to(device)
    model.load_state_dict(torch.load("trained_models/"+conf['TRAINING'].get("name_model")))

    data_loader = getDataLoader(conf, "train")

    if not os.path.exists(path):
        os.makedirs(path)


    for sample in tqdm.tqdm(data_loader):
        s=torch.stack(sample["x"],dim=0)
        s=s.permute(1,0,4,2,3).to(device)
        
        y=model(s)
        y=y.cpu()
        residual = torch.abs(y[0].permute(1,2,0).detach() - sample["x"][len(sample["x"])//2][0])
        residual = residual.numpy()

        frame = sample["referencePath"][0].split("/")[-1].split(".")[-2]
        video = sample["referencePath"][0].split("/")[-2]

        inf_name = f"vid{video}_{frame}_inf.png"
        res_name = f"vid{video}_{frame}_res.png"

        out = y[0].permute(1,2,0).detach().numpy() * 255
        residual = residual * 255

        cv2.imwrite(os.path.join(path, inf_name), out)
        cv2.imwrite(os.path.join(path, res_name), residual)

        del s 
        del y

        if(testing):
            break