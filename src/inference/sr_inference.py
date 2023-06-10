from __future__ import absolute_import
import torch
import cv2
from sr_utils.sr_utils import sanitizeInput, sanitizeGT
import os
from data.REDS_loader import REDS_loader
import albumentations as A
from model.generator import Generator
import numpy
import torchvision.transforms.functional

def inference(conf):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    num_vid = conf["INFERENCE"].getint("video_number")
    path = conf["INFERENCE"]["save_path"]
    model = Generator(conf).to(device)
    model.load_state_dict(torch.load("trained_models/"+conf['TRAINING'].get("name_model")))

    if not os.path.exists(path):
        os.makedirs(path)

    model.eval()

    loader= REDS_loader(conf,A.Compose([]),"train")

    for i in range(100):
        img=loader. __getitem__(num_vid*100+i)["x"]

        for element,_ in enumerate(img):
            img[element]=torch.Tensor(img[element])
            img[element]=img[element].unsqueeze(1)
            img[element]=img[element].permute(1,3,2,0)
            img[element]=torchvision.transforms.functional.crop(img[element], 0, 0, conf['DEFAULT'].getint("image_height"), conf['DEFAULT'].getint("image_width")) 
            
        x = torch.stack(img,dim=0)
        x=x.permute(1,0,2,3,4).to(device)

        Ohat = model(x)
        name = f'vid{num_vid}_{i}.png'

        filename=os.path.join(path,name)
        Ohat=Ohat.squeeze().permute(2,1,0)
        cv2.imwrite(filename, numpy.array(Ohat.to("cpu").detach().numpy()))
        del Ohat
    return 
