
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

@torch.no_grad()
def inference(conf, testing = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    #num_vid = conf["INFERENCE"].getint("video_number")
    path = conf["INFERENCE"]["save_path"]
    model = Generator(conf)
    model.load_state_dict(torch.load("trained_models/"+conf['TRAINING'].get("name_model")))

    data_loader = getDataLoader(conf, "train")

    model.eval()

    if not os.path.exists(path):
        os.makedirs(path)


    for sample in tqdm.tqdm(data_loader):
        s=torch.stack(sample["x"],dim=0)
        s=s.permute(1,0,4,2,3).to(device)

        ups= torch.nn.Upsample(size=(720, 1280), mode='bilinear', align_corners=None, recompute_scale_factor=None)
        upsampled = s[:,len(sample["x"])//2,:,:,:].cpu()
        upsampled = ups(upsampled)
        # MODELLO ! S ! ebbasta
        model.to(device)
        y=model(s)
        model = model.to("cpu")
        #S ! y 
        y=torch.nn.functional.interpolate(y, size=(180,320), mode='bilinear', align_corners=None, recompute_scale_factor=None)
  
        for i in range(conf['INFERENCE'].getint("n_updates")):
            torch.cuda.empty_cache()
            print("trying to swap stuff")
            s[:,model.center_frame_index,:,:,:]=y
            print("swapped!")
            del y
            model = model.to(device)
            y=model(s)
            model = model.to("cpu")
            y=torch.nn.functional.interpolate(y, size=(180,320), mode='bilinear', align_corners=None, recompute_scale_factor=None)
 
        y=y.to("cpu")
        y=ups(y)
        residual = torch.abs(y.detach() - upsampled)
        residual = residual.numpy()

        for i in range(y.shape[0]):
            frame = sample["referencePath"][i].split("/")[-1].split(".")[-2]
            video = sample["referencePath"][i].split("/")[-2]

            inf_name = f"vid{video}_{frame}_inf.png"
            res_name = f"vid{video}_{frame}_res.png"
            original_name = f"vid{video}_{frame}_orig.png"

            out = y[i,:,:,:].permute(1,2,0).detach().numpy() * 255
            singleres = residual[i,:,:,:].transpose((1,2,0)) * 255
            orig = upsampled[i,:,:,:].detach().numpy()
            orig = orig.transpose((1,2,0)) * 255
            cv2.imwrite(os.path.join(path, inf_name), out)
            cv2.imwrite(os.path.join(path, res_name), singleres)
            cv2.imwrite(os.path.join(path, original_name), orig)

        del s 
        del y
        torch.cuda.empty_cache()

        if(testing):
            break
