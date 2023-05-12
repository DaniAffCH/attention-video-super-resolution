from torch.utils.data import Dataset
import cv2
import os
import albumentations as A
import torch

"""
    Methods
    -------
    __getitem__(idx):
        It takes an index (possibly provided from torch data loader) and returns a dictionary.
        - x represents the list of neighbors blurred images (including the target one)
        - y represents the sharp image (which refers to the target)
        - referencePath contains the description of the target processed i.e. the video it belongs to and the frame number  
"""
class REDS_loader(Dataset):
    def __init__(self, conf, transform):
        self.sharpdir =  os.path.join(conf["DATASET"]["root"], "train_sharp/train/train_sharp")
        self.blurdir = os.path.join(conf["DATASET"]["root"], "train_blur/train/train_blur")
        self.transform = transform

        maxsharp = max([int(i) for i in os.listdir(self.sharpdir)])
        maxblur = max([int(i) for i in os.listdir(self.blurdir)])

        assert(maxsharp == maxblur)

        self.neighborsWindow = conf["DEFAULT"].getint("n_neighbors")
        self.samples = [f"{el:03d}/{frame:08d}" for el in range(maxblur+1) for frame in range(self.neighborsWindow, 100-self.neighborsWindow )]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video, frame = self.samples[idx].split("/")
        window = list(range( int(frame) - self.neighborsWindow, int(frame) + self.neighborsWindow + 1))
        
        dic = {f"image{i}":cv2.imread(f"{self.blurdir}/{video}/{elem:08d}.png") for i, elem in enumerate(window)}
        dic["image"] = cv2.imread(f"{self.sharpdir}/{video}/{frame}.png")
        trans = self.transform(**dic)

        target_sharp = trans["image"]
        del trans["image"]
        neighbors_images_blur = list(trans.values())

        return {"x":neighbors_images_blur, "y":target_sharp, "referencePath": f"{video}/{frame}.png"}

def getDataLoader(conf):
    targetdict = {f"image{k}":"image" for k in range(conf["DEFAULT"].getint("n_neighbors") * 2 + 1)}

    transform = A.Compose([
        A.ToFloat(max_value=255, always_apply=True, p=1.0),
        A.Rotate(limit = 60, p=0.3),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomCrop(conf['DEFAULT'].getint("image_height"), conf['DEFAULT'].getint("image_width"))
    ], additional_targets = targetdict)

    rl = REDS_loader(conf, transform)

    data_loader = torch.utils.data.DataLoader(
        rl,
        batch_size=conf['DEFAULT'].getint("batch_size")
    )

    return data_loader
