from torch.utils.data import Dataset
import cv2
import os

# TODO: add transformations
class REDS_loader(Dataset):
    def __init__(self, conf):
        self.sharpdir =  os.path.join(conf["DATASET"]["root"], "train_sharp/train/train_sharp")
        self.blurdir = os.path.join(conf["DATASET"]["root"], "train_blur/train/train_blur")

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

        neighbors_images_blur = [cv2.imread(f"{self.blurdir}/{video}/{elem:08d}.png") for elem in window]
        target_sharp = cv2.imread(f"{self.sharpdir}/{video}/{frame}.png")

        return {"x":neighbors_images_blur, "y":target_sharp, "referencePath": f"{video}/{frame}.png"}

