from torch.utils.data import Dataset
import cv2
import os

# TODO: generalize it
class SharpLoader(Dataset):
    def __init__(self, conf):
        self.dir =  os.path.join(conf["DATASET"]["root"], "train_sharp/train/train_sharp")
        self.samples = [f"{el}/{frame}" for el in os.listdir(self.dir) for frame in os.listdir(os.path.join(self.dir, el))]
        self.neighborsWindow = conf["DEFAULT"].getint("n_neighbors")

    def __len__(self):
        return len(os.listdir(self.dir))

    def __getitem__(self, idx):
        video, frame = self.samples[idx].split("/")
        frame, _ = frame.split(".")
        window = list(range( max(0, int(frame) - self.neighborsWindow ), min(99, int(frame) + self.neighborsWindow ) + 1))
        neighbors_images = [cv2.imread(f"{self.dir}/{video}/{idx:08d}.png") for idx in window]

        return neighbors_images

