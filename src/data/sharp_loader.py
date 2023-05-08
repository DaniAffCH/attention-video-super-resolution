from torch.utils.data import Dataset
import cv2
import os

# TODO: generalize it
class SharpLoader(Dataset):
    def __init__(self, conf):
        self.dir =  os.path.join(conf["DATASET"]["root"], "train_sharp/train/train_sharp")

        maxel = max([int(i) for i in os.listdir(self.dir)])
        self.neighborsWindow = conf["DEFAULT"].getint("n_neighbors")
        self.samples = [f"{el:03d}/{frame:08d}" for el in range(maxel+1) for frame in range(self.neighborsWindow, 100-self.neighborsWindow )]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video, frame = self.samples[idx].split("/")
        window = list(range( int(frame) - self.neighborsWindow, int(frame) + self.neighborsWindow + 1))

        neighbors_images = [cv2.imread(f"{self.dir}/{video}/{elem:08d}.png") for elem in window]

        return {"images":neighbors_images, "referencePath": f"{video}/{frame}.png"}

