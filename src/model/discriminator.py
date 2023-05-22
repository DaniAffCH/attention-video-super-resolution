from __future__ import absolute_import

from torch import nn 
from model.common_layers import VGGBlock
import json
import time
class Discriminator(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.FMsize = (conf['DEFAULT'].getint("image_width"), conf['DEFAULT'].getint("image_height"))

        self.structure = json.loads(conf['DISCRIMINATOR']['FE_structure'])
        modules = list()
        self.channels = 3

        for el in self.structure:
            modules.append(VGGBlock(self.channels, el["out"], el["stride"]))
            self.channels = el["out"]
            self.FMsize = (self.FMsize[0]//el["stride"], self.FMsize[1]//el["stride"]) 

        self.featureExtractor = nn.ModuleList(modules)
        self.fc1 = nn.Linear(self.channels * self.FMsize[0] * self.FMsize[1], 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self,x):
        st = et = 0

        if self.conf['DEFAULT'].getboolean("time_debugging"):
            st = time.time()

        for l in range(len(self.featureExtractor)):
            x = self.featureExtractor[l](x)

        if self.conf['DEFAULT'].getboolean("time_debugging"):
            et = time.time()
            print(f"[TIME] Discriminator feature extractor: {et-st} s")
            st = time.time()

        x = x.reshape(x.size(0), -1)
        x = nn.functional.leaky_relu(self.fc1(x))
        x = self.fc2(x)

        if self.conf['DEFAULT'].getboolean("time_debugging"):
            et = time.time()
            print(f"[TIME] Discriminator fully connected: {et-st} s")
            st = time.time()

        return x