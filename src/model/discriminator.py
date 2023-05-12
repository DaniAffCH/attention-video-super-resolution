from __future__ import absolute_import

from torch import nn 
class Discriminator(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.inputSize = (conf['DEFAULT'].getint("image_width"), conf['DEFAULT'].getint("image_height"))
        

    def forward(self,x):
        pass