from __future__ import absolute_import
from torch import nn

class ConvBlockBase(nn.Module):
    """
    (d,num_features,h,w)---->(d,num_features,h,w)
    """

    def __init__(self, num_feat, useResidual = True):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1) 
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_feat)
        self.bn2 = nn.BatchNorm2d(num_feat)
        self.res = useResidual


    def forward(self, x):
        tmp = self.conv1(x)
        tmp = nn.functional.leaky_relu(self.bn1(tmp))
        tmp = self.conv2(tmp)
        tmp = nn.functional.leaky_relu(self.bn2(tmp))
        return tmp + x if self.res else tmp