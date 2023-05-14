from __future__ import absolute_import
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision import ops
from sr_utils.sr_utils import bilinear_upsample

import torch

class ConvNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation=True):
        if(activation):
            super(ConvNorm, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )
        else:
            super(ConvNorm, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
                nn.BatchNorm2d(out_channels),
            )

class ConvBlockBase(nn.Module):
    """
    (d,num_features,h,w)---->(d,num_features,h,w)
    """

    def __init__(self, num_feat, useResidual = True):
        super().__init__()
        self.s = nn.Sequential(
            ConvNorm(num_feat, num_feat, 3),
            ConvNorm(num_feat, num_feat, 3)
        )
        self.res = useResidual


    def forward(self, x):
        tmp = self.s(x)
        return tmp + x if self.res else tmp

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, useResidual=True):
        super().__init__()
        self.middle_channels = in_channels + (out_channels-in_channels)//2
        self.res = useResidual and in_channels == out_channels and stride == 1

        self.s = nn.Sequential(
            ConvNorm(in_channels, self.middle_channels, 3),
            ConvNorm(self.middle_channels, self.middle_channels, 3, stride),
            ConvNorm(self.middle_channels, out_channels, 3)
        )

    def forward(self,x):
        tmp = self.s(x)
        return tmp + x if self.res else tmp
    
class DeformConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super().__init__()
        self.kernel_size=_pair(kernel_size)
        self.groups=self.kernel_size[0]*self.kernel_size[1]
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))

        self.conv_for_offset=nn.Conv2d(in_channels,2*self.kernel_size[0]*self.kernel_size[1], 3, 1, 1) 

    def forward(self,input,offset):
        offset=self.conv_for_offset(offset)
        out=ops.deform_conv2d(input,offset,self.weight,None,stride=1,padding=1,dilation=1)
        return out
    

class Alignment(nn.Module):
    def __init__(self,num_features,num_level=3):   
        super().__init__()
        #need a conv for each level
        self.first_conv=nn.ModuleList([])
        self.second_conv=nn.ModuleList([])
        self.lrelu=nn.LeakyReLU()
        self.deform_conv=nn.ModuleList([])
        self.feat_conv=nn.ModuleList([])
        self.final_conv_offset=nn.Conv2d(2*num_features, num_features, 3, 1, 1)
        self.final_deform_conv=DeformConvBlock(num_features,num_features,kernel_size=3)
        for i in range(num_level): #each level has its weights
            self.first_conv.append(nn.Conv2d(2*num_features, num_features, 3, 1, 1))
            if(i==2):
                self.second_conv.append(nn.Conv2d(num_features, num_features, 3, 1, 1))
            else: #double of the feature because the concatenation of the previous level offset
                self.second_conv.append(nn.Conv2d(2*num_features, num_features, 3, 1, 1))
                self.feat_conv.append(nn.Conv2d(2*num_features, num_features, 3, 1, 1))  #we use also the aligned feature of the previous level to predict the next
            self.deform_conv.append(DeformConvBlock(num_features,num_features,kernel_size=3))
            


    def forward(self,central_frame_feature_list,neighb_feature_list):
        #first we have to concatenate the offset starting from the lowest level of the pyramid
        level=3
        upsampled_off=None
        upsampled_feat=None
        while(level):
            offset=torch.cat([central_frame_feature_list[level-1],neighb_feature_list[level-1]],dim=1) #now we have 2*num_features channels
            offset=self.lrelu(self.first_conv[level-1](offset)) #learn the offset, it says were the kernel must be applied for the convolution, as if the feature were aligned (the kernels goes to the same part of the object)
            if level==3:
                offset=self.second_conv[level-1](offset)
            else: 
                offset=self.second_conv[level-1](torch.cat([offset,upsampled_off],dim=1))
            
            #deformable convolution DConv(F t+i, ∆P lt+i ) 
            aligned_feat=self.deform_conv[level-1](neighb_feature_list[level-1],offset)

            if (level<3):
                aligned_feat=self.feat_conv[level-1](torch.cat([aligned_feat,upsampled_feat],dim=1))

            if(level>1):
                upsampled_off=bilinear_upsample(offset,2)
                upsampled_feat=bilinear_upsample(aligned_feat,2)
            level-=1
        #last cascading
        final_offset=torch.cat([aligned_feat,central_frame_feature_list[0]],dim=1)
        final_offset=self.final_conv_offset(final_offset)
        aligned_feat=self.final_deform_conv(aligned_feat,final_offset)
        return aligned_feat

class AttentionModule(nn.Module):
   
    def __init__(self,num_features,center_frame_index,num_frames):
        super().__init__()
        self.center_frame_index=center_frame_index
        self.query=nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.keys=nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.sigmoid=nn.Sigmoid()
        self.fusion=nn.Conv2d(num_features*num_frames,num_features,1,1,0)
        self.spatial_attention=nn.Conv2d(num_features*num_frames,num_features,1,1,0)


    def forward(self,aligned_feat):
        b, t, c, h, w = aligned_feat.size()
        #temporal correlation(attention) for each couple of neighbors. Multihead attention would be more complex but useless(not so much temporal distance)
        query=self.query(aligned_feat[:,self.center_frame_index,:,:,:].clone())
        keys=self.keys(aligned_feat.view(-1,c,h,w))
        keys=keys.view(b,t,-1,h,w)

        correlation=[]
        for i in range(t):
            key=keys[:,i,:,:,:]
            corr=key*query         #cross product attention
            correlation.append(corr) #re-add the temporal dimension to do the fusion after this
            
        correlation=torch.cat(correlation,dim=1)
        correlation=self.sigmoid(correlation)      #temporal attention map 
        aligned_feat=aligned_feat.view(b,t*c,h,w)*correlation   #pixel wise mult to the original features
        #fusion
        fused_feature=self.fusion(aligned_feat)

        #spatial attention
        sp_attn=self.spatial_attention(aligned_feat)
        sp_attn=self.sigmoid(sp_attn)             #mask
        important_features=fused_feature*sp_attn
        return important_features