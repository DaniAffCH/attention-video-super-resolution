from __future__ import absolute_import

from torch import nn 
import torch
from sr_utils.sr_utils import bilinear_upsample
from torchvision import ops
from torch.nn.modules.utils import _pair
class Feature_Extraction(nn.Module):            #we need residual for the vanishing of the gradient or can be replaced by pre-trained VGG
    """
    (d,num_features,h,w)---->(d,num_features,h,w)
    """

    def __init__(self, num_feat):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1) 
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x=self.conv2(self.relu(self.conv1(x)))
        return x
    
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
        for i in range(num_level): #each level has is weights
            self.first_conv.append(nn.Conv2d(2*num_features, num_features, 3, 1, 1))
            if(i==2):
                self.second_conv.append(nn.Conv2d(num_features, num_features, 3, 1, 1))
                self.feat_conv.append(nn.Conv2d(num_features, num_features, 3, 1, 1)) 
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
            if(level>1):
                upsampled_off=bilinear_upsample(offset,2)
                upsampled_feat=bilinear_upsample(aligned_feat,2)

            level-=1
        return aligned_feat


class AttentionModule(nn.Module):
   
    def __init__(self,num_features):
        super().__init__()
        self.num_features=num_features


    def forward(self,x):
        return x
        

class Generator(nn.Module):
    def __init__(self,num_frame,num_extr_blocks,num_ch_in,num_features):
        super().__init__()
        self.num_ch_in=num_ch_in
        self.num_frame=num_frame
        self.num_features=num_features
        self.num_extr_blocks=num_extr_blocks
        self.center_frame_index = num_frame // 2

        #blocks
        self.first_conv=nn.Conv2d(num_ch_in, num_features, 3, 1, 1)    
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.feat_ex = nn.ModuleList([])
        for i in range(num_extr_blocks):
            self.feat_ex.append(Feature_Extraction(num_features))
        
        #prepare the features for the pyramid
        self.l1_to_l2=nn.Conv2d(num_features, num_features, 3, 2, 1) #stride 2 to half the dimensions
        self.l2_to_l2=nn.Conv2d(num_features, num_features, 3, 1, 1) 
        self.l2_to_l3=nn.Conv2d(num_features, num_features, 3, 2, 1) 
        self.l3_to_l3=nn.Conv2d(num_features, num_features, 3, 1, 1) 

        self.align=Alignment(num_features)
        self.attn=AttentionModule(num_features)


    def forward(self,x):
        """
        (b,t,c,h,w) b is the batch size, t is the time dimension
        b = number of target frame with his neighboirs considered at time
        t = number of neighbors (including itself)

        """
        b, t, c, h, w = x.size()
        z=x.contiguous().view(-1,c,h,w) #to do the 2d convolution 
        #L1 first layer of the pyramid
        #(d,3,h,w)---->(d,num_features,h,w)
        l1=self.lrelu(self.first_conv(z))
        for l,fe in enumerate(self.feat_ex):
            l1=fe(l1)
        #L2
        l2=self.lrelu(self.l1_to_l2(l1))
        l2=self.lrelu(self.l2_to_l2(l2))
        #L3
        l3=self.lrelu(self.l2_to_l3(l2))
        l3=self.lrelu(self.l3_to_l3(l3))
        #turn back to 5 dimension with a pack of features for each frame
        l1 = l1.view(b, t, -1, h, w)       
        l2 = l2.view(b, t, -1, h//2, w//2) 
        l3 = l3.view(b, t, -1, h//4, w//4)
        #feature alignment
        #need 2 list, one with the features from  all the levels of the pyramid belonging to the central frame 
        #(the one we want to upgrade)
        #another for the neighbor
        central_frame_feature_list=[ 
            l1[:, self.center_frame_index, :, :, :].clone(),
            l2[:, self.center_frame_index, :, :, :].clone(),
            l3[:, self.center_frame_index, :, :, :].clone()
        ]
        aligned_feature_list=[]
        for i in range(t):            #align each frame features to the center frame features
            neighb_feature_list=[ 
                l1[:, i, :, :, :].clone(),
                l2[:, i, :, :, :].clone(),
                l3[:, i, :, :, :].clone()
            ]
            aligned_feature=self.align(central_frame_feature_list,neighb_feature_list)
            aligned_feature_list.append(aligned_feature)
        aligned_tensor=torch.stack(aligned_feature_list,dim=1)

        #fusion of the features using cross temporal and spatial attention information
        fused_feature=self.attn(aligned_tensor)

        #reconstruction phase



        return fused_feature