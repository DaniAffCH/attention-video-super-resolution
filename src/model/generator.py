from __future__ import absolute_import

from torch import nn 
import torch
from model.common_layers import ConvBlockBase, Alignment, AttentionModule
import time
class Generator(nn.Module):
    def __init__(self,conf):
        super().__init__()
        self.conf = conf
        self.num_ch_in=3
        self.num_frames=2*conf["DEFAULT"].getint("n_neighbors")+1
        self.num_features=conf["GENERATOR"].getint("num_features")
        self.num_extr_blocks=conf["GENERATOR"].getint("convolutional_stages")
        self.center_frame_index = self.num_frames // 2

        #blocks
        #deconv it's like to learn the inverse of a blur convolution
        #self.deblur=nn.ConvTranspose2d(self.num_ch_in, self.num_ch_in, 3, stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        self.first_conv=nn.Conv2d(self.num_ch_in, self.num_features, 3, 1, 1)    
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.feat_ex = nn.ModuleList([])
        for _ in range(self.num_extr_blocks):
            self.feat_ex.append(ConvBlockBase(self.num_features))
        
        #prepare the features for the pyramid
        self.l1_to_l2=nn.Conv2d(self.num_features, self.num_features, 3, 2, 1) #stride 2 to half the dimensions
        self.l2_to_l2=nn.Conv2d(self.num_features, self.num_features, 3, 1, 1) 
        self.l2_to_l3=nn.Conv2d(self.num_features, self.num_features, 3, 2, 1) 
        self.l3_to_l3=nn.Conv2d(self.num_features, self.num_features, 3, 1, 1) 

        self.align=Alignment(self.num_features)
        self.attn=AttentionModule(self.num_features,self.center_frame_index,self.num_frames)

        self.finalconv = ConvBlockBase(self.num_features)

        self.upsampler = torch.nn.Upsample(size=(1024,576), mode='bilinear', align_corners=None, recompute_scale_factor=None)

        self.restore=nn.Conv2d(self.num_features,3,1,1,0)


    def forward(self,x):
        """
        (b,t,c,h,w) b is the batch size, t is the time dimension
        b = number of target frame with his neighboirs considered at time
        t = number of neighbors (including itself)

        """
        st = et = 0
        b, t, c, h, w = x.size()
        z=x.contiguous().view(-1,c,h,w) #to do the 2d convolution 
        #L1 first layer of the pyramid
        #(d,3,h,w)---->(d,num_features,h,w)
        #z=self.lrelu(self.deblur(z))
        
        if self.conf['DEFAULT'].getboolean("time_debugging"):
            st = time.time()

        l1=self.lrelu(self.first_conv(z))
        for fe in self.feat_ex:
            l1=fe(l1)

        if self.conf['DEFAULT'].getboolean("time_debugging"):
            et = time.time()
            print(f"[TIME] Generator feature extractor: {et-st} s")
            st = time.time()

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
            aligned_feature=self.align(central_frame_feature_list,neighb_feature_list) # bottneck
            aligned_feature_list.append(aligned_feature)
        aligned_tensor=torch.stack(aligned_feature_list,dim=1) #now this are the tokens for the attention layer

        if self.conf['DEFAULT'].getboolean("time_debugging"):
            et = time.time()
            print(f"[TIME] Generator feature alignment: {et-st} s")
            st = time.time()

        #fusion of the features using cross temporal and spatial attention information
        fused_feature=self.attn(aligned_tensor)

        if self.conf['DEFAULT'].getboolean("time_debugging"):
            et = time.time()
            print(f"[TIME] Generator attention: {et-st} s")
            st = time.time()

        #reconstruction phase
        fused_feature = self.finalconv(fused_feature)
        residual=self.restore(fused_feature) #this has to be pixel-shuffled in order to get bigger

        upsampled_x=x[:,self.center_frame_index,:,:,:]

        upsampled_x = self.upsampler(upsampled_x)
        residual = self.upsampler(residual)

        image_hq=upsampled_x+residual

        if self.conf['DEFAULT'].getboolean("time_debugging"):
            et = time.time()
            print(f"[TIME] Generator image restoring: {et-st} s")
            st = time.time()


        return image_hq
