from __future__ import absolute_import

from torch import nn 


class Feature_Extraction(nn.Module):
    """
    (d,num_features,h,w)---->(d,num_features,h,w)
    """

    def __init__(self, num_feat):
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True) 
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
    
class AttentionModule(nn.Module):
   
    def __init__(self,num_features):
        self.num_features=num_features


    def forward(self,x):
        return x
        

class Generator(nn.Module):
    def __init__(self,num_frame,num_extr_blocks,num_ch_in,num_features):

        self.num_ch_in=num_ch_in
        self.num_frame=num_frame
        self.num_features=num_features

        #blocks
        self.first_conv=nn.Conv2d(num_ch_in, num_features, 3, 1, 1)    
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.feat_ex = nn.ModuleList([])
        for i in range(num_extr_blocks):
            self.feat_ex.append(Feature_Extraction(num_features))
        
        #prepare the features for the pyramid
        self.l1_to_l2=nn.Conv2d(num_features, num_features, 3, 2, 1, bias=True) #stride 2 to half the dimensions
        self.l2_to_l2=nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True) 
        self.l2_to_l3=nn.Conv2d(num_features, num_features, 3, 2, 1, bias=True) 
        self.l3_to_l3=nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True) 


    def forward(self,x):
        """
        (b,t,c,h,w) b is the batch size, t is the time dimension
        b = number of target frame with his neighboirs considered at time
        t = number of neighbors (including itself)

        """
        b, t, c, h, w = x.size()
        z=x.view(-1,c,h,w) #to do the 2d convolution 

        #L1 first layer of the pyramid
        #(d,3,h,w)---->(d,num_features,h,w)
        l1=self.lrelu(self.first_conv(z))
        for fe in enumerate(self.feat_ex):
            l1=fe(l1)
        l1=self.feat_ex(l1)
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