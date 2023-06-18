# attention-video-super-resolution
This repository contains a deep learning model that achieves video super resolution task by leveraging the cross-attention and dynamic filtering. The model predicts each frame by taking into account not only the low resolution itself, but also its neighboring frames. In order to capture the dependencies among nearby frames, the models uses Vision Transformers.

--------------- TODO: foto piu sghecia ---------------  

 Additionally, the entire network is designed to be scalable, enabling customization based on the number of neighbors and features used, so as to handle the performance-inference time tradeoff.
 
## Architecture
The architecture is formed by three principal blocks: the feature extractor, the align module and the spatio-temporal attention layer.
- The feature extractor is a cascade of convolutional layers, which provides the desired number of features.
- The align module uses a pyramidal representation of frames in order to learn the offsets at different scales and to use these for the deformable convolutions.
Such offsets are obtained by combining the features of each frame in the same level of the pyramid, and then applying a deformable convolution. The result is upsampled until we get the final set of aligned features.
- The final spatio-temporal layer performs a feature fusion by first computing the correlation between the center frame and all its neighbors. Afterwards, it performs a convolution to fuse the features, and finally it computes the spatial attention for each frame using pooling operations, and multiplying the attention map for the features frame-wise.
At the end, the residual is computed by another convolutional stage and added to the original upsampled image.

--------------- TODO: foto architettura ---------------

## Results 

## Installation

## Usage

## Acknowledgment
