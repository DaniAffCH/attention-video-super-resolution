# attention-video-super-resolution
This repository contains a deep learning model that achieves video super-resolution tasks by leveraging cross-attention and dynamic filtering. The model predicts each frame by taking into account not only the low resolution itself but also its neighboring frames. In order to capture the dependencies among nearby frames, the models use Vision Transformers.

--------------- TODO: foto piu sghecia ---------------  

 Additionally, the entire network is designed to be scalable, enabling customization based on the number of neighbors and features used, so as to handle the performance-inference time tradeoff.
 
## Architecture
The architecture aims to predict the residual that enhances image sharpness. The network consists of three principal blocks:
- The **Feature Extractor** is a series of convolutional layers that extract the desired number of features from the input.
- The **Align Module** uses a pyramidal representation of frames to learn offsets at different scales. These offsets are used for deformable convolutions. The align module combines the features of each frame at the same level of the pyramid and applies deformable convolutions to obtain aligned features. The result is upsampled until the final set of aligned features is obtained.
- The **Spatiotemporal Attention** Layer performs feature fusion by first computing the correlation between the center frame and all its neighbors. Then, it applies a convolution to fuse the features. Finally, it computes the spatial attention for each frame using pooling operations and multiplies the attention map with the frame-wise features.
- The actual residual is computed by another convolutional stage and added to the original upsampled image.

--------------- TODO: foto architettura ---------------

## Results 

## Installation

## Usage

## Acknowledgment
