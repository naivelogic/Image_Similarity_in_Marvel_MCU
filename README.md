# Similar Image Search

### Objective
The overall goal of this project is to __provide a query image and find the closest image(s) in our database__. In other words, given an image and a library of images, retireve the closes K similar images to queried image. 


### Algorithm & Experiments 

In order to perform the similar inmage search, below are some of the algorithms used to determine which K images in the database is _similar_ to queried image. 
* Structured SIMilarity (SSIM) Index [1]


#### Requirements
Those new to python libraries, follow this guide to [install Keras + Tensorflow](https://keras.io/#installation) as both are needed if training a new model. 


#### Sources
* [1] Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, ["Image quality assessment: From error visibility to structural similarity,"](https://ece.uwaterloo.ca/~z70wang/publications/ssim.html) IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, Apr. 2004.


-----
# Notes

### Regional Maximum activations of convolutions (R-MAC)

source: [End-to-end Learning of Deep Visual Representations for Image Retrieval](https://arxiv.org/pdf/1610.07940.pdf) - A. Gordo et al.

__Regional Maximum Activations of Convolutions (R-MAC)__ - computes based descriptiors of several images regions at different scales that are sum-aggregated into a compact feature vector of fixed length, and is therefore moderately robust to scale and translation. 

In other words, _R-MAC_ is a global inmage representaiton used for image retrieval.
* Uses a _"fully convolutional" CNN_ as a powerful local feature extractor
    * Advantage: encodes images at high resolutions without distoring aspect ratio
    * uses CNN pretrained on ImageNet on AlexNet and VGG16 network architectures
    * R-MAC pipeline integrated in a single CNN 
* __Local features__ are then _max-pooled_ across several multi-scale overlapping regions, obtained from a _rigid grid_ covering the image
* __Normalization Pipeline__ on _Region-level features_ are independetly `l2`-normalized + whitened with PCA + `l2`-normalized again. Region descriptors are _sum-aggregated_ and `l2`-normalized again. 
* __Global image represenation__ is obtained in a compact vector whose size is 2`k` dimensions - depending on network architecture
