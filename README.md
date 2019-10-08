# Similar Image Retrieval System (Example - Marvel MCU Movies)

### Objective
The overall goal of this project is train a set of models to determine when provided a new image (in this case a frame from one of the MCU movies), we seek to provide an output of the closest __similar images__ related to the queries image/scene. In other words, given an image and a library of images, retireve the closes K similar images to queried image. 


![](https://cdn-images-1.medium.com/max/2000/1*DcfRFa1ShCK7SkoMC2dHfA.jpeg)

__Dataset__ - For the Image Processing, i utilized Marvel's trailers posted on YouTube. Refer to the image_processing file in the image folder on downloading and capturing images.Such as:
* [YouTube] [MCU Complete Recap](https://www.youtube.com/watch?v=4eMW0TKNlpQ)
* [YouTube] [The Entire MCU Timeline Explained](https://www.youtube.com/watch?v=SY86xyG-hDY&t=1088s)
* [Youtbe] [Marvel Cinematic Universe 10 Year Recap](https://www.youtube.com/watch?v=wYXav05fy4w)


### Algorithm & Experiments 

In order to perform the similar inmage search, below are some of the algorithms used to determine which K images in the database is _similar_ to queried image. 
* __Structural Similarity (SSIM) Index__
* __Locality Sensitive Hashing (LSH)__
* various methods of __CNN distance feature extraction__, such as:Euclidean, Cosine, CityBlock, Manhattan and L2 regularization 

#### Structural Similarity (SSIM) Index
__Structureal Similarity (SSIM) Index__ is an image quality metric that assesses the visual impact of three charachteristics[1]:
1. __Luminance__
2. __Contrast__
3. __Structure__

https://towardsdatascience.com/automatic-image-quality-assessment-in-python-391a6be52c11

[Detailing the SSIM Algorithm]
SSIM provides an image quality index based on the computations of the luminance, contrast and strucutre terms. 


#### Locality Sensitive Hashing (LSH)
While SSIM is to calculate the similarities among documents. __Locality Sensitive Hashing (LSH)__ is to find __near duplicates__ pairs of images _(viz. approximate nearest neighbor)_ in the library. Essentially, when 2 images are compared, LSH computes local relationships between the two imates on the feature map. The closer the features, the more likely the images would result in similar hashes (viz. reduce representation of data).  As descirbed in [3] below is the high level steps when cmoputing the LSH Algorithm:

![](https://miro.medium.com/max/952/1*27nQOTC79yfh5lzmL06Ieg.png)



#### Transfer Learning using feature extraxtrion 
[ ] TODO

![](https://github.com/naivelogic/Image_Similarity_in_Marvel_MCU/blob/master/dev/marvel_image_retrievel_sample.png)


### Requirements
* Those new to python libraries, follow this guide to [install Keras + Tensorflow](https://keras.io/#installation) as both are needed if training a new model. 

* For the Image Processing, i utilized Marvel's trailers posted on YouTube. Refer to the image_processing file in the image folder on downloading and capturing images. To do this you'll need to download `pytube` by running the following command in the terminal. [pytube documentation](https://python-pytube.readthedocs.io/en/latest/user/install.html)

```
$ pip install pytube

```




### Related Work
* [1] [Math Works - SSIM](https://www.mathworks.com/help/images/ref/ssim.html)
* [2] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). [Image quality assessment: from error visibility to structural similarity](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf). IEEE transactions on image processing, 13(4), 600–612.
* [3] [Locality Sensitive Hashing](https://santhoshhari.github.io/Locality-Sensitive-Hashing/) - Application of Locality Sensitive Hashing to Audio Fingerprinting


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
