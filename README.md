# Similar Image Retrieval System (Example - Marvel MCU Movies)

### Objective
The overall goal of this project is train a set of models to determine when provided a new image (in this case a frame from one of the MCU movies), we seek to provide an output of the closest __similar images__ related to the queries image/scene. In other words, given an image and a library of images, retireve the closes K similar images to queried image. This objective is achieved by:
* leverage pre-trained models to extract features of images and index the nearest neighbor
* Retrieve similar images by calculating similarity distances between inquired image and images in the databse. 

![](https://cdn-images-1.medium.com/max/2000/1*DcfRFa1ShCK7SkoMC2dHfA.jpeg)

### Project Overview
For all Marvel Cinematic Universe (MCU) fans, and more boadly applied Machine Learning practictioners, this project seeks to utilize unsupervised learning and pretrained models to retrieve a list of similar images based on the inquired image. In short, we are building a unspervided deep learning Image Similarity Recommendation system. Moreover, described in the notebooks, the project can be used to tuned whatever image repository you have available. 

#### Key Steps
To summerize the processes involved in this image similarity and retrieval system, at the highest level we are using a pre-trained deep learning model to to extract features from a provided image library (in this case Marvel Cinematic Movies) into a list of numbers (array vector) describing each image. Then we will experiment with various distance functions to best calculate the similarities between a queried image and all the other feature vectors from the image library to determine the images that are most similar. 

Below are the key steps used in this project:
* Collect data and establish Image library _(in thes project we scrapped various YouTube videos)_
* Normalized, resize and preprocess images 
* Index image library and append meta data
* Compute image similarity score _(similarity measures and algorithms described in the below section)_
* Extract image features with pretrained model (e.g., CNN model like VGG50)
* Save compressed feature matrix from compiled model 
* Display predicted similary images from image library based off of a new image


#### Marvel Dataset
For the Image Processing, i utilized Marvel's trailers posted on YouTube. Refer to the image_processing file in the image folder on downloading and capturing images.Such as:
* Experiment 1:
  * [YouTube] [MCU Complete Recap](https://www.youtube.com/watch?v=4eMW0TKNlpQ)
  * [YouTube] [The Entire MCU Timeline Explained](https://www.youtube.com/watch?v=SY86xyG-hDY&t=1088s)
  * [Youtbe] [Marvel Cinematic Universe 10 Year Recap](https://www.youtube.com/watch?v=wYXav05fy4w)
* Experiment 2:
  * Went to all [YouTube] [Marevel Entertainment](https://www.youtube.com/user/MARVEL) channel and for each movie scraped images scenes as jpg files in each movie tile specific folder. Such as:
  
    ```
    - ./images/
      - antman1
      - antman2
      - avengers1
      - avengers2
    ```


### Algorithm & Experiments 

In order to perform the similar inmage search, below are some of the algorithms used to determine which K images in the database is _similar_ to queried image. 
* __Mean Squared Error__ - calculates the average squared differences (viz. errors) between images. The closer MSE is to 0, the more similar. 
* __Structural Similarity (SSIM) Index__
* __Locality Sensitive Hashing (LSH)__ - creates image feature hash table that computes the similarity probability and returns a relevance rank of images indexed from the image library
* various methods of __CNN distance feature extraction__, such as:Euclidean, Cosine, CityBlock, Manhattan and L2 regularization 

#### Transfer Learning using feature extraxtrion 
While experimenting, we experimented with various pre-trained deep learning architecture like ImageNet and VGG to generate features from images and similarity metrics. 

Below is an example output of the image similarity retrieval system:

![](https://github.com/naivelogic/Image_Similarity_in_Marvel_MCU/blob/master/dev/marvel_image_retrievel_sample._tony.png)

![](https://github.com/naivelogic/Image_Similarity_in_Marvel_MCU/blob/master/dev/marvel_image_retrievel_sample.png)

Just scrapping various trailer images of the MCU movies, we randomly selected a MCU movie image on google, in this case Guardians of the Galaxy (2014) and which retrieved similar images from the image library from that scene. Apparently the inquired image is not in the image library, however to solve that, we can just use the [x] [Duplicate Hash] notebook that uses the Locality Sensitive Hashing (LSH) function disussed above for identifying duplicative images. 

-----

## Results

#### Model Training Results

|   Model   | Input  Size |  Loss  | Accuracy | top 5  accuracy |  Date    |
|:---------:|:-----------:|:------:|:--------:|:---------------:|:--------:|
| ResNet34  |   224x224   | 1.011  |   0.723  |      0.935      | 10/17/19 |
| ResNet50  |   224x224   | 1.018x |   0.737  |      0.936      | 10/22/19 |
| ResNet18  |   224x224   |  x.xxx |   x.xxx  |      x.xxx      |          |
| Squeeznet |             |        |          |                 |          |



#### Image Similarity Results


| Distance Measure | mAP@5 | mAP@10 | mAP@20 | notes |
|------------------|-------|--------|--------|-------|
| Euclidean        |       |        |        |       |
| Cosine           |       |        |        |       |
| Manhattan        |       |        |        |       |

------

### Requirements
* Those new to python libraries, follow this guide to [install Keras + Tensorflow](https://keras.io/#installation) as both are needed if training a new model. 

If starting fresh:
* Download Anaconda and Python 3 - refer here for [setup assistance](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)
* conda install -c conda-forge tensorflow
* pip inesll keras
* clone repository

* For the Image Processing, i utilized Marvel's trailers posted on YouTube. Refer to the image_processing file in the image folder on downloading and capturing images. To do this you'll need to download `pytube` by running the following command in the terminal. [pytube documentation](https://python-pytube.readthedocs.io/en/latest/user/install.html)

```
$ pip install pytube

```

----
### Next Steps, TODOs, Outstanding Features
* [ ] TODO: Model and Similarity Metric Evaluations 
* [ ] Formalize experimental notebooks and scripts
* [ ] Feature: Web crawl using [Azure: Bing Image Search](https://azure.microsoft.com/en-us/services/cognitive-services/bing-image-search-api/) to find similar images. 



----
## Related Work & References
* [1] [Math Works - SSIM](https://www.mathworks.com/help/images/ref/ssim.html)
* [2] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). [Image quality assessment: from error visibility to structural similarity](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf). IEEE transactions on image processing, 13(4), 600–612.
* [3] [Locality Sensitive Hashing](https://santhoshhari.github.io/Locality-Sensitive-Hashing/) - Application of Locality Sensitive Hashing to Audio Fingerprinting
