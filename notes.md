# Various notes from Research and Experiments 

----
## Models and Algorithms

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



-----
### Renaiming Images in Folder

```python
import os

#folder which we want to rename
folder = './images/ironman2/'

for root, dirs, files in os.walk(folder):
    for i,f in enumerate(files):
        
        #original name of images
        absname = os.path.join(root, f)
        
        #name we want to replace with
        name = str(int(i)+1)+'.jpg'
        
        newname = os.path.join(root, name)
        
        #renaming images in provided folder
        os.rename(absname, newname)
```

### Various ways for reading images and file paths

```python
directory='../images/test'
items=[]
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        items.append(filename)
```


```python
path = '/home/redne/git_repos/marvel_movies/images/avengers4/'
image_files = [(path + f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
image_files
```


```python
import os
IMAGE_DIR = '/home/redne/git_repos/POE/mnt/poeBlob/train_images/'

os.chdir(IMAGE_DIR)
os.getcwd()

image_files = os.listdir()
print(len(image_files))
```


```python
import os, shutil
original_dataset_dir = '/home/redne/.kaggle/dogs_vs_cats/train/'

base_dir = '/home/redne/.kaggle/dogs_vs_cats/cats_and_dogs_small' 
#os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train') 
#os.mkdir(train_dir) 

train_cats_dir = os.path.join(train_dir, 'cats') 
#os.mkdir(train_cats_dir)


fnames = ['cat.{}.jpg'.format(i) for i in range(1000)] 
for fname in fnames: 
    src = os.path.join(original_dataset_dir, fname) 
    dst = os.path.join(train_cats_dir, fname) 
    shutil.copyfile(src, dst)
```


----
# Deep Learning Models

### ResNet Keras Parameters 

```python
keras.applications.resnet.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```
https://keras.io/applications/#build-inceptionv3-over-a-custom-input-tensor

<h3 id="arguments_3">Arguments</h3>
<ul>
<li>include_top: whether to include the fully-connected layer at the top of the network.</li>
<li>weights: one of <code>None</code> (random initialization) or <code>'imagenet'</code> (pre-training on ImageNet).</li>
<li>input_tensor: optional Keras tensor (i.e. output of <code>layers.Input()</code>) to use as image input for the model.</li>
<li>input_shape: optional shape tuple, only to be specified
    if <code>include_top</code> is <code>False</code> (otherwise the input shape
    has to be <code>(224, 224, 3)</code> (with <code>'channels_last'</code> data format)
    or <code>(3, 224, 224)</code> (with <code>'channels_first'</code> data format).
    It should have exactly 3 inputs channels,
    and width and height should be no smaller than 32.
    E.g. <code>(200, 200, 3)</code> would be one valid value.</li>
<li>pooling: Optional pooling mode for feature extraction
    when <code>include_top</code> is <code>False</code>.<ul>
<li><code>None</code> means that the output of the model will be
    the 4D tensor output of the
    last convolutional block.</li>
<li><code>'avg'</code> means that global average pooling
    will be applied to the output of the
    last convolutional block, and thus
    the output of the model will be a 2D tensor.</li>
<li><code>'max'</code> means that global max pooling will
    be applied.</li>
</ul>
</li>
<li>classes: optional number of classes to classify images 
    into, only to be specified if <code>include_top</code> is <code>True</code>, and 
    if no <code>weights</code> argument is specified.</li>
</ul>


-----
# Model Notes

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
