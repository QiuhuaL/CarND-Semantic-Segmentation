# Semantic Segmentation
### Introductionn 
In this project, we are to label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
The folloqing python packages are used for the project: 
 - [Python 3](https://www.python.org/)
 - [TensorFlow with GPU](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

##### Dataset
The Kitti Road dataset (http://www.cvlibs.net/datasets/kitti/eval_road.php) was downloaded from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

#### Architecture
The Fully Convolutional Network implemented in this project uses the architecture described in [Long et al.](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)  It is based on the pre-trained VGG model, and the fully connected layers was transformed nto fully convolution layers.  It combines coarse, high layer information with fine, low layers information. 

It first fuses the predictions from the convolution layer on top of VGG layer 4 with the predications computed on convolutiona layer on top of layer 7, by adding a 2* sampling layer and summing both predictions together.  The fusion is continued by fusing predictions from pool layer 3 and the results from the first step, and the final upsampling layer made the predictions back to the image.

### Implementation
The fully convolutional network was built in the code `main.py` . It implements the modules to load the pre-trained vgg model in the function `load_vgg`. The different layers to learn the features of the images were built in the function `layers`. The built network was optimized by minimizing the cross-entropy loss in the function `optimize`, and the network was trained with the function `train_nn`, where the loss of the network was printed while the network is training.     


The network was trained using Adam optimizer.  The parameters chosen are:
 - `epochs = 80`
 - `batch_size = 4`
 - `keep_prob = 0.4`
 - `learning_rate = 0.0001`

To run the project, use the following command from the terminal:
```
python main.py
```

### Results
[//]: # (Image References)
[image1]: ./images/um_000032.png
[image2]: ./images/uu_000015.png
[image3]: ./images/uu_000099.png
[image4]: ./images/umm_000077.png

Example sementic segmentation results on the road recognitions are shown below.  It shows that the architecture that combines multi-level resolution layers successfully find the road from the images at the pixel level. 

![sample][image1]
![sample][image2]
![sample][image3]
![sample][image4]


