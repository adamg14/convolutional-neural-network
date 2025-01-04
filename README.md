# Convolutional Neural Network

## Description

A python implimentation of a convolution neural network for classification of images into label groups. A convolutional neural network is a specialised neural network for identification
of internala patterns within images regardless of attributes like reflection. This results in a higher accuracy than a stanard neural network which looks at an image as a singular data point.

## Project Structure Breakdown

-[convolutional-nn.py](convolutional-nn.py) - The cifar10 image dataset imported from keras, split into training and testing data. Image preprocessing is applied normalising the pixel data.
The set of labels (the classes for classification) is defined. The CNN architecture is defined with the following layers:
  - A convolutional layer - defining the amount of filters, the sample size of the filters, the activation function used (relu), and the shape of the data going into the next layer.
  - A max-pooling layer - pooling (aggregating) 2x2 samples with a stride of 2.
  - These two layers are repeated shrinking by a factor of 2.
  - The layers are then flattened into a scalar and passed into a densly connected layer, similar to a traditional neural network. This layer has 10 output nodes. Using the relu
    activation layer each representing a probability distribution for the probability that the input image belongs to that class.
- The model is then compiled using the adam optimiser and cross entropy loss function. The model is then trained using the partitioned training data
- using the ImageDataGenerator method from tensorflow.keras, training images are transformed to test how the model deals with classification of images that has been augmented.
  The convolutional characteristics of the neural network means the accuracy of the model should be mostly unaffected by this.

-[pretrained_model.py](pretrained_model.py) - This script connects a pretrained convolutional model to a custom densely connected layers, as pre-trained convolutional models are available
with high accuracy without the need to be trained locally. Data partitioned into training, testing and validation. Data preprocessing is applied to the image data to ensure they are uniform shape.
The images are then shuffled and placed into batches. The convolutional base of the neural network is imported from keras and training on this layers is frozen, the one in use is 
the MobileNet V2 developed by Google. A global pooling layer and a dense prediction layer with an output node is added to the model. Once the model's architure has been defined, 
the model is then compiled, trained and evaluated. Finally, the model is saved so re-training is not needed when it is imported by another script [dogs_vs_cats.h5](dogs_vs_cats.h5).

## Dependencies

This project uses python verison 3.10, make sure this installed on your machine, then used (either by setting it as your default python version
or setting a virtual environment) to install the packages and run the scripts. [Python Download Link](https://www.python.org/downloads/)
Use the pip python package manager to install the following dependencies for the project:

```bash
pip3.10 install numpy
pip3.10 install matplotlib
pip3.10 install tensorflow
```

## Usage

```bash
python3.10 convolutional-nn.py
```

```bash
python3.10 pretrained_models.py
```


