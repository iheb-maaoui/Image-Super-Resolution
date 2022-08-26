# Single Image Super Resolution

## Project Description

The goal of this project is exploit the power of Generative Adversarial Networks in order to upscale a low resolution image by a factor of 4 in order to transform it into a high resolution image. This task is not an easy task espacially if our images have a very low resolution which complicate recovering photo-realistic textures from such kind of images.

I will implement a Super Resolution Generative Adversarial Network which upscale images with the shape (128,128,3) to the shape of (512,512,3).

## Super Resolution Generative Adversarial Network

### Components

The idea of GANs is that we want to make 2 neural networks compete against each other in the hope that this competition will push them to excel. In our case it is composed of : 

* A Generator: 
We use  16 blocks deep ResNet that Takes a low resolution image, pass it through a deep convolutional networks whose output will be upscaled using two Upscale layers before outputting a high resolution image at the very end. Before the traning stage, it will output random images with (512,512,3) as shape.

* A Discriminator:

Takes a high resolution image as input, it can be either a fake image from the generator or a real image from the training set as input, and must guess whether the input image is fake or real.

### Training 

During training, the generator and the discriminator have opposite goals: the discriminator tries to tell fake images from real images, while the generator tries to produce images that look real enough to trick the discriminator.

Each training iteration is divided into two phases:

Training the discriminator (Binary classifier which detect real images) using the binary cross-entropy loss. on a batch of real and fake images having respectively 1 and 0 as labels.
Backpropagation only optimizes the weights of the discriminator during this phase (We don't want to train the generator).

In the second phase, we train the generator. We first use it to produce another batch of fake images, and once again the discriminator is used to tell whether the images are fake or real. This time we do not add real images in the batch, and all the labels are set to 1 (real): in other words, we want the generator to produce images that the discriminator will (wrongly) believe to be real which generates the competition between the 2 components.

Intuition: The better the discriminator gets at guessing fake images, the more it encourages the generator to be more creative in creating better photo-realistic images.


##### Used Loss function

The loss used to train the discriminator is the classic binary cross-entropy loss. For the training of the generator we use a weighted sum between:

* An adversarial loss which is basically a crossentropy loss which caracterize the ability of the generator to accuratly classify pixels of high resolution generated image (0 or 1 classification) = BCE(generated_HR, ground_truth_HR).

* A content loss which represents a Mean squared Error loss calculated on feature maps of the VGG network, which are
more invariant to changes in pixel space = MSE(feature_maps_generated_HR,feature maps_Ground_truth_HR)

Perceptual loss = Content loss + 0.001 Adversarial loss