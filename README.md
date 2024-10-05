# Image Colorization using GAN

This project implements a Generative Adversarial Network (GAN) for image colorization, leveraging the COCO dataset. The model takes grayscale images as input and generates colorized images using a combination of a U-Net generator and a Patch Discriminator.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Procedure](#training-procedure)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [Future Work](#future-work)

## Introduction

Colorizing black and white images is a challenging problem in computer vision. This project utilizes GANs to generate realistic color images from grayscale inputs. The GAN consists of a generator that predicts color information and a discriminator that evaluates the authenticity of the generated images.

## Dependencies

To run this project, you will need the following Python packages:

- `torch`
- `torchvision`
- `numpy`
- `PIL`
- `skimage`
- `matplotlib`
- `fastai`

You can install the required packages in your Google Colab notebook using:

```python
!pip install fastai==2.4
```

## Dataset

This model uses a subset of the COCO dataset, specifically the sample images for training and validation. The dataset is automatically downloaded and extracted using the `fastai` library.

```python
from fastai.data.external import untar_data, URLs
coco_path = untar_data(URLs.COCO_SAMPLE)
```

The training set consists of 8,000 images, while the validation set includes 2,000 images, chosen randomly from the available samples.

## Model Architecture

The model consists of two main components:

1. **Generator**: A U-Net architecture (`SimpleUNet`) that takes a grayscale image as input and generates the corresponding color channels (a and b in the LAB color space).

2. **Discriminator**: A Patch Discriminator (`PatchDiscriminator`) that evaluates the authenticity of generated images by distinguishing between real and fake images.

## Training Procedure

The training process involves alternating between updating the discriminator and generator based on their respective losses. The discriminator is trained to correctly classify real and fake images, while the generator aims to fool the discriminator into classifying its outputs as real.

The training function is defined as follows:

```python
def train_gan(generator, discriminator, train_dl, opt_G, opt_D, criterion_GAN, criterion_L1, device, epochs=20, lambda_L1=100):
    ...
```

You can adjust the number of epochs and other hyperparameters to optimize performance.

## Usage

After training the model, you can visualize the results by retrieving specific images from the data loader. The following functions are provided:

- `get_nth_image(data_loader, n)`: Fetches the \(n\)-th image from the data loader.
- `visualize_nth_image(generator, data_loader, n, device)`: Visualizes the grayscale input, real color image, and the generated color image.

To visualize the \(n\)-th image, use:

```python
visualize_nth_image(generator, train_dl, n, device)
```

## Results

The model aims to produce visually appealing colorized images from grayscale inputs. During training, the loss values for both the generator and discriminator are logged for monitoring performance.

## Acknowledgments

- This project is inspired by advancements in image processing and deep learning techniques.
- The COCO dataset is a widely used dataset in the computer vision community, providing a rich source of images for training models.

## Future Work
  - Extend the model to higher-resolution images.
  - Improve colorization quality by experimenting with deeper networks or different GAN architectures.
