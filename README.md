# Image Colorization with GANs

This project implements an image colorization model using Generative Adversarial Networks (GANs) in PyTorch. It allows you to input grayscale images and generate colored versions by training on a dataset of color images. The model uses a U-Net-based generator and a PatchGAN-based discriminator to achieve this task. This project can be run easily using Google Colab.

## Model Overview

The architecture consists of two main components:
1. **Generator (U-Net)**
2. **Discriminator (PatchGAN)**

### Generator

The generator is a U-Net, which is a convolutional neural network designed for image-to-image tasks. It takes a grayscale image (single channel) as input and generates the corresponding color information (two channels representing the chrominance components in the L*a*b color space).

- **U-Net Architecture**: 
    - A series of downsampling and upsampling layers with skip connections. 
    - The downsampling path extracts features from the input, while the upsampling path reconstructs the output image, using skip connections to combine high-resolution features from the encoder with those in the decoder.
    - Pretrained ResNet-18 is used as the encoder to leverage transfer learning, improving the generator’s capability.

### Discriminator

The discriminator is a PatchGAN, a type of GAN discriminator that classifies patches of an image as real or fake rather than the entire image. This helps the discriminator focus on local image details like textures and edges, which is crucial for tasks like colorization.

- **PatchGAN Architecture**:
    - A convolutional neural network that processes both the real and generated (fake) images.
    - Instead of producing a single scalar value as output, it outputs a matrix where each value corresponds to a patch of the input image, determining whether the patch is real or fake.

### Loss Functions

- **Generator Loss**: 
  - GAN loss (BCE loss) ensures that the generator produces images that are indistinguishable from real images to the discriminator.
  - L1 Loss ensures the generated images are close to the ground truth in pixel space, helping retain high-level content.
  
- **Discriminator Loss**: 
  - GAN loss (BCE loss) is used to determine how well the discriminator can distinguish between real and generated images.

## Dataset

We use the [COCO dataset](https://cocodataset.org/) for training and validation. This dataset contains diverse images, making it a good fit for colorization tasks.

The dataset is directly accessed using the `fastai` library, which simplifies downloading and managing the data.

```python
from fastai.data.external import untar_data, URLs

coco_path = untar_data(URLs.COCO_SAMPLE)
coco_path = str(coco_path) + "/train_sample"
```

## How to Use

### Run on Google Colab

You can run the entire project on Google Colab. Just upload the notebook and start training the model. The code is designed to work seamlessly in Colab, so you don't need to install any dependencies manually.

## Results

After training, you can visualize the results by inputting grayscale images and checking the generated colorized output.

## Conclusion

This project demonstrates the power of GANs in image colorization tasks. The U-Net generator combined with the PatchGAN discriminator provides a robust approach for generating high-quality color images from grayscale inputs.

## Future Work
  - Extend the model to higher-resolution images.
  - Improve colorization quality by experimenting with deeper networks or different GAN architectures.
