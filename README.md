# Progressive Growing of GANs for Improved Quality, Stability, and Variation

This is PyTorch implementation of ProgressiveGAN described in paper ["Progressive Growing of GANs for Improved Quality, Stability, and Variation"](https://arxiv.org/abs/1710.10196).

Work is in progress. Weight normalization still don't work well.

# Usage

## Config

Use `options` class for set up model before training.

* exp_name - model name
* batch - batch size
* latent - size of latent space vector
* isize - final generating image size
* device_ids - GPU ids. Use list for initialize
* device - use GPU or CPU for training
* data_path - path of dataset
* epochs - number of epochs
* lr_d - lerning rate of discriminator
* lr_g - lerning rate of generator
* lr_decay_epoch - []

## Training

To begin trainig use:

```sh
python train.py
```

## Runing

There is no ability to explicitly use trained network.

# Example of generating cats

![Training process](https://github.com/ArMaxik/Progressive_growing_of_GANs-Pytorch_implementation/blob/master/illustrations/training.gif?raw=true)

![Generated example](https://github.com/ArMaxik/Progressive_growing_of_GANs-Pytorch_implementation/blob/master/illustrations/result.png?raw=true)

# Compatability

* Python 3.7.3
* PyTorch 1.7.1
* CUDA 10.1
* CUDNN 7.6.3

# Acknowledgement
* https://github.com/nashory/pggan-pytorch