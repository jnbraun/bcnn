# bcnn

[![Build Status](https://travis-ci.org/jnbraun/bcnn.svg?branch=master)](https://travis-ci.org/jnbraun/bcnn/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

bcnn is a pure C implementation of Convolutional Neural Networks (widely used in Deep Learning applications such as object detection and recognition in images).
It is aimed to be lightweight, concise and easy to install.
It supports both CPU and GPU (CUDA) computation.

## Dependencies:
### Minimal build (CPU with or without SSE2 acceleration):
No external dependency (only requires bip (image processing library) and bh (helpers library) already included).

### GPU build: 
Requires CUDA libraries (cudart, cublas, curand) and a GPU with compute capability 2.0 at least.

## Build:
- On linux systems: clone the repository and simply type: 
```bash
make
```
You may want to edit the following lines of the Makefile at your convenience:
```
CUDA=1
CUDNN=0
DEBUG=0
USE_SSE2=1
CUDA_PATH=/usr/local/cuda
ARCH= --gpu-architecture=compute_20 --gpu-code=compute_20
```

- On windows systems: Use cmake to generate the project (choose x64 configuration if using CUDA lib), then build the solution.
Only tested with msvc2010 and 2013.

## Features:

* Currently implemented layers: 
	- Convolutional
	- Deconvolutional
	- Fully-connected
	- Activation functions: relu, tanh, abs, ramp, softplus, leaky-relu, clamp.
	- Softmax
	- Max-pooling
	- Dropout
	- Batch normalization
	- Layers concatenation
* Learning by SGD.
* Online data augmentation (crop, rotation, distortion)

## License:

Released under MIT license.
