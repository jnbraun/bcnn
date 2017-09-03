# bcnn

[![Build Status](https://travis-ci.org/jnbraun/bcnn.svg?branch=master)](https://travis-ci.org/jnbraun/bcnn/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

#### ***  Contributors are welcomed ! ***

bcnn is a plain C / Cuda implementation of Convolutional Neural Networks (widely used for Deep Learning applications such as object detection and recognition in images).

It is aimed to be easy to build with a very limited number of dependencies (standalone if only used on CPU) and designed with 'hackability' in mind.

At the current state, it can run on CPU and Nvidia's GPU. CuDNNv5 is now supported.

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
ARCH= --gpu-architecture=compute_50 --gpu-code=compute_50
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
* Learning algorithms: SGD, Adam.
* Online data augmentation (crop, rotation, distortion)

## How to use it:

* Use the command line tool bcnn-cl with configuration file: see an example [here](https://github.com/jnbraun/bcnn/tree/master/examples/mnist_cl).

* Or use the static library and write your own code: see an example [there](https://github.com/jnbraun/bcnn/tree/master/examples/mnist).

## License:

Released under MIT license.
