# BCNN

[![Build Status](https://travis-ci.org/jnbraun/bcnn.svg?branch=master)](https://travis-ci.org/jnbraun/bcnn/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Introduction
BCNN (Bare Convolutional Neural Networks) is a minimalist framework designed to prototype, train and deploy convolutional neural networks for embedded computer vision applications. 

### Features
* Written in C99. Clean C API designed to be integrated in C or C++ codebase. **No** Python.
* Lightweight: the minimal build requires **no** external dependency.
* Modular: Can leverage a Blas library such as OpenBLAS on CPU. Can also run on Nvidia's GPU. CuDNN is supported to offer maximal training speed.
* Fast: Optimized inference speed using AVX and ARM Neon SIMD instructions. Particularly efficient on ARMv8 architecture.
* Flexible: Supports multi inputs / outputs / branches. Provides the commonly used operators to build state-of-the-art CNN architectures (ResNet, DenseNet, MobileNet, [Yolo](https://github.com/jnbraun/bcnn/tree/generic_layer/examples/yolo) ...)
* [Command line tool](https://github.com/jnbraun/bcnn/tree/generic_layer/examples/mnist_cl) to train / evaluate models via simple configuration file.
* Online data augmentation via bip: a fast image processing library (usable as standalone module).
* (Experimental) Model converters from Caffe->bcnn and bcnn->TensorFlow Lite.

## Getting started
Download or clone the repository:
```
git clone https://github.com/jnbraun/bcnn.git
```

You need to have cmake installed in order to build the library.

### [Optional] Dependencies 
#### CPU
* Minimal build: no external dependency.
* Build with Blas: requires a blas library (OpenBLAS is preferred).

#### GPU 
Requires CUDA libraries (cudart, cublas, curand) and a GPU with compute capability 2.0 at least. CuDNN is optional but supported.

### Build
* User configuration: Depending on you system, you may want to edit the following lines of the CMakeLists.txt:
```
# User configuration settings
option(USE_AVX "Build with AVX instructions" ON)
option(USE_CUDA "Build with CUDA libraries" OFF)
option(USE_CUDNN "Build with CuDNN library" OFF)
option(USE_BLAS "Build with BLAS library" ON)
option(USE_NEON "Build with Neon instructions" OFF)
```

* [Optional] When building with CUDA and / or CuDNN, you may need to adjust the following line depending on the compute capability of your GPU:
```
# Uncomment the proper line according to the system cuda arch
set(CUDA_ARCH 
    #"-gencode arch=compute_30,code=sm_30;"
    #"-gencode arch=compute_35,code=sm_35;"
    "-gencode arch=compute_50,code=sm_50;"
    "-gencode arch=compute_50,code=compute_50;"
    "-gencode arch=compute_52,code=sm_52;"
    #"-gencode arch=compute_60,code=sm_60;"
    #"-gencode arch=compute_61,code=sm_61;"
)
```

* Build:
```
cd path/to/bcnn
mkdir build
cd build/
cmake ../
make
```

## How to use it

* Use the command line tool bcnn-cl with configuration file: see an example [here](https://github.com/jnbraun/bcnn/tree/master/examples/mnist_cl).

* Or use the static library and write your own code: see an example [there](https://github.com/jnbraun/bcnn/tree/master/examples/mnist).

## License

Released under MIT license.
