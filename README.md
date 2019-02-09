# BCNN

[![Build Status](https://travis-ci.org/jnbraun/bcnn.svg?branch=master)](https://travis-ci.org/jnbraun/bcnn/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Introduction
BCNN (Bare Convolutional Neural Networks) is a minimalist framework designed to prototype, train and deploy convolutional neural networks for embedded computer vision applications. 

### Features
* Written in C99. Clean C API designed to be used in C and C++ codebase. **No** Python.
* Lightweight: the minimal build requires **no** external dependency.
* Modular: Can leverage a Blas library such as OpenBLAS on CPU. Can also run on Nvidia's GPU. CuDNN is supported to offer maximal training speed.
* Fast: Optimized inference speed using AVX and ARM Neon SIMD instructions. Particularly efficient on ARMv8 architecture.
* Flexible design: Supports multi inputs / outputs / branches. Provide the commonly used operators to build state-of-the-art CNN architectures (ResNet, DenseNet, MobileNet, Yolo ...)
* [Command line tool](https://github.com/jnbraun/bcnn/tree/generic_layer/examples/mnist_cl) to train / evaluate models via simple configuration file.
* Online data augmentation via a fast image processing library (usable as standalone module)

## Build:
Download or clone the repository:
```
git clone https://github.com/jnbraun/bcnn.git
```

You need to have cmake installed in order to build the library.

### Dependencies (optional):
#### CPU build
* Minimal build: no external dependency.
* Build with Blas: requires a blas library (OpenBLAS is preferred).

#### GPU build: 
Requires CUDA libraries (cudart, cublas, curand) and a GPU with compute capability 2.0 at least. CuDNN is optional but supported.

* User configuration: Depending on you build configuration, you may want to edit the following lines of the CMakeLists.txt:
```
# User configuration settings
option(USE_AVX "Build with AVX instructions" ON)
option(USE_CUDA "Build with CUDA libraries" OFF)
option(USE_CUDNN "Build with CuDNN library" OFF)
option(USE_BLAS "Build with BLAS library" ON)
# Building examples
option(BUILD_EXAMPLES "Build examples ON" ON)
```

* [Optional] When building with CUDA and/or CuDNN, you may need to adjust the following line depending on the compute capability of your GPU:
```
set(CUDA_NVCC_FLAGS "-arch=compute_50; -code=sm_50; -lcuda -lcudart -lcublas -lcurand")
```

* Build:
```
cd <bcnn-root-dir>
mkdir build
cd build/
cmake ../
make
```

## How to use it:

* Use the command line tool bcnn-cl with configuration file: see an example [here](https://github.com/jnbraun/bcnn/tree/master/examples/mnist_cl).

* Or use the static library and write your own code: see an example [there](https://github.com/jnbraun/bcnn/tree/master/examples/mnist).

## License:

Released under MIT license.
