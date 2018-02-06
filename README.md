# bcnn

[![Build Status](https://travis-ci.org/jnbraun/bcnn.svg?branch=master)](https://travis-ci.org/jnbraun/bcnn/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

#### ***  Contributors are welcomed ! ***

bcnn (Bare CNN) is a minimalist implementation of Convolutional Neural Networks in plain C and Cuda.

It is aimed to be easy to build with a very limited number of dependencies (standalone if only used on CPU) and designed with 'hackability' in mind.

At the current state, it can run on CPU and Nvidia's GPU. CuDNN versions >= 5 (up to 7) are supported.

## Dependencies:
### CPU build
* Minimal build: no external dependency (only requires bip (image processing library) and bh (helpers library) already included).

* Build with Blas: requires a blas library (OpenBLAS is preferred).

### GPU build: 
Requires CUDA libraries (cudart, cublas, curand) and a GPU with compute capability 2.0 at least. CuDNN is optional but supported.

## Build:
Download or clone the repository:
```
git clone --recursive https://github.com/jnbraun/bcnn.git
```

### Linux:
Install cmake.

* User configuration: Depending on you build configuration, you may want to edit the following lines of ./CMakeLists.txt:
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
cmake --build ./
```

### Windows:
Use cmake to generate the project (choose x64 configuration if using CUDA lib), then build the solution.
Tested with msvc2010 and msvc2013 only.

## Features:

* Currently implemented layers: 
    - Convolution
    - Transposed convolution (aka Deconvolution)
    - Depthwise separable convolution
    - Fully-connected
    - Activation functions: relu, tanh, abs, ramp, softplus, leaky-relu, clamp.
    - Softmax
    - Max-pooling
    - Dropout
    - Batch normalization
* Learning algorithms: SGD, Adam.
* Online data augmentation (crop, rotation, distortion, flip)

## How to use it:

* Use the command line tool bcnn-cl with configuration file: see an example [here](https://github.com/jnbraun/bcnn/tree/master/examples/mnist_cl).

* Or use the static library and write your own code: see an example [there](https://github.com/jnbraun/bcnn/tree/master/examples/mnist).

## License:

Released under MIT license.
