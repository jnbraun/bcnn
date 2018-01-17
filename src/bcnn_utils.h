/*
* Copyright (c) 2016 Jean-Noel Braun.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#ifndef BH_UTILS_H
#define BH_UTILS_H

/* Cuda include */
#ifdef BCNN_USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#ifdef BCNN_USE_CUDNN
#include <cudnn.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int state;
    float r;
} bcnn_gauss_gen;

float bcnn_rng_gaussian(bcnn_gauss_gen *g);

void get_binary_row(float *row, unsigned int *bin_row, int size);

void get_binary_col(float *col, unsigned int *bin_col, int n, int k);

void get_binary_col_unrolled(float *col, unsigned int *b_col, int n, int k);


#ifdef BCNN_USE_CUDA

cublasHandle_t bcnn_cublas_handle();

#define bcnn_cuda_check(RET) {                                                  \
    if ((RET) != cudaSuccess) {                                                 \
        fprintf(stderr, "[ERROR] [CUDA] %s\n", cudaGetErrorString((RET)));      \
        exit((RET));                                                            \
    }                                                                           \
}
#define bcnn_cublas_check(RET) {                                                \
    if ((RET) != CUBLAS_STATUS_SUCCESS) {                                       \
        fprintf(stderr, "[ERROR] [CUBLAS] %d\n", (int)(RET));                   \
        exit((RET));                                                            \
    }                                                                           \
}

#define bcnn_curand_check(RET) {                                                \
    if ((RET) != CURAND_STATUS_SUCCESS) {                                       \
        fprintf(stderr, "[ERROR] [CURAND] %d\n", (int)(RET));                   \
        exit((RET));                                                            \
    }                                                                           \
}

dim3 bcnn_cuda_gridsize(unsigned int n);

int *bcnn_cuda_malloc_i32(int n);

float *bcnn_cuda_malloc_f32(int n);

float *bcnn_cuda_memcpy_f32(float *x, int n);

void bcnn_cuda_fill_with_random(float *x_gpu, int n);

void bcnn_cuda_free(void *x_gpu);

void bcnn_cuda_memcpy_host2dev(float *x_gpu, float *x, int n);

void bcnn_cuda_memcpy_dev2host(float *x_gpu, float *x, int n);

void bcnn_cuda_set_device(int id);

#ifdef BCNN_USE_CUDNN
#define bcnn_cudnn_check(RET) {                                                 \
    if ((RET) != CUDNN_STATUS_SUCCESS) {                                        \
        fprintf(stderr, "[ERROR] [CUDNN] %s\n", cudnnGetErrorString((RET)));    \
        exit((RET));                                                            \
    }                                                                           \
}
cudnnHandle_t bcnn_cudnn_handle();
#endif // BCNN_USE_CUDNN

#endif //BCNN_USE_CUDA

#ifdef __cplusplus
}
#endif

#endif //BH_UTILS_H