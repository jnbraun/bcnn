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

#include "bcnn/bcnn.h"
#include "bcnn_utils.h"
#include "bh_log.h"

#include <bh/bh.h>
#include <bh/bh_error.h>
#include <bh/bh_string.h>

float bcnn_rng_gaussian(bcnn_gauss_gen *g) {
    float v1, v2, s, m;

    if (g->state) {
        g->state = 0;
        return g->r;
    } else {
        do {
            v1 = 2 * (float)rand() / RAND_MAX - 1;
            v2 = 2 * (float)rand() / RAND_MAX - 1;
            s = v1 * v1 + v2 * v2;
        } while (s >= 1.0f || s == 0.0f);
        g->state = 1;
        m = sqrtf(-2.0f * logf(s) / s);
        g->r = v2 * m;
        return v1 * m;
    }
}

#ifdef BCNN_USE_CUDA
#ifdef BCNN_USE_CUDNN
cudnnHandle_t bcnn_cudnn_handle() {
    static int init = 0;
    static cudnnHandle_t handle;
    if (!init) {
        cudnnCreate(&handle);
        init = 1;
    }
    return handle;
}
#endif

cublasHandle_t bcnn_cublas_handle() {
    static int init = 0;
    static cublasHandle_t handle;
    if (!init) {
        cublasCreate(&handle);
        init = 1;
    }
    return handle;
}

dim3 bcnn_cuda_gridsize(unsigned int n) {
    unsigned int k = (n - 1) / (BCNN_CUDA_THREADS) + 1;
    unsigned int x = k;
    unsigned int y = 1;
    dim3 d;

    if (x > 65535) {
        x = (unsigned int)ceil(sqrt((float)k));
        y = (n - 1) / (x * (BCNN_CUDA_THREADS)) + 1;
    }

    d.x = x;
    d.y = y;
    d.z = 1;

    return d;
}

int *bcnn_cuda_malloc_i32(int n) {
    int *x_gpu;
    size_t size = sizeof(int) * n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);

    bcnn_cuda_check(status);
    return x_gpu;
}

float *bcnn_cuda_malloc_f32(int n) {
    float *x_gpu;
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);

    bcnn_cuda_check(status);
    return x_gpu;
}

float *bcnn_cuda_memcpy_f32(float *x, int n) {
    float *x_gpu;
    size_t size = sizeof(float) * n;

    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    bcnn_cuda_check(status);
    if (x) {
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        bcnn_cuda_check(status);
    }

    if (!x_gpu) {
        // bh_error("Cuda malloc failed", BCNN_CUDA_FAILED_ALLOC);
        fprintf(stderr, "[ERROR] Cuda malloc failed\n");
    }

    return x_gpu;
}

void bcnn_cuda_memcpy_f32_noalloc(float *x, float *x_gpu, int n) {
    size_t size = sizeof(float) * n;

    bh_check(x_gpu != NULL, "Invalid pointer to gpu memory");

    if (x) {
        cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        bcnn_cuda_check(status);
    }
}

void bcnn_cuda_fill_with_random(float *x_gpu, int n) {
    static curandGenerator_t gen;
    static int init = 0;
    if (!init) {
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, time(0));
        init = 1;
    }
    curandGenerateUniform(gen, x_gpu, n);
    bcnn_cuda_check(cudaPeekAtLastError());
}

void bcnn_cuda_free(void *x_gpu) {
    cudaError_t status = cudaFree(x_gpu);
    bcnn_cuda_check(status);
}

void bcnn_cuda_memcpy_host2dev(float *x_gpu, float *x, int n) {
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    bcnn_cuda_check(status);
}

void bcnn_cuda_memcpy_dev2host(float *x_gpu, float *x, int n) {
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    bcnn_cuda_check(status);
}

void bcnn_cuda_set_device(int id) { bcnn_cuda_check(cudaSetDevice(id)); }

#endif