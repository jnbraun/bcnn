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
#include <bh/bh.h>
#include <bh/bh_string.h>
#include <bh/bh_error.h>

#include "bcnn/bcnn.h"

int bcnn_get_tensor_size(bcnn_tensor *tensor)
{
    return tensor->w * tensor->h * tensor->c * tensor->b;
}

float bcnn_rng_gaussian(bcnn_gauss_gen *g)
{
    float v1, v2, s, m;
    
    if (g->state) {
        g->state = 0;
        return g->r;
    }
    else {
        do {
            v1 = 2 * (float)rand() / RAND_MAX - 1;
            v2 = 2 * (float)rand() / RAND_MAX - 1;
            s = v1 * v1 + v2 * v2;
        }
        while (s >= 1.0f || s == 0.0f);
        g->state = 1;
        m = sqrtf(-2.0f * logf(s) / s);
        g->r = v2 * m;
        return v1 * m;
    }
}

void get_binary_row(float *row, uint32_t *bin_row, int size)
{
    int i, j;
    uint32_t rvalue, sign;
    for (i = 0; i < size; i += BITS_IN_UINT32) {
        rvalue=0;
        for (j = 0;j < BITS_IN_UINT32; ++j) {
            sign = (row[i+j]>=0);
            rvalue |= (sign << j);
        }
        bin_row[i / BITS_IN_UINT32] = rvalue;
    }
}

void get_binary_col(float *col, uint32_t *bin_col, int n, int k)
{           
    int x, y, b;
    uint32_t rvalue, sign;
    for (y = 0; y < (n / BITS_IN_UINT32); y++) {
        for (x = 0; x < k; ++x) {          
            rvalue=0;    
            for (b=0; b < BITS_IN_UINT32; ++b){
                sign = (col[(y * BITS_IN_UINT32 + b) * k + x]>=0); 
                rvalue |= (sign << b);
            }
            bin_col[y * k + x] = rvalue;
        }
    }    
}

void get_binary_col_unrolled(float* col, uint32_t * b_col, int n, int k)
{        
    int y, b, x;
    float *col_0, *col_1, *col_2, *col_3;
    uint32_t *y_col_pt = NULL, *pnter = NULL;
    //register uint32_t rvalue0,rvalue1, rvalue2, rvalue3;
    /*register uint32_t sign0, sign1, sign2, sign3, sign4, sign5, sign6, sign7,
          sign8, sign9, sign10, sign11, sign12, sign13, sign14, sign15;*/

    for (y = 0; y < (n / BITS_IN_UINT32); y++) {
      y_col_pt = &b_col[y * k];
      for (x = 0; x < k; x += 4) {          
        register uint32_t rvalue0 = 0, rvalue1 = 0, rvalue2 = 0, rvalue3 = 0;
           
        for(b=0; b<BITS_IN_UINT32; b+=4) {
          register uint32_t sign0, sign1, sign2, sign3, sign4, sign5, sign6, sign7,
          sign8, sign9, sign10, sign11, sign12, sign13, sign14, sign15;

          col_0 = &col[(y*BITS_IN_UINT32+b)*k + x];
          col_1 = &col[(y*BITS_IN_UINT32+b+1)*k + x];
          col_2 = &col[(y*BITS_IN_UINT32+b+2)*k + x];
          col_3 = &col[(y*BITS_IN_UINT32+b+3)*k + x];

          sign0 = (*col_0>=0);          
          sign1 = (*col_1>=0);          
          sign2 = (*col_2>=0);          
          sign3 = (*col_3>=0);          
         
          BIT_SET(rvalue0, b, sign0);
          BIT_SET(rvalue0, (b+1), sign1);
          BIT_SET(rvalue0, (b+2), sign2);
          BIT_SET(rvalue0, (b+3), sign3);

          sign4 = (*(col_0+1)>=0);          
          sign5 = (*(col_1+1)>=0);          
          sign6 = (*(col_2+1)>=0);          
          sign7 = (*(col_3+1)>=0);          
         
          BIT_SET(rvalue1, b, sign4);
          BIT_SET(rvalue1, (b+1), sign5);
          BIT_SET(rvalue1, (b+2), sign6);
          BIT_SET(rvalue1, (b+3), sign7);

          sign8 = (*(col_0+2)>=0);          
          sign9 = (*(col_1+2)>=0);          
          sign10 = (*(col_2+2)>=0);          
          sign11 = (*(col_3+2)>=0);          
         
          BIT_SET(rvalue2, b, sign8);
          BIT_SET(rvalue2, (b+1), sign9);
          BIT_SET(rvalue2, (b+2), sign10);
          BIT_SET(rvalue2, (b+3), sign11);

          sign12 = (*(col_0+3)>=0);          
          sign13 = (*(col_1+3)>=0);          
          sign14 = (*(col_2+3)>=0);          
          sign15 = (*(col_3+3)>=0);          
         
          BIT_SET(rvalue3, b, sign12);
          BIT_SET(rvalue3, (b+1), sign13);
          BIT_SET(rvalue3, (b+2), sign14);
          BIT_SET(rvalue3, (b+3), sign15);
        }
        pnter = &y_col_pt[x];
        *pnter = rvalue0;   
        *(pnter+1) = rvalue1;        
        *(pnter+2) = rvalue2;        
        *(pnter+3) = rvalue3;        
      }
    }     
}


#ifdef BCNN_USE_CUDA
#ifdef BCNN_USE_CUDNN
cudnnHandle_t bcnn_cudnn_handle()
{
    static int init = 0;
    static cudnnHandle_t handle;
    if (!init) {
        cudnnCreate(&handle);
        init = 1;
    }
    return handle;
}
#endif

cublasHandle_t bcnn_cublas_handle()
{
    static int init = 0;
    static cublasHandle_t handle;
    if (!init) {
        cublasCreate(&handle);
        init = 1;
    }
    return handle;
}


dim3 bcnn_cuda_gridsize(unsigned int n)
{
    unsigned int k = (n - 1) / (BCNN_CUDA_THREADS)+1;
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


int *bcnn_cuda_malloc_i32(int n)
{
    int *x_gpu;
    size_t size = sizeof(int)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);

    bcnn_cuda_check(status);
    return x_gpu;
}

float *bcnn_cuda_malloc_f32(int n)
{
    float *x_gpu;
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);

    bcnn_cuda_check(status);
    return x_gpu;
}


float *bcnn_cuda_memcpy_f32(float *x, int n)
{
    float *x_gpu;
    size_t size = sizeof(float)* n;

    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    bcnn_cuda_check(status);

    if (x) {
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        bcnn_cuda_check(status);
    }

    if (!x_gpu) {
        //bh_error("Cuda malloc failed", BCNN_CUDA_FAILED_ALLOC);
        fprintf(stderr, "[ERROR] Cuda malloc failed\n");
    }

    return x_gpu;
}

void bcnn_cuda_fill_with_random(float *x_gpu, int n)
{
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

void bcnn_cuda_free(void *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    bcnn_cuda_check(status);
}

void bcnn_cuda_memcpy_host2dev(float *x_gpu, float *x, int n)
{
    size_t size = sizeof(float)* n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    bcnn_cuda_check(status);
}

void bcnn_cuda_memcpy_dev2host(float *x_gpu, float *x, int n)
{
    size_t size = sizeof(float)* n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    bcnn_cuda_check(status);
}

void bcnn_cuda_set_device(int id)
{
    bcnn_cuda_check(cudaSetDevice(id));
}

#endif