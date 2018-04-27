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

#ifdef BCNN_USE_CUDA

#include "bcnn_avgpool_layer.h"
#include "bcnn_utils.h"

__global__ void _bcnn_forward_avgpool_layer_kernel(int sz, int c, int h, int w,
                                                    float *src,
                                                    float *dst) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= sz) {
        return;
    }
    int k = id % c;
    id /= c;
    int b = id;
    int idx = (k + c * b);
    for (int i = 0; i < w * h; ++i) {
        int offset = w * h * idx + i;
        dst[idx] += src[offset];
    }
    dst[idx] /= w * h;
}

int bcnn_forward_avgpool_layer_gpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                   bcnn_tensor *dst_tensor) {
    int sz = bcnn_tensor_get_size(dst_tensor);

    _bcnn_forward_avgpool_layer_kernel<<<bcnn_cuda_gridsize(sz),
                                         BCNN_CUDA_THREADS>>>(
        sz, src_tensor->c, src_tensor->h, src_tensor->w, 
        src_tensor->data_gpu, dst_tensor->data_gpu);
    bcnn_cuda_check(cudaPeekAtLastError());

    return BCNN_SUCCESS;
}

__global__ void _bcnn_backward_avgpool_layer_kernel(int sz, int c, int h, int w,
                                                    float *src_grad_data,
                                                    float *dst_grad_data) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= sz) {
        return;
    }
    int k = id % c;
    id /= c;
    int b = id;
    int idx = (k + c * b);
    for (int i = 0; i < w * h; ++i) {
        int offset = w * h * idx + i;
        src_grad_data[i] += dst_grad_data[idx] / (w * h);
    }
}

int bcnn_backward_avgpool_layer_gpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                    bcnn_tensor *dst_tensor) {
    int sz = bcnn_tensor_get_size(dst_tensor);

    _bcnn_backward_avgpool_layer_kernel<<<bcnn_cuda_gridsize(sz),
                                          BCNN_CUDA_THREADS>>>(
        sz, src_tensor->c, src_tensor->h, src_tensor->w,
        src_tensor->grad_data_gpu, dst_tensor->grad_data_gpu);
    bcnn_cuda_check(cudaPeekAtLastError());

    return BCNN_SUCCESS;
}

#endif