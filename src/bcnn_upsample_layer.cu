/*
* Copyright (c) 2016-present Jean-Noel Braun.
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

#include "bcnn_mat.h"
#include "bcnn_upsample_layer.h"
#include "bcnn_utils.h"

__global__ void bcnn_forward_upsample_cuda_kernel(size_t dst_sz, float *src,
                                                  int w, int h, int c, int n,
                                                  int size, float *dst) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= dst_sz) {
        return;
    }
    int dst_idx = i;
    int dst_w = i % (w * size);
    i = i / (w * size);
    int dst_h = i % (h * size);
    i = i / (h * size);
    int dst_c = i % c;
    i = i / c;
    int b = i % n;
    int src_w = dst_w / size;
    int src_h = dst_h / size;
    int src_c = dst_c;

    int src_idx = b * w * h * c + src_c * w * h + src_h * w + src_w;

    dst[dst_idx] += src[src_idx];
}

void bcnn_forward_upsample_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_upsample_param *param = (bcnn_upsample_param *)node->param;
    bcnn_cuda_fill_f32(bcnn_tensor_size(dst_tensor), 0, dst_tensor->data_gpu,
                       1);
    size_t size = src_tensor->w * src_tensor->h * src_tensor->c *
                  src_tensor->n * param->size * param->size;
    bcnn_forward_upsample_cuda_kernel<<<bcnn_cuda_blocks(size),
                                        BCNN_CUDA_THREADS>>>(
        size, src_tensor->data_gpu, src_tensor->w, src_tensor->h, src_tensor->c,
        src_tensor->n, param->size, dst_tensor->data_gpu);
    bcnn_cuda_check(cudaPeekAtLastError());
    return;
}

__global__ void bcnn_backward_upsample_cuda_kernel(size_t dst_sz, float *src,
                                                   int w, int h, int c, int n,
                                                   int size, float *dst) {
    size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= dst_sz) {
        return;
    }
    int dst_idx = i;
    int dst_w = i % (w * size);
    i = i / (w * size);
    int dst_h = i % (h * size);
    i = i / (h * size);
    int dst_c = i % c;
    i = i / c;
    int b = i % n;
    int in_w = dst_w / size;
    int in_h = dst_h / size;
    int in_c = dst_c;
    int src_idx = b * w * h * c + in_c * w * h + in_h * w + in_w;
    src[src_idx] += dst[dst_idx];
}

void bcnn_backward_upsample_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_upsample_param *param = (bcnn_upsample_param *)node->param;
    size_t size = src_tensor->w * src_tensor->h * src_tensor->c *
                  src_tensor->n * param->size * param->size;
    bcnn_backward_upsample_cuda_kernel<<<bcnn_cuda_blocks(size),
                                         BCNN_CUDA_THREADS>>>(
        size, src_tensor->grad_data_gpu, src_tensor->w, src_tensor->h,
        src_tensor->c, src_tensor->n, param->size, dst_tensor->grad_data_gpu);
    bcnn_cuda_check(cudaPeekAtLastError());
    return;
}

#endif  // BCNN_USE_CUDA