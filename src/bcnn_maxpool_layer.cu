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

#include "bcnn_maxpool_layer.h"
#include "bcnn_utils.h"

__global__ void bcnn_forward_maxpool_layer_kernel(int n, int in_h, int in_w,
                                                  int in_c, int stride,
                                                  int size, float *input,
                                                  float *output, int *indexes) {
    int h = (in_h - 1) / stride + 1;
    int w = (in_w - 1) / stride + 1;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) {
        return;
    }

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int out_index = j + w * (i + h * (k + c * b));
    float max = -INFINITY;
    int max_i = -1;
    int l, m;
    for (l = 0; l < size; ++l) {
        for (m = 0; m < size; ++m) {
            int cur_h = i * stride + l;
            int cur_w = j * stride + m;
            int index = cur_w + in_w * (cur_h + in_h * (k + b * in_c));
            int valid =
                (cur_h >= 0 && cur_h < in_h && cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;
            max_i = (val > max) ? index : max_i;
            max = (val > max) ? val : max;
        }
    }
    output[out_index] = max;
    indexes[out_index] = max_i;
}

void bcnn_forward_maxpool_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_maxpool_param *param = (bcnn_maxpool_param *)node->param;
#ifdef BCNN_USE_CUDNN
    float zero = 0.0f, one = 1.0f;
    bcnn_cudnn_check(
        cudnnPoolingForward(bcnn_cudnn_handle(), param->pooling_desc, &one,
                            param->src_tensor_desc, src_tensor->data_gpu, &zero,
                            param->dst_tensor_desc, dst_tensor->data_gpu));
#else
    int sz = bcnn_tensor_size(dst_tensor);

    bcnn_forward_maxpool_layer_kernel<<<bcnn_cuda_gridsize(sz),
                                        BCNN_CUDA_THREADS>>>(
        sz, src_tensor->h, src_tensor->w, src_tensor->c, param->stride,
        param->size, src_tensor->data_gpu, dst_tensor->data_gpu,
        param->indexes_gpu);
    bcnn_cuda_check(cudaPeekAtLastError());
#endif

    return;
}

__global__ void bcnn_backward_maxpool_layer_kernel(int n, int in_h, int in_w,
                                                   int in_c, int stride,
                                                   int size, float *diff,
                                                   float *prev_delta,
                                                   int *indexes) {
    int h = (in_h - 1) / stride + 1;
    int w = (in_w - 1) / stride + 1;
    int c = in_c;
    int area = (size - 1) / stride;

    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) {
        return;
    }

    int index = id;
    int j = id % in_w;
    id /= in_w;
    int i = id % in_h;
    id /= in_h;
    int k = id % in_c;
    id /= in_c;
    int b = id;

    int w_offset = (-size - 1) / 2 + 1;
    int h_offset = (-size - 1) / 2 + 1;

    float d = 0;
    int l, m;
    for (l = -area; l < area + 1; ++l) {
        for (m = -area; m < area + 1; ++m) {
            int out_w = (j - w_offset) / stride + m;
            int out_h = (i - h_offset) / stride + l;
            int out_index = out_w + w * (out_h + h * (k + c * b));
            int valid = (out_w >= 0 && out_w < w && out_h >= 0 && out_h < h);
            d += (valid && indexes[out_index] == index) ? diff[out_index] : 0;
        }
    }
    prev_delta[index] += d;
}

void bcnn_backward_maxpool_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_maxpool_param *param = (bcnn_maxpool_param *)node->param;
#ifdef BCNN_USE_CUDNN
    float zero = 0.0f, one = 1.0f;
    bcnn_cudnn_check(cudnnPoolingBackward(
        bcnn_cudnn_handle(), param->pooling_desc, &one, param->dst_tensor_desc,
        dst_tensor->data_gpu, param->dst_tensor_desc, dst_tensor->grad_data_gpu,
        param->src_tensor_desc, src_tensor->data_gpu, &zero,
        param->src_tensor_desc, src_tensor->grad_data_gpu));
#else
    int sz = bcnn_tensor_size(src_tensor);

    bcnn_backward_maxpool_layer_kernel<<<bcnn_cuda_gridsize(sz),
                                         BCNN_CUDA_THREADS>>>(
        sz, src_tensor->h, src_tensor->w, src_tensor->c, param->stride,
        param->size, dst_tensor->grad_data_gpu, src_tensor->grad_data_gpu,
        param->indexes_gpu);
    bcnn_cuda_check(cudaPeekAtLastError());
#endif

    return;
}

#endif