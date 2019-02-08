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

#include "bcnn_activation_layer.h"

#include <bh/bh_macros.h>

#include "bcnn_utils.h"

__global__ void bcnn_forward_activation_layer_kernel(float *x, int sz,
                                                     bcnn_activation a) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < sz) {
        switch (a) {
            case BCNN_ACT_TANH:
                x[i] = (exp(2 * x[i]) - 1) / (exp(2 * x[i]) + 1);
                break;
            case BCNN_ACT_RELU:
                x[i] = x[i] * (x[i] > 0);
                break;
            case BCNN_ACT_LRELU:
                x[i] = (x[i] > 0 ? x[i] : 0.1f * x[i]);
                break;
            case BCNN_ACT_RAMP:
                x[i] = x[i] * (x[i] > 0) + 0.1 * x[i];
                break;
            case BCNN_ACT_CLAMP:
                x[i] = bh_clamp(x[i], 0, 1);
                break;
            case BCNN_ACT_LOGISTIC:
                x[i] = 1.0f / (1.0f + (float)exp(-x[i]));
                break;
            case BCNN_ACT_NONE:
                break;
            default:
                break;
        }
    }
    return;
}

void bcnn_forward_activation_gpu(float *x, int sz, bcnn_activation a) {
    bcnn_forward_activation_layer_kernel<<<bcnn_cuda_blocks(sz),
                                           BCNN_CUDA_THREADS>>>(x, sz, a);
    return;
}

void bcnn_forward_activation_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_activation_param *param = (bcnn_activation_param *)node->param;
    int sz = bcnn_tensor_size(dst_tensor);

    dst_tensor->data_gpu = src_tensor->data_gpu;
    bcnn_forward_activation_gpu(dst_tensor->data_gpu, sz, param->activation);
    bcnn_cuda_check(cudaPeekAtLastError());

    return;
}

__global__ void bcnn_backward_activation_layer_kernel(float *x, float *dx,
                                                      int sz,
                                                      bcnn_activation a) {
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (i < sz) {
        switch (a) {
            case BCNN_ACT_TANH:
                dx[i] *= (1 - x[i] * x[i]);
                break;
            case BCNN_ACT_RELU:
                dx[i] *= ((float)(x[i] > 0));
                break;
            case BCNN_ACT_LRELU:
                dx[i] *= (x[i] > 0 ? 1.0f : 0.1f);
                break;
            case BCNN_ACT_RAMP:
                dx[i] *= ((float)(x[i] > 0) + 0.1f);
                break;
            case BCNN_ACT_CLAMP:
                dx[i] *= (float)(x[i] > 0.0f && (x[i] < 1.0f));
                break;
            case BCNN_ACT_LOGISTIC:
                dx[i] *= (1 - x[i]) * x[i];
                break;
            case BCNN_ACT_NONE:
                break;
            default:
                break;
        }
    }
}

void bcnn_backward_activation_gpu(float *x, float *dx, int sz,
                                  bcnn_activation a) {
    bcnn_backward_activation_layer_kernel<<<bcnn_cuda_blocks(sz),
                                            BCNN_CUDA_THREADS>>>(x, dx, sz, a);
    return;
}

void bcnn_backward_activation_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_activation_param *param = (bcnn_activation_param *)node->param;
    int sz = bcnn_tensor_size(dst_tensor);

    bcnn_backward_activation_gpu(
        dst_tensor->data_gpu, dst_tensor->grad_data_gpu, sz, param->activation);
    bcnn_cuda_check(cudaPeekAtLastError());
    src_tensor->grad_data_gpu = dst_tensor->grad_data_gpu;

    return;
}

#endif