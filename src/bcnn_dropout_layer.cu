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
#include "bcnn_dropout_layer.h"
#include "bcnn_utils.h"

__global__ void _bcnn_dropout_layer_kernel(float *input, int size, float *rand,
                                           float prob, float scale) {
    int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (id < size) {
        input[id] = (rand[id] < prob) ? 0 : input[id] * scale;
    }
}

void bcnn_forward_dropout_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_dropout_param *param = (bcnn_dropout_param *)node->param;
    int size = bcnn_tensor_size(src_tensor);

    if (net->mode != TRAIN) {
        return;
    }
    bcnn_cuda_fill_with_random(param->rand_gpu, size);
    _bcnn_dropout_layer_kernel<<<bcnn_cuda_gridsize(size), BCNN_CUDA_THREADS>>>(
        src_tensor->data_gpu, size, param->rand_gpu, param->dropout_rate,
        param->scale);
    bcnn_cuda_check(cudaPeekAtLastError());

    return;
}

void bcnn_backward_dropout_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_dropout_param *param = (bcnn_dropout_param *)node->param;
    int size = bcnn_tensor_size(src_tensor);

    if (!src_tensor->grad_data_gpu) {
        return;
    }
    _bcnn_dropout_layer_kernel<<<bcnn_cuda_gridsize(size), BCNN_CUDA_THREADS>>>(
        src_tensor->grad_data_gpu, size, param->rand_gpu, param->dropout_rate,
        param->scale);
    bcnn_cuda_check(cudaPeekAtLastError());

    return;
}

#endif