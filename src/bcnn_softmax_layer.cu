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

#include "bcnn_softmax_layer.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"

__global__ void _bcnn_forward_softmax_layer_kernel(int n, int batch,
                                                   float *input,
                                                   float *output) {
    float sum = 0.f;
    float maxf = -INFINITY;
    int b = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;

    if (b >= batch) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        int val = input[i + b * n];
        maxf = (val > maxf) ? val : maxf;
    }
    for (int i = 0; i < n; ++i) {
        sum += exp(input[i + b * n] - maxf);
    }
    sum = (sum != 0) ? maxf + log(sum) : maxf - 100.f;
    for (int i = 0; i < n; ++i) {
        output[i + b * n] = exp(input[i + b * n] - sum);
    }
}

void bcnn_forward_softmax_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    int src_size = bcnn_tensor_size3d(src_tensor);
    int batch_size = dst_tensor->n;

    _bcnn_forward_softmax_layer_kernel<<<bcnn_cuda_gridsize(batch_size),
                                         BCNN_CUDA_THREADS>>>(
        src_size, batch_size, src_tensor->data_gpu, dst_tensor->data_gpu);
    bcnn_cuda_check(cudaPeekAtLastError());

    return;
}

void bcnn_backward_softmax_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    int size = bcnn_tensor_size(src_tensor);

    bcnn_cuda_axpy(size, 1, dst_tensor->grad_data_gpu, 1,
                   src_tensor->grad_data_gpu, 1);

    return;
}

#endif