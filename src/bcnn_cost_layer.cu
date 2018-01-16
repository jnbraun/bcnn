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

#include <bh/bh.h>

#include "bcnn/bcnn.h"
#include "bcnn_mat.h"

int bcnn_forward_cost_layer_gpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *label_node, bcnn_node *dst_node)
{
    int i, j, offset, j_best, n, d;
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    bcnn_tensor label = label_node->tensor;
    int input_size = src.w * src.h * src.c;
    int batch_size = src.n;
    int sz = src.n * input_size;
    float p_max;
    float *src_data_cpu = NULL;
    // If no truth available, do nothing
    if (!label.data)
        return 0;

    bcnn_cuda_copy_f32(sz, src.data_gpu, 1, dst.grad_data_gpu, 1);
    bcnn_cuda_axpy(sz, -1, label.data_gpu, 1, dst.grad_data_gpu, 1);

    switch (layer->loss_metric) {
    case COST_ERROR:
        *(dst.data) = 0.0f;
        src_data_cpu = (float *)calloc(sz, sizeof(float));
        bcnn_cuda_memcpy_dev2host(src.data_gpu, src_data_cpu, sz);
        bcnn_cuda_memcpy_dev2host(label.data_gpu, label.data, sz);
        for (i = 0; i < batch_size; ++i) {
            offset = i * input_size;
            p_max = FLT_MIN;
            j_best = 0;
            for (j = 0; j < input_size; ++j) {
                if (src_data_cpu[offset + j] > p_max) {
                    p_max = src_data_cpu[offset + j];
                    j_best = j;
                }
            }
            if (label.data[offset + j_best] == 0)
                *(dst.data) += 1.0f;
        }
        bh_free(src_data_cpu);
        break;
    case COST_SSE:
        bcnn_cuda_memcpy_dev2host(dst.grad_data_gpu, dst.grad_data, sz);
        *(dst.data) = bcnn_dot(sz, dst.grad_data, dst.grad_data);
        break;
    case COST_MSE:
        bcnn_cuda_memcpy_dev2host(dst.grad_data_gpu, dst.grad_data, sz);
        *(dst.data) = bcnn_dot(sz, dst.grad_data, dst.grad_data);
        *(dst.data) /= input_size;
        break;
    case COST_CRPS:
        *(dst.data) = 0.0f;
        src_data_cpu = (float *)calloc(sz, sizeof(float));
        bcnn_cuda_memcpy_dev2host(src.data_gpu, src_data_cpu, sz);
        bcnn_cuda_memcpy_dev2host(label.data_gpu, label.data, sz);
        for (i = 0; i < batch_size; ++i) {
            offset = i * input_size;
            for (j = 1; j < input_size; ++j) {
                if (src_data_cpu[offset + j] < src_data_cpu[offset + j - 1]) {
                    src_data_cpu[offset + j] = src_data_cpu[offset + j - 1];
                }
            }
        }
        bcnn_axpy(sz, -1, label.data, src_data_cpu);
        *(dst.data) = bcnn_dot(sz, src_data_cpu, src_data_cpu);
        bh_free(src_data_cpu);
        break;
    case COST_LOGLOSS:
        *(dst.data) = 0.0f;
        src_data_cpu = (float *)calloc(sz, sizeof(float));
        bcnn_cuda_memcpy_dev2host(src.data_gpu, src_data_cpu, sz);
        bcnn_cuda_memcpy_dev2host(label.data_gpu, label.data, sz);
        for (i = 0; i < batch_size; ++i) {
            offset = i * input_size;
            for (j = 0; j < input_size; ++j) {
                if (label.data[offset + j] > 0.0f) {
                    *(dst.data) += (float)-log(bh_clamp(src_data_cpu[offset + j], 1e-8f, 1.0f - 1e-8f));
                }
            }
        }
        bh_free(src_data_cpu);
        bcnn_cuda_memcpy_dev2host(dst.grad_data_gpu, dst.grad_data, sz);
        break;
    case COST_DICE:
        src_data_cpu = (float *)calloc(sz, sizeof(float));
        bcnn_cuda_memcpy_dev2host(src.data_gpu, src_data_cpu, sz);
        bcnn_cuda_memcpy_dev2host(label.data_gpu, label.data, sz);
        *(dst.data) = 0.0f;
        for (i = 0; i < batch_size; ++i) {
            offset = i * input_size;
            n = 0;
            d = 0;
            for (j = 0; j < input_size; ++j) {
                n += label.data[offset + j] * (src_data_cpu[offset + j] > 0.5f);
                d += label.data[offset + j] + (src_data_cpu[offset + j] > 0.5f);
            }
            *(dst.data) += (float)(2.0f * n + 1.0f) / (d + 1.0f);
        }
        bh_free(src_data_cpu);
        bcnn_cuda_memcpy_dev2host(dst.grad_data_gpu, dst.grad_data, sz);
        break;
    }

    return BCNN_SUCCESS;
}


int bcnn_backward_cost_layer_gpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int input_size = src.w * src.h * src.c;
    int sz = src.n * input_size;

    bcnn_cuda_axpy(sz, layer->scale, dst.grad_data_gpu, 1, src.grad_data_gpu, 1);

    return BCNN_SUCCESS;
}

#endif