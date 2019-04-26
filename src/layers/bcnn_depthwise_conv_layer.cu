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
#include "bcnn_depthwise_conv_layer.h"

#include "bcnn_mat.h"
#include "bcnn_tensor.h"
#include "bcnn_utils.h"

/* Depthwise Separable convolution */

__global__ void _bcnn_forward_depthwise_conv_weight_kernel(
    int nthreads, float *src_data, float *weight_data, int channels, int dst_h,
    int dst_w, int src_h, int src_w, int kernel_sz, int stride, int pad,
    float *dst_data) {
    int i, n, c, h, w, kh, kw, h_in, w_in, offset;
    float value;
    float *weight = NULL;

    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads;
         i += blockDim.x * gridDim.x) {
        n = i / channels / dst_h / dst_w;
        c = (i / dst_h / dst_w) % channels;
        h = (i / dst_w) % dst_h;
        w = i % dst_w;
        weight = weight_data + c * kernel_sz * kernel_sz;
        value = 0;
        for (kh = 0; kh < kernel_sz; ++kh) {
            for (kw = 0; kw < kernel_sz; ++kw) {
                h_in = -pad + h * stride + kh;
                w_in = -pad + w * stride + kw;
                if ((h_in >= 0) && (h_in < src_h) && (w_in >= 0) &&
                    (w_in < src_w)) {
                    offset = ((n * channels + c) * src_h + h_in) * src_w + w_in;
                    value += (*weight) * src_data[offset];
                }
                ++weight;
            }
        }
        dst_data[i] = value;
    }
}

void bcnn_forward_depthwise_conv_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_depthwise_conv_param *param = (bcnn_depthwise_conv_param *)node->param;
    int sz = bcnn_tensor_size(dst_tensor);

    _bcnn_forward_depthwise_conv_weight_kernel<<<bcnn_cuda_blocks(sz),
                                                 BCNN_CUDA_THREADS>>>(
        sz, (float *)src_tensor->data_gpu, (float *)weights->data_gpu,
        dst_tensor->c, dst_tensor->h, dst_tensor->w, src_tensor->h,
        src_tensor->w, param->size, param->stride, param->pad,
        (float *)dst_tensor->data_gpu);
    bcnn_cuda_check(cudaPeekAtLastError());

    bcnn_cuda_add_bias((float *)dst_tensor->data_gpu, (float *)biases->data_gpu,
                       dst_tensor->n, src_tensor->c,
                       dst_tensor->h * dst_tensor->w);

    bcnn_forward_activation_gpu((float *)dst_tensor->data_gpu, sz,
                                param->activation);

    return;
}

__global__ void _bcnn_backward_depthwise_conv_weight_kernel(
    int nthreads, float *dst_grad, float *src_data, int batch_size,
    const int channels, int dst_h, int dst_w, const int src_h, const int src_w,
    int kernel_sz, int stride, int pad, float *weight_diff) {
    int i, n, c, h, w, kw, kh, h_out_s, w_out_s, h_out, w_out, offset;
    float *p_weight_diff = NULL;

    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads;
         i += blockDim.x * gridDim.x) {
        n = i / channels / src_h / src_w;
        c = (i / src_h / src_w) % channels;
        h = (i / src_w) % src_h;
        w = i % src_w;
        p_weight_diff = weight_diff + c * kernel_sz * kernel_sz;
        for (kh = 0; kh < kernel_sz; ++kh) {
            for (kw = 0; kw < kernel_sz; ++kw) {
                h_out_s = h + pad - kh;
                w_out_s = w + pad - kw;
                if (((h_out_s % stride) == 0) && ((w_out_s % stride) == 0)) {
                    h_out = h_out_s / stride;
                    w_out = w_out_s / stride;
                    if ((h_out >= 0) && (h_out < dst_h) && (w_out >= 0) &&
                        (w_out < dst_w)) {
                        offset = ((n * channels + c) * dst_h + h_out) * dst_w +
                                 w_out;
                        *p_weight_diff += src_data[i] * dst_grad[offset];
                    }
                }
                ++p_weight_diff;
            }
        }
    }
}

__global__ void _bcnn_backward_depthwise_conv_data_kernel(
    int nthreads, float *dst_grad, float *weight_data, int batch_size,
    const int channels, int dst_h, int dst_w, const int src_h, const int src_w,
    int kernel_sz, int stride, int pad, float *src_grad) {
    int i, n, c, h, w, kw, kh, h_out_s, w_out_s, h_out, w_out, offset;
    float value = 0.0f;
    float *weight = NULL;

    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads;
         i += blockDim.x * gridDim.x) {
        n = i / channels / src_h / src_w;
        c = (i / src_h / src_w) % channels;
        h = (i / src_w) % src_h;
        w = i % src_w;
        weight = weight_data + c * kernel_sz * kernel_sz;
        value = 0.0f;
        for (kh = 0; kh < kernel_sz; ++kh) {
            for (kw = 0; kw < kernel_sz; ++kw) {
                h_out_s = h + pad - kh;
                w_out_s = w + pad - kw;
                if (((h_out_s % stride) == 0) && ((w_out_s % stride) == 0)) {
                    h_out = h_out_s / stride;
                    w_out = w_out_s / stride;
                    if ((h_out >= 0) && (h_out < dst_h) && (w_out >= 0) &&
                        (w_out < dst_w)) {
                        offset = ((n * channels + c) * dst_h + h_out) * dst_w +
                                 w_out;
                        value += (*weight) * dst_grad[offset];
                    }
                }
                ++weight;
            }
        }
        src_grad[i] += value;
    }
}

void bcnn_backward_depthwise_conv_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_depthwise_conv_param *param = (bcnn_depthwise_conv_param *)node->param;
    int src_sz = bcnn_tensor_size(src_tensor);
    int dst_sz = bcnn_tensor_size(dst_tensor);

    if ((float *)src_tensor->grad_data_gpu)
        bcnn_cuda_fill_f32(src_sz, 0.0f, (float *)src_tensor->grad_data_gpu, 1);

    bcnn_backward_activation_gpu(
        (float *)dst_tensor->data_gpu, (float *)dst_tensor->grad_data_gpu,
        dst_tensor->w * dst_tensor->h * dst_tensor->c * dst_tensor->n,
        param->activation);

    bcnn_cuda_grad_bias((float *)biases->grad_data_gpu,
                        (float *)dst_tensor->grad_data_gpu, src_tensor->n,
                        src_tensor->c, dst_tensor->w * dst_tensor->h);

    _bcnn_backward_depthwise_conv_weight_kernel<<<bcnn_cuda_blocks(src_sz),
                                                  BCNN_CUDA_THREADS>>>(
        src_sz, (float *)dst_tensor->grad_data_gpu,
        (float *)src_tensor->data_gpu, src_tensor->n, src_tensor->c,
        dst_tensor->h, dst_tensor->w, src_tensor->h, src_tensor->w, param->size,
        param->stride, param->pad, (float *)weights->grad_data_gpu);
    bcnn_cuda_check(cudaPeekAtLastError());

    if ((float *)src_tensor->grad_data_gpu) {
        _bcnn_backward_depthwise_conv_data_kernel<<<bcnn_cuda_blocks(src_sz),
                                                    BCNN_CUDA_THREADS>>>(
            src_sz, (float *)dst_tensor->grad_data_gpu,
            (float *)weights->data_gpu, src_tensor->n, src_tensor->c,
            dst_tensor->h, dst_tensor->w, src_tensor->h, src_tensor->w,
            param->size, param->stride, param->pad,
            (float *)src_tensor->grad_data_gpu);
        bcnn_cuda_check(cudaPeekAtLastError());
    }

    return;
}

#endif