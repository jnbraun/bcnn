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

#include <bh/bh_timer.h>
#include "bcnn/bcnn.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"


/* Depthwise Separable convolution */

__global__ void _bcnn_forward_depthwise_sep_conv_weight_kernel(int nthreads,
     float *src_data, float *weight_data, int channels, int dst_h,
     int dst_w, int src_h, int src_w, int kernel_sz, int stride,
     int pad, float *dst_data)
{
   int i, n, c, h, w, kh, kw, h_in, w_in, offset;
   float value;
   float *weight = NULL;

    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += blockDim.x * gridDim.x) {
        n = i / channels / dst_h / dst_w;
        c = (i / dst_h / dst_w) % channels;
        h = (i / dst_w) % dst_h;
        w = i % dst_w;
        weight = weight_data + c * kernel_sz * kernel_sz;
        value = 0;
        /*if (h * layer->stride - layer->pad >= 0 && (h * layer->stride - layer->pad + layer->size) < src.h &&
            w * layer->stride - layer->pad >= 0 && (w * layer->stride - layer->pad + layer->size) < src.w) {
                
        }*/
        for (kh = 0; kh < kernel_sz; ++kh) {
            for (kw = 0; kw < kernel_sz; ++kw) {
                h_in = -pad + h * stride + kh;
                w_in = -pad + w * stride + kw;
                if ((h_in >= 0) && (h_in < src_h) && (w_in >= 0) && (w_in < src_w)) {
                    offset = ((n * channels + c) * src_h + h_in) * src_w + w_in;
                    value += (*weight) * src_data[offset];
                }
                ++weight;
            }
        }
        dst_data[i] = value;
    }
}


int bcnn_forward_depthwise_sep_conv_layer_gpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int sz = bcnn_tensor_get_size(&dst);
    /*bh_timer t = { 0 };
    bh_timer_start(&t);*/
    
    _bcnn_forward_depthwise_sep_conv_weight_kernel<<<bcnn_cuda_blocks(sz), BCNN_CUDA_THREADS>>>(
           sz, src.data_gpu, layer->weight_gpu, dst.c,
           dst.h, dst.w, src.h, src.w, layer->size, layer->stride,
           layer->pad, dst.data_gpu);
    bcnn_cuda_check(cudaPeekAtLastError());
    
    bcnn_cuda_add_bias(dst.data_gpu, layer->bias_gpu, dst.n, src.c, dst.h * dst.w);
    
    bcnn_forward_activation_gpu(dst.data_gpu, sz, layer->activation);
    /*bh_timer_stop(&t);
    fprintf(stderr, "sepconv-forward-time %lf sec\n", bh_timer_get_msec(&t) / 1000);*/
            
   return BCNN_SUCCESS;
}


__global__ void _bcnn_backward_depthwise_sep_conv_weight_kernel(int nthreads,
     float *dst_grad, float *src_data,
     int batch_size, const int channels, int dst_h, int dst_w, const int src_h, const int src_w,
     int kernel_sz, int stride, int pad, float *weight_diff)
{
 
    int i, n, c, h, w, kw, kh, h_out_s, w_out_s, h_out, w_out, offset; 
    float value = 0.0f;
    float *p_weight_diff = NULL;

    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += blockDim.x * gridDim.x) {
        n = i / channels / src_h / src_w;
        c = (i / src_h / src_w) % channels;
        h = (i / src_w) % src_h;
        w = i % src_w;
        p_weight_diff = weight_diff + c * kernel_sz * kernel_sz;
        value = 0.0f;
        for (kh = 0; kh < kernel_sz; ++kh) {
            for (kw = 0; kw < kernel_sz; ++kw) {
                h_out_s = h + pad - kh;
                w_out_s = w + pad - kw;
                if (((h_out_s % stride) == 0) && ((w_out_s % stride) == 0)) {
                    h_out = h_out_s / stride;
                    w_out = w_out_s / stride;
                    if ((h_out >= 0) && (h_out < dst_h) && (w_out >= 0) && (w_out < dst_w)) {
                        offset = ((n * channels + c) * dst_h + h_out) * dst_w + w_out;
                        *p_weight_diff += src_data[i] * dst_grad[offset];
                    }
                }
                ++p_weight_diff;
            }
        }
    }
}


__global__ void _bcnn_backward_depthwise_sep_conv_data_kernel(int nthreads,
     float *dst_grad, float *weight_data,
     int batch_size, const int channels, int dst_h, int dst_w, const int src_h, const int src_w,
     int kernel_sz, int stride, int pad, float *src_grad)
{
 
    int i, n, c, h, w, kw, kh, h_out_s, w_out_s, h_out, w_out, offset; 
    float value = 0.0f;
    float *weight = NULL;

    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += blockDim.x * gridDim.x) {
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
                    if ((h_out >= 0) && (h_out < dst_h) && (w_out >= 0) && (w_out < dst_w)) {
                        offset = ((n * channels + c) * dst_h + h_out) * dst_w + w_out;
                        value += (*weight) * dst_grad[offset];
                    }
                }
                ++weight;
            }
        }
        src_grad[i] += value;
    }
}


int bcnn_backward_depthwise_sep_conv_layer_gpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int src_sz = bcnn_tensor_get_size(&src);
    int dst_sz = bcnn_tensor_get_size(&dst);
    /*bh_timer t = { 0 };
    bh_timer_start(&t);*/
    
    if (src.grad_data_gpu)
        bcnn_cuda_fill_f32(src_sz, 0.0f, src.grad_data_gpu, 1);
    
    bcnn_backward_activation_gpu(dst.data_gpu, dst.grad_data_gpu,
        dst.w * dst.h * dst.c * dst.n,
        layer->activation);

    bcnn_cuda_grad_bias(layer->bias_diff_gpu, dst.grad_data_gpu, src.n, src.c, dst.w * dst.h);

    _bcnn_backward_depthwise_sep_conv_weight_kernel<<<bcnn_cuda_blocks(src_sz), BCNN_CUDA_THREADS>>>(
         src_sz, dst.grad_data_gpu, src.data_gpu,
         src.n, src.c, dst.h, dst.w, src.h, src.w,
         layer->size, layer->stride, layer->pad, layer->weight_diff_gpu);
    bcnn_cuda_check(cudaPeekAtLastError());
    
    if (src.grad_data_gpu) {
        _bcnn_backward_depthwise_sep_conv_data_kernel<<<bcnn_cuda_blocks(src_sz), BCNN_CUDA_THREADS>>>(
             src_sz, dst.grad_data_gpu, layer->weight_gpu, src.n, src.c,
             dst.h, dst.w, src.h, src.w, layer->size, layer->stride, layer->pad, src.grad_data_gpu);
        bcnn_cuda_check(cudaPeekAtLastError());
    }
   
    /*bh_timer_stop(&t);
    fprintf(stderr, "sepconv-backward-time %lf sec\n", bh_timer_get_msec(&t) / 1000);*/
   
   return BCNN_SUCCESS;
}


#endif