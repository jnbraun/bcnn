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

// im2col and col2im functions from caffe
// Reference https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu

__global__ void im2col_gpu_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_col) 
{
    int i, j, w, h, w_out, h_index, h_out, channel_in, channel_out;
    int h_in, w_in;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float *data_col_ptr = NULL;
    const float *data_im_ptr = NULL;

    for(; index < n; index += blockDim.x * gridDim.x) {
        w_out = index % width_col;
        h_index = index / width_col;
        h_out = h_index % height_col;
        channel_in = h_index / height_col;
        channel_out = channel_in * ksize * ksize;
        h_in = h_out * stride - pad;
        w_in = w_out * stride - pad;
        data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;
        for (i = 0; i < ksize; ++i) {
            for (j = 0; j < ksize; ++j) {
                h = h_in + i;
                w = w_in + j;
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : 0;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void bcnn_im2col_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad, float *data_col)
{
    pad = pad ? ksize/2 : 0;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    im2col_gpu_kernel<<</*bcnn_cuda_blocks(num_kernels)*/(num_kernels + BCNN_CUDA_THREADS - 1) / BCNN_CUDA_THREADS, BCNN_CUDA_THREADS>>>(
                            num_kernels, im, height, width, ksize, pad,
                            stride, height_col,
                            width_col, data_col);
}


__global__ void col2im_gpu_kernel(const int n, const float* data_col,
        const int height, const int width, const int ksize,
        const int pad,
        const int stride,
        const int height_col, const int width_col,
        float *data_im)
{
    int w, h, c, w_col_start, w_col_end, h_col_start, h_col_end;
    int offset, coeff_h_col, coeff_w_col, h_col, w_col;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    float val;

    for (; index < n; index += blockDim.x * gridDim.x) {
        val = 0;
        w = index % width + pad;
        h = (index / width) % height + pad;
        c = index / (width * height);

        // compute the start and end of the output
        w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
        w_col_end = bh_min(w / stride + 1, width_col);
        h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
        h_col_end = bh_min(h / stride + 1, height_col);

        // equivalent implementation
        offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
        coeff_h_col = (1 - stride * ksize * height_col) * width_col;
        coeff_w_col = (1 - stride * height_col * width_col);
        for (h_col = h_col_start; h_col < h_col_end; ++h_col) {
            for (w_col = w_col_start; w_col < w_col_end; ++w_col) {
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
            }
        }
        data_im[index] += val;
    }
}

void bcnn_col2im_gpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im)
{
    int height_col, width_col, num_kernels;

    pad = pad ? ksize/2 : 0;
    height_col = (height + 2 * pad - ksize) / stride + 1;
    width_col = (width + 2 * pad - ksize) / stride + 1;
    num_kernels = channels * height * width;

    col2im_gpu_kernel<<<(num_kernels + BCNN_CUDA_THREADS - 1) / BCNN_CUDA_THREADS,
                             BCNN_CUDA_THREADS>>>(
                            num_kernels, data_col, height, width, ksize, pad,
                            stride, height_col,
                            width_col, data_im);
}


__global__ void _bcnn_forward_bias_kernel(float *output, float *bias, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if (offset < size)
        output[(batch * n+filter) * size + offset] += bias[filter];
}

__global__ void ConvolutionDepthwiseBiasForward(int nthreads,
     float *bias_data, int channels,
     const int dst_h, const int dst_w, float* const dst_data)
{
    int i, c;

    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += blockDim.x * gridDim.x) {
        c = (i / dst_h / dst_w) % channels;
        dst_data[i] += bias_data[c];
    }
}

int bcnn_forward_bias_gpu(float *output, float *bias, int batch, int n, int size)
{
    dim3 dimGrid((size - 1) / BCNN_CUDA_THREADS + 1, n, batch);
    dim3 dimBlock(BCNN_CUDA_THREADS, 1, 1);

    _bcnn_forward_bias_kernel<<<dimGrid, dimBlock>>>(output, bias, n, size);
    bcnn_cuda_check(cudaPeekAtLastError());
    return BCNN_SUCCESS;
}

__global__ void _bcnn_backward_bias_kernel(float *bias_diff, float *diff, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if (offset < size)
        bias_diff[filter] += diff[(batch*n+filter)*size + offset];
}

int bcnn_backward_bias_gpu(float *bias_diff, float *diff, int batch, int n, int size)
{
    dim3 dimGrid((size - 1) / BCNN_CUDA_THREADS + 1, n, batch);
    dim3 dimBlock(BCNN_CUDA_THREADS, 1, 1);

    _bcnn_backward_bias_kernel<<<dimGrid, dimBlock>>>(bias_diff, diff, n, size);
    bcnn_cuda_check(cudaPeekAtLastError());
    return BCNN_SUCCESS;
}


int bcnn_forward_conv_layer_gpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = dst.n;
    int sz;
    /*bh_timer t = { 0 };
    bh_timer_start(&t);*/
    
#ifdef BCNN_USE_CUDNN
    float alpha = 1.0f, beta = 0.0f;
    bcnn_cudnn_check(cudnnConvolutionForward(bcnn_cudnn_handle(),
                                        &alpha,
                                        layer->src_tensor_desc,
                                        src.data_gpu,
                                        layer->filter_desc,
                                        layer->weight_gpu,
                                        layer->conv_desc,
                                        layer->fwd_algo,
                                        layer->conv_workspace_gpu,
                                        layer->workspace_size,
                                        &beta,
                                        layer->dst_tensor_desc,
                                        dst.data_gpu));
    bcnn_cudnn_check(cudnnAddTensor(bcnn_cudnn_handle(), &alpha,
              layer->bias_desc, layer->bias_gpu,
              &alpha,
              layer->dst_tensor_desc, dst.data_gpu));
#else
    int i, w_sz, out_sz, out_spatial_dim;

    out_sz = batch_size * dst.w * dst.h * dst.c;
    w_sz = layer->size * layer->size * src.c;
    out_spatial_dim = dst.w * dst.h;
    sz = src.c * src.h * src.w;

    bcnn_cuda_fill_f32(out_sz, 0, dst.data_gpu, 1);
    for (i = 0; i < batch_size; ++i) {
        if (layer->size == 1)
            layer->conv_workspace_gpu = src.data_gpu + i * sz;
        else {
            bcnn_im2col_gpu(src.data_gpu + i * sz,
                src.c, src.h, src.w,
                layer->size, layer->stride, layer->pad, layer->conv_workspace_gpu);
        }
        bcnn_cuda_gemm(0, 0, layer->num, out_spatial_dim, w_sz, 1.0f,
            layer->weight_gpu, w_sz, layer->conv_workspace_gpu, out_spatial_dim, 1.0f,
            dst.data_gpu + i * layer->num * out_spatial_dim, out_spatial_dim);
    }
    bcnn_forward_bias_gpu(dst.data_gpu, layer->bias_gpu, batch_size, layer->num, out_spatial_dim);
#endif

    sz = dst.w * dst.h * dst.c * batch_size;
    bcnn_forward_activation_gpu(dst.data_gpu, sz, layer->activation);
    
    /*bh_timer_stop(&t);
    fprintf(stderr, "conv-forward-time %lf sec\n", bh_timer_get_msec(&t) / 1000);*/

    return BCNN_SUCCESS;
}


int bcnn_backward_conv_layer_gpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = dst.n;
#ifndef BCNN_USE_CUDNN
    int i, sz = src.w * src.h * src.c;
    int w_sz = layer->size * layer->size * src.c;
    int out_spatial_dim = dst.w * dst.h;
#else
    float one = 1.0f, zero = 0.0f;
#endif
    /*bh_timer t = { 0 };
    bh_timer_start(&t);*/

    bcnn_backward_activation_gpu(dst.data_gpu, dst.grad_data_gpu,
        dst.w * dst.h * dst.c * batch_size,
        layer->activation);

#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnConvolutionBackwardBias(bcnn_cudnn_handle(),
              &one,
              layer->dst_tensor_desc,  dst.grad_data_gpu,
              &one,
              layer->bias_desc, layer->bias_diff_gpu));
    bcnn_cudnn_check(cudnnConvolutionBackwardFilter(bcnn_cudnn_handle(),
                                            &one,
                                            layer->src_tensor_desc,
                                            src.data_gpu,
                                            layer->dst_tensor_desc,
                                            dst.grad_data_gpu,
                                            layer->conv_desc,
                                            layer->bwd_filter_algo,
                                            layer->conv_workspace_gpu,
                                            layer->workspace_size,
                                            &one,
                                            layer->filter_desc,
                                            layer->weight_diff_gpu));
    if (src.grad_data_gpu) {
        bcnn_cudnn_check(cudnnConvolutionBackwardData(bcnn_cudnn_handle(),
                                            &one,
                                            layer->filter_desc,
                                            layer->weight_gpu,
                                            layer->dst_tensor_desc,
                                            dst.grad_data_gpu,
                                            layer->conv_desc,
                                            layer->bwd_data_algo,
                                            layer->conv_workspace_gpu,
                                            layer->workspace_size,
                                            &zero,
                                            layer->src_tensor_desc,
                                            src.grad_data_gpu));
    }
#else
    bcnn_backward_bias_gpu(layer->bias_diff_gpu, dst.grad_data_gpu, batch_size, layer->num, out_spatial_dim);
    for (i = 0; i < batch_size; ++i) {
        if (layer->size == 1)
            layer->conv_workspace_gpu = src.data_gpu + i * sz;
        else {
            bcnn_im2col_gpu(src.data_gpu + i * sz,
                src.c, src.h, src.w,
                layer->size, layer->stride, layer->pad, layer->conv_workspace_gpu);
        }
        bcnn_cuda_gemm(0, 1, layer->num, w_sz, out_spatial_dim, 1,
            dst.grad_data_gpu + i * layer->num * out_spatial_dim, out_spatial_dim, layer->conv_workspace_gpu, out_spatial_dim, 1,
            layer->weight_diff_gpu, w_sz);

        if (src.grad_data_gpu) {
            if (layer->size == 1) {
                bcnn_cuda_gemm(1, 0, w_sz, out_spatial_dim, layer->num, 1,
                    layer->weight_gpu, w_sz, dst.grad_data_gpu + i * out_spatial_dim * layer->num, out_spatial_dim, 0,
                    src.grad_data_gpu + i * sz, out_spatial_dim);
            }
            else {
                bcnn_cuda_gemm(1, 0, w_sz, out_spatial_dim, layer->num, 1,
                    layer->weight_gpu, w_sz, dst.grad_data_gpu + i * out_spatial_dim * layer->num, out_spatial_dim, 0,
                    layer->conv_workspace_gpu, out_spatial_dim);
                bcnn_col2im_gpu(layer->conv_workspace_gpu,
                    src.c, src.h, src.w,
                    layer->size, layer->stride, layer->pad, src.grad_data_gpu + i * sz);
            }
        }
    }
#endif
    /*bh_timer_stop(&t);
    fprintf(stderr, "conv-backward-time %lf sec\n", bh_timer_get_msec(&t) / 1000);*/
    return BCNN_SUCCESS;
}


int bcnn_forward_deconv_layer_gpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int i, m, n, k, sz;
    int batch_size = dst.n;

    sz = batch_size * dst.w * dst.h * dst.c;
    bcnn_cuda_fill_f32(sz, 0, dst.data_gpu, 1);

    m = layer->num * layer->size * layer->size;
    k = src.c;
    n = src.w * src.h;
    sz = src.c * src.h * src.w;
    for (i = 0; i < batch_size; ++i){
        bcnn_cuda_gemm(1, 0, m, n, k, 1.0f, layer->weight_gpu, m, src.data_gpu + i * sz, n, 0.0f, layer->conv_workspace_gpu, n);
        bcnn_col2im_gpu(layer->conv_workspace_gpu, layer->num, dst.h, dst.w, layer->size,
            layer->stride, 0, dst.data_gpu + i * layer->num * dst.w * dst.h);
    }

    bcnn_forward_bias_gpu(dst.data_gpu, layer->bias_gpu, batch_size, layer->num, dst.w * dst.h);
    
    sz = dst.w * dst.h * dst.c * batch_size;
    bcnn_forward_activation_gpu(dst.data_gpu, sz, layer->activation);

    return BCNN_SUCCESS;
}

int bcnn_backward_deconv_layer_gpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int i, sz = src.w * src.h * src.c;
    int m = src.c;
    int n = layer->size * layer->size * dst.c;
    int k = src.w * src.h;
    int batch_size = src.n;
    float *a = NULL, *b = NULL, *c = NULL, *pdst = NULL;
    float alpha = 1.0f / batch_size;

    bcnn_backward_activation_gpu(dst.data_gpu, dst.grad_data_gpu,
        dst.w * dst.h * dst.c * batch_size,
        layer->activation);

    bcnn_backward_bias_gpu(layer->bias_diff_gpu, dst.grad_data_gpu, batch_size, layer->num,
        dst.h * dst.w);
    
    for (i = 0; i < batch_size; ++i) {
        a = src.data_gpu + i * src.c * src.w * src.h;
        b = layer->conv_workspace_gpu;
        c = layer->weight_diff_gpu;

        pdst = dst.grad_data_gpu + i * dst.c * dst.w * dst.h;

        bcnn_im2col_gpu(pdst, dst.c, dst.h, dst.w,
            layer->size, layer->stride, 0, layer->conv_workspace_gpu);
        bcnn_cuda_gemm(0, 1, m, n, k, alpha, a, k, b, k, 1.0f, c, n);

        if (src.grad_data_gpu) {
            a = layer->weight_gpu;
            b = layer->conv_workspace_gpu;
            c = src.grad_data_gpu + i * sz;
            bcnn_cuda_gemm(0, 0, src.c, k, n, 1.0f, a, n, b, k, 0.0f, c, k);
        }
    }
    return 0;
}


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
    
    /*ConvolutionDepthwiseBiasForward<<<bcnn_cuda_blocks(sz), BCNN_CUDA_THREADS>>>(
            sz, layer->bias_gpu, dst.c, dst.h, dst.w, dst.data_gpu);
    bcnn_cuda_check(cudaPeekAtLastError());*/
    bcnn_forward_bias_gpu(dst.data_gpu, layer->bias_gpu, dst.n, src.c, dst.h * dst.w);
    
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


__global__ void ConvolutionDepthwiseBiasBackward(const int nthreads,
     float *top_diff, int batch_size, int channels,
     int dst_h, int dst_w, float *buffer_data)
{
    int i, c, n, h, w, offset;

    for (i = blockIdx.x * blockDim.x + threadIdx.x; i < nthreads; i += blockDim.x * gridDim.x) {
        c = i / batch_size / dst_h / dst_w;
        n = (i / dst_h / dst_w) % batch_size;
        h = (i / dst_w) % dst_h;
        w = i % dst_w;
        offset = ((n * channels + c) * dst_h + h) * dst_w + w;
        buffer_data[i] = top_diff[offset];
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

    bcnn_backward_bias_gpu(layer->bias_diff_gpu, dst.grad_data_gpu, src.n, src.c, dst.w * dst.h);

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