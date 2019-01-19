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

#ifndef BCNN_CONV_LAYER_H
#define BCNN_CONV_LAYER_H

#include <bcnn/bcnn.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bcnn_conv_param {
    int num;
    int size;
    int stride;
    int pad;
    int num_groups;
    int batch_norm;
    size_t workspace_size;
    bcnn_activation activation;
    bcnn_tensor saved_mean;
    bcnn_tensor saved_variance;
    bcnn_tensor running_mean;
    bcnn_tensor running_variance;
    float *conv_workspace;
    float *workspace;  // embedded batchnorm
    float *x_norm;
    float *adam_m;
    float *adam_v;
#ifdef BCNN_USE_CUDA
    float *conv_workspace_gpu;
    float *bn_workspace_gpu;
    float *x_norm_gpu;
    float *adam_m_gpu;
    float *adam_v_gpu;
#ifdef BCNN_USE_CUDNN
    cudnnTensorDescriptor_t src_tensor_desc;
    cudnnTensorDescriptor_t dst_tensor_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t fwd_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
#endif
#endif
} bcnn_conv_param;

void bcnn_forward_conv_layer(bcnn_net *net, bcnn_node *node);
void bcnn_backward_conv_layer(bcnn_net *net, bcnn_node *node);

#ifdef __cplusplus
}
#endif

#endif  // BCNN_CONV_LAYER_H