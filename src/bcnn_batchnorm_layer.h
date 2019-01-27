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

#ifndef BCNN_BATCHNORM_LAYER_H
#define BCNN_BATCHNORM_LAYER_H

#include <bcnn/bcnn.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bcnn_batchnorm_param {
    bcnn_tensor saved_mean;
    bcnn_tensor saved_variance;
    float *workspace;
    float *x_norm;
#ifdef BCNN_USE_CUDA
    float *workspace_gpu;
    float *x_norm_gpu;
#ifdef BCNN_USE_CUDNN
    cudnnTensorDescriptor_t dst_tensor_desc;
    cudnnTensorDescriptor_t bias_desc;
#endif
#endif
} bcnn_batchnorm_param;

void bcnn_forward_batchnorm_layer(bcnn_net *net, bcnn_node *node);
void bcnn_backward_batchnorm_layer(bcnn_net *net, bcnn_node *node);
void bcnn_release_param_batchnorm_layer(bcnn_node *node);

void bcnn_forward_batchnorm_cpu(bcnn_tensor *src_tensor,
                                bcnn_tensor *dst_tensor, bcnn_tensor *bn_mean,
                                bcnn_tensor *bn_var, bcnn_tensor *bn_scales,
                                bcnn_tensor *biases, bcnn_tensor *saved_mean,
                                bcnn_tensor *saved_var, float *x_norm,
                                float *workspace, bcnn_state state);
void bcnn_backward_batchnorm_cpu(bcnn_tensor *src_tensor,
                                 bcnn_tensor *dst_tensor, bcnn_tensor *bn_mean,
                                 bcnn_tensor *bn_var, bcnn_tensor *bn_scales,
                                 bcnn_tensor *biases, bcnn_tensor *saved_mean,
                                 bcnn_tensor *saved_var, float *x_norm,
                                 float *workspace, bcnn_state state);
void bcnn_forward_batchnorm_layer_cpu(bcnn_net *net, bcnn_node *node);
void bcnn_backward_batchnorm_layer_cpu(bcnn_net *net, bcnn_node *node);

#ifdef BCNN_USE_CUDA

void bcnn_forward_batchnorm_gpu(bcnn_tensor *src_tensor,
                                bcnn_tensor *dst_tensor, bcnn_tensor *bn_mean,
                                bcnn_tensor *bn_var, bcnn_tensor *bn_scales,
                                bcnn_tensor *biases, bcnn_tensor *saved_mean,
                                bcnn_tensor *saved_var, float *x_norm_gpu,
                                float *workspace_gpu, bcnn_state state
#ifdef BCNN_USE_CUDNN
                                ,
                                cudnnTensorDescriptor_t dst_tensor_desc,
                                cudnnTensorDescriptor_t bias_desc
#endif
                                );
void bcnn_backward_batchnorm_gpu(bcnn_tensor *src_tensor,
                                 bcnn_tensor *dst_tensor, bcnn_tensor *bn_mean,
                                 bcnn_tensor *bn_var, bcnn_tensor *bn_scales,
                                 bcnn_tensor *biases, bcnn_tensor *saved_mean,
                                 bcnn_tensor *saved_variance, float *x_norm_gpu,
                                 float *workspace_gpu, bcnn_state state
#ifdef BCNN_USE_CUDNN
                                 ,
                                 cudnnTensorDescriptor_t dst_tensor_desc,
                                 cudnnTensorDescriptor_t bias_desc
#endif
                                 );
void bcnn_forward_batchnorm_layer_gpu(bcnn_net *net, bcnn_node *node);
void bcnn_backward_batchnorm_layer_gpu(bcnn_net *net, bcnn_node *node);
#endif

#ifdef __cplusplus
}
#endif

#endif  // BCNN_BATCHNORM_LAYER_H