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

#include <bh/bh_macros.h>
#include <bh/bh_string.h>

#ifdef BCNN_USE_BLAS
#include "cblas.h"
#endif

#include "bcnn_activation_layer.h"
#include "bcnn_batchnorm_layer.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"

bcnn_status bcnn_add_convolutional_layer(bcnn_net *net, int n, int size,
                                         int stride, int pad, int num_groups,
                                         int batch_norm, bcnn_filler_type init,
                                         bcnn_activation activation,
                                         int quantize, char *src_id,
                                         char *dst_id) {
    int i, sz, k, l;
    bcnn_node node = {0};
    float std_init = 0.0f;
    bcnn_gauss_gen g = {0};
#ifdef BCNN_USE_CUDNN
    size_t cudnn_wrk_sz = 0;
#endif
    bcnn_tensor dst_tensor = {0};

    if (net->num_nodes > 0) {
        int is_src_node_found = 0;
        for (i = net->num_tensors - 1; i >= 0; --i) {
            if (strcmp(net->tensors[i].name, src_id) == 0) {
                bcnn_node_add_input(net, &node, i);
                is_src_node_found = 1;
                break;
            }
        }
        BCNN_CHECK_AND_LOG(
            net->log_ctx, is_src_node_found, BCNN_INVALID_PARAMETER,
            "Convolution layer: invalid input node name %s", src_id);
    } else {
        BCNN_CHECK_AND_LOG(
            net->log_ctx, bcnn_tensor_size(&net->tensors[0]) > 0,
            BCNN_INVALID_PARAMETER,
            "Invalid input size of the network. "
            "Hint: you can use 'bcnn_net_set_input_shape' to set the "
            "network input size");
        bcnn_node_add_input(net, &node, 0);
    }

    // Setup layer
    BCNN_CHECK_AND_LOG(net->log_ctx,
                       net->tensors[node.src[0]].c % num_groups == 0,
                       BCNN_INVALID_PARAMETER,
                       "Number of input channels has to be a multiple of the "
                       "number of groups");
    BCNN_CHECK_AND_LOG(net->log_ctx, n % num_groups == 0,
                       BCNN_INVALID_PARAMETER,
                       "Number of output channels has to be a multiple of the "
                       "number of groups");
    node.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    node.layer->type = CONVOLUTIONAL;
    node.layer->num = n;
    node.layer->stride = stride;
    node.layer->size = size;
    node.layer->pad = pad;
    node.layer->num_groups = num_groups;
    int num_channels_per_group = net->tensors[node.src[0]].c / num_groups;
    // Create weights tensor
    bcnn_tensor weights = {0};
    char weights_name[256];
    sprintf(weights_name, "%s_w", src_id);
    bcnn_tensor_create(&weights, n, num_channels_per_group, size, size, 1,
                       weights_name);
    bcnn_tensor_filler w_filler = {
        .range = (size * size * num_channels_per_group), .type = init};
    bcnn_tensor_fill(&weights, w_filler);
    bcnn_net_add_tensor(net, weights);
    bcnn_node_add_input(net, &node, net->num_tensors - 1);
    // Create bias tensor
    bcnn_tensor biases = {0};
    char biases_name[256];
    sprintf(biases_name, "%s_b", src_id);
    bcnn_tensor_create(&biases, 1, 1, 1, n, 1, biases_name);
    bcnn_net_add_tensor(net, biases);
    bcnn_node_add_input(net, &node, net->num_tensors - 1);
    if (net->learner.optimizer == ADAM) {
        int weights_size = bcnn_tensor_size(&weights);
        node.layer->adam_m = (float *)calloc(weights_size, sizeof(float));
        node.layer->adam_v = (float *)calloc(weights_size, sizeof(float));
    }
    bcnn_tensor_set_shape(
        &dst_tensor, net->tensors[node.src[0]].n, node.layer->num,
        (net->tensors[node.src[0]].h + 2 * node.layer->pad - node.layer->size) /
                node.layer->stride +
            1,
        (net->tensors[node.src[0]].w + 2 * node.layer->pad - node.layer->size) /
                node.layer->stride +
            1,
        1);
    bcnn_tensor_allocate(&dst_tensor);
    bh_strfill(&dst_tensor.name, dst_id);
    // Add node to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(net, &node, net->num_tensors - 1);
    sz = net->tensors[node.dst[0]].w * net->tensors[node.dst[0]].h *
         num_channels_per_group * size * size;
    node.layer->conv_workspace = (float *)calloc(sz, sizeof(float));
    node.layer->gemm_ctx = net->gemm_ctx;

    if (batch_norm) {
        node.layer->batch_norm = 1;
        int sz = bcnn_tensor_size(&net->tensors[node.dst[0]]);
        int channels = net->tensors[node.dst[0]].c;
        char saved_mean_name[256], saved_var_name[256], running_mean_name[256],
            running_var_name[256], scales_name[256];
        sprintf(saved_mean_name, "%s_sav_mean", src_id);
        bcnn_tensor_create(&node.layer->saved_mean, 1, 1, 1, channels, 1,
                           saved_mean_name);
        sprintf(saved_var_name, "%s_sav_var", src_id);
        bcnn_tensor_create(&node.layer->saved_variance, 1, 1, 1, channels, 1,
                           saved_var_name);

        // Global mean and variance tensors for batch norm
        sprintf(running_mean_name, "%s_run_mean", src_id);
        sprintf(running_var_name, "%s_run_var", src_id);
        sprintf(scales_name, "%s_scales", src_id);
        bcnn_tensor running_mean = {0};
        bcnn_tensor_create(&running_mean, 1, 1, 1, channels, 0,
                           running_mean_name);  // no gradients
        bcnn_net_add_tensor(net, running_mean);
        bcnn_node_add_input(net, &node, net->num_tensors - 1);
        bcnn_tensor running_variance = {0};
        bcnn_tensor_create(&running_variance, 1, 1, 1, channels, 0,
                           running_var_name);  // no gradients
        bcnn_net_add_tensor(net, running_variance);
        bcnn_node_add_input(net, &node, net->num_tensors - 1);
        bcnn_tensor scales = {0};
        bcnn_tensor_create(&scales, 1, 1, 1, channels, 1, scales_name);
        bcnn_tensor_filler filler = {.value = 1.0f, .type = FIXED};
        bcnn_tensor_fill(&scales, filler);
        bcnn_net_add_tensor(net, scales);
        bcnn_node_add_input(net, &node, net->num_tensors - 1);
        // Internal workspace for batch norm
        node.layer->x_norm = (float *)calloc(sz, sizeof(float));
        node.layer->workspace = (float *)calloc(sz, sizeof(float));
    }
#ifdef BCNN_USE_CUDA
    if (net->learner.optimizer == ADAM) {
        int weights_size = bcnn_tensor_size(&weights);
        node.layer->adam_m_gpu =
            bcnn_cuda_memcpy_f32(node.layer->adam_m, weights_size);
        node.layer->adam_v_gpu =
            bcnn_cuda_memcpy_f32(node.layer->adam_v, weights_size);
    }
#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&node.layer->src_tensor_desc));
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&node.layer->dst_tensor_desc));
    bcnn_cudnn_check(cudnnCreateFilterDescriptor(&node.layer->filter_desc));
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&node.layer->bias_desc));
    bcnn_cudnn_check(cudnnCreateConvolutionDescriptor(&node.layer->conv_desc));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(
        node.layer->src_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        net->tensors[node.dst[0]].n, net->tensors[node.src[0]].c,
        net->tensors[node.src[0]].h, net->tensors[node.src[0]].w));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(
        node.layer->dst_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        net->tensors[node.dst[0]].n, net->tensors[node.dst[0]].c,
        net->tensors[node.dst[0]].h, net->tensors[node.dst[0]].w));
    bcnn_cudnn_check(cudnnSetFilter4dDescriptor(
        node.layer->filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        node.layer->num, num_channels_per_group, node.layer->size,
        node.layer->size));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(
        node.layer->bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
        net->tensors[node.dst[0]].c, 1, 1));
#if CUDNN_MAJOR >= 6
    bcnn_cudnn_check(cudnnSetConvolution2dDescriptor(
        node.layer->conv_desc, node.layer->pad, node.layer->pad,
        node.layer->stride, node.layer->stride, 1, 1, CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));
#else
    bcnn_cudnn_check(cudnnSetConvolution2dDescriptor(
        node.layer->conv_desc, node.layer->pad, node.layer->pad,
        node.layer->stride, node.layer->stride, 1, 1, CUDNN_CROSS_CORRELATION));
#endif  // CUDNN_MAJOR
#if CUDNN_MAJOR >= 7
    bcnn_cudnn_check(cudnnSetConvolutionGroupCount(node.layer->conv_desc,
                                                   node.layer->num_groups));
#else
    BCNN_CHECK_AND_LOG(net->log_ctx, node.layer->num_groups == 1,
                       BCNN_INVALID_PARAMETER,
                       "CUDNN version doesn't support groups > 1");
#endif  // CUDNN_MAJOR
    bcnn_cudnn_check(cudnnGetConvolutionForwardAlgorithm(
        bcnn_cudnn_handle(), node.layer->src_tensor_desc,
        node.layer->filter_desc, node.layer->conv_desc,
        node.layer->dst_tensor_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0,
        &node.layer->fwd_algo));
    bcnn_cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm(
        bcnn_cudnn_handle(), node.layer->filter_desc,
        node.layer->dst_tensor_desc, node.layer->conv_desc,
        node.layer->src_tensor_desc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0, &node.layer->bwd_data_algo));
    bcnn_cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithm(
        bcnn_cudnn_handle(), node.layer->src_tensor_desc,
        node.layer->dst_tensor_desc, node.layer->conv_desc,
        node.layer->filter_desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
        &node.layer->bwd_filter_algo));
    bcnn_cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(
        bcnn_cudnn_handle(), node.layer->src_tensor_desc,
        node.layer->filter_desc, node.layer->conv_desc,
        node.layer->dst_tensor_desc, node.layer->fwd_algo, &cudnn_wrk_sz));
    node.layer->workspace_size =
        bh_max(node.layer->workspace_size, cudnn_wrk_sz);
    bcnn_cudnn_check(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        bcnn_cudnn_handle(), node.layer->src_tensor_desc,
        node.layer->dst_tensor_desc, node.layer->conv_desc,
        node.layer->filter_desc, node.layer->bwd_filter_algo, &cudnn_wrk_sz));
    node.layer->workspace_size =
        bh_max(node.layer->workspace_size, cudnn_wrk_sz);
    bcnn_cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(
        bcnn_cudnn_handle(), node.layer->filter_desc,
        node.layer->dst_tensor_desc, node.layer->conv_desc,
        node.layer->src_tensor_desc, node.layer->bwd_data_algo, &cudnn_wrk_sz));
    node.layer->workspace_size =
        bh_max(node.layer->workspace_size, cudnn_wrk_sz);
    net->workspace_size =
        bh_max(net->workspace_size, node.layer->workspace_size);
#else
    node.layer->workspace_size =
        net->tensors[node.dst[0]].w * net->tensors[node.dst[0]].h *
        net->tensors[node.src[0]].c / num_groups * size * size;
    net->workspace_size =
        bh_max(net->workspace_size, node.layer->workspace_size);
#endif  // BCNN_USE_CUDNN
    if (node.layer->batch_norm) {
        int sz = bcnn_tensor_size(&net->tensors[node.dst[0]]);
        node.layer->x_norm_gpu = bcnn_cuda_memcpy_f32(node.layer->x_norm, sz);
        node.layer->bn_workspace_gpu =
            bcnn_cuda_memcpy_f32(node.layer->workspace, sz);
    }
#endif  // BCNN_USE_CUDA
    node.layer->activation = activation;
    bcnn_net_add_node(net, node);
    BCNN_INFO(
        net->log_ctx,
        "[Convolutional] input_shape= %dx%dx%d nb_filters= %d kernel_size= %d "
        "stride= %d padding= %d groups= %d output_shape= %dx%dx%d",
        net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
        net->tensors[node.src[0]].c, n, size, stride, pad, num_groups,
        net->tensors[node.dst[0]].w, net->tensors[node.dst[0]].h,
        net->tensors[node.dst[0]].c);

    return 0;
}

int bcnn_forward_conv_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                bcnn_tensor *dst_tensor, bcnn_tensor *weights,
                                bcnn_tensor *biases, bcnn_tensor *bn_mean,
                                bcnn_tensor *bn_var, bcnn_tensor *bn_scales) {
    int i, j, m, n, k, sz;
    float *a = NULL, *b = NULL, *c = NULL;
    int batch_size = src_tensor->n;

    sz = bcnn_tensor_size(dst_tensor);
    memset(dst_tensor->data, 0, sz * sizeof(float));

    m = layer->num / layer->num_groups;
    k = layer->size * layer->size * src_tensor->c / layer->num_groups;
    n = dst_tensor->w * dst_tensor->h;

    sz = src_tensor->c * src_tensor->h * src_tensor->w;
    b = layer->conv_workspace;
    int wsz = bcnn_tensor_size(weights);
    for (i = 0; i < batch_size; ++i) {
        for (j = 0; j < layer->num_groups; ++j) {
            a = weights->data + j * wsz / layer->num_groups;
            c = dst_tensor->data + (i * layer->num_groups + j) * n * m;
            float *src = src_tensor->data +
                         (i * layer->num_groups + j) * sz / layer->num_groups;
            if (layer->size == 1) {
                b = src;
            } else {
                bcnn_im2col(src, src_tensor->c / layer->num_groups,
                            src_tensor->h, src_tensor->w, layer->size,
                            layer->pad, layer->stride, b);
            }
#if BCNN_USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                        1.0f, a, k, b, n, 1.0f, c, n);
#else
            bcnn_gemm(layer->gemm_ctx, 0, 0, m, n, k, 1.0f, a, k, b, n, 1.0f, c,
                      n);
#endif
        }
    }
    if (layer->batch_norm) {  // inplace batch norm
        bcnn_forward_batchnorm_layer_cpu(layer, dst_tensor, dst_tensor, bn_mean,
                                         bn_var, bn_scales, biases);
    } else {
        bcnn_add_bias(dst_tensor->data, biases->data, batch_size, layer->num,
                      dst_tensor->w * dst_tensor->h);
    }

    sz = dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size;
    bcnn_forward_activation_cpu(dst_tensor->data, sz, layer->activation);
    return BCNN_SUCCESS;
}

int bcnn_backward_conv_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                 bcnn_tensor *dst_tensor, bcnn_tensor *weights,
                                 bcnn_tensor *biases, bcnn_tensor *bn_mean,
                                 bcnn_tensor *bn_var, bcnn_tensor *bn_scales) {
    int batch_size = src_tensor->n;
    int i, sz = src_tensor->w * src_tensor->h * src_tensor->c;
    int m = layer->num / layer->num_groups;
    int n = layer->size * layer->size * src_tensor->c / layer->num_groups;
    int k = dst_tensor->w * dst_tensor->h;
    float *a = NULL, *b = NULL, *c = NULL;

    bcnn_backward_activation_cpu(
        dst_tensor->data, dst_tensor->grad_data,
        dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size,
        layer->activation);

    if (layer->batch_norm) {  // inplace batch norm
        bcnn_backward_batchnorm_layer_cpu(layer, dst_tensor, dst_tensor,
                                          bn_mean, bn_var, bn_scales, biases);
    } else {
        bcnn_grad_bias(biases->grad_data, dst_tensor->grad_data, batch_size,
                       layer->num, k);
    }
    int wsz = bcnn_tensor_size(weights);
    for (i = 0; i < batch_size; ++i) {
        for (int j = 0; j < layer->num_groups; ++j) {
            a = dst_tensor->grad_data + (i * layer->num_groups + j) * m * k;
            b = layer->conv_workspace;
            c = weights->grad_data + j * wsz / layer->num_groups;
            float *src = src_tensor->data +
                         (i * layer->num_groups + j) * sz / layer->num_groups;
            if (layer->size == 1) {
                b = src;
            } else {
                bcnn_im2col(src, src_tensor->c / layer->num_groups,
                            src_tensor->h, src_tensor->w, layer->size,
                            layer->pad, layer->stride, b);
            }
#if BCNN_USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f,
                        a, k, b, k, 1.0f, c, n);
#else
            bcnn_gemm(layer->gemm_ctx, 0, 1, m, n, k, 1.0f, a, k, b, k, 1.0f, c,
                      n);
#endif

            if (src_tensor->grad_data) {
                a = weights->data + j * wsz / layer->num_groups;
                b = dst_tensor->grad_data + (i * layer->num_groups + j) * m * k;
                c = layer->conv_workspace;
                float *src_grad =
                    src_tensor->grad_data +
                    (i * layer->num_groups + j) * sz / layer->num_groups;
                if (layer->size == 1) {
#if BCNN_USE_BLAS
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, k,
                                m, 1.0f, a, n, b, k, 0.0f, src_grad, k);
#else
                    bcnn_gemm(layer->gemm_ctx, 1, 0, n, k, m, 1.0f, a, n, b, k,
                              0.0f, src_grad, k);
#endif
                } else {
#if BCNN_USE_BLAS
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, k,
                                m, 1.0f, a, n, b, k, 0.0f, c, k);
#else
                    bcnn_gemm(layer->gemm_ctx, 1, 0, n, k, m, 1.0f, a, n, b, k,
                              0.0f, c, k);
#endif
                    bcnn_col2im(layer->conv_workspace,
                                src_tensor->c / layer->num_groups,
                                src_tensor->h, src_tensor->w, layer->size,
                                layer->pad, layer->stride, src_grad);
                }
            }
        }
    }
    return BCNN_SUCCESS;
}

#ifdef BCNN_USE_CUDA
int bcnn_forward_conv_layer_gpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                bcnn_tensor *dst_tensor, bcnn_tensor *weights,
                                bcnn_tensor *biases, bcnn_tensor *bn_mean,
                                bcnn_tensor *bn_var, bcnn_tensor *bn_scales) {
    int batch_size = dst_tensor->n;
    int sz;

#ifdef BCNN_USE_CUDNN
    float alpha = 1.0f, beta = 0.0f;
    bcnn_cudnn_check(cudnnConvolutionForward(
        bcnn_cudnn_handle(), &alpha, layer->src_tensor_desc,
        src_tensor->data_gpu, layer->filter_desc, weights->data_gpu,
        layer->conv_desc, layer->fwd_algo, layer->conv_workspace_gpu,
        layer->workspace_size, &beta, layer->dst_tensor_desc,
        dst_tensor->data_gpu));
    if (!layer->batch_norm) {
        bcnn_cudnn_check(cudnnAddTensor(
            bcnn_cudnn_handle(), &alpha, layer->bias_desc, biases->data_gpu,
            &alpha, layer->dst_tensor_desc, dst_tensor->data_gpu));
    }
#else
    int i, w_sz, out_sz, dst_sz2d;
    out_sz = batch_size * dst_tensor->w * dst_tensor->h * dst_tensor->c;
    w_sz = layer->size * layer->size * src_tensor->c / layer->num_groups;
    dst_sz2d = dst_tensor->w * dst_tensor->h;
    sz = src_tensor->c * src_tensor->h * src_tensor->w / layer->num_groups;

    bcnn_cuda_fill_f32(out_sz, 0, dst_tensor->data_gpu, 1);
    for (i = 0; i < batch_size; ++i) {
        for (int j = 0; j < layer->num_groups; ++j) {
            if (layer->size == 1)
                layer->conv_workspace_gpu =
                    src_tensor->data_gpu + (i * layer->num_groups + j) * sz;
            else {
                bcnn_cuda_im2col(
                    src_tensor->data_gpu + (i * layer->num_groups + j) * sz,
                    src_tensor->c / layer->num_groups, src_tensor->h,
                    src_tensor->w, layer->size, layer->stride, layer->pad,
                    layer->conv_workspace_gpu);
            }
            bcnn_cuda_gemm(0, 0, layer->num / layer->num_groups, dst_sz2d, w_sz,
                           1.0f, weights->data_gpu, w_sz,
                           layer->conv_workspace_gpu, dst_sz2d, 1.0f,
                           dst_tensor->data_gpu +
                               (i * layer->num_groups + j) * layer->num /
                                   layer->num_groups * dst_sz2d,
                           dst_sz2d);
        }
    }
    if (!layer->batch_norm) {
        bcnn_cuda_add_bias(dst_tensor->data_gpu, biases->data_gpu, batch_size,
                           layer->num, dst_sz2d);
    }
#endif
    if (layer->batch_norm) {
        bcnn_forward_batchnorm_layer_gpu(layer, dst_tensor, dst_tensor, bn_mean,
                                         bn_var, bn_scales, biases);
    }
    sz = dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size;
    bcnn_forward_activation_gpu(dst_tensor->data_gpu, sz, layer->activation);

    return BCNN_SUCCESS;
}

int bcnn_backward_conv_layer_gpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                 bcnn_tensor *dst_tensor, bcnn_tensor *weights,
                                 bcnn_tensor *biases, bcnn_tensor *bn_mean,
                                 bcnn_tensor *bn_var, bcnn_tensor *bn_scales) {
    int batch_size = dst_tensor->n;
#ifndef BCNN_USE_CUDNN
    int i;
    int sz = src_tensor->w * src_tensor->h * src_tensor->c / layer->num_groups;
    int w_sz = bcnn_tensor_size(weights);
    int n = layer->size * layer->size * src_tensor->c / layer->num_groups;
    int dst_sz2d = dst_tensor->w * dst_tensor->h;
#else
    float one = 1.0f, zero = 0.0f;
#endif

    bcnn_backward_activation_gpu(
        dst_tensor->data_gpu, dst_tensor->grad_data_gpu,
        dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size,
        layer->activation);

    if (layer->batch_norm) {
        bcnn_backward_batchnorm_layer_gpu(layer, dst_tensor, dst_tensor,
                                          bn_mean, bn_var, bn_scales, biases);
    } else {
#ifndef BCNN_USE_CUDNN
        bcnn_cuda_grad_bias(biases->grad_data_gpu, dst_tensor->grad_data_gpu,
                            batch_size, layer->num, dst_sz2d);
#endif
    }

#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnConvolutionBackwardBias(
        bcnn_cudnn_handle(), &one, layer->dst_tensor_desc,
        dst_tensor->grad_data_gpu, &one, layer->bias_desc,
        biases->grad_data_gpu));
    bcnn_cudnn_check(cudnnConvolutionBackwardFilter(
        bcnn_cudnn_handle(), &one, layer->src_tensor_desc, src_tensor->data_gpu,
        layer->dst_tensor_desc, dst_tensor->grad_data_gpu, layer->conv_desc,
        layer->bwd_filter_algo, layer->conv_workspace_gpu,
        layer->workspace_size, &one, layer->filter_desc,
        weights->grad_data_gpu));
    if (src_tensor->grad_data_gpu) {
        bcnn_cudnn_check(cudnnConvolutionBackwardData(
            bcnn_cudnn_handle(), &one, layer->filter_desc, weights->data_gpu,
            layer->dst_tensor_desc, dst_tensor->grad_data_gpu, layer->conv_desc,
            layer->bwd_data_algo, layer->conv_workspace_gpu,
            layer->workspace_size, &zero, layer->src_tensor_desc,
            src_tensor->grad_data_gpu));
    }
#else
    for (i = 0; i < batch_size; ++i) {
        for (int j = 0; j < layer->num_groups; ++j) {
            if (layer->size == 1)
                layer->conv_workspace_gpu = src_tensor->data_gpu + i * sz;
            else {
                bcnn_cuda_im2col(
                    src_tensor->data_gpu + (i * layer->num_groups + j) * sz,
                    src_tensor->c / layer->num_groups, src_tensor->h,
                    src_tensor->w, layer->size, layer->stride, layer->pad,
                    layer->conv_workspace_gpu);
            }
            bcnn_cuda_gemm(
                0, 1, layer->num / layer->num_groups, n, dst_sz2d, 1,
                dst_tensor->grad_data_gpu + (i * layer->num_groups + j) *
                                                layer->num / layer->num_groups *
                                                dst_sz2d,
                dst_sz2d, layer->conv_workspace_gpu, dst_sz2d, 1,
                weights->grad_data_gpu + j * w_sz / layer->num_groups, n);
            if (src_tensor->grad_data_gpu) {
                if (layer->size == 1) {
                    bcnn_cuda_gemm(
                        1, 0, n, dst_sz2d, layer->num / layer->num_groups, 1,
                        weights->data_gpu + j * w_sz / layer->num_groups, n,
                        dst_tensor->grad_data_gpu +
                            (i * layer->num_groups + j) * layer->num /
                                layer->num_groups * dst_sz2d,
                        dst_sz2d, 0,
                        src_tensor->grad_data_gpu +
                            (i * layer->num_groups + j) * sz,
                        dst_sz2d);
                } else {
                    bcnn_cuda_gemm(
                        1, 0, n, dst_sz2d, layer->num / layer->num_groups, 1,
                        weights->data_gpu + j * w_sz / layer->num_groups, n,
                        dst_tensor->grad_data_gpu +
                            (i * layer->num_groups + j) * layer->num /
                                layer->num_groups * dst_sz2d,
                        dst_sz2d, 0, layer->conv_workspace_gpu, dst_sz2d);
                    bcnn_cuda_col2im(layer->conv_workspace_gpu,
                                     src_tensor->c / layer->num_groups,
                                     src_tensor->h, src_tensor->w, layer->size,
                                     layer->stride, layer->pad,
                                     src_tensor->grad_data_gpu +
                                         (i * layer->num_groups + j) * sz);
                }
            }
        }
    }
#endif

    return BCNN_SUCCESS;
}
#endif

int bcnn_forward_conv_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_tensor *bn_mean = NULL;
    bcnn_tensor *bn_var = NULL;
    bcnn_tensor *bn_scales = NULL;
    if (node->layer->batch_norm == 1) {
        bn_mean = &net->tensors[node->src[3]];
        bn_var = &net->tensors[node->src[4]];
        bn_scales = &net->tensors[node->src[5]];
    }
#ifdef BCNN_USE_CUDA
    return bcnn_forward_conv_layer_gpu(node->layer, src, dst, weights, biases,
                                       bn_mean, bn_var, bn_scales);
#else
    return bcnn_forward_conv_layer_cpu(node->layer, src, dst, weights, biases,
                                       bn_mean, bn_var, bn_scales);
#endif
}

int bcnn_backward_conv_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_tensor *bn_mean = NULL;
    bcnn_tensor *bn_var = NULL;
    bcnn_tensor *bn_scales = NULL;
    if (node->layer->batch_norm == 1) {
        bn_mean = &net->tensors[node->src[3]];
        bn_var = &net->tensors[node->src[4]];
        bn_scales = &net->tensors[node->src[5]];
    }
#ifdef BCNN_USE_CUDA
    return bcnn_backward_conv_layer_gpu(node->layer, src, dst, weights, biases,
                                        bn_mean, bn_var, bn_scales);
#else
    return bcnn_backward_conv_layer_cpu(node->layer, src, dst, weights, biases,
                                        bn_mean, bn_var, bn_scales);
#endif
}