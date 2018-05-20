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

#include "bcnn_activation_layer.h"
#include "bcnn_batchnorm_layer.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"

#ifdef BCNN_USE_BLAS
#include "cblas.h"
#endif

#include <bh/bh_mem.h>
#include <bh/bh_string.h>
#include <bh/bh_timer.h>

#include "bh_log.h"

int bcnn_add_convolutional_layer(bcnn_net *net, int n, int size, int stride,
                                 int pad, int batch_norm, bcnn_filler_type init,
                                 bcnn_activation activation, int quantize,
                                 char *src_id, char *dst_id) {
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
                bcnn_node_add_input(&node, i);
                is_src_node_found = 1;
                break;
            }
        }
        bh_check(is_src_node_found,
                 "Convolution layer: invalid input node name %s", src_id);
    } else {
        bh_check(bcnn_tensor_get_size(&net->tensors[0]) > 0,
                 "Invalid input size of the network. "
                 "Hint: you can use 'bcnn_net_set_input_shape' to set the "
                 "network input size");
        bcnn_node_add_input(&node, 0);
    }

    // Setup layer
    node.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    node.layer->type = CONVOLUTIONAL;
    node.layer->num = n;
    node.layer->stride = stride;
    node.layer->size = size;
    node.layer->pad = pad;

    // Setup layer weights
    char weights_name[256];
    sprintf(weights_name, "%s_w", src_id);
    bcnn_tensor_create(&node.layer->weights, 1, 1, 1,
                       net->tensors[node.src[0]].c * n * size * size, 1,
                       weights_name);
    bcnn_tensor_filler w_filler = {
        .range = (size * size * net->tensors[node.src[0]].c), .type = init};
    bcnn_tensor_fill(&node.layer->weights, w_filler);
    // Setup layer biases
    char biases_name[256];
    sprintf(biases_name, "%s_b", src_id);
    bcnn_tensor_create(&node.layer->biases, 1, 1, 1, n, 1, biases_name);

    // Create weights tensor
    /*bcnn_tensor weights = {0};
    char weights_name[256];
    sprintf(weights_name, "%s_w", src_id);
    bcnn_tensor_create(&weights, 1, 1, 1,
                       net->tensors[node.src[0]].c * n * size * size, 1,
                       weights_name);
    bcnn_tensor_filler w_filler = {
        .range = (size * size * net->tensors[node.src[0]].c), .type = init};
    bcnn_tensor_fill(&weights, w_filler);
    bcnn_net_add_tensor(net, weights);
    bcnn_node_add_input(&node, net->num_tensors - 1);
    // Create bias tensor
    bcnn_tensor biases = {0};
    char biases_name[256];
    sprintf(biases_name, "%s_b", src_id);
    bcnn_tensor_create(&biases, 1, 1, 1, n, 1, biases_name);
    bcnn_net_add_tensor(net, biases);
    bcnn_node_add_input(&node, net->num_tensors - 1);*/
    if (net->learner.optimizer == ADAM) {
        int weights_size = bcnn_tensor_get_size(&node.layer->weights);
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
    bcnn_node_add_output(&node, net->num_tensors - 1);
    sz = net->tensors[node.dst[0]].w * net->tensors[node.dst[0]].h *
         net->tensors[node.src[0]].c * size * size;
    node.layer->conv_workspace = (float *)calloc(sz, sizeof(float));

    if (batch_norm) {
        node.layer->batch_norm = 1;
        int sz = bcnn_tensor_get_size(&net->tensors[node.dst[0]]);
        int channels = net->tensors[node.dst[0]].c;
        char saved_mean_name[256], saved_var_name[256], running_mean_name[256],
            running_var_name[256], scales_name[256];
        sprintf(saved_mean_name, "%s_sav_mean", src_id);
        bcnn_tensor_create(&node.layer->saved_mean, 1, 1, 1, channels, 1,
                           saved_mean_name);
        sprintf(saved_var_name, "%s_sav_var", src_id);
        bcnn_tensor_create(&node.layer->saved_variance, 1, 1, 1, channels, 1,
                           saved_var_name);
        sprintf(running_mean_name, "%s_run_mean", src_id);
        bcnn_tensor_create(&node.layer->running_mean, 1, 1, 1, channels, 0,
                           running_mean_name);  // no gradients
        sprintf(running_var_name, "%s_run_var", src_id);
        bcnn_tensor_create(&node.layer->running_variance, 1, 1, 1, channels, 0,
                           running_var_name);  // no gradients
        node.layer->x_norm = (float *)calloc(sz, sizeof(float));
        node.layer->bn_workspace = (float *)calloc(sz, sizeof(float));
        sprintf(scales_name, "%s_scales", src_id);
        bcnn_tensor_create(&node.layer->scales, 1, 1, 1, channels, 1,
                           scales_name);
        bcnn_tensor_filler filler = {.value = 1.0f, .type = FIXED};
        bcnn_tensor_fill(&node.layer->scales, filler);
    }
#ifdef BCNN_USE_CUDA
    if (net->learner.optimizer == ADAM) {
        int weights_size = bcnn_tensor_get_size(&node.layer->weights);
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
        node.layer->num, net->tensors[node.src[0]].c, node.layer->size,
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
#endif
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
    node.layer->workspace_size = net->tensors[node.dst[0]].w *
                                 net->tensors[node.dst[0]].h *
                                 net->tensors[node.src[0]].c * size * size;
    net->workspace_size =
        bh_max(net->workspace_size, node.layer->workspace_size);
    if (node.layer->batch_norm) {
        int sz = bcnn_tensor_get_size(&net->tensors[node.dst[0]]);
        node.layer->x_norm_gpu = bcnn_cuda_memcpy_f32(node.layer->x_norm, sz);
        node.layer->bn_workspace_gpu =
            bcnn_cuda_memcpy_f32(node.layer->bn_workspace, sz);
    }
#endif
#endif
    node.layer->activation = activation;
    bcnn_net_add_node(net, node);
    bh_log_info(
        "[Convolutional] input_shape= %dx%dx%d nb_filters= %d kernel_size= %d "
        "stride= %d padding= %d output_shape= %dx%dx%d",
        net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
        net->tensors[node.src[0]].c, n, size, stride, pad,
        net->tensors[node.dst[0]].w, net->tensors[node.dst[0]].h,
        net->tensors[node.dst[0]].c);

    return 0;
}

int bcnn_forward_conv_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                bcnn_tensor *dst_tensor) {
    int i, j, m, n, k, sz;
    float *a = NULL, *b = NULL, *c = NULL;
    int batch_size = src_tensor->n;

    sz = bcnn_tensor_get_size(dst_tensor);
    memset(dst_tensor->data, 0, sz * sizeof(float));

    m = layer->num;
    k = layer->size * layer->size * src_tensor->c;
    n = dst_tensor->w * dst_tensor->h;

    sz = src_tensor->c * src_tensor->h * src_tensor->w;
    a = layer->weights.data;
    b = layer->conv_workspace;
    c = dst_tensor->data;

    /*for (int j = src_tensor->h / 2 * src_tensor->w / 2;
         j < src_tensor->h / 2 * src_tensor->w / 2 + 20; ++j) {
        fprintf(stderr, "%f ", src_tensor->data[j]);
    }
    fprintf(stderr, "\n");*/
    for (i = 0; i < batch_size; ++i) {
        if (layer->size == 1) {
            b = src_tensor->data + i * sz;
        } else {
            bcnn_im2col(src_tensor->data + i * sz, src_tensor->c, src_tensor->h,
                        src_tensor->w, layer->size, layer->pad, layer->stride,
                        b);
        }
#if BCNN_USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a,
                    k, b, n, 1.0f, c, n);
#else
        bcnn_gemm(0, 0, m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
#endif
        c += n * m;
    }

    /*for (int j = dst_tensor->h / 2 * dst_tensor->w / 2;
         j < dst_tensor->h / 2 * dst_tensor->w / 2 + 20; ++j) {
        fprintf(stderr, "%f ", dst_tensor->data[j]);
    }
    fprintf(stderr, "\n");*/

    if (layer->batch_norm) {  // inplace batch norm
        bcnn_forward_batchnorm_layer_cpu(layer, dst_tensor, dst_tensor);
    } else {
        bcnn_add_bias(dst_tensor->data, layer->biases.data, batch_size,
                      layer->num, dst_tensor->w * dst_tensor->h);
    }

    sz = dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size;
    bcnn_forward_activation_cpu(dst_tensor->data, sz, layer->activation);

    return BCNN_SUCCESS;
}

int bcnn_backward_conv_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                 bcnn_tensor *dst_tensor) {
    int batch_size = src_tensor->n;
    int i, sz = src_tensor->w * src_tensor->h * src_tensor->c;
    int m = layer->num;
    int n = layer->size * layer->size * src_tensor->c;
    int k = dst_tensor->w * dst_tensor->h;
    float *a = NULL, *b = NULL, *c = NULL;

    bcnn_backward_activation_cpu(
        dst_tensor->data, dst_tensor->grad_data,
        dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size,
        layer->activation);

    if (layer->batch_norm) {  // inplace batch norm
        bcnn_backward_batchnorm_layer_cpu(layer, dst_tensor, dst_tensor);
    } else {
        bcnn_grad_bias(layer->biases.grad_data, dst_tensor->grad_data,
                       batch_size, layer->num, k);
    }

    for (i = 0; i < batch_size; ++i) {
        a = dst_tensor->grad_data + i * m * k;
        b = layer->conv_workspace;
        c = layer->weights.grad_data;

        if (layer->size == 1) {
            b = src_tensor->data + i * sz;
        } else {
            bcnn_im2col(src_tensor->data + i * sz, src_tensor->c, src_tensor->h,
                        src_tensor->w, layer->size, layer->pad, layer->stride,
                        b);
        }
#if BCNN_USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a,
                    k, b, k, 1.0f, c, n);
#else
        bcnn_gemm(0, 1, m, n, k, 1.0f, a, k, b, k, 1.0f, c, n);
#endif

        if (src_tensor->grad_data) {
            a = layer->weights.data;
            b = dst_tensor->grad_data + i * m * k;
            c = layer->conv_workspace;

            if (layer->size == 1) {
#if BCNN_USE_BLAS
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, k, m,
                            1.0f, a, n, b, k, 0.0f,
                            src_tensor->grad_data + i * sz, k);
#else
                bcnn_gemm(1, 0, n, k, m, 1.0f, a, n, b, k, 0.0f,
                          src_tensor->grad_data + i * sz, k);
#endif
            } else {
#if BCNN_USE_BLAS
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, k, m,
                            1.0f, a, n, b, k, 0.0f, c, k);
#else
                bcnn_gemm(1, 0, n, k, m, 1.0f, a, n, b, k, 0.0f, c, k);
#endif
                bcnn_col2im(layer->conv_workspace, src_tensor->c, src_tensor->h,
                            src_tensor->w, layer->size, layer->pad,
                            layer->stride, src_tensor->grad_data + i * sz);
            }
        }
    }

    return BCNN_SUCCESS;
}

#ifdef BCNN_USE_CUDA

int bcnn_forward_conv_layer_gpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                bcnn_tensor *dst_tensor) {
    int batch_size = dst_tensor->n;
    int sz;

#ifdef BCNN_USE_CUDNN
    float alpha = 1.0f, beta = 0.0f;
    bcnn_cudnn_check(cudnnConvolutionForward(
        bcnn_cudnn_handle(), &alpha, layer->src_tensor_desc,
        src_tensor->data_gpu, layer->filter_desc, layer->weights.data_gpu,
        layer->conv_desc, layer->fwd_algo, layer->conv_workspace_gpu,
        layer->workspace_size, &beta, layer->dst_tensor_desc,
        dst_tensor->data_gpu));
    if (!layer->batch_norm) {
        bcnn_cudnn_check(
            cudnnAddTensor(bcnn_cudnn_handle(), &alpha, layer->bias_desc,
                           layer->biases.data_gpu, &alpha,
                           layer->dst_tensor_desc, dst_tensor->data_gpu));
    }
#else
    int i, w_sz, out_sz, out_spatial_dim;
    out_sz = batch_size * dst_tensor->w * dst_tensor->h * dst_tensor->c;
    w_sz = layer->size * layer->size * src_tensor->c;
    out_spatial_dim = dst_tensor->w * dst_tensor->h;
    sz = src_tensor->c * src_tensor->h * src_tensor->w;

    bcnn_cuda_fill_f32(out_sz, 0, dst_tensor->data_gpu, 1);
    for (i = 0; i < batch_size; ++i) {
        if (layer->size == 1)
            layer->conv_workspace_gpu = src_tensor->data_gpu + i * sz;
        else {
            bcnn_cuda_im2col(src_tensor->data_gpu + i * sz, src_tensor->c,
                             src_tensor->h, src_tensor->w, layer->size,
                             layer->stride, layer->pad,
                             layer->conv_workspace_gpu);
        }
        bcnn_cuda_gemm(0, 0, layer->num, out_spatial_dim, w_sz, 1.0f,
                       layer->weights.data_gpu, w_sz, layer->conv_workspace_gpu,
                       out_spatial_dim, 1.0f,
                       dst_tensor->data_gpu + i * layer->num * out_spatial_dim,
                       out_spatial_dim);
    }
    if (!layer->batch_norm) {
        bcnn_cuda_add_bias(dst_tensor->data_gpu, layer->biases.data_gpu,
                           batch_size, layer->num, out_spatial_dim);
    }
#endif
    if (layer->batch_norm) {
        bcnn_forward_batchnorm_layer_gpu(layer, dst_tensor, dst_tensor);
    }
    sz = dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size;
    bcnn_forward_activation_gpu(dst_tensor->data_gpu, sz, layer->activation);

    return BCNN_SUCCESS;
}

int bcnn_backward_conv_layer_gpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                 bcnn_tensor *dst_tensor) {
    int batch_size = dst_tensor->n;
#ifndef BCNN_USE_CUDNN
    int i, sz = src_tensor->w * src_tensor->h * src_tensor->c;
    int w_sz = layer->size * layer->size * src_tensor->c;
    int out_spatial_dim = dst_tensor->w * dst_tensor->h;
#else
    float one = 1.0f, zero = 0.0f;
#endif

    bcnn_backward_activation_gpu(
        dst_tensor->data_gpu, dst_tensor->grad_data_gpu,
        dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size,
        layer->activation);

#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnConvolutionBackwardBias(
        bcnn_cudnn_handle(), &one, layer->dst_tensor_desc,
        dst_tensor->grad_data_gpu, &one, layer->bias_desc,
        layer->biases.grad_data_gpu));
    bcnn_cudnn_check(cudnnConvolutionBackwardFilter(
        bcnn_cudnn_handle(), &one, layer->src_tensor_desc, src_tensor->data_gpu,
        layer->dst_tensor_desc, dst_tensor->grad_data_gpu, layer->conv_desc,
        layer->bwd_filter_algo, layer->conv_workspace_gpu,
        layer->workspace_size, &one, layer->filter_desc,
        layer->weights.grad_data_gpu));
    if (src_tensor->grad_data_gpu) {
        bcnn_cudnn_check(cudnnConvolutionBackwardData(
            bcnn_cudnn_handle(), &one, layer->filter_desc,
            layer->weights.data_gpu, layer->dst_tensor_desc,
            dst_tensor->grad_data_gpu, layer->conv_desc, layer->bwd_data_algo,
            layer->conv_workspace_gpu, layer->workspace_size, &zero,
            layer->src_tensor_desc, src_tensor->grad_data_gpu));
    }
#else
    if (layer->batch_norm) {
        bcnn_backward_batchnorm_layer_cpu(layer, dst_tensor, dst_tensor);
    } else {
        bcnn_cuda_grad_bias(layer->biases.grad_data_gpu,
                            dst_tensor->grad_data_gpu, batch_size, layer->num,
                            out_spatial_dim);
    }
    for (i = 0; i < batch_size; ++i) {
        if (layer->size == 1)
            layer->conv_workspace_gpu = src_tensor->data_gpu + i * sz;
        else {
            bcnn_cuda_im2col(src_tensor->data_gpu + i * sz, src_tensor->c,
                             src_tensor->h, src_tensor->w, layer->size,
                             layer->stride, layer->pad,
                             layer->conv_workspace_gpu);
        }
        bcnn_cuda_gemm(
            0, 1, layer->num, w_sz, out_spatial_dim, 1,
            dst_tensor->grad_data_gpu + i * layer->num * out_spatial_dim,
            out_spatial_dim, layer->conv_workspace_gpu, out_spatial_dim, 1,
            layer->weights.grad_data_gpu, w_sz);

        if (src_tensor->grad_data_gpu) {
            if (layer->size == 1) {
                bcnn_cuda_gemm(1, 0, w_sz, out_spatial_dim, layer->num, 1,
                               layer->weights.data_gpu, w_sz,
                               dst_tensor->grad_data_gpu +
                                   i * out_spatial_dim * layer->num,
                               out_spatial_dim, 0,
                               src_tensor->grad_data_gpu + i * sz,
                               out_spatial_dim);
            } else {
                bcnn_cuda_gemm(1, 0, w_sz, out_spatial_dim, layer->num, 1,
                               layer->weights.data_gpu, w_sz,
                               dst_tensor->grad_data_gpu +
                                   i * out_spatial_dim * layer->num,
                               out_spatial_dim, 0, layer->conv_workspace_gpu,
                               out_spatial_dim);
                bcnn_cuda_col2im(layer->conv_workspace_gpu, src_tensor->c,
                                 src_tensor->h, src_tensor->w, layer->size,
                                 layer->stride, layer->pad,
                                 src_tensor->grad_data_gpu + i * sz);
            }
        }
    }
#endif

    return BCNN_SUCCESS;
}

#endif

int bcnn_forward_conv_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    // bcnn_tensor *weights = &net->tensors[node->src[1]];
    // bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_conv_layer_gpu(node->layer, src, dst);
#else
    return bcnn_forward_conv_layer_cpu(node->layer, src, dst);
#endif
}

int bcnn_backward_conv_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    // bcnn_tensor *weights = &net->tensors[node->src[1]];
    // bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_conv_layer_gpu(node->layer, src, dst);
#else
    return bcnn_backward_conv_layer_cpu(node->layer, src, dst);
#endif
}