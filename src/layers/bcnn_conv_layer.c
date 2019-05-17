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

#include <bh/bh_macros.h>
#include <bh/bh_string.h>

#ifdef BCNN_USE_BLAS
#include "cblas.h"
#endif

#include "bcnn_activation_layer.h"
#include "bcnn_batchnorm_layer.h"
#include "bcnn_conv_layer.h"
#include "bcnn_learner.h"
#include "bcnn_mat.h"
#include "bcnn_net.h"
#include "bcnn_tensor.h"
#include "bcnn_utils.h"

#include <bh/bh_timer.h>

bcnn_status bcnn_add_convolutional_layer(bcnn_net *net, int n, int size,
                                         int stride, int pad, int num_groups,
                                         int batch_norm, bcnn_filler_type init,
                                         bcnn_activation activation,
                                         int quantize, const char *src_id,
                                         const char *dst_id) {
    bcnn_node node = {0};
    bcnn_tensor dst_tensor = {0};

    if (net->num_nodes > 0) {
        int is_src_node_found = 0;
        for (int i = net->num_tensors - 1; i >= 0; --i) {
            if (strcmp(net->tensors[i].name, src_id) == 0) {
                bcnn_node_add_input(net, &node, i);
                is_src_node_found = 1;
                break;
            }
        }
        BCNN_CHECK_AND_LOG(
            net->log_ctx, is_src_node_found, BCNN_INVALID_PARAMETER,
            "Convolution layer: invalid input node name %s\n", src_id);
    } else {
        BCNN_CHECK_AND_LOG(net->log_ctx, bcnn_tensor_size(&net->tensors[0]) > 0,
                           BCNN_INVALID_PARAMETER,
                           "Invalid input size of the network. "
                           "Hint: Use 'bcnn_set_input_shape' to set the "
                           "network input size\n");
        bcnn_node_add_input(net, &node, 0);
    }
    BCNN_CHECK_AND_LOG(net->log_ctx,
                       net->tensors[node.src[0]].c % num_groups == 0,
                       BCNN_INVALID_PARAMETER,
                       "Number of input channels has to be a multiple of the "
                       "number of groups\n");
    BCNN_CHECK_AND_LOG(net->log_ctx, n % num_groups == 0,
                       BCNN_INVALID_PARAMETER,
                       "Number of output channels has to be a multiple of the "
                       "number of groups\n");
    int num_channels_per_group = net->tensors[node.src[0]].c / num_groups;
    // Create weights tensor
    bcnn_tensor weights = {0};
    char weights_name[256];
    sprintf(weights_name, "%s_w", src_id);
    bcnn_tensor_create(&weights, n, num_channels_per_group, size, size, 1,
                       weights_name, net->mode);
    bcnn_tensor_filler w_filler = {
        .range = (size * size * num_channels_per_group), .type = init};
    bcnn_tensor_fill(&weights, w_filler);
    BCNN_CHECK_STATUS(bcnn_net_add_tensor(net, weights));
    BCNN_CHECK_STATUS(bcnn_node_add_input(net, &node, net->num_tensors - 1));
    // Create bias tensor
    bcnn_tensor biases = {0};
    char biases_name[256];
    sprintf(biases_name, "%s_b", src_id);
    bcnn_tensor_create(&biases, 1, 1, 1, n, 1, biases_name, net->mode);
    BCNN_CHECK_STATUS(bcnn_net_add_tensor(net, biases));
    BCNN_CHECK_STATUS(bcnn_node_add_input(net, &node, net->num_tensors - 1));
    // Fill nodes param
    node.type = BCNN_LAYER_CONV2D;
    node.param_size = sizeof(bcnn_conv_param);
    node.param = (bcnn_conv_param *)calloc(1, node.param_size);
    bcnn_conv_param *param = (bcnn_conv_param *)node.param;
    param->activation = activation;
    param->pad = pad;
    param->num = n;
    param->size = size;
    param->stride = stride;
    param->num_groups = num_groups;
    node.forward = bcnn_forward_conv_layer;
    node.backward = bcnn_backward_conv_layer;
    node.update = bcnn_update_conv_layer;
    node.release_param = bcnn_release_param_conv_layer;
    if (net->learner != NULL) {
        if (net->learner->optimizer == BCNN_OPTIM_ADAM) {
            int weights_size = bcnn_tensor_size(&weights);
            param->adam_m = (float *)calloc(weights_size, sizeof(float));
            param->adam_v = (float *)calloc(weights_size, sizeof(float));
        }
    }
    bcnn_tensor_set_shape(
        &dst_tensor, net->tensors[node.src[0]].n, param->num,
        (net->tensors[node.src[0]].h + 2 * param->pad - param->size) /
                param->stride +
            1,
        (net->tensors[node.src[0]].w + 2 * param->pad - param->size) /
                param->stride +
            1,
        1);
    BCNN_CHECK_STATUS(bcnn_tensor_allocate(&dst_tensor, net->mode));
    bh_strfill(&dst_tensor.name, dst_id);
    // Add tensor to net
    BCNN_CHECK_STATUS(bcnn_net_add_tensor(net, dst_tensor));
    // Add tensor output index to node
    BCNN_CHECK_STATUS(bcnn_node_add_output(net, &node, net->num_tensors - 1));
    int sz_wk = net->tensors[node.dst[0]].w * net->tensors[node.dst[0]].h *
                num_channels_per_group * size * size;
    param->conv_workspace = (float *)calloc(sz_wk, sizeof(float));
    if (batch_norm) {
        param->batch_norm = 1;
        int sz = bcnn_tensor_size(&net->tensors[node.dst[0]]);
        int channels = net->tensors[node.dst[0]].c;
        char saved_mean_name[256], saved_var_name[256], running_mean_name[256],
            running_var_name[256], scales_name[256];
        sprintf(saved_mean_name, "%s_sav_mean", src_id);
        bcnn_tensor_create(&param->saved_mean, 1, 1, 1, channels, 1,
                           saved_mean_name, net->mode);
        sprintf(saved_var_name, "%s_sav_var", src_id);
        bcnn_tensor_create(&param->saved_variance, 1, 1, 1, channels, 1,
                           saved_var_name, net->mode);

        // Global mean and variance tensors for batch norm
        sprintf(running_mean_name, "%s_run_mean", src_id);
        sprintf(running_var_name, "%s_run_var", src_id);
        sprintf(scales_name, "%s_scales", src_id);
        bcnn_tensor running_mean = {0};
        bcnn_tensor_create(&running_mean, 1, 1, 1, channels, 0,
                           running_mean_name, net->mode);  // no gradients
        BCNN_CHECK_STATUS(bcnn_net_add_tensor(net, running_mean));
        BCNN_CHECK_STATUS(
            bcnn_node_add_input(net, &node, net->num_tensors - 1));
        bcnn_tensor running_variance = {0};
        bcnn_tensor_create(&running_variance, 1, 1, 1, channels, 0,
                           running_var_name, net->mode);  // no gradients
        BCNN_CHECK_STATUS(bcnn_net_add_tensor(net, running_variance));
        BCNN_CHECK_STATUS(
            bcnn_node_add_input(net, &node, net->num_tensors - 1));
        bcnn_tensor scales = {0};
        bcnn_tensor_create(&scales, 1, 1, 1, channels, 1, scales_name,
                           net->mode);
        bcnn_tensor_filler filler = {.value = 1.0f, .type = BCNN_FILLER_FIXED};
        bcnn_tensor_fill(&scales, filler);
        BCNN_CHECK_STATUS(bcnn_net_add_tensor(net, scales));
        BCNN_CHECK_STATUS(
            bcnn_node_add_input(net, &node, net->num_tensors - 1));
        // Internal workspace for batch norm
        param->x_norm = (float *)calloc(sz, sizeof(float));
        param->workspace = (float *)calloc(sz, sizeof(float));
    }
#ifdef CONV3X3
    // Special case for conv 3x3/s1
    if (param->size == 3 && param->stride == 1 && param->batch_norm == 0 &&
        param->num_groups == 1) {
        // TEST
        /*for (int i = 0; i < bcnn_tensor_size(&weights); ++i) {
            weights.data[i] =
                2 * ((float)(i) / bcnn_tensor_size(&weights) - 0.5);
        }*/
        bh_free(param->conv_workspace);
        int src_c_div4 = bh_div_up(net->tensors[node.src[0]].c, 4);
        int dst_c_div4 = bh_div_up(n, 4);
        param->workspace_size = net->num_threads * CONV_TILED *
                                (src_c_div4 + bh_div_up(n, 4) + 1) *
                                CONV3x3_SRC_BLOCK;
        fprintf(stderr, "wk size %ld\n", param->workspace_size);
        param->conv_workspace =
            (float *)calloc(param->workspace_size, sizeof(float));
        param->weights_workspace =
            (float *)calloc(src_c_div4 * dst_c_div4 * 256, sizeof(float));
        param->biases_workspace = (float *)calloc(
            bh_round_up(bcnn_tensor_size(&biases), 4), sizeof(float));
        memcpy(param->biases_workspace, biases.data,
               bcnn_tensor_size(&biases) * sizeof(float));
        // fprintf(stderr, "%d %d\n", src_c_div4, dst_c_div4);
        // bcnn_conv3x3_convert_weights(weights.data, param->weights_workspace,
        //                             net->tensors[node.src[0]].c, n);
        // Reshape src tensor
        size_t src_sz = net->tensors[node.src[0]].w *
                        net->tensors[node.src[0]].h * src_c_div4 * 4 *
                        net->tensors[node.src[0]].n;
        param->src_workspace = (float *)calloc(src_sz, sizeof(float));
        size_t dst_sz = net->tensors[node.dst[0]].w *
                        net->tensors[node.dst[0]].h * dst_c_div4 * 4 *
                        net->tensors[node.dst[0]].n;
        param->dst_workspace = (float *)calloc(dst_sz, sizeof(float));
    }
#endif
#ifdef BCNN_USE_CUDA
    if (net->learner != NULL) {
        if (net->learner->optimizer == BCNN_OPTIM_ADAM) {
            int weights_size = bcnn_tensor_size(&weights);
            param->adam_m_gpu =
                bcnn_cuda_memcpy_f32(param->adam_m, weights_size);
            param->adam_v_gpu =
                bcnn_cuda_memcpy_f32(param->adam_v, weights_size);
        }
    }
#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&param->src_tensor_desc));
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&param->dst_tensor_desc));
    bcnn_cudnn_check(cudnnCreateFilterDescriptor(&param->filter_desc));
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&param->bias_desc));
    bcnn_cudnn_check(cudnnCreateConvolutionDescriptor(&param->conv_desc));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(
        param->src_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        net->tensors[node.dst[0]].n, net->tensors[node.src[0]].c,
        net->tensors[node.src[0]].h, net->tensors[node.src[0]].w));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(
        param->dst_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        net->tensors[node.dst[0]].n, net->tensors[node.dst[0]].c,
        net->tensors[node.dst[0]].h, net->tensors[node.dst[0]].w));
    bcnn_cudnn_check(cudnnSetFilter4dDescriptor(
        param->filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, param->num,
        num_channels_per_group, param->size, param->size));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(
        param->bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
        net->tensors[node.dst[0]].c, 1, 1));
#if CUDNN_MAJOR >= 6
    bcnn_cudnn_check(cudnnSetConvolution2dDescriptor(
        param->conv_desc, param->pad, param->pad, param->stride, param->stride,
        1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
#else
    bcnn_cudnn_check(cudnnSetConvolution2dDescriptor(
        param->conv_desc, param->pad, param->pad, param->stride, param->stride,
        1, 1, CUDNN_CROSS_CORRELATION));
#endif  // CUDNN_MAJOR
#if CUDNN_MAJOR >= 7
    bcnn_cudnn_check(
        cudnnSetConvolutionGroupCount(param->conv_desc, param->num_groups));
#else
    BCNN_CHECK_AND_LOG(net->log_ctx, param->num_groups == 1,
                       BCNN_INVALID_PARAMETER,
                       "CUDNN version doesn't support groups > 1\n");
#endif  // CUDNN_MAJOR
    bcnn_cudnn_check(cudnnGetConvolutionForwardAlgorithm(
        bcnn_cudnn_handle(), param->src_tensor_desc, param->filter_desc,
        param->conv_desc, param->dst_tensor_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &param->fwd_algo));
    bcnn_cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm(
        bcnn_cudnn_handle(), param->filter_desc, param->dst_tensor_desc,
        param->conv_desc, param->src_tensor_desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &param->bwd_data_algo));
    bcnn_cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithm(
        bcnn_cudnn_handle(), param->src_tensor_desc, param->dst_tensor_desc,
        param->conv_desc, param->filter_desc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
        &param->bwd_filter_algo));
    size_t cudnn_wrk_sz = 0;
    bcnn_cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(
        bcnn_cudnn_handle(), param->src_tensor_desc, param->filter_desc,
        param->conv_desc, param->dst_tensor_desc, param->fwd_algo,
        &cudnn_wrk_sz));
    param->workspace_size = bh_max(param->workspace_size, cudnn_wrk_sz);
    bcnn_cudnn_check(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        bcnn_cudnn_handle(), param->src_tensor_desc, param->dst_tensor_desc,
        param->conv_desc, param->filter_desc, param->bwd_filter_algo,
        &cudnn_wrk_sz));
    param->workspace_size = bh_max(param->workspace_size, cudnn_wrk_sz);
    bcnn_cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(
        bcnn_cudnn_handle(), param->filter_desc, param->dst_tensor_desc,
        param->conv_desc, param->src_tensor_desc, param->bwd_data_algo,
        &cudnn_wrk_sz));
    param->workspace_size = bh_max(param->workspace_size, cudnn_wrk_sz);
    bcnn_cuda_context *cuda_ctx = (bcnn_cuda_context *)net->cuda_ctx;
    cuda_ctx->workspace_size =
        bh_max(cuda_ctx->workspace_size, param->workspace_size);
#else
    param->workspace_size =
        net->tensors[node.dst[0]].w * net->tensors[node.dst[0]].h *
        net->tensors[node.src[0]].c / num_groups * size * size;
    bcnn_cuda_context *cuda_ctx = (bcnn_cuda_context *)net->cuda_ctx;
    cuda_ctx->workspace_size =
        bh_max(cuda_ctx->workspace_size, param->workspace_size);
#endif  // BCNN_USE_CUDNN
    if (param->batch_norm) {
        int sz = bcnn_tensor_size(&net->tensors[node.dst[0]]);
        param->x_norm_gpu = bcnn_cuda_memcpy_f32(param->x_norm, sz);
        param->bn_workspace_gpu = bcnn_cuda_memcpy_f32(param->workspace, sz);
    }
#endif  // BCNN_USE_CUDA
    bcnn_net_add_node(net, node);
    BCNN_INFO(net->log_ctx,
              "[Conv2d] input_shape= %dx%dx%d filters= %d kernel_size= %d "
              "stride= %d padding= %d groups= %d batchnorm= %d output_shape= "
              "%dx%dx%d\n",
              net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
              net->tensors[node.src[0]].c, n, size, stride, pad, num_groups,
              batch_norm, net->tensors[node.dst[0]].w,
              net->tensors[node.dst[0]].h, net->tensors[node.dst[0]].c);

    return 0;
}

void bcnn_forward_conv_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_tensor *bn_mean = NULL;
    bcnn_tensor *bn_var = NULL;
    bcnn_tensor *bn_scales = NULL;
    bcnn_conv_param *param = (bcnn_conv_param *)node->param;
    if (param->batch_norm == 1) {
        bn_mean = &net->tensors[node->src[3]];
        bn_var = &net->tensors[node->src[4]];
        bn_scales = &net->tensors[node->src[5]];
    }
    int batch_size = src_tensor->n;

    int sz = bcnn_tensor_size(dst_tensor);
    memset(dst_tensor->data, 0, sz * sizeof(float));

    int m = param->num / param->num_groups;
    int k = param->size * param->size * src_tensor->c / param->num_groups;
    int n = dst_tensor->w * dst_tensor->h;

    sz = src_tensor->c * src_tensor->h * src_tensor->w;
    float *b = param->conv_workspace;
    int wsz = bcnn_tensor_size(weights);
#ifdef CONV3X3
    // Special case for conv 3x3/s1
    if (param->size == 3 && param->stride == 1 && param->batch_norm == 0 &&
        param->num_groups == 1) {
        bh_timer t = {0};
        bh_timer_start(&t);
        bcnn_nchw_to_nc4hw4(param->src_workspace, src_tensor->data,
                            src_tensor->h * src_tensor->w, src_tensor->c);
        bh_timer_stop(&t);
        fprintf(stderr, "pack %f\n", bh_timer_get_msec(&t));
        bh_timer_start(&t);
        bcnn_conv3x3s1_kernel(
            param->src_workspace, src_tensor->w, src_tensor->h, src_tensor->c,
            param->dst_workspace, dst_tensor->w, dst_tensor->h, dst_tensor->c,
            batch_size, param->pad, param->weights_workspace,
            param->biases_workspace, param->conv_workspace,
            param->workspace_size, net->num_threads);
        bh_timer_stop(&t);
        fprintf(stderr, "conv3x3 %f\n", bh_timer_get_msec(&t));
        bh_timer_start(&t);
        bcnn_nc4hw4_to_nchw(dst_tensor->data, param->dst_workspace,
                            dst_tensor->w * dst_tensor->h, dst_tensor->c);
        bh_timer_stop(&t);
        fprintf(stderr, "unpack %f\n", bh_timer_get_msec(&t));
    } else {
#endif
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < param->num_groups; ++j) {
                float *a = weights->data + j * wsz / param->num_groups;
                float *c =
                    dst_tensor->data + (i * param->num_groups + j) * n * m;
                float *src =
                    src_tensor->data +
                    (i * param->num_groups + j) * sz / param->num_groups;
                if (param->size == 1) {
                    b = src;
                } else {
#ifdef BCNN_USE_OPENMP
                    bcnn_im2col_mt(src, src_tensor->c / param->num_groups,
                                   src_tensor->h, src_tensor->w, param->size,
                                   param->pad, param->stride, b);
#else
                bcnn_im2col(src, src_tensor->c / param->num_groups,
                            src_tensor->h, src_tensor->w, param->size,
                            param->pad, param->stride, b);
#endif
                }
#if BCNN_USE_BLAS
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
                            1.0f, a, k, b, n, 1.0f, c, n);
#else
            bcnn_gemm(net->gemm_ctx, 0, 0, m, n, k, 1.0f, a, k, b, n, 1.0f, c,
                      n);
#endif
            }
        }
#ifdef CONV3X3
    }
#endif
    // TEST
    int spatial = dst_tensor->w * dst_tensor->h;
    for (int i = 0; i < 5; i++) {
        fprintf(stderr, "%f %f %f %f\n", dst_tensor->data[spatial * i],
                dst_tensor->data[spatial * i + 1],
                dst_tensor->data[spatial * i + 2],
                dst_tensor->data[spatial * i + 3]);
    }
    if (param->batch_norm) {  // inplace batch norm
        bcnn_forward_batchnorm_cpu(dst_tensor, dst_tensor, bn_mean, bn_var,
                                   bn_scales, biases, &param->saved_mean,
                                   &param->saved_variance, param->x_norm,
                                   param->workspace, net->mode);
    } else {
        bcnn_add_bias(dst_tensor->data, biases->data, batch_size, param->num,
                      dst_tensor->w * dst_tensor->h);
    }

    sz = dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size;
    bcnn_forward_activation_cpu(dst_tensor->data, sz, param->activation);
    return;
}

void bcnn_backward_conv_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_tensor *bn_mean = NULL;
    bcnn_tensor *bn_var = NULL;
    bcnn_tensor *bn_scales = NULL;
    bcnn_conv_param *param = (bcnn_conv_param *)node->param;
    if (param->batch_norm == 1) {
        bn_mean = &net->tensors[node->src[3]];
        bn_var = &net->tensors[node->src[4]];
        bn_scales = &net->tensors[node->src[5]];
    }
    int batch_size = src_tensor->n;
    int i, sz = src_tensor->w * src_tensor->h * src_tensor->c;
    int m = param->num / param->num_groups;
    int n = param->size * param->size * src_tensor->c / param->num_groups;
    int k = dst_tensor->w * dst_tensor->h;
    float *a = NULL, *b = NULL, *c = NULL;

    bcnn_backward_activation_cpu(
        dst_tensor->data, dst_tensor->grad_data,
        dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size,
        param->activation);

    if (param->batch_norm) {  // inplace batch norm
        bcnn_backward_batchnorm_cpu(dst_tensor, dst_tensor, bn_mean, bn_var,
                                    bn_scales, biases, &param->saved_mean,
                                    &param->saved_variance, param->x_norm,
                                    param->workspace, net->mode);
    } else {
        bcnn_grad_bias(biases->grad_data, dst_tensor->grad_data, batch_size,
                       param->num, k);
    }
    int wsz = bcnn_tensor_size(weights);
    for (i = 0; i < batch_size; ++i) {
        for (int j = 0; j < param->num_groups; ++j) {
            a = dst_tensor->grad_data + (i * param->num_groups + j) * m * k;
            b = param->conv_workspace;
            c = weights->grad_data + j * wsz / param->num_groups;
            float *src = src_tensor->data +
                         (i * param->num_groups + j) * sz / param->num_groups;
            if (param->size == 1) {
                b = src;
            } else {
                bcnn_im2col(src, src_tensor->c / param->num_groups,
                            src_tensor->h, src_tensor->w, param->size,
                            param->pad, param->stride, b);
            }
#if BCNN_USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f,
                        a, k, b, k, 1.0f, c, n);
#else
            bcnn_gemm(net->gemm_ctx, 0, 1, m, n, k, 1.0f, a, k, b, k, 1.0f, c,
                      n);
#endif

            if (src_tensor->grad_data) {
                a = weights->data + j * wsz / param->num_groups;
                b = dst_tensor->grad_data + (i * param->num_groups + j) * m * k;
                c = param->conv_workspace;
                float *src_grad =
                    src_tensor->grad_data +
                    (i * param->num_groups + j) * sz / param->num_groups;
                if (param->size == 1) {
#if BCNN_USE_BLAS
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, k,
                                m, 1.0f, a, n, b, k, 0.0f, src_grad, k);
#else
                    bcnn_gemm(net->gemm_ctx, 1, 0, n, k, m, 1.0f, a, n, b, k,
                              0.0f, src_grad, k);
#endif
                } else {
#if BCNN_USE_BLAS
                    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, k,
                                m, 1.0f, a, n, b, k, 0.0f, c, k);
#else
                    bcnn_gemm(net->gemm_ctx, 1, 0, n, k, m, 1.0f, a, n, b, k,
                              0.0f, c, k);
#endif
                    bcnn_col2im(param->conv_workspace,
                                src_tensor->c / param->num_groups,
                                src_tensor->h, src_tensor->w, param->size,
                                param->pad, param->stride, src_grad);
                }
            }
        }
    }
    return;
}

#ifdef BCNN_USE_CUDA
void bcnn_forward_conv_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_tensor *bn_mean = NULL;
    bcnn_tensor *bn_var = NULL;
    bcnn_tensor *bn_scales = NULL;
    bcnn_conv_param *param = (bcnn_conv_param *)node->param;
    if (param->batch_norm == 1) {
        bn_mean = &net->tensors[node->src[3]];
        bn_var = &net->tensors[node->src[4]];
        bn_scales = &net->tensors[node->src[5]];
    }
    int batch_size = dst_tensor->n;
    int sz;

#ifdef BCNN_USE_CUDNN
    float alpha = 1.0f, beta = 0.0f;
    bcnn_cudnn_check(cudnnConvolutionForward(
        bcnn_cudnn_handle(), &alpha, param->src_tensor_desc,
        src_tensor->data_gpu, param->filter_desc, weights->data_gpu,
        param->conv_desc, param->fwd_algo, param->conv_workspace_gpu,
        param->workspace_size, &beta, param->dst_tensor_desc,
        dst_tensor->data_gpu));
    if (!param->batch_norm) {
        bcnn_cudnn_check(cudnnAddTensor(
            bcnn_cudnn_handle(), &alpha, param->bias_desc, biases->data_gpu,
            &alpha, param->dst_tensor_desc, dst_tensor->data_gpu));
    }
#else
    int i, w_sz, out_sz, dst_sz2d;
    out_sz = batch_size * dst_tensor->w * dst_tensor->h * dst_tensor->c;
    w_sz = param->size * param->size * src_tensor->c / param->num_groups;
    dst_sz2d = dst_tensor->w * dst_tensor->h;
    sz = src_tensor->c * src_tensor->h * src_tensor->w / param->num_groups;

    bcnn_cuda_fill_f32(out_sz, 0, dst_tensor->data_gpu, 1);
    for (i = 0; i < batch_size; ++i) {
        for (int j = 0; j < param->num_groups; ++j) {
            if (param->size == 1)
                param->conv_workspace_gpu =
                    src_tensor->data_gpu + (i * param->num_groups + j) * sz;
            else {
                bcnn_cuda_im2col(
                    src_tensor->data_gpu + (i * param->num_groups + j) * sz,
                    src_tensor->c / param->num_groups, src_tensor->h,
                    src_tensor->w, param->size, param->stride, param->pad,
                    param->conv_workspace_gpu);
            }
            bcnn_cuda_gemm(0, 0, param->num / param->num_groups, dst_sz2d, w_sz,
                           1.0f, weights->data_gpu, w_sz,
                           param->conv_workspace_gpu, dst_sz2d, 1.0f,
                           dst_tensor->data_gpu +
                               (i * param->num_groups + j) * param->num /
                                   param->num_groups * dst_sz2d,
                           dst_sz2d);
        }
    }
    if (!param->batch_norm) {
        bcnn_cuda_add_bias(dst_tensor->data_gpu, biases->data_gpu, batch_size,
                           param->num, dst_sz2d);
    }
#endif
    if (param->batch_norm) {
        bcnn_forward_batchnorm_gpu(dst_tensor, dst_tensor, bn_mean, bn_var,
                                   bn_scales, biases, &param->saved_mean,
                                   &param->saved_variance, param->x_norm_gpu,
                                   param->bn_workspace_gpu, net->mode
#ifdef BCNN_USE_CUDNN
                                   ,
                                   param->dst_tensor_desc, param->bias_desc
#endif
                                   );
    }
    sz = dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size;
    bcnn_forward_activation_gpu(dst_tensor->data_gpu, sz, param->activation);

    return;
}

void bcnn_backward_conv_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_tensor *bn_mean = NULL;
    bcnn_tensor *bn_var = NULL;
    bcnn_tensor *bn_scales = NULL;
    bcnn_conv_param *param = (bcnn_conv_param *)node->param;
    if (param->batch_norm == 1) {
        bn_mean = &net->tensors[node->src[3]];
        bn_var = &net->tensors[node->src[4]];
        bn_scales = &net->tensors[node->src[5]];
    }
    int batch_size = dst_tensor->n;
#ifndef BCNN_USE_CUDNN
    int i;
    int sz = src_tensor->w * src_tensor->h * src_tensor->c / param->num_groups;
    int w_sz = bcnn_tensor_size(weights);
    int n = param->size * param->size * src_tensor->c / param->num_groups;
    int dst_sz2d = dst_tensor->w * dst_tensor->h;
#else
    float one = 1.0f, zero = 0.0f;
#endif

    bcnn_backward_activation_gpu(
        dst_tensor->data_gpu, dst_tensor->grad_data_gpu,
        dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size,
        param->activation);

    if (param->batch_norm) {
        bcnn_backward_batchnorm_gpu(dst_tensor, dst_tensor, bn_mean, bn_var,
                                    bn_scales, biases, &param->saved_mean,
                                    &param->saved_variance, param->x_norm_gpu,
                                    param->bn_workspace_gpu, net->mode
#ifdef BCNN_USE_CUDNN
                                    ,
                                    param->dst_tensor_desc, param->bias_desc
#endif
                                    );
    } else {
#ifndef BCNN_USE_CUDNN
        bcnn_cuda_grad_bias(biases->grad_data_gpu, dst_tensor->grad_data_gpu,
                            batch_size, param->num, dst_sz2d);
#endif
    }

#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnConvolutionBackwardBias(
        bcnn_cudnn_handle(), &one, param->dst_tensor_desc,
        dst_tensor->grad_data_gpu, &one, param->bias_desc,
        biases->grad_data_gpu));
    bcnn_cudnn_check(cudnnConvolutionBackwardFilter(
        bcnn_cudnn_handle(), &one, param->src_tensor_desc, src_tensor->data_gpu,
        param->dst_tensor_desc, dst_tensor->grad_data_gpu, param->conv_desc,
        param->bwd_filter_algo, param->conv_workspace_gpu,
        param->workspace_size, &one, param->filter_desc,
        weights->grad_data_gpu));
    if (src_tensor->grad_data_gpu) {
        bcnn_cudnn_check(cudnnConvolutionBackwardData(
            bcnn_cudnn_handle(), &one, param->filter_desc, weights->data_gpu,
            param->dst_tensor_desc, dst_tensor->grad_data_gpu, param->conv_desc,
            param->bwd_data_algo, param->conv_workspace_gpu,
            param->workspace_size, &zero, param->src_tensor_desc,
            src_tensor->grad_data_gpu));
    }
#else
    for (i = 0; i < batch_size; ++i) {
        for (int j = 0; j < param->num_groups; ++j) {
            if (param->size == 1)
                param->conv_workspace_gpu = src_tensor->data_gpu + i * sz;
            else {
                bcnn_cuda_im2col(
                    src_tensor->data_gpu + (i * param->num_groups + j) * sz,
                    src_tensor->c / param->num_groups, src_tensor->h,
                    src_tensor->w, param->size, param->stride, param->pad,
                    param->conv_workspace_gpu);
            }
            bcnn_cuda_gemm(
                0, 1, param->num / param->num_groups, n, dst_sz2d, 1,
                dst_tensor->grad_data_gpu +
                    (i * param->num_groups + j) * param->num /
                        param->num_groups * dst_sz2d,
                dst_sz2d, param->conv_workspace_gpu, dst_sz2d, 1,
                weights->grad_data_gpu + j * w_sz / param->num_groups, n);
            if (src_tensor->grad_data_gpu) {
                if (param->size == 1) {
                    bcnn_cuda_gemm(
                        1, 0, n, dst_sz2d, param->num / param->num_groups, 1,
                        weights->data_gpu + j * w_sz / param->num_groups, n,
                        dst_tensor->grad_data_gpu +
                            (i * param->num_groups + j) * param->num /
                                param->num_groups * dst_sz2d,
                        dst_sz2d, 0, src_tensor->grad_data_gpu +
                                         (i * param->num_groups + j) * sz,
                        dst_sz2d);
                } else {
                    bcnn_cuda_gemm(
                        1, 0, n, dst_sz2d, param->num / param->num_groups, 1,
                        weights->data_gpu + j * w_sz / param->num_groups, n,
                        dst_tensor->grad_data_gpu +
                            (i * param->num_groups + j) * param->num /
                                param->num_groups * dst_sz2d,
                        dst_sz2d, 0, param->conv_workspace_gpu, dst_sz2d);
                    bcnn_cuda_col2im(param->conv_workspace_gpu,
                                     src_tensor->c / param->num_groups,
                                     src_tensor->h, src_tensor->w, param->size,
                                     param->stride, param->pad,
                                     src_tensor->grad_data_gpu +
                                         (i * param->num_groups + j) * sz);
                }
            }
        }
    }
#endif

    return;
}
#endif

void bcnn_forward_conv_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_forward_conv_layer_gpu(net, node);
#else
    return bcnn_forward_conv_layer_cpu(net, node);
#endif
}

void bcnn_backward_conv_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_backward_conv_layer_gpu(net, node);
#else
    return bcnn_backward_conv_layer_cpu(net, node);
#endif
}

void bcnn_update_conv_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_conv_param *param = (bcnn_conv_param *)node->param;
    int batch_size = net->batch_size;
    int weights_size = bcnn_tensor_size(weights);
    int biases_size = bcnn_tensor_size(biases);
    switch (net->learner->optimizer) {
        case BCNN_OPTIM_ADAM: {
#ifdef BCNN_USE_CUDA
            bcnn_adam_update_gpu(
                weights->data_gpu, biases->data_gpu, weights->grad_data_gpu,
                biases->grad_data_gpu, param->adam_m_gpu, param->adam_v_gpu,
                weights_size, biases_size, batch_size, net->learner->seen,
                net->learner->beta1, net->learner->beta2,
                net->learner->learning_rate, net->learner->momentum,
                net->learner->decay);
#else
            bcnn_adam_update_cpu(weights->data, biases->data,
                                 weights->grad_data, biases->grad_data,
                                 param->adam_m, param->adam_v, weights_size,
                                 biases_size, batch_size, net->learner->seen,
                                 net->learner->beta1, net->learner->beta2,
                                 net->learner->learning_rate,
                                 net->learner->momentum, net->learner->decay);
#endif
            break;
        }
        case BCNN_OPTIM_SGD: {
#ifdef BCNN_USE_CUDA
            bcnn_sgd_update_gpu(weights->data_gpu, biases->data_gpu,
                                weights->grad_data_gpu, biases->grad_data_gpu,
                                weights_size, biases_size, batch_size,
                                net->learner->learning_rate,
                                net->learner->momentum, net->learner->decay);
#else
            bcnn_sgd_update_cpu(weights->data, biases->data, weights->grad_data,
                                biases->grad_data, weights_size, biases_size,
                                batch_size, net->learner->learning_rate,
                                net->learner->momentum, net->learner->decay);
#endif
            break;
        }
        default: { break; }
    }
}

void bcnn_release_param_conv_layer(bcnn_node *node) {
    bcnn_conv_param *param = (bcnn_conv_param *)node->param;
    bcnn_tensor_destroy(&param->saved_mean);
    bcnn_tensor_destroy(&param->saved_variance);
    bh_free(param->conv_workspace);
    bh_free(param->x_norm);
    bh_free(param->workspace);
    bh_free(param->adam_m);
    bh_free(param->adam_v);
    bh_free(param->weights_workspace);
    bh_free(param->biases_workspace);
    bh_free(param->src_workspace);
    bh_free(param->dst_workspace);
#ifdef BCNN_USE_CUDA
    if (param->x_norm_gpu) {
        bcnn_cuda_free(param->x_norm_gpu);
    }
    if (param->bn_workspace_gpu) {
        bcnn_cuda_free(param->bn_workspace_gpu);
    }
    if (param->adam_m_gpu) {
        bcnn_cuda_free(param->adam_m_gpu);
    }
    if (param->adam_v_gpu) {
        bcnn_cuda_free(param->adam_v_gpu);
    }
// param->conv_workspace_gpu is alloc'd / free'd at the struct bcnn_net level
#ifdef BCNN_USE_CUDNN
    cudnnDestroyTensorDescriptor(param->src_tensor_desc);
    cudnnDestroyTensorDescriptor(param->dst_tensor_desc);
    cudnnDestroyTensorDescriptor(param->bias_desc);
    cudnnDestroyFilterDescriptor(param->filter_desc);
    cudnnDestroyConvolutionDescriptor(param->conv_desc);
#endif
#endif
}