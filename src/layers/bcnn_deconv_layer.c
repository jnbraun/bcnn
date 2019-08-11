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

#include "bcnn_deconv_layer.h"

#ifdef BCNN_USE_BLAS
#include "cblas.h"
#endif

#include <bh/bh_mem.h>
#include <bh/bh_string.h>

#include "bcnn_activation_layer.h"
#include "bcnn_learner.h"
#include "bcnn_mat.h"
#include "bcnn_net.h"
#include "bcnn_tensor.h"
#include "bcnn_utils.h"

/* Deconv layer */
bcnn_status bcnn_add_deconvolutional_layer(
    bcnn_net *net, int n, int size, int stride, int pad, bcnn_filler_type init,
    bcnn_activation activation, const char *src_id, const char *dst_id) {
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
            "Deconvolution layer: invalid input node name %s\n", src_id);
    } else {
        bcnn_node_add_input(net, &node, 0);
    }

    // Fill nodes param
    node.type = BCNN_LAYER_TRANSPOSE_CONV2D;
    node.param_size = sizeof(bcnn_deconv_param);
    node.param = (bcnn_deconv_param *)calloc(1, node.param_size);
    bcnn_deconv_param *param = (bcnn_deconv_param *)node.param;
    param->activation = activation;
    param->pad = pad;
    param->num = n;
    param->size = size;
    param->stride = stride;
    node.forward = bcnn_forward_deconv_layer;
    node.backward = bcnn_backward_deconv_layer;
    node.update = bcnn_update_deconv_layer;
    node.release_param = bcnn_release_param_deconv_layer;

    // Create weights tensor
    bcnn_tensor weights = {0};
    char weights_name[256];
    sprintf(weights_name, "%s_w", src_id);
    bcnn_tensor_create(&weights, 1, 1, 1,
                       net->tensors[node.src[0]].c * n * size * size, 1,
                       weights_name, net->mode);
    bcnn_tensor_filler w_filler = {
        .range = (size * size * net->tensors[node.src[0]].c), .type = init};
    bcnn_tensor_fill(&weights, w_filler);
    bcnn_net_add_tensor(net, weights);
    bcnn_node_add_input(net, &node, net->num_tensors - 1);
    // Create bias tensor
    bcnn_tensor biases = {0};
    char biases_name[256];
    sprintf(biases_name, "%s_b", src_id);
    bcnn_tensor_create(&biases, 1, 1, 1, n, 1, biases_name, net->mode);
    bcnn_net_add_tensor(net, biases);
    bcnn_node_add_input(net, &node, net->num_tensors - 1);

    bcnn_tensor_set_shape(&dst_tensor, net->tensors[node.src[0]].n, param->num,
                          param->stride * (net->tensors[node.src[0]].h - 1) +
                              param->size - 2 * param->pad,
                          param->stride * (net->tensors[node.src[0]].w - 1) +
                              param->size - 2 * param->pad,
                          1);
    bcnn_tensor_allocate(&dst_tensor, net->mode);
    bh_strfill(&dst_tensor.name, dst_id);
    // Add node to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(net, &node, net->num_tensors - 1);
    int sz = net->tensors[node.dst[0]].w * net->tensors[node.dst[0]].h *
             net->tensors[node.src[0]].c * size * size;
    param->conv_workspace =
        (float *)bh_align_calloc(sz * sizeof(float), align_offset_);
    if (net->learner != NULL) {
        if (net->learner->optimizer == BCNN_OPTIM_ADAM) {
            int weights_size = bcnn_tensor_size(&weights);
            param->adam_m = (float *)bh_align_calloc(
                weights_size * sizeof(float), align_offset_);
            param->adam_v = (float *)bh_align_calloc(
                weights_size * sizeof(float), align_offset_);
        }
    }
#ifdef BCNN_USE_CUDA
    param->conv_workspace_gpu = bcnn_cuda_memcpy_f32(param->conv_workspace, sz);
    if (net->learner != NULL) {
        if (net->learner->optimizer == BCNN_OPTIM_ADAM) {
            int weights_size = bcnn_tensor_size(&weights);
            param->adam_m_gpu =
                bcnn_cuda_memcpy_f32(param->adam_m, weights_size);
            param->adam_v_gpu =
                bcnn_cuda_memcpy_f32(param->adam_v, weights_size);
        }
    }
#endif

    bcnn_net_add_node(net, node);

    BCNN_INFO(
        net->log_ctx,
        "[Deconvolutional] input_shape= %dx%dx%d nb_filters= %d kernel_size= "
        "%d stride= %d output_shape= %dx%dx%d\n",
        net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
        net->tensors[node.src[0]].c, n, size, stride,
        net->tensors[node.dst[0]].w, net->tensors[node.dst[0]].h,
        net->tensors[node.dst[0]].c);

    return BCNN_SUCCESS;
}

void bcnn_forward_deconv_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_deconv_param *param = (bcnn_deconv_param *)node->param;
    int batch_size = src_tensor->n;
    int i, m, n, k, sz;

    sz = batch_size * dst_tensor->w * dst_tensor->h * dst_tensor->c;

    bcnn_fill_f32(sz, 0.0f, dst_tensor->data);

    m = param->num * param->size * param->size;
    k = src_tensor->c;
    n = src_tensor->w * src_tensor->h;
    sz = src_tensor->c * src_tensor->h * src_tensor->w;
    for (i = 0; i < batch_size; ++i) {
#if BCNN_USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, 1.0f,
                    weights->data, m, src_tensor->data + i * sz, n, 0.0f,
                    param->conv_workspace, n);
#else
        bcnn_gemm(net->gemm_ctx, 1, 0, m, n, k, 1.0f, weights->data, m,
                  src_tensor->data + i * sz, n, 0.0f, param->conv_workspace, n,
                  net->num_threads);
#endif
        bcnn_col2im(
            param->conv_workspace, param->num, dst_tensor->h, dst_tensor->w,
            param->size, 0, param->stride,
            dst_tensor->data + i * param->num * dst_tensor->w * dst_tensor->h);
    }

    bcnn_add_bias(dst_tensor->data, biases->data, batch_size, param->num,
                  dst_tensor->w * dst_tensor->h, net->num_threads);

    sz = dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size;
    // TODO: prelu not supported
    bcnn_forward_activation_cpu(dst_tensor->data, sz, NULL,
                                dst_tensor->w * dst_tensor->h, dst_tensor->c,
                                param->activation);

    return;
}

void bcnn_backward_deconv_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_deconv_param *param = (bcnn_deconv_param *)node->param;
    int batch_size = src_tensor->n;
    int i, sz = src_tensor->w * src_tensor->h * src_tensor->c;
    int m = src_tensor->c;
    int n = param->size * param->size * dst_tensor->c;
    int k = src_tensor->w * src_tensor->h;
    float *pdst = NULL;
    float alpha = 1.0f / batch_size;
    // TODO: prelu not supported
    bcnn_backward_activation_cpu(
        dst_tensor->data, dst_tensor->grad_data,
        dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size, NULL, NULL,
        dst_tensor->w * dst_tensor->h, dst_tensor->c, param->activation);

    bcnn_grad_bias(biases->grad_data, dst_tensor->grad_data, batch_size,
                   param->num, dst_tensor->w * dst_tensor->h);

    for (i = 0; i < batch_size; ++i) {
        pdst = dst_tensor->grad_data +
               i * param->num * dst_tensor->w * dst_tensor->h;
        bcnn_im2col(pdst, dst_tensor->c, dst_tensor->h, dst_tensor->w,
                    param->size, 0, param->stride, param->conv_workspace);
#if BCNN_USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, alpha,
                    src_tensor->data +
                        i * src_tensor->c * src_tensor->h * src_tensor->w,
                    k, param->conv_workspace, k, 1.0f, weights->grad_data, n);
#else
        bcnn_gemm(net->gemm_ctx, 0, 1, m, n, k, alpha,
                  src_tensor->data +
                      i * src_tensor->c * src_tensor->h * src_tensor->w,
                  k, param->conv_workspace, k, 1.0f, weights->grad_data, n,
                  net->num_threads);
#endif
        if (src_tensor->grad_data) {
#if BCNN_USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        src_tensor->c, k, n, 1.0f, weights->data, n,
                        param->conv_workspace, k, 0.0f,
                        src_tensor->grad_data + i * sz, k);
#else
            bcnn_gemm(net->gemm_ctx, 0, 0, src_tensor->c, k, n, 1.0f,
                      weights->data, n, param->conv_workspace, k, 0.0f,
                      src_tensor->grad_data + i * sz, k, net->num_threads);
#endif
        }
    }
    return;
}

#ifdef BCNN_USE_CUDA
void bcnn_forward_deconv_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_deconv_param *param = (bcnn_deconv_param *)node->param;
    int i, m, n, k, sz;
    int batch_size = dst_tensor->n;

    sz = batch_size * dst_tensor->w * dst_tensor->h * dst_tensor->c;
    bcnn_cuda_fill_f32(sz, 0, dst_tensor->data_gpu, 1);

    m = param->num * param->size * param->size;
    k = src_tensor->c;
    n = src_tensor->w * src_tensor->h;
    sz = src_tensor->c * src_tensor->h * src_tensor->w;
    for (i = 0; i < batch_size; ++i) {
        bcnn_cuda_gemm(1, 0, m, n, k, 1.0f, weights->data_gpu, m,
                       src_tensor->data_gpu + i * sz, n, 0.0f,
                       param->conv_workspace_gpu, n);
        bcnn_cuda_col2im(param->conv_workspace_gpu, param->num, dst_tensor->h,
                         dst_tensor->w, param->size, param->stride, 0,
                         dst_tensor->data_gpu +
                             i * param->num * dst_tensor->w * dst_tensor->h);
    }

    bcnn_cuda_add_bias(dst_tensor->data_gpu, biases->data_gpu, batch_size,
                       param->num, dst_tensor->w * dst_tensor->h);

    sz = dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size;
    bcnn_forward_activation_gpu(dst_tensor->data_gpu, sz, param->activation);

    return;
}

void bcnn_backward_deconv_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_deconv_param *param = (bcnn_deconv_param *)node->param;
    int i, sz = src_tensor->w * src_tensor->h * src_tensor->c;
    int m = src_tensor->c;
    int n = param->size * param->size * dst_tensor->c;
    int k = src_tensor->w * src_tensor->h;
    int batch_size = src_tensor->n;
    float *a = NULL, *b = NULL, *c = NULL, *pdst = NULL;
    float alpha = 1.0f / batch_size;

    bcnn_backward_activation_gpu(
        dst_tensor->data_gpu, dst_tensor->grad_data_gpu,
        dst_tensor->w * dst_tensor->h * dst_tensor->c * batch_size,
        param->activation);

    bcnn_cuda_grad_bias(biases->grad_data_gpu, dst_tensor->grad_data_gpu,
                        batch_size, param->num, dst_tensor->h * dst_tensor->w);

    for (i = 0; i < batch_size; ++i) {
        a = src_tensor->data_gpu +
            i * src_tensor->c * src_tensor->w * src_tensor->h;
        b = param->conv_workspace_gpu;
        c = weights->grad_data_gpu;

        pdst = dst_tensor->grad_data_gpu +
               i * dst_tensor->c * dst_tensor->w * dst_tensor->h;

        bcnn_cuda_im2col(pdst, dst_tensor->c, dst_tensor->h, dst_tensor->w,
                         param->size, param->stride, 0,
                         param->conv_workspace_gpu);
        bcnn_cuda_gemm(0, 1, m, n, k, alpha, a, k, b, k, 1.0f, c, n);

        if (src_tensor->grad_data_gpu) {
            a = weights->data_gpu;
            b = param->conv_workspace_gpu;
            c = src_tensor->grad_data_gpu + i * sz;
            bcnn_cuda_gemm(0, 0, src_tensor->c, k, n, 1.0f, a, n, b, k, 0.0f, c,
                           k);
        }
    }
    return;
}
#endif

void bcnn_forward_deconv_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_forward_deconv_layer_gpu(net, node);
#else
    return bcnn_forward_deconv_layer_cpu(net, node);
#endif
}

void bcnn_backward_deconv_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_backward_deconv_layer_gpu(net, node);
#else
    return bcnn_backward_deconv_layer_cpu(net, node);
#endif
}

void bcnn_update_deconv_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *weights = &net->tensors[node->src[1]];
    bcnn_tensor *biases = &net->tensors[node->src[2]];
    bcnn_deconv_param *param = (bcnn_deconv_param *)node->param;
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

void bcnn_release_param_deconv_layer(bcnn_node *node) {
    bcnn_deconv_param *param = (bcnn_deconv_param *)node->param;
    bh_align_free(param->conv_workspace);
    bh_align_free(param->adam_m);
    bh_align_free(param->adam_v);
#ifdef BCNN_USE_CUDA
    if (param->adam_m_gpu) {
        bcnn_cuda_free(param->adam_m_gpu);
    }
    if (param->adam_v_gpu) {
        bcnn_cuda_free(param->adam_v_gpu);
    }
// param->conv_workspace_gpu is alloc'd / free'd at the struct bcnn_net level
#endif
}