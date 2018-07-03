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

#include "bcnn_batchnorm_layer.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"

#include <bh/bh_log.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>

#include <bh/bh_log.h>

bcnn_status bcnn_add_batchnorm_layer(bcnn_net *net, char *src_id,
                                     char *dst_id) {
    int i, sz, channels;
    bcnn_node node = {0};
    bcnn_tensor dst_tensor = {0};

    BCNN_CHECK_AND_LOG(
        net->log_ctx, net->num_nodes >= 1, BCNN_INVALID_PARAMETER,
        "Batchnorm layer can't be the first layer of the network");

    int is_src_node_found = 0;
    for (i = net->num_tensors - 1; i >= 0; --i) {
        if (strcmp(net->tensors[i].name, src_id) == 0) {
            bcnn_node_add_input(net, &node, i);
            is_src_node_found = 1;
            break;
        }
    }
    BCNN_CHECK_AND_LOG(net->log_ctx, is_src_node_found, BCNN_INVALID_PARAMETER,
                       "Batchnorm layer: invalid input node name %s", src_id);
    bcnn_tensor_set_shape(
        &dst_tensor, net->tensors[node.src[0]].n, net->tensors[node.src[0]].c,
        net->tensors[node.src[0]].h, net->tensors[node.src[0]].w, 1);
    bcnn_tensor_allocate(&dst_tensor);
    bh_strfill(&dst_tensor.name, dst_id);
    // Add node to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(net, &node, net->num_tensors - 1);

    sz = bcnn_tensor_size(&net->tensors[node.dst[0]]);
    // Setup layer
    node.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    node.layer->type = BATCHNORM;
    channels = net->tensors[node.dst[0]].c;
    char saved_mean_name[256], saved_var_name[256], running_mean_name[256],
        running_var_name[256], scales_name[256], biases_name[256];
    sprintf(saved_mean_name, "%s_sav_mean", src_id);
    sprintf(saved_var_name, "%s_sav_var", src_id);
    sprintf(running_mean_name, "%s_run_mean", src_id);
    sprintf(running_var_name, "%s_run_var", src_id);
    sprintf(scales_name, "%s_scales", src_id);
    sprintf(biases_name, "%s_b", src_id);
    bcnn_tensor_create(&node.layer->saved_mean, 1, 1, 1, channels, 1,
                       saved_mean_name);
    bcnn_tensor_create(&node.layer->saved_variance, 1, 1, 1, channels, 1,
                       saved_var_name);
    bcnn_tensor running_mean = {0};
    bcnn_tensor_create(&running_mean, 1, 1, 1, channels, 0,
                       running_mean_name);  // no gradients
    bcnn_net_add_tensor(net, running_mean);
    bcnn_node_add_input(net, &node, net->num_tensors - 1);
    bcnn_tensor running_var = {0};
    bcnn_tensor_create(&running_var, 1, 1, 1, channels, 0,
                       running_var_name);  // no gradients
    bcnn_net_add_tensor(net, running_var);
    bcnn_node_add_input(net, &node, net->num_tensors - 1);
    bcnn_tensor scales = {0};
    bcnn_tensor_create(&scales, 1, 1, 1, channels, 1, scales_name);
    bcnn_tensor_filler filler = {.value = 1.0f, .type = FIXED};
    bcnn_tensor_fill(&scales, filler);
    bcnn_net_add_tensor(net, scales);
    bcnn_node_add_input(net, &node, net->num_tensors - 1);
    bcnn_tensor biases = {0};
    bcnn_tensor_create(&biases, 1, 1, 1, channels, 1, biases_name);
    bcnn_net_add_tensor(net, biases);
    bcnn_node_add_input(net, &node, net->num_tensors - 1);
    // Internal data
    node.layer->x_norm = (float *)calloc(sz, sizeof(float));
    node.layer->bn_workspace = (float *)calloc(sz, sizeof(float));
#ifdef BCNN_USE_CUDA
    node.layer->x_norm_gpu =
        bcnn_cuda_memcpy_f32(net->tensors[node.dst[0]].data, sz);
    node.layer->bn_workspace_gpu =
        bcnn_cuda_memcpy_f32(node.layer->bn_workspace, sz);
#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(
        &node.layer->dst_tensor_desc));  // same desc for x, dx, dy
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&node.layer->bias_desc));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(
        node.layer->dst_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        net->tensors[node.dst[0]].n, channels, net->tensors[node.dst[0]].h,
        net->tensors[node.dst[0]].w));
    bcnn_cudnn_check(
        cudnnSetTensor4dDescriptor(node.layer->bias_desc, CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT, 1, channels, 1, 1));
#endif
#endif

    // Add connection to net
    bcnn_net_add_node(net, node);

    BCNN_INFO(net->log_ctx,
              "[Batchnorm] input_shape= %dx%dx%d output_shape= %dx%dx%d",
              net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
              net->tensors[node.src[0]].c, net->tensors[node.dst[0]].w,
              net->tensors[node.dst[0]].h, net->tensors[node.dst[0]].c);

    return BCNN_SUCCESS;
}

static void _mean_variance_forward(float *x, int b, int c, int wxh, float *mean,
                                   float *var) {
    float scale = 1.0f / (b * wxh);
    int i, j, k;
    float s = 0.0f;

    for (i = 0; i < c; ++i) {
        mean[i] = 0;
        var[i] = 0;
        for (j = 0; j < b; ++j) {
            k = j * c * wxh + i * wxh;
            bcnn_vsum(wxh, x + k, &s);
            mean[i] += s;
            var[i] += bcnn_dot(wxh, x + k, x + k);
        }
        // TODO: check which option is faster here
        // mean[i] *= scale;
        // var[i] = var[i] * scale - mean[i] * mean[i];
    }
    bcnn_scal(c, scale, mean);
    bcnn_varmean(c, mean, scale, var);
}

static void _norm_forward(float *x, float *mean, float *variance, int b, int c,
                          int wxh) {
    int k, j, i, ind;

    for (k = 0; k < b; ++k) {
        for (j = 0; j < c; ++j) {
            for (i = 0; i < wxh; ++i) {
                ind = k * c * wxh + j * wxh + i;
                x[ind] = (x[ind] - mean[j]) / (sqrtf(variance[j] + 0.000001f));
            }
        }
    }
}

int bcnn_forward_batchnorm_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                     bcnn_tensor *dst_tensor,
                                     bcnn_tensor *bn_mean, bcnn_tensor *bn_var,
                                     bcnn_tensor *bn_scales,
                                     bcnn_tensor *biases) {
    int batch_size = src_tensor->n;
    int sz = dst_tensor->w * dst_tensor->h * dst_tensor->c;

    if (src_tensor != dst_tensor) {
        bcnn_copy_f32(sz * batch_size, src_tensor->data, dst_tensor->data);
    }
    bcnn_copy_f32(sz * batch_size, dst_tensor->data, layer->bn_workspace);

    if (layer->net_state) {
        _mean_variance_forward(dst_tensor->data, batch_size, dst_tensor->c,
                               dst_tensor->h * dst_tensor->w,
                               layer->saved_mean.data,
                               layer->saved_variance.data);

        bcnn_scal(dst_tensor->c, 0.9f, bn_mean->data);
        bcnn_axpy(dst_tensor->c, 0.1f, layer->saved_mean.data, bn_mean->data);
        bcnn_scal(dst_tensor->c, 0.9f, bn_var->data);
        bcnn_axpy(dst_tensor->c, 0.1f, layer->saved_variance.data,
                  bn_var->data);

        _norm_forward(dst_tensor->data, layer->saved_mean.data,
                      layer->saved_variance.data, batch_size, dst_tensor->c,
                      dst_tensor->h * dst_tensor->w);
        bcnn_copy_f32(batch_size * sz, dst_tensor->data, layer->x_norm);
    } else {
        // Normalize with global mean / variance
        _norm_forward(dst_tensor->data, bn_mean->data, bn_var->data, batch_size,
                      dst_tensor->c, dst_tensor->h * dst_tensor->w);
    }
    // dst <- scale * dst + bias
    bcnn_scales(dst_tensor->data, bn_scales->data, batch_size, dst_tensor->c,
                dst_tensor->h * dst_tensor->w);
    bcnn_add_bias(dst_tensor->data, biases->data, batch_size, dst_tensor->c,
                  dst_tensor->h * dst_tensor->w);
    return BCNN_SUCCESS;
}

static void _mean_variance_backward(float *x, float *grad, float *mean,
                                    float *var, int b, int c, int wxh,
                                    float *mean_diff, float *var_diff) {
    int i, j, k;
    float s = 0.0f;

    for (i = 0; i < c; ++i) {
        mean_diff[i] = 0;
        var_diff[i] = 0;
        for (j = 0; j < b; ++j) {
            k = j * c * wxh + i * wxh;
            bcnn_vsum(wxh, grad + k, &s);
            mean_diff[i] += s;
            var_diff[i] += bcnn_shiftdot(wxh, x + k, mean[i], grad + k, 0.0f);
        }
        mean_diff[i] *= (-1.0f / sqrtf(var[i] + 0.00001f));
    }
    bcnn_varnorm(c, var, -0.5f, var_diff);
}

static void _normalize_backward(float *x, float *mean, float *var,
                                float *mean_delta, float *var_diff, int b,
                                int c, int wxh, float *grad) {
    int i, j, k, ind;

    for (j = 0; j < b; ++j) {
        for (i = 0; i < c; ++i) {
            for (k = 0; k < wxh; ++k) {
                ind = j * c * wxh + i * wxh + k;
                grad[ind] =
                    grad[ind] * 1.0f / (sqrtf(var[i] + 0.00001f)) +
                    var_diff[i] * 2.0f * (x[ind] - mean[i]) / (wxh * b) +
                    mean_delta[i] / (wxh * b);
            }
        }
    }
}

int bcnn_backward_batchnorm_layer_cpu(bcnn_layer *layer,
                                      bcnn_tensor *src_tensor,
                                      bcnn_tensor *dst_tensor,
                                      bcnn_tensor *bn_mean, bcnn_tensor *bn_var,
                                      bcnn_tensor *bn_scales,
                                      bcnn_tensor *bn_biases) {
    int batch_size = src_tensor->n;
    int sz = dst_tensor->w * dst_tensor->h * dst_tensor->c;

    if (!layer->net_state) {
        layer->saved_mean.data = bn_mean->data;
        layer->saved_variance.data = bn_var->data;
    }
    bcnn_grad_bias(bn_biases->grad_data, dst_tensor->grad_data, batch_size,
                   dst_tensor->c, dst_tensor->h * dst_tensor->w);
    bcnn_grad_scales(layer->x_norm, dst_tensor->grad_data, batch_size,
                     dst_tensor->c, dst_tensor->h * dst_tensor->w,
                     bn_scales->grad_data);
    bcnn_scales(dst_tensor->grad_data, bn_scales->data, batch_size,
                dst_tensor->c, dst_tensor->h * dst_tensor->w);
    _mean_variance_backward(
        layer->bn_workspace, dst_tensor->grad_data, layer->saved_mean.data,
        layer->saved_variance.data, batch_size, dst_tensor->c,
        dst_tensor->w * dst_tensor->h, layer->saved_mean.grad_data,
        layer->saved_variance.grad_data);
    _normalize_backward(layer->bn_workspace, layer->saved_mean.data,
                        layer->saved_variance.data, layer->saved_mean.grad_data,
                        layer->saved_variance.grad_data, batch_size,
                        dst_tensor->c, dst_tensor->w * dst_tensor->h,
                        dst_tensor->grad_data);

    if (src_tensor->grad_data && src_tensor != dst_tensor) {
        bcnn_copy_f32(sz * batch_size, dst_tensor->grad_data,
                      src_tensor->grad_data);
    }
    return BCNN_SUCCESS;
}

int bcnn_forward_batchnorm_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
    bcnn_tensor *bn_mean = &net->tensors[node->src[1]];
    bcnn_tensor *bn_var = &net->tensors[node->src[2]];
    bcnn_tensor *bn_scales = &net->tensors[node->src[3]];
    bcnn_tensor *bn_bias = &net->tensors[node->src[4]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_batchnorm_layer_gpu(node->layer, src, dst, bn_mean,
                                            bn_var, bn_scales, bn_bias);
#else
    return bcnn_forward_batchnorm_layer_cpu(node->layer, src, dst, bn_mean,
                                            bn_var, bn_scales, bn_bias);
#endif
}

int bcnn_backward_batchnorm_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
    bcnn_tensor *bn_mean = &net->tensors[node->src[1]];
    bcnn_tensor *bn_var = &net->tensors[node->src[2]];
    bcnn_tensor *bn_scales = &net->tensors[node->src[3]];
    bcnn_tensor *bn_bias = &net->tensors[node->src[4]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_batchnorm_layer_gpu(node->layer, src, dst, bn_mean,
                                             bn_var, bn_scales, bn_bias);
#else
    return bcnn_backward_batchnorm_layer_cpu(node->layer, src, dst, bn_mean,
                                             bn_var, bn_scales, bn_bias);
#endif
}
