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

#include <bh/bh_error.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>

#include "bh_log.h"

int bcnn_add_batchnorm_layer(bcnn_net *net, char *src_id, char *dst_id) {
    int i, sz, channels;
    bcnn_connection conn = {0};
    bcnn_node dst_node = {0};

    bh_check(net->nb_connections >= 1,
             "Batchnorm layer can't be the first layer of the network");

    int is_src_node_found = 0;
    for (i = net->num_nodes - 1; i >= 0; --i) {
        if (strcmp(net->nodes[i].id, src_id) == 0) {
            bcnn_connection_add_src_node(&conn, i);
            is_src_node_found = 1;
            break;
        }
    }
    bh_check(is_src_node_found, "Batchnorm layer: invalid input node name %s",
             src_id);

    bh_strfill(&dst_node.id, dst_id);
    bcnn_tensor_set_shape(&dst_node.tensor, net->nodes[conn.src[0]].tensor.n,
                          net->nodes[conn.src[0]].tensor.c,
                          net->nodes[conn.src[0]].tensor.h,
                          net->nodes[conn.src[0]].tensor.w, 1);
    bcnn_tensor_allocate(&dst_node.tensor);
    // Add node to net
    bcnn_net_add_node(net, dst_node);
    // Add node pointer to connection
    bcnn_connection_add_dst_node(&conn, net->num_nodes - 1);

    sz = bcnn_tensor_get_size(&net->nodes[conn.dst[0]].tensor);
    // Setup layer
    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = BATCHNORM;
    channels = net->nodes[conn.dst[0]].tensor.c;
    bcnn_tensor_create(&conn.layer->saved_mean, 1, 1, 1, channels, 1);
    bcnn_tensor_create(&conn.layer->saved_variance, 1, 1, 1, channels, 1);
    bcnn_tensor_create(&conn.layer->running_mean, 1, 1, 1, channels,
                       0);  // no gradients
    bcnn_tensor_create(&conn.layer->running_variance, 1, 1, 1, channels,
                       0);  // no gradients
    conn.layer->x_norm = (float *)calloc(sz, sizeof(float));
    conn.layer->bn_workspace = (float *)calloc(sz, sizeof(float));
    bcnn_tensor_create(&conn.layer->scales, 1, 1, 1, channels, 1);
    bcnn_tensor_filler filler = {.value = 1.0f, .type = FIXED};
    bcnn_tensor_fill(&conn.layer->scales, filler);
    bcnn_tensor_create(&conn.layer->biases, 1, 1, 1, channels, 1);
#ifdef BCNN_USE_CUDA
    conn.layer->x_norm_gpu =
        bcnn_cuda_memcpy_f32(net->nodes[conn.dst[0]].tensor.data, sz);
    conn.layer->bn_workspace_gpu =
        bcnn_cuda_memcpy_f32(conn.layer->bn_workspace, sz);
#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(
        &conn.layer->src_tensor_desc));  // same desc for x, dx, dy
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&conn.layer->dst_tensor_desc));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(
        conn.layer->src_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        net->nodes[conn.dst[0]].tensor.n, channels,
        net->nodes[conn.dst[0]].tensor.h, net->nodes[conn.dst[0]].tensor.w));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(
        conn.layer->dst_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
        channels, 1, 1));
#endif
#endif

    // Add connection to net
    bcnn_net_add_connection(net, conn);

    bh_log_info(
        "[Batchnorm] input_shape= %dx%dx%d output_shape= %dx%dx%d",
        net->nodes[conn.src[0]].tensor.w, net->nodes[conn.src[0]].tensor.h,
        net->nodes[conn.src[0]].tensor.c, net->nodes[conn.dst[0]].tensor.w,
        net->nodes[conn.dst[0]].tensor.h, net->nodes[conn.dst[0]].tensor.c);

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

// int bcnn_forward_batchnorm_layer_cpu(bcnn_connection *conn)
int bcnn_forward_batchnorm_layer_cpu(bcnn_layer *layer, bcnn_node *src_node,
                                     bcnn_node *dst_node) {
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = src.n;
    int sz = dst.w * dst.h * dst.c;

    bcnn_copy_f32(sz * batch_size, src.data, dst.data);
    bcnn_copy_f32(sz * batch_size, dst.data, layer->bn_workspace);

    if (layer->net_state) {
        _mean_variance_forward(dst.data, batch_size, dst.c, dst.h * dst.w,
                               layer->saved_mean.data,
                               layer->saved_variance.data);

        bcnn_scal(dst.c, 0.9f, layer->running_mean.data);
        bcnn_axpy(dst.c, 0.1f, layer->saved_mean.data,
                  layer->running_mean.data);
        bcnn_scal(dst.c, 0.9f, layer->running_variance.data);
        bcnn_axpy(dst.c, 0.1f, layer->saved_variance.data,
                  layer->running_variance.data);

        _norm_forward(dst.data, layer->saved_mean.data,
                      layer->saved_variance.data, batch_size, dst.c,
                      dst.h * dst.w);
        bcnn_copy_f32(batch_size * sz, dst.data, layer->x_norm);
    } else {
        // Normalize with global mean / variance
        _norm_forward(dst.data, layer->running_mean.data,
                      layer->running_variance.data, batch_size, dst.c,
                      dst.h * dst.w);
    }

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

int bcnn_backward_batchnorm_layer_cpu(bcnn_layer *layer, bcnn_node *src_node,
                                      bcnn_node *dst_node) {
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = src.n;
    int sz = dst.w * dst.h * dst.c;

    if (!layer->net_state) {
        layer->saved_mean.data = layer->running_mean.data;
        layer->saved_variance.data = layer->running_variance.data;
    }

    _mean_variance_backward(
        layer->bn_workspace, dst.grad_data, layer->saved_mean.data,
        layer->saved_variance.data, batch_size, dst.c, dst.w * dst.h,
        layer->saved_mean.grad_data, layer->saved_variance.grad_data);
    _normalize_backward(layer->bn_workspace, layer->saved_mean.data,
                        layer->saved_variance.data, layer->saved_mean.grad_data,
                        layer->saved_variance.grad_data, batch_size, dst.c,
                        dst.w * dst.h, dst.grad_data);

    if (src.grad_data)
        bcnn_copy_f32(sz * batch_size, dst.grad_data, src.grad_data);

    return BCNN_SUCCESS;
}

int bcnn_forward_batchnorm_layer(bcnn_net *net, bcnn_connection *conn) {
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_batchnorm_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_forward_batchnorm_layer_cpu(conn->layer, src, dst);
#endif
}

int bcnn_backward_batchnorm_layer(bcnn_net *net, bcnn_connection *conn) {
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_batchnorm_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_backward_batchnorm_layer_cpu(conn->layer, src, dst);
#endif
}
