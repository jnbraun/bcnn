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

#include "bcnn_cost_layer.h"

#include <float.h>
#include <math.h>

#include <bh/bh_macros.h>
#include <bh/bh_string.h>

#include "bcnn_mat.h"
#include "bcnn_net.h"
#include "bcnn_tensor.h"
#include "bcnn_utils.h"

bcnn_status bcnn_add_cost_layer(bcnn_net *net, bcnn_loss loss,
                                bcnn_loss_metric loss_metric, float scale,
                                const char *src_id, const char *label_id,
                                const char *dst_id) {
    int sz, i;
    bcnn_node node = {0};
    bcnn_tensor dst_tensor = {0};

    BCNN_CHECK_AND_LOG(net->log_ctx, net->num_nodes >= 1,
                       BCNN_INVALID_PARAMETER,
                       "Cost layer can't be the first layer of the network");
    int is_src_node_found = 0;
    for (i = net->num_tensors - 1; i >= 0; --i) {
        if (strcmp(net->tensors[i].name, src_id) == 0) {
            bcnn_node_add_input(net, &node, i);
            is_src_node_found = 1;
            break;
        }
    }
    BCNN_CHECK_AND_LOG(net->log_ctx, is_src_node_found, BCNN_INVALID_PARAMETER,
                       "Cost layer: invalid input node name %s", src_id);

    // Fill nodes param
    node.type = BCNN_LAYER_COST;
    node.param_size = sizeof(bcnn_cost_param);
    node.param = (bcnn_cost_param *)calloc(1, node.param_size);
    bcnn_cost_param *param = (bcnn_cost_param *)node.param;
    param->scale = scale;
    param->loss = loss;
    param->loss_metric = loss_metric;
    node.forward = bcnn_forward_cost_layer;
    node.backward = bcnn_backward_cost_layer;

    // Setup label node
    bcnn_tensor_set_shape(&net->tensors[1], net->tensors[node.src[0]].n,
                          net->tensors[node.src[0]].c,
                          net->tensors[node.src[0]].h,
                          net->tensors[node.src[0]].w, 0);
    bcnn_tensor_allocate(&net->tensors[1], net->mode);
    // Add pointer to label node to connection
    bcnn_node_add_input(net, &node, /*label_id=*/1);

    // Create output node
    bcnn_tensor_set_shape(
        &dst_tensor, net->tensors[node.src[0]].n, net->tensors[node.src[0]].c,
        net->tensors[node.src[0]].h, net->tensors[node.src[0]].w, 1);
    bcnn_tensor_allocate(&dst_tensor, net->mode);
    // bh_strfill(&dst_tensor.name, dst_id);
    dst_tensor.name = dst_id;
    // Add node to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(net, &node, net->num_tensors - 1);

    bcnn_net_add_node(net, node);
    return 0;
}

static void bcnn_huber_loss(int n, float *x, float *label, float *error,
                            float *grad_error, float hdelta) {
    int i;
    float e = 0.0f;

    for (i = 0; i < n; ++i) {
        e = x[i] - label[i];
        if (fabs(e) > hdelta) {
            grad_error[i] = (e > 0) ? 1.0f : -1.0f;
            *error += 2.0f * hdelta * fabs(e) - hdelta * hdelta;
        } else {
            grad_error[i] = e;
            *error += e * e;
        }
    }
    return;
}

static void bcnn_euclidean_loss_forward(bcnn_tensor *src_tensor,
                                        bcnn_tensor *label,
                                        bcnn_tensor *dst_tensor) {
    int size = bcnn_tensor_size(src_tensor);
#ifdef BCNN_USE_CUDA
    bcnn_cuda_copy_f32(size, src_tensor->data_gpu, 1, dst_tensor->grad_data_gpu,
                       1);
    bcnn_cuda_axpy(size, -1, label->data_gpu, 1, dst_tensor->grad_data_gpu, 1);
#else
    bcnn_copy_f32(size, src_tensor->data, dst_tensor->grad_data);
    bcnn_axpy(size, -1, label->data, dst_tensor->grad_data);
#endif
}

static void bcnn_euclidean_loss_backward(bcnn_tensor *src_tensor,
                                         bcnn_tensor *dst_tensor,
                                         bcnn_cost_param *param) {
    int size = bcnn_tensor_size(src_tensor);
#ifdef BCNN_USE_CUDA
    bcnn_cuda_axpy(size, param->scale, dst_tensor->grad_data_gpu, 1,
                   src_tensor->grad_data_gpu, 1);
#else
    bcnn_axpy(size, param->scale, dst_tensor->grad_data, src_tensor->grad_data);
#endif
}

void bcnn_compute_error(bcnn_cost_param *param, bcnn_tensor *src_tensor,
                        bcnn_tensor *label, bcnn_tensor *dst_tensor) {
    int input_size = src_tensor->w * src_tensor->h * src_tensor->c;
    int batch_size = src_tensor->n;
    int sz = src_tensor->n * input_size;
#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_dev2host(dst_tensor->grad_data_gpu, dst_tensor->grad_data,
                              sz);
#endif

    switch (param->loss_metric) {
        case BCNN_METRIC_ERROR_RATE:
            *(dst_tensor->data) = 0.0f;
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(src_tensor->data_gpu, src_tensor->data,
                                      sz);
            bcnn_cuda_memcpy_dev2host(label->data_gpu, label->data, sz);
#endif
            for (int i = 0; i < batch_size; ++i) {
                int offset = i * input_size;
                float p_max = FLT_MIN;
                int j_best = 0;
                for (int j = 0; j < input_size; ++j) {
                    if (src_tensor->data[offset + j] > p_max) {
                        p_max = src_tensor->data[offset + j];
                        j_best = j;
                    }
                }
                if (label->data[offset + j_best] == 0) {
                    *(dst_tensor->data) += 1.0f;
                }
            }
            break;
        case BCNN_METRIC_SSE:
            *(dst_tensor->data) =
                bcnn_dot(sz, dst_tensor->grad_data, dst_tensor->grad_data);
            break;
        case BCNN_METRIC_MSE:
            *(dst_tensor->data) =
                bcnn_dot(sz, dst_tensor->grad_data, dst_tensor->grad_data);
            *(dst_tensor->data) /= input_size;
            break;
        case BCNN_METRIC_CRPS:
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(src_tensor->data_gpu, src_tensor->data,
                                      sz);
            bcnn_cuda_memcpy_dev2host(label->data_gpu, label->data, sz);
#endif
            *(dst_tensor->data) = 0.0f;
            float *input_cpu = (float *)calloc(sz, sizeof(float));
            for (int i = 0; i < batch_size; ++i) {
                int offset = i * input_size;
                for (int j = 1; j < input_size; ++j) {
                    if (src_tensor->data[offset + j] <
                        src_tensor->data[offset + j - 1]) {
                        input_cpu[offset + j] =
                            src_tensor->data[offset + j - 1];
                    }
                }
            }
            *(dst_tensor->data) =
                bcnn_dot(sz, dst_tensor->grad_data, dst_tensor->grad_data);
            bh_free(input_cpu);
            break;
        case BCNN_METRIC_LOGLOSS:
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(src_tensor->data_gpu, src_tensor->data,
                                      sz);
            bcnn_cuda_memcpy_dev2host(label->data_gpu, label->data, sz);
#endif
            *(dst_tensor->data) = 0.0f;
            for (int i = 0; i < batch_size; ++i) {
                int offset = i * input_size;
                for (int j = 0; j < input_size; ++j) {
                    if (label->data[offset + j] > 0.0f) {
                        *(dst_tensor->data) += (float)-log(bh_clamp(
                            src_tensor->data[offset + j], 1e-8f, 1.0f - 1e-8f));
                    }
                }
            }
            break;
        case BCNN_METRIC_DICE:
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(src_tensor->data_gpu, src_tensor->data,
                                      sz);
            bcnn_cuda_memcpy_dev2host(label->data_gpu, label->data, sz);
#endif
            *(dst_tensor->data) = 0.0f;
            for (int i = 0; i < batch_size; ++i) {
                int offset = i * input_size;
                int n = 0;
                int d = 0;
                for (int j = 0; j < input_size; ++j) {
                    n += (int)(label->data[offset + j] *
                               (src_tensor->data[offset + j] > 0.5f));
                    d += (int)(label->data[offset + j] +
                               (src_tensor->data[offset + j] > 0.5f));
                }
                *(dst_tensor->data) += (float)(2.0f * n + 1.0f) / (d + 1.0f);
            }
            break;
    }
}

void bcnn_forward_cost_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_tensor *label = &net->tensors[1];
    bcnn_cost_param *param = (bcnn_cost_param *)node->param;
    // If no truth available, do nothing
    if (!label->data) {
        return;
    }
    switch (param->loss) {
        case BCNN_LOSS_EUCLIDEAN:
            bcnn_euclidean_loss_forward(src_tensor, label, dst_tensor);
            break;
        case BCNN_LOSS_LIFTED_STRUCT:
            bcnn_lifted_struct_loss_forward(net, node);
            break;
    }

    bcnn_compute_error(param, src_tensor, label, dst_tensor);

    return;
}

void bcnn_backward_cost_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_cost_param *param = (bcnn_cost_param *)node->param;
    switch (param->loss) {
        case BCNN_LOSS_EUCLIDEAN:
            bcnn_euclidean_loss_backward(src_tensor, dst_tensor, param);
            break;
        case BCNN_LOSS_LIFTED_STRUCT:
            bcnn_lifted_struct_loss_backward(net, node);
            break;
    }

    return;
}
