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

#include "bcnn_activation_layer.h"

#include <math.h>
#include <string.h>

#include <bh/bh_log.h>
#include <bh/bh_macros.h>

#include "bcnn_learner.h"
#include "bcnn_net.h"
#include "bcnn_tensor.h"
#include "bcnn_utils.h"

bcnn_status bcnn_add_activation_layer(bcnn_net *net, bcnn_activation type,
                                      const char *src_id) {
    bcnn_node node = {0};

    BCNN_CHECK_AND_LOG(
        net->log_ctx, net->num_nodes >= 1, BCNN_INVALID_PARAMETER,
        "Activation layer can't be the first layer of the network\n");
    int is_src_node_found = 0;
    for (int i = net->num_tensors - 1; i >= 0; --i) {
        if (strcmp(net->tensors[i].name, src_id) == 0) {
            bcnn_node_add_input(net, &node, i);
            bcnn_node_add_output(net, &node, i);
            is_src_node_found = 1;
            break;
        }
    }
    BCNN_CHECK_AND_LOG(net->log_ctx, is_src_node_found, BCNN_INVALID_PARAMETER,
                       "Activation layer: invalid input node name %s\n",
                       src_id);

    node.type = BCNN_LAYER_ACTIVATION;
    node.param_size = sizeof(bcnn_activation_param);
    node.param = (bcnn_activation_param *)calloc(1, node.param_size);
    bcnn_activation_param *param = (bcnn_activation_param *)node.param;
    param->activation = type;
    node.forward = bcnn_forward_activation_layer;
    node.backward = bcnn_backward_activation_layer;
    node.update = bcnn_update_activation_layer;
    if (type == BCNN_ACT_PRELU) {
        char weights_name[256];
        sprintf(weights_name, "%s_w_prelu", src_id);
        bcnn_tensor weights = {0};
        bcnn_tensor_create(&weights, 1, 1, 1, net->tensors[node.src[0]].c, 1,
                           weights_name, net->mode);
        bcnn_net_add_tensor(net, weights);
        bcnn_node_add_input(net, &node, net->num_tensors - 1);
    }

    bcnn_net_add_node(net, node);

    char node_opname[256];
    snprintf(node_opname, 256, BH_LOG_BOLDBLUE "[%s]" BH_LOG_RESET,
             bcnn_act2str(type));
    BCNN_INFO(net->log_ctx,
              "%-48s %-8s (%4d x%4d x%4d) -> %-8s (%4d x%4d x%4d)\n",
              node_opname, net->tensors[node.src[0]].name,
              net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
              net->tensors[node.src[0]].c, net->tensors[node.dst[0]].name,
              net->tensors[node.dst[0]].w, net->tensors[node.dst[0]].h,
              net->tensors[node.dst[0]].c);

    return BCNN_SUCCESS;
}

void bcnn_forward_activation_cpu(float *x, int sz, float *slope,
                                 int spatial_size, int channels,
                                 bcnn_activation a) {
    switch (a) {
        case BCNN_ACT_TANH:
            for (int i = 0; i < sz; ++i) {
                x[i] = (float)(exp(2 * x[i]) - 1) / ((float)exp(2 * x[i]) + 1);
            }
            break;
        case BCNN_ACT_RELU:
            for (int i = 0; i < sz; ++i) {
                x[i] = x[i] * (x[i] > 0);
            }
            break;
        case BCNN_ACT_LRELU:
            for (int i = 0; i < sz; ++i) {
                x[i] = (x[i] > 0 ? x[i] : 0.1f * x[i]);
            }
            break;
        case BCNN_ACT_RAMP:
            for (int i = 0; i < sz; ++i) {
                x[i] = x[i] * (x[i] > 0) + 0.1f * x[i];
            }
            break;
        case BCNN_ACT_SOFTPLUS:
            for (int i = 0; i < sz; ++i) {
                x[i] = (float)log(1.0f + (float)exp(x[i]));
            }
            break;
        case BCNN_ACT_ABS:
            for (int i = 0; i < sz; ++i) {
                x[i] = (float)fabs(x[i]);
            }
            break;
        case BCNN_ACT_CLAMP:
            for (int i = 0; i < sz; ++i) {
                x[i] = bh_clamp(x[i], 0, 1);
            }
            break;
        case BCNN_ACT_LOGISTIC:
            for (int i = 0; i < sz; ++i) {
                x[i] = 1.0f / (1.0f + (float)exp(-x[i]));
            }
            break;
        case BCNN_ACT_PRELU:
            for (int i = 0; i < sz; ++i) {
                int c = (i / spatial_size) % channels;
                x[i] = (x[i] > 0 ? x[i] : slope[c] * x[i]);
            }
            break;
        case BCNN_ACT_NONE:
            break;
        default:
            break;
    }
    return;
}

void bcnn_forward_activation_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_activation_param *param = (bcnn_activation_param *)node->param;
    bcnn_tensor *weights = NULL;
    if (param->activation == BCNN_ACT_PRELU) {
        weights = &net->tensors[node->src[1]];
    }
    int sz = bcnn_tensor_size(dst_tensor);
    dst_tensor->data = src_tensor->data;
    bcnn_forward_activation_cpu(dst_tensor->data, sz, weights->data,
                                dst_tensor->w * dst_tensor->h, dst_tensor->c,
                                param->activation);

    return;
}

void bcnn_backward_activation_cpu(float *x, float *dx, int sz, float *slope,
                                  float *grad_slope, int spatial_size,
                                  int channels, bcnn_activation a) {
    switch (a) {
        case BCNN_ACT_TANH:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= (1 - x[i] * x[i]);
            }
            break;
        case BCNN_ACT_RELU:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= ((float)(x[i] > 0));
            }
            break;
        case BCNN_ACT_LRELU:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= (x[i] > 0 ? 1.0f : 0.1f);
            }
            break;
        case BCNN_ACT_RAMP:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= ((float)(x[i] > 0) + 0.1f);
            }
            break;
        case BCNN_ACT_SOFTPLUS:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= 1.0f / (1.0f + (float)exp(-x[i]));
            }
            break;
        case BCNN_ACT_ABS:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= (x[i] >= 0 ? 1.0f : -1.0f);
            }
            break;
        case BCNN_ACT_CLAMP:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= ((float)(x[i] > 0.0f && x[i] < 1.0f));
            }
            break;
        case BCNN_ACT_LOGISTIC:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= (1 - x[i]) * x[i];
            }
            break;
        case BCNN_ACT_NONE:
            break;
        case BCNN_ACT_PRELU: {
            for (int i = 0; i < sz; ++i) {
                int c = (i / spatial_size) % channels;
                grad_slope[c] += dx[i] * x[i] * (x[i] < 0);
            }
            for (int i = 0; i < sz; ++i) {
                int c = (i / spatial_size) % channels;
                dx[i] *= (x[i] > 0 ? 1.0f : slope[c]);
            }
            break;
        }
        default:
            break;
    }
    return;
}

void bcnn_backward_activation_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_activation_param *param = (bcnn_activation_param *)node->param;
    bcnn_tensor *weights = NULL;
    if (param->activation == BCNN_ACT_PRELU) {
        weights = &net->tensors[node->src[1]];
    }
    int sz = bcnn_tensor_size(dst_tensor);
    bcnn_backward_activation_cpu(dst_tensor->data, dst_tensor->grad_data, sz,
                                 weights->data, weights->grad_data,
                                 dst_tensor->w * dst_tensor->h, dst_tensor->c,
                                 param->activation);
    src_tensor->grad_data = dst_tensor->grad_data;

    return;
}

void bcnn_forward_activation_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_forward_activation_layer_gpu(net, node);
#else
    return bcnn_forward_activation_layer_cpu(net, node);
#endif
}

void bcnn_backward_activation_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_backward_activation_layer_gpu(net, node);
#else
    return bcnn_backward_activation_layer_cpu(net, node);
#endif
}

void bcnn_update_activation_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_activation_param *param = (bcnn_activation_param *)node->param;
    if (param->activation == BCNN_ACT_PRELU) {
        bcnn_tensor *weights = &net->tensors[node->src[1]];
        if (net->learner->optimizer == BCNN_OPTIM_SGD) {
#ifdef BCNN_USE_CUDA
            bcnn_sgd_update_gpu(weights->data_gpu, NULL, weights->grad_data_gpu,
                                NULL, bcnn_tensor_size(weights), 0, weights->n,
                                net->learner->learning_rate,
                                net->learner->momentum, net->learner->decay);
#else
            bcnn_sgd_update_cpu(weights->data, NULL, weights->grad_data, NULL,
                                bcnn_tensor_size(weights), 0, weights->n,
                                net->learner->learning_rate,
                                net->learner->momentum, net->learner->decay);
#endif
        } else if (net->learner->optimizer == BCNN_OPTIM_ADAM) {
#ifdef BCNN_USE_CUDA
            bcnn_sgd_update_gpu(weights->data_gpu, NULL, weights->grad_data_gpu,
                                NULL, bcnn_tensor_size(weights), 0, weights->n,
                                net->learner->learning_rate,
                                net->learner->momentum, net->learner->decay);
#else
            bcnn_sgd_update_cpu(weights->data, NULL, weights->grad_data, NULL,
                                bcnn_tensor_size(weights), 0, weights->n,
                                net->learner->learning_rate,
                                net->learner->momentum, net->learner->decay);
#endif
        }
    }
    return;
}