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

#include <bh/bh_macros.h>
#include "bcnn_learner.h"

bcnn_status bcnn_add_activation_layer(bcnn_net *net, bcnn_activation type,
                                      char *src_id) {
    bcnn_node node = {0};
    char type_name[256];
    int i;

    BCNN_CHECK_AND_LOG(
        net->log_ctx, net->num_nodes >= 1, BCNN_INVALID_PARAMETER,
        "Activation layer can't be the first layer of the network");
    node.type = ACTIVATION;
    node.param_size = sizeof(bcnn_activation_param);
    node.param = (bcnn_activation_param *)calloc(1, node.param_size);
    bcnn_activation_param *param = (bcnn_activation_param *)node.param;
    param->activation = type;
    node.forward = bcnn_forward_activation_layer;
    node.backward = bcnn_backward_activation_layer;
    node.update = bcnn_update_activation_layer;

    int is_src_node_found = 0;
    for (i = net->num_tensors - 1; i >= 0; --i) {
        if (strcmp(net->tensors[i].name, src_id) == 0) {
            bcnn_node_add_input(net, &node, i);
            bcnn_node_add_output(net, &node, i);
            is_src_node_found = 1;
            break;
        }
    }
    BCNN_CHECK_AND_LOG(net->log_ctx, is_src_node_found, BCNN_INVALID_PARAMETER,
                       "Activation layer: invalid input node name %s", src_id);
    if (type == PRELU) {
        char weights_name[256];
        sprintf(weights_name, "%s_w_prelu", src_id);
        bcnn_tensor weights = {0};
        bcnn_tensor_create(&weights, 1, 1, 1, net->tensors[node.src[0]].c, 1,
                           weights_name, net->state);
        bcnn_net_add_tensor(net, weights);
        bcnn_node_add_input(net, &node, net->num_tensors - 1);
    }

    bcnn_net_add_node(net, node);

    switch (type) {
        case TANH:
            sprintf(type_name, "Tanh");
            break;
        case RELU:
            sprintf(type_name, "ReLU");
            break;
        case RAMP:
            sprintf(type_name, "Ramp");
            break;
        case SOFTPLUS:
            sprintf(type_name, "Softplus");
            break;
        case LRELU:
            sprintf(type_name, "Leaky-ReLU");
            break;
        case ABS:
            sprintf(type_name, "AbsVal");
            break;
        case CLAMP:
            sprintf(type_name, "Clamp");
            break;
        case PRELU:
            sprintf(type_name, "PReLU");
            break;
        default:
            sprintf(type_name, "None");
            break;
    }

    BCNN_INFO(net->log_ctx,
              "[Activation] input_shape= %dx%dx%d function= %s output_shape= "
              "%dx%dx%d",
              net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
              net->tensors[node.src[0]].c, type_name,
              net->tensors[node.dst[0]].w, net->tensors[node.dst[0]].h,
              net->tensors[node.dst[0]].c);

    return BCNN_SUCCESS;
}

void bcnn_forward_activation_cpu(float *x, int sz, bcnn_activation a) {
    switch (a) {
        case TANH:
            for (int i = 0; i < sz; ++i) {
                x[i] = (float)(exp(2 * x[i]) - 1) / ((float)exp(2 * x[i]) + 1);
            }
            break;
        case RELU:
            for (int i = 0; i < sz; ++i) {
                x[i] = x[i] * (x[i] > 0);
            }
            break;
        case LRELU:
            for (int i = 0; i < sz; ++i) {
                x[i] = (x[i] > 0 ? x[i] : 0.1f * x[i]);
            }
            break;
        case RAMP:
            for (int i = 0; i < sz; ++i) {
                x[i] = x[i] * (x[i] > 0) + 0.1f * x[i];
            }
            break;
        case SOFTPLUS:
            for (int i = 0; i < sz; ++i) {
                x[i] = (float)log(1.0f + (float)exp(x[i]));
            }
            break;
        case ABS:
            for (int i = 0; i < sz; ++i) {
                x[i] = (float)fabs(x[i]);
            }
            break;
        case CLAMP:
            for (int i = 0; i < sz; ++i) {
                x[i] = bh_clamp(x[i], 0, 1);
            }
            break;
        case LOGISTIC:
            for (int i = 0; i < sz; ++i) {
                x[i] = 1.0f / (1.0f + (float)exp(-x[i]));
            }
            break;
        case NONE:
            break;
        default:
            break;
    }
    return;
}

static void bcnn_forward_prelu(float *x, float *slope, int size,
                               int spatial_size, int channels) {
    for (int i = 0; i < size; ++i) {
        int c = (i / spatial_size) % channels;
        x[i] = (x[i] > 0 ? x[i] : slope[c] * x[i]);
    }
}

void bcnn_forward_activation_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_activation_param *param = (bcnn_activation_param *)node->param;
    bcnn_tensor *weights = NULL;
    if (param->activation == PRELU) {
        weights = &net->tensors[node->src[1]];
    }
    int sz = bcnn_tensor_size(dst_tensor);
    dst_tensor->data = src_tensor->data;
    if (param->activation == PRELU) {
        bcnn_forward_prelu(dst_tensor->data, weights->data, sz,
                           dst_tensor->w * dst_tensor->h, dst_tensor->c);
    } else {
        bcnn_forward_activation_cpu(dst_tensor->data, sz, param->activation);
    }
    return;
}

void bcnn_backward_activation_cpu(float *x, float *dx, int sz,
                                  bcnn_activation a) {
    switch (a) {
        case TANH:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= (1 - x[i] * x[i]);
            }
            break;
        case RELU:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= ((float)(x[i] > 0));
            }
            break;
        case LRELU:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= (x[i] > 0 ? 1.0f : 0.1f);
            }
            break;
        case RAMP:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= ((float)(x[i] > 0) + 0.1f);
            }
            break;
        case SOFTPLUS:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= 1.0f / (1.0f + (float)exp(-x[i]));
            }
            break;
        case ABS:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= (x[i] >= 0 ? 1.0f : -1.0f);
            }
            break;
        case CLAMP:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= ((float)(x[i] > 0.0f && x[i] < 1.0f));
            }
            break;
        case LOGISTIC:
            for (int i = 0; i < sz; ++i) {
                dx[i] *= (1 - x[i]) * x[i];
            }
            break;
        case NONE:
            break;
        default:
            break;
    }
    return;
}

static void bcnn_backward_prelu(float *x, float *dx, float *slope,
                                float *grad_slope, int size, int spatial_size,
                                int channels) {
    for (int i = 0; i < size; ++i) {
        int c = (i / spatial_size) % channels;
        grad_slope[c] += dx[i] * x[i] * (x[i] < 0);
    }
    for (int i = 0; i < size; ++i) {
        int c = (i / spatial_size) % channels;
        dx[i] *= (x[i] > 0 ? 1.0f : slope[c]);
    }
}

void bcnn_backward_activation_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_activation_param *param = (bcnn_activation_param *)node->param;
    bcnn_tensor *weights = NULL;
    if (param->activation == PRELU) {
        weights = &net->tensors[node->src[1]];
    }
    int sz = bcnn_tensor_size(dst_tensor);

    if (param->activation == PRELU) {
        bcnn_backward_prelu(dst_tensor->data, dst_tensor->grad_data,
                            weights->data, weights->grad_data, sz,
                            dst_tensor->w * dst_tensor->h, dst_tensor->c);
    } else {
        bcnn_backward_activation_cpu(dst_tensor->data, dst_tensor->grad_data,
                                     sz, param->activation);
    }
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
    if (param->activation = PRELU) {
        bcnn_tensor *weights = &net->tensors[node->src[1]];
        if (net->learner.optimizer == SGD) {
#ifdef BCNN_USE_CUDA
            bcnn_sgd_update_gpu(weights->data_gpu, NULL, weights->grad_data_gpu,
                                NULL, bcnn_tensor_size(weights), 0, weights->n,
                                net->learner.learning_rate,
                                net->learner.momentum, net->learner.decay);
#else
            bcnn_sgd_update_cpu(weights->data, NULL, weights->grad_data, NULL,
                                bcnn_tensor_size(weights), 0, weights->n,
                                net->learner.learning_rate,
                                net->learner.momentum, net->learner.decay);
#endif
        } else if (net->learner.optimizer == ADAM) {
#ifdef BCNN_USE_CUDA
            bcnn_sgd_update_gpu(weights->data_gpu, NULL, weights->grad_data_gpu,
                                NULL, bcnn_tensor_size(weights), 0, weights->n,
                                net->learner.learning_rate,
                                net->learner.momentum, net->learner.decay);
#else
            bcnn_sgd_update_cpu(weights->data, NULL, weights->grad_data, NULL,
                                bcnn_tensor_size(weights), 0, weights->n,
                                net->learner.learning_rate,
                                net->learner.momentum, net->learner.decay);
#endif
        }
    }
    return;
}