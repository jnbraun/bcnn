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
#include "bcnn_dropout_layer.h"

#include <bh/bh_string.h>

#include "bcnn_utils.h"

bcnn_status bcnn_add_dropout_layer(bcnn_net *net, float rate, char *src_id) {
    bcnn_node node = {0};

    BCNN_CHECK_AND_LOG(net->log_ctx, net->num_nodes >= 1,
                       BCNN_INVALID_PARAMETER,
                       "Dropout layer can't be the first layer of the network");

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
                       "Dropout layer: invalid input node name %s", src_id);

    node.type = DROPOUT;
    node.param_size = sizeof(bcnn_dropout_param);
    node.param = (bcnn_dropout_param *)calloc(1, node.param_size);
    bcnn_dropout_param *param = (bcnn_dropout_param *)node.param;
    param->dropout_rate = rate;
    int sz = bcnn_tensor_size(&net->tensors[node.src[0]]);
    param->rand = (float *)calloc(sz, sizeof(float));
    param->scale = 1.0f / (1.0f - rate);
#ifdef BCNN_USE_CUDA
    param->rand_gpu = bcnn_cuda_memcpy_f32(param->rand, sz);
#endif
    node.forward = bcnn_forward_dropout_layer;
    node.backward = bcnn_backward_dropout_layer;
    node.release_param = bcnn_release_param_dropout_layer;

    bcnn_net_add_node(net, node);

    BCNN_INFO(net->log_ctx,
              "[Dropout] input_shape= %dx%dx%d rate= %f output_shape= %dx%dx%d",
              net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
              net->tensors[node.src[0]].c, rate, net->tensors[node.dst[0]].w,
              net->tensors[node.dst[0]].h, net->tensors[node.dst[0]].c);
    return 0;
}

void bcnn_forward_dropout_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_dropout_param *param = (bcnn_dropout_param *)node->param;
    int sz = bcnn_tensor_size(src_tensor);

    if (net->state != TRAIN) {
        return;
    }
    for (int i = 0; i < sz; ++i) {
        float r = (float)rand() / RAND_MAX;
        param->rand[i] = r;
        if (r < param->dropout_rate) {
            src_tensor->data[i] = 0;
        } else {
            src_tensor->data[i] *= param->scale;
        }
    }
    return;
}

void bcnn_forward_dropout_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_forward_dropout_layer_gpu(net, node);
#else
    return bcnn_forward_dropout_layer_cpu(net, node);
#endif
}

void bcnn_backward_dropout_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_dropout_param *param = (bcnn_dropout_param *)node->param;
    int sz = bcnn_tensor_size(src_tensor);

    if (!src_tensor->grad_data) {
        return;
    }
    for (int i = 0; i < sz; ++i) {
        float r = param->rand[i];
        if (r < param->dropout_rate) {
            src_tensor->grad_data[i] = 0;
        } else {
            src_tensor->grad_data[i] *= param->scale;
        }
    }
    return;
}

void bcnn_backward_dropout_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_dropout_layer_gpu(net, node);
#else
    return bcnn_backward_dropout_layer_cpu(net, node);
#endif
}

void bcnn_release_param_dropout_layer(bcnn_node *node) {
    bcnn_dropout_param *param = (bcnn_dropout_param *)node->param;
    bh_free(param->rand);
#ifdef BCNN_USE_CUDA
    bh_free(param->rand_gpu);
#endif
    return;
}