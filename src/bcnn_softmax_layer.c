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
#include "bcnn_softmax_layer.h"

#include <bh/bh_log.h>
#include <bh/bh_macros.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>
#include "bcnn_utils.h"

bcnn_status bcnn_add_softmax_layer(bcnn_net *net, char *src_id, char *dst_id) {
    int num_nodes = net->num_nodes + 1;
    int sz, i;
    bcnn_node node = {0};
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
            "Full-connected layer: invalid input node name %s", src_id);
    } else {
        bcnn_node_add_input(net, &node, 0);
    }

    // Setup output node
    bcnn_tensor_set_shape(&dst_tensor,
                          net->tensors[node.src[0]].n,  // batch size
                          net->tensors[node.src[0]].c,  // depth
                          net->tensors[node.src[0]].h,  // height
                          net->tensors[node.src[0]].w,  // width
                          1);
    bcnn_tensor_allocate(&dst_tensor, net->state);
    bh_strfill(&dst_tensor.name, dst_id);
    // Add node to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(net, &node, net->num_tensors - 1);

    node.type = SOFTMAX;
    node.forward = bcnn_forward_softmax_layer;
    node.backward = bcnn_backward_softmax_layer;

    bcnn_net_add_node(net, node);

    BCNN_INFO(net->log_ctx,
              "[Softmax] input_shape= %dx%dx%d output_shape= %dx%dx%d",
              net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
              net->tensors[node.src[0]].c, net->tensors[node.dst[0]].w,
              net->tensors[node.dst[0]].h, net->tensors[node.dst[0]].c);

    return BCNN_SUCCESS;
}

void bcnn_forward_softmax_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    int b, i, batch_size = src_tensor->n;
    int src_size = bcnn_tensor_size3d(src_tensor);
    float vmax = -FLT_MAX;
    float sum = 0.0f;

    if (src_tensor->w * src_tensor->h == 1) {
        for (b = 0; b < batch_size; ++b) {
            vmax = -FLT_MAX;
            sum = 0.0f;
            for (i = 0; i < src_size; ++i) {
                if (src_tensor->data[b * src_size + i] > vmax) {
                    vmax = src_tensor->data[b * src_size + i];
                }
            }
            for (i = 0; i < src_size; ++i) {
                sum += (float)exp(src_tensor->data[b * src_size + i] - vmax);
            }
            if (sum) {
                sum = vmax + (float)log(sum);
            } else
                sum = vmax - 100.0f;
            for (i = 0; i < src_size; ++i) {
                dst_tensor->data[b * src_size + i] =
                    (float)exp(src_tensor->data[b * src_size + i] - sum);
            }
        }
    } else {
        for (b = 0; b < batch_size; ++b) {
            for (i = 0; i < src_tensor->w * src_tensor->h; ++i) {
                int c;
                vmax = -FLT_MAX;
                sum = 0.0f;
                for (c = 0; c < src_tensor->c; ++c) {
                    vmax = bh_max(
                        vmax,
                        src_tensor
                            ->data[b * src_size +
                                   c * src_tensor->w * src_tensor->h + i]);
                }
                for (c = 0; c < src_tensor->c; ++c) {
                    sum += (float)exp(
                        src_tensor
                            ->data[b * src_size +
                                   c * src_tensor->w * src_tensor->h + i] -
                        vmax);
                }
                if (sum) {
                    sum = vmax + (float)log(sum);
                } else {
                    sum = vmax - 100.0f;
                }
                for (c = 0; c < src_tensor->c; ++c) {
                    dst_tensor->data[b * src_size +
                                     c * src_tensor->w * src_tensor->h + i] =
                        (float)exp(
                            src_tensor
                                ->data[b * src_size +
                                       c * src_tensor->w * src_tensor->h + i] -
                            sum);
                }
            }
        }
    }
    return;
}

void bcnn_backward_softmax_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    int sz = bcnn_tensor_size(src_tensor);

    for (int i = 0; i < sz; ++i) {
        src_tensor->grad_data[i] += dst_tensor->grad_data[i];
    }
    return;
}

void bcnn_forward_softmax_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_forward_softmax_layer_gpu(net, node);
#else
    return bcnn_forward_softmax_layer_cpu(net, node);
#endif
}

void bcnn_backward_softmax_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_backward_softmax_layer_gpu(net, node);
#else
    return bcnn_backward_softmax_layer_cpu(net, node);
#endif
}