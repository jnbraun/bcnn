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
#include "bcnn_softmax_layer.h"

#include <bh/bh_mem.h>
#include <bh/bh_string.h>

#include "bh_log.h"

int bcnn_add_softmax_layer(bcnn_net *net, char *src_id, char *dst_id) {
    int nb_connections = net->nb_connections + 1;
    int sz, i;
    bcnn_connection conn = {0};
    bcnn_node dst_node = {0};

    if (net->nb_connections > 0) {
        int is_src_node_found = 0;
        for (i = net->num_nodes - 1; i >= 0; --i) {
            if (strcmp(net->nodes[i].id, src_id) == 0) {
                bcnn_connection_add_src_node(&conn, i);
                is_src_node_found = 1;
                break;
            }
        }
        bh_check(is_src_node_found,
                 "Full-connected layer: invalid input node name %s", src_id);
    } else {
        bcnn_connection_add_src_node(&conn, 0);
    }

    // Setup output node
    bh_strfill(&dst_node.id, dst_id);
    bcnn_tensor_set_shape(&dst_node.tensor,
                          net->nodes[conn.src[0]].tensor.n,  // batch size
                          net->nodes[conn.src[0]].tensor.c,  // depth
                          net->nodes[conn.src[0]].tensor.h,  // height
                          net->nodes[conn.src[0]].tensor.w,  // width
                          1);
    bcnn_tensor_allocate(&dst_node.tensor);
    // Add node to net
    bcnn_net_add_node(net, dst_node);
    // Add node pointer to connection
    bcnn_connection_add_dst_node(&conn, net->num_nodes - 1);

    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = SOFTMAX;

    bcnn_net_add_connection(net, conn);

    bh_log_info(
        "[Softmax] input_shape= %dx%dx%d output_shape= %dx%dx%d",
        net->nodes[conn.src[0]].tensor.w, net->nodes[conn.src[0]].tensor.h,
        net->nodes[conn.src[0]].tensor.c, net->nodes[conn.dst[0]].tensor.w,
        net->nodes[conn.dst[0]].tensor.h, net->nodes[conn.dst[0]].tensor.c);

    return BCNN_SUCCESS;
}

int bcnn_forward_softmax_layer_cpu(bcnn_layer *layer, bcnn_node *src_node,
                                   bcnn_node *dst_node) {
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int b, i, batch_size = src.n;
    int src_size = bcnn_tensor_get_size3d(&src);
    float vmax = -FLT_MAX;
    float sum = 0.0f;

    if (src.w * src.h == 1) {
        for (b = 0; b < batch_size; ++b) {
            vmax = -FLT_MAX;
            sum = 0.0f;
            for (i = 0; i < src_size; ++i) {
                if (src.data[b * src_size + i] > vmax) {
                    vmax = src.data[b * src_size + i];
                }
            }
            for (i = 0; i < src_size; ++i) {
                sum += (float)exp(src.data[b * src_size + i] - vmax);
            }
            if (sum) {
                sum = vmax + (float)log(sum);
            } else
                sum = vmax - 100.0f;
            for (i = 0; i < src_size; ++i) {
                dst.data[b * src_size + i] =
                    (float)exp(src.data[b * src_size + i] - sum);
            }
        }
    } else {
        for (b = 0; b < batch_size; ++b) {
            for (i = 0; i < src.w * src.h; ++i) {
                int c;
                vmax = -FLT_MAX;
                sum = 0.0f;
                for (c = 0; c < src.c; ++c) {
                    vmax = bh_max(
                        vmax, src.data[b * src_size + c * src.w * src.h + i]);
                }
                for (c = 0; c < src.c; ++c) {
                    sum += (float)exp(
                        src.data[b * src_size + c * src.w * src.h + i] - vmax);
                }
                if (sum) {
                    sum = vmax + (float)log(sum);
                } else {
                    sum = vmax - 100.0f;
                }
                for (c = 0; c < src.c; ++c) {
                    dst.data[b * src_size + c * src.w * src.h + i] = (float)exp(
                        src.data[b * src_size + c * src.w * src.h + i] - sum);
                }
            }
        }
    }
    return BCNN_SUCCESS;
}

int bcnn_backward_softmax_layer_cpu(bcnn_layer *layer, bcnn_node *src_node,
                                    bcnn_node *dst_node) {
    int i;
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int sz = bcnn_tensor_get_size(&src);

    for (i = 0; i < sz; ++i) src.grad_data[i] += dst.grad_data[i];

    return BCNN_SUCCESS;
}

int bcnn_forward_softmax_layer(bcnn_net *net, bcnn_connection *conn) {
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_softmax_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_forward_softmax_layer_cpu(conn->layer, src, dst);
#endif
}

int bcnn_backward_softmax_layer(bcnn_net *net, bcnn_connection *conn) {
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_softmax_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_backward_softmax_layer_cpu(conn->layer, src, dst);
#endif
}