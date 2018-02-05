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
#include "bcnn_dropout_layer.h"

#include <bh/bh_error.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>

#include "bcnn_utils.h"
#include "bh_log.h"

int bcnn_add_dropout_layer(bcnn_net *net, float rate, char *src_id) {
    int sz = 0, i;
    bcnn_connection conn = {0};

    bh_check(net->nb_connections >= 1,
             "Dropout layer can't be the first layer of the network");

    int is_src_node_found = 0;
    for (i = net->num_nodes - 1; i >= 0; --i) {
        if (strcmp(net->nodes[i].id, src_id) == 0) {
            bcnn_connection_add_src_node(&conn, i);
            bcnn_connection_add_dst_node(&conn, i);
            is_src_node_found = 1;
            break;
        }
    }
    bh_check(is_src_node_found, "Dropout layer: invalid input node name %s",
             src_id);

    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = DROPOUT;
    conn.layer->dropout_rate = rate;
    sz = bcnn_tensor_get_size(&net->nodes[conn.src[0]].tensor);
    conn.layer->rand = (float *)calloc(sz, sizeof(float));
    conn.layer->scale = 1.0f / (1.0f - rate);
#ifdef BCNN_USE_CUDA
    conn.layer->rand_gpu = bcnn_cuda_memcpy_f32(conn.layer->rand, sz);
#endif

    bcnn_net_add_connection(net, conn);

    bh_log_info(
        "[Dropout] input_shape= %dx%dx%d rate= %f output_shape= %dx%dx%d",
        net->nodes[conn.src[0]].tensor.w, net->nodes[conn.src[0]].tensor.h,
        net->nodes[conn.src[0]].tensor.c, rate,
        net->nodes[conn.dst[0]].tensor.w, net->nodes[conn.dst[0]].tensor.h,
        net->nodes[conn.dst[0]].tensor.c);
    return 0;
}

int bcnn_forward_dropout_layer_cpu(bcnn_layer *layer, bcnn_node *src_node,
                                   bcnn_node *dst_node) {
    bcnn_tensor src = src_node->tensor;
    int i, sz = bcnn_tensor_get_size(&src);
    float r;

    if (!layer->net_state)  // state != train
        return BCNN_SUCCESS;

    for (i = 0; i < sz; ++i) {
        r = (float)rand() / RAND_MAX;
        layer->rand[i] = r;
        if (r < layer->dropout_rate) {
            src.data[i] = 0;
        } else {
            src.data[i] *= layer->scale;
        }
    }
    return BCNN_SUCCESS;
}

int bcnn_forward_dropout_layer(bcnn_net *net, bcnn_connection *conn) {
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_dropout_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_forward_dropout_layer_cpu(conn->layer, src, dst);
#endif
}

int bcnn_backward_dropout_layer_cpu(bcnn_layer *layer, bcnn_node *src_node,
                                    bcnn_node *dst_node) {
    bcnn_tensor src = src_node->tensor;
    int i, sz = bcnn_tensor_get_size(&src);
    float r;

    if (!src.grad_data) {
        return BCNN_SUCCESS;
    }

    for (i = 0; i < sz; ++i) {
        r = layer->rand[i];
        if (r < layer->dropout_rate) {
            src.grad_data[i] = 0;
        } else {
            src.grad_data[i] *= layer->scale;
        }
    }
    return BCNN_SUCCESS;
}

int bcnn_backward_dropout_layer(bcnn_net *net, bcnn_connection *conn) {
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_dropout_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_backward_dropout_layer_cpu(conn->layer, src, dst);
#endif
}