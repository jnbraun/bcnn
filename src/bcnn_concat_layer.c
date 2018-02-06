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

#include "bcnn_concat_layer.h"

#include <bh/bh_error.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>

#include "bcnn_mat.h"
#include "bh_log.h"

int bcnn_add_concat_layer(bcnn_net *net, char *src_id1, char *src_id2,
                          char *dst_id) {
    int i, sz, ind_concat = -1;
    bcnn_connection conn = {0};
    bcnn_node dst_node = {0};
    int is_src_node1_found = 0, is_src_node2_found = 0;

    bh_check(net->nb_connections >= 1,
             "Concat layer can't be the first layer of the network");

    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = CONCAT;

    for (i = net->num_nodes - 1; i >= 0; --i) {
        if (strcmp(net->nodes[i].id, src_id1) == 0) {
            bcnn_connection_add_src_node(&conn, i);
            is_src_node1_found = 1;
        }
        if (strcmp(net->nodes[i].id, src_id2) == 0) {
            bcnn_connection_add_src_node(&conn, i);
            is_src_node2_found = 1;
        }
        if (is_src_node1_found && is_src_node2_found) {
            break;
        }
    }
    bh_check(is_src_node1_found, "Concat layer: invalid input node name %s",
             src_id1);
    bh_check(is_src_node2_found, "Concat layer: invalid input node name %s",
             src_id2);
    // Check spatial dimensions consistency
    bh_check(
        net->nodes[conn.src[0]].tensor.w == net->nodes[conn.src[1]].tensor.w,
        "Concat layer: inconsistent width size between node %s (w = %d) and "
        "node %s (w = %d)",
        src_id1, net->nodes[conn.src[0]].tensor.w, src_id2,
        net->nodes[conn.src[1]].tensor.w);
    bh_check(
        net->nodes[conn.src[0]].tensor.h == net->nodes[conn.src[1]].tensor.h,
        "Concat layer: inconsistent width size between node %s (w = %d) and "
        "node %s (w = %d)",
        src_id1, net->nodes[conn.src[0]].tensor.h, src_id2,
        net->nodes[conn.src[1]].tensor.h);

    // Setup output node
    bh_strfill(&dst_node.id, dst_id);
    bcnn_tensor_set_shape(
        &dst_node.tensor, net->nodes[conn.src[0]].tensor.n,
        net->nodes[conn.src[0]].tensor.c + net->nodes[conn.src[1]].tensor.c,
        net->nodes[conn.src[0]].tensor.h, net->nodes[conn.src[0]].tensor.w, 1);
    bcnn_tensor_allocate(&dst_node.tensor);
    // Add node to net
    bcnn_net_add_node(net, dst_node);
    // Add node pointer to connection
    bcnn_connection_add_dst_node(&conn, net->num_nodes - 1);
    // Add connection to net
    bcnn_net_add_connection(net, conn);

    bh_log_info(
        "[Concat] input1_shape= %dx%dx%d input2_shape= %dx%dx%d output_shape= "
        "%dx%dx%d",
        net->nodes[conn.src[0]].tensor.w, net->nodes[conn.src[0]].tensor.h,
        net->nodes[conn.src[0]].tensor.c, net->nodes[conn.src[1]].tensor.w,
        net->nodes[conn.src[1]].tensor.h, net->nodes[conn.src[1]].tensor.c,
        net->nodes[conn.dst[0]].tensor.w, net->nodes[conn.dst[0]].tensor.h,
        net->nodes[conn.dst[0]].tensor.c);

    return BCNN_SUCCESS;
}

int bcnn_forward_concat_layer_cpu(bcnn_node *src0_node, bcnn_node *src1_node,
                                  bcnn_node *dst_node) {
    int j;
    bcnn_tensor src0 = src0_node->tensor;
    bcnn_tensor src1 = src1_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int src0_sz = bcnn_tensor_get_size3d(&src0);
    int src1_sz = bcnn_tensor_get_size3d(&src1);
    int dst_sz = bcnn_tensor_get_size3d(&dst);

    for (j = 0; j < src0.n; ++j) {
        bcnn_copy_f32(src0_sz, src0.data + j * src0_sz, dst.data + j * dst_sz);
    }
    for (j = 0; j < src1.n; ++j) {
        bcnn_copy_f32(src1_sz, src1.data + j * src1_sz,
                      dst.data + src0_sz + j * dst_sz);
    }

    return BCNN_SUCCESS;
}

int bcnn_backward_concat_layer_cpu(bcnn_node *src0_node, bcnn_node *src1_node,
                                   bcnn_node *dst_node) {
    int j;
    bcnn_tensor src0 = src0_node->tensor;
    bcnn_tensor src1 = src1_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int src0_sz = bcnn_tensor_get_size3d(&src0);
    int src1_sz = bcnn_tensor_get_size3d(&src1);
    int dst_sz = bcnn_tensor_get_size3d(&dst);

    for (j = 0; j < src0.n; ++j) {
        bcnn_axpy(src0_sz, 1.0f, dst.grad_data + j * dst_sz,
                  src0.grad_data + j * src0_sz);
    }
    for (j = 0; j < src1.n; ++j) {
        bcnn_axpy(src1_sz, 1.0f, dst.grad_data + src0_sz + j * dst_sz,
                  src1.grad_data + j * src1_sz);
    }

    return BCNN_SUCCESS;
}

#ifdef BCNN_USE_CUDA

int bcnn_forward_concat_layer_gpu(bcnn_node *src0_node, bcnn_node *src1_node,
                                  bcnn_node *dst_node) {
    int j;
    bcnn_tensor src0 = src0_node->tensor;
    bcnn_tensor src1 = src1_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int src0_sz = bcnn_tensor_get_size3d(&src0);
    int src1_sz = bcnn_tensor_get_size3d(&src1);
    int dst_sz = bcnn_tensor_get_size3d(&dst);

    for (j = 0; j < src0.n; ++j) {
        bcnn_cuda_copy_f32(src0_sz, src0.data_gpu + j * src0_sz, 1,
                           dst.data_gpu + j * dst_sz, 1);
    }
    for (j = 0; j < src0.n; ++j) {
        bcnn_cuda_copy_f32(src1_sz, src1.data_gpu + j * src1_sz, 1,
                           dst.data_gpu + src0_sz + j * dst_sz, 1);
    }

    return BCNN_SUCCESS;
}

int bcnn_backward_concat_layer_gpu(bcnn_node *src0_node, bcnn_node *src1_node,
                                   bcnn_node *dst_node) {
    int j;
    bcnn_tensor src0 = src0_node->tensor;
    bcnn_tensor src1 = src1_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int src0_sz = bcnn_tensor_get_size3d(&src0);
    int src1_sz = bcnn_tensor_get_size3d(&src1);
    int dst_sz = bcnn_tensor_get_size3d(&dst);

    for (j = 0; j < src0.n; ++j) {
        bcnn_cuda_axpy(src0_sz, 1.0f, dst.grad_data_gpu + j * dst_sz, 1,
                       src0.grad_data_gpu + j * src0_sz, 1);
    }
    for (j = 0; j < src0.n; ++j) {
        bcnn_cuda_axpy(src1_sz, 1.0f, dst.grad_data_gpu + src0_sz + j * dst_sz,
                       1, src1.grad_data_gpu + j * src1_sz, 1);
    }

    return BCNN_SUCCESS;
}

#endif

int bcnn_forward_concat_layer(bcnn_net *net, bcnn_connection *conn) {
    bh_check(conn->num_src == 2, "Concat layer: invalid setup");
    bcnn_node *src0 = &net->nodes[conn->src[0]];
    bcnn_node *src1 = &net->nodes[conn->src[1]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_concat_layer_gpu(src0, src1, dst);
#else
    return bcnn_forward_concat_layer_cpu(src0, src1, dst);
#endif
}

int bcnn_backward_concat_layer(bcnn_net *net, bcnn_connection *conn) {
    bh_check(conn->num_src == 2, "Concat layer: invalid setup");
    bcnn_node *src0 = &net->nodes[conn->src[0]];
    bcnn_node *src1 = &net->nodes[conn->src[1]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_concat_layer_gpu(src0, src1, dst);
#else
    return bcnn_backward_concat_layer_cpu(src0, src1, dst);
#endif
}
