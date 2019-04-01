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

#include "bcnn_concat_layer.h"

#include <bh/bh_string.h>

#include "bcnn_mat.h"
#include "bcnn_net.h"
#include "bcnn_tensor.h"
#include "bcnn_utils.h"

bcnn_status bcnn_add_concat_layer(bcnn_net *net, int num_src,
                                  char *const *src_ids, const char *dst_id) {
    int i, sz, ind_concat = -1;
    bcnn_node node = {0};
    bcnn_tensor dst_tensor = {0};
    int is_src_node1_found = 0, is_src_node2_found = 0;

    BCNN_CHECK_AND_LOG(
        net->log_ctx, net->num_nodes >= 1, BCNN_INVALID_PARAMETER,
        "Concat layer can't be the first layer of the network\n");

    node.type = BCNN_LAYER_CONCAT;
    node.forward = bcnn_forward_concat_layer;
    node.backward = bcnn_backward_concat_layer;
    for (int i = 0; i < num_src; ++i) {
        int tid = -1;
        BCNN_CHECK_AND_LOG(
            net->log_ctx,
            (tid = bcnn_get_tensor_index_by_name(net, src_ids[i])) >= 0,
            BCNN_INVALID_PARAMETER,
            "Concat layer: invalid input node name %s\n", src_ids[i]);
        bcnn_node_add_input(net, &node, tid);
    }

    // Check spatial dimensions consistency
    int out_c = net->tensors[node.src[0]].c;
    for (int i = 1; i < node.num_src; ++i) {
        BCNN_CHECK_AND_LOG(
            net->log_ctx,
            (net->tensors[node.src[0]].w == net->tensors[node.src[i]].w) &&
                (net->tensors[node.src[0]].h == net->tensors[node.src[i]].h),
            BCNN_INVALID_PARAMETER,
            "Concat layer: inconsistent spatial sizes between node %s (%dx%d) "
            "and node %s (%dx%d)\n",
            src_ids[0], net->tensors[node.src[0]].w,
            net->tensors[node.src[0]].h, src_ids[i],
            net->tensors[node.src[i]].w, net->tensors[node.src[i]].h);
        out_c += net->tensors[node.src[i]].c;
    }

    // Setup output tensor
    bcnn_tensor_set_shape(&dst_tensor, net->tensors[node.src[0]].n, out_c,
                          net->tensors[node.src[0]].h,
                          net->tensors[node.src[0]].w, 1);
    bcnn_tensor_allocate(&dst_tensor, net->mode);
    bh_strfill(&dst_tensor.name, dst_id);
    // Add tensor to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(net, &node, net->num_tensors - 1);
    // Add node to net
    bcnn_net_add_node(net, node);
    BCNN_INFO(net->log_ctx,
              "[Concat] inputs_shape= %dx%d output_shape= "
              "%dx%dx%d\n",
              net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
              net->tensors[node.dst[0]].w, net->tensors[node.dst[0]].h,
              net->tensors[node.dst[0]].c);
    return BCNN_SUCCESS;
}

void bcnn_forward_concat_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    int dst_sz = bcnn_tensor_size3d(dst_tensor);
    int dst_offset = 0;
    for (int i = 0; i < node->num_src; ++i) {
        bcnn_tensor *src_tensor = &net->tensors[node->src[i]];
        int src_sz = bcnn_tensor_size3d(src_tensor);
        for (int j = 0; j < src_tensor->n; ++j) {
            bcnn_copy_f32(src_sz, src_tensor->data + j * src_sz,
                          dst_tensor->data + dst_offset + j * dst_sz);
        }
        dst_offset += src_sz;
    }
    return;
}

void bcnn_backward_concat_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    int dst_sz = bcnn_tensor_size3d(dst_tensor);
    int dst_offset = 0;
    for (int i = 0; i < node->num_src; ++i) {
        bcnn_tensor *src_tensor = &net->tensors[node->src[i]];
        int src_sz = bcnn_tensor_size3d(src_tensor);
        if (src_tensor->grad_data) {
            for (int j = 0; j < src_tensor->n; ++j) {
                bcnn_axpy(src_sz, 1.0f,
                          dst_tensor->grad_data + dst_offset + j * dst_sz,
                          src_tensor->grad_data + j * src_sz);
            }
        }
        dst_offset += src_sz;
    }
    return;
}

#ifdef BCNN_USE_CUDA

void bcnn_forward_concat_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    int dst_sz = bcnn_tensor_size3d(dst_tensor);
    int dst_offset = 0;
    for (int i = 0; i < node->num_src; ++i) {
        bcnn_tensor *src_tensor = &net->tensors[node->src[i]];
        int src_sz = bcnn_tensor_size3d(src_tensor);
        for (int j = 0; j < src_tensor->n; ++j) {
            bcnn_cuda_copy_f32(src_sz, src_tensor->data_gpu + j * src_sz, 1,
                               dst_tensor->data_gpu + j * dst_sz, 1);
        }
        dst_offset += src_sz;
    }
    return;
}

void bcnn_backward_concat_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    int dst_sz = bcnn_tensor_size3d(dst_tensor);
    int dst_offset = 0;
    for (int i = 0; i < node->num_src; ++i) {
        bcnn_tensor *src_tensor = &net->tensors[node->src[i]];
        int src_sz = bcnn_tensor_size3d(src_tensor);
        if (src_tensor->grad_data) {
            for (int j = 0; j < src_tensor->n; ++j) {
                bcnn_cuda_axpy(src_sz, 1.0f, dst_tensor->grad_data_gpu +
                                                 dst_offset + j * dst_sz,
                               1, src_tensor->grad_data_gpu + j * src_sz, 1);
            }
        }
        dst_offset += src_sz;
    }
    return;
}

#endif

void bcnn_forward_concat_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_forward_concat_layer_gpu(net, node);
#else
    return bcnn_forward_concat_layer_cpu(net, node);
#endif
}

void bcnn_backward_concat_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_backward_concat_layer_gpu(net, node);
#else
    return bcnn_backward_concat_layer_cpu(net, node);
#endif
}
