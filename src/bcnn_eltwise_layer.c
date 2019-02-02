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

#include "bcnn_eltwise_layer.h"

#include <bh/bh_string.h>

#include "bcnn_activation_layer.h"
#include "bcnn_mat.h"
#include "bcnn_net.h"
#include "bcnn_tensor.h"
#include "bcnn_utils.h"

bcnn_status bcnn_add_eltwise_layer(bcnn_net *net, bcnn_activation activation,
                                   const char *src_id1, const char *src_id2,
                                   const char *dst_id) {
    bcnn_node node = {0};
    bcnn_tensor dst_tensor = {0};
    int is_src_node1_found = 0, is_src_node2_found = 0;

    node.type = ELTWISE;
    node.param_size = sizeof(bcnn_eltwise_param);
    node.param = (bcnn_eltwise_param *)calloc(1, sizeof(node.param_size));
    bcnn_eltwise_param *param = (bcnn_eltwise_param *)node.param;
    param->activation = activation;
    node.forward = bcnn_forward_eltwise_layer;
    node.backward = bcnn_backward_eltwise_layer;
    for (int i = net->num_tensors - 1; i >= 0; --i) {
        if (strcmp(net->tensors[i].name, src_id1) == 0) {
            bcnn_node_add_input(net, &node, i);
            is_src_node1_found = 1;
        }
        if (strcmp(net->tensors[i].name, src_id2) == 0) {
            bcnn_node_add_input(net, &node, i);
            is_src_node2_found = 1;
        }
        if (is_src_node1_found && is_src_node2_found) {
            break;
        }
    }
    BCNN_CHECK_AND_LOG(net->log_ctx, is_src_node1_found, BCNN_INVALID_PARAMETER,
                       "Eltwise layer: invalid input node name %s", src_id1);
    BCNN_CHECK_AND_LOG(net->log_ctx, is_src_node2_found, BCNN_INVALID_PARAMETER,
                       "Eltwise layer: invalid input node name %s", src_id2);
    // Check spatial dimensions consistency
    BCNN_CHECK_AND_LOG(
        net->log_ctx,
        net->tensors[node.src[0]].w == net->tensors[node.src[1]].w &&
            net->tensors[node.src[0]].h == net->tensors[node.src[1]].h &&
            net->tensors[node.src[0]].c == net->tensors[node.src[1]].c,
        BCNN_INVALID_PARAMETER,
        "Eltwise layer: inconsistent sizes between tensor %s and tensor %s",
        src_id1, src_id2);
    // Setup output tensor
    bcnn_tensor_set_shape(
        &dst_tensor, net->tensors[node.src[0]].n, net->tensors[node.src[0]].c,
        net->tensors[node.src[0]].h, net->tensors[node.src[0]].w, 1);
    bcnn_tensor_allocate(&dst_tensor, net->mode);
    bh_strfill(&dst_tensor.name, dst_id);
    // Add tensor to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(net, &node, net->num_tensors - 1);
    // Add node to net
    bcnn_net_add_node(net, node);

    BCNN_INFO(
        net->log_ctx,
        "[EltWise] input1_shape= %dx%dx%d input2_shape= %dx%dx%d output_shape= "
        "%dx%dx%d",
        net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
        net->tensors[node.src[0]].c, net->tensors[node.src[1]].w,
        net->tensors[node.src[1]].h, net->tensors[node.src[1]].c,
        net->tensors[node.dst[0]].w, net->tensors[node.dst[0]].h,
        net->tensors[node.dst[0]].c);
    return BCNN_SUCCESS;
}

void bcnn_forward_eltwise_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src0_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *src1_tensor = &net->tensors[node->src[1]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_eltwise_param *param = (bcnn_eltwise_param *)node->param;
    int sz = bcnn_tensor_size(dst_tensor);

    bcnn_copy_f32(sz, src0_tensor->data, dst_tensor->data);
    bcnn_axpy(sz, 1.0f, src1_tensor->data, dst_tensor->data);
    bcnn_forward_activation_cpu(dst_tensor->data, sz, param->activation);

    return;
}

void bcnn_backward_eltwise_layer_cpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src0_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *src1_tensor = &net->tensors[node->src[1]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_eltwise_param *param = (bcnn_eltwise_param *)node->param;
    int sz = bcnn_tensor_size(dst_tensor);

    bcnn_backward_activation_cpu(dst_tensor->data, dst_tensor->grad_data, sz,
                                 param->activation);
    bcnn_axpy(sz, 1.0f, dst_tensor->grad_data, src0_tensor->grad_data);
    bcnn_axpy(sz, 1.0f, dst_tensor->grad_data, src1_tensor->grad_data);

    return;
}

#ifdef BCNN_USE_CUDA
void bcnn_forward_eltwise_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src0_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *src1_tensor = &net->tensors[node->src[1]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_eltwise_param *param = (bcnn_eltwise_param *)node->param;
    int sz = bcnn_tensor_size(dst_tensor);

    bcnn_cuda_copy_f32(sz, src0_tensor->data_gpu, 1, dst_tensor->data_gpu, 1);
    bcnn_cuda_axpy(sz, 1.0f, src1_tensor->data_gpu, 1, dst_tensor->data_gpu, 1);
    bcnn_forward_activation_gpu(dst_tensor->data_gpu, sz, param->activation);

    return;
}

void bcnn_backward_eltwise_layer_gpu(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src0_tensor = &net->tensors[node->src[0]];
    bcnn_tensor *src1_tensor = &net->tensors[node->src[1]];
    bcnn_tensor *dst_tensor = &net->tensors[node->dst[0]];
    bcnn_eltwise_param *param = (bcnn_eltwise_param *)node->param;
    int sz = bcnn_tensor_size(dst_tensor);

    bcnn_backward_activation_gpu(
        dst_tensor->data_gpu, dst_tensor->grad_data_gpu, sz, param->activation);
    bcnn_cuda_axpy(sz, 1.0f, dst_tensor->grad_data_gpu, 1,
                   src0_tensor->grad_data_gpu, 1);
    bcnn_cuda_axpy(sz, 1.0f, dst_tensor->grad_data_gpu, 1,
                   src1_tensor->grad_data_gpu, 1);

    return;
}
#endif

void bcnn_forward_eltwise_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_forward_eltwise_layer_gpu(net, node);
#else
    return bcnn_forward_eltwise_layer_cpu(net, node);
#endif
}

void bcnn_backward_eltwise_layer(bcnn_net *net, bcnn_node *node) {
#ifdef BCNN_USE_CUDA
    return bcnn_backward_eltwise_layer_gpu(net, node);
#else
    return bcnn_backward_eltwise_layer_cpu(net, node);
#endif
}