/*
 * Copyright (c) 2016-2018 Jean-Noel Braun.
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

#include "bcnn_lrn_layer.h"
#include "bcnn_mat.h"
#include "bcnn_utils.h"

#include <bh/bh_macros.h>
#include <bh/bh_string.h>

bcnn_status bcnn_add_lrn_layer(bcnn_net *net, int local_size, float alpha,
                               float beta, float k, char *src_id,
                               char *dst_id) {
    bcnn_node node = {0};
    bcnn_tensor dst_tensor = {0};

    if (net->num_nodes > 0) {
        int is_src_node_found = 0;
        for (int i = net->num_tensors - 1; i >= 0; --i) {
            if (strcmp(net->tensors[i].name, src_id) == 0) {
                bcnn_node_add_input(net, &node, i);
                is_src_node_found = 1;
                break;
            }
        }
        BCNN_CHECK_AND_LOG(net->log_ctx, is_src_node_found,
                           BCNN_INVALID_PARAMETER,
                           "LRN layer: invalid input node name %s", src_id);
    } else {
        bcnn_node_add_input(net, &node, 0);
    }

    bcnn_tensor_set_shape(&dst_tensor,
                          net->tensors[node.src[0]].n,  // batch size
                          net->tensors[node.src[0]].c,  // depth
                          net->tensors[node.src[0]].h,  // height
                          net->tensors[node.src[0]].w,  // width
                          1);
    bcnn_tensor_allocate(&dst_tensor);
    bh_strfill(&dst_tensor.name, dst_id);
    // Add node to net
    bcnn_net_add_tensor(net, dst_tensor);
    // Add tensor output index to node
    bcnn_node_add_output(net, &node, net->num_tensors - 1);

    node.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    node.layer->type = LRN;
    int sz = bcnn_tensor_size(&net->tensors[node.src[0]]);
    // x_norm stores the intermediate sum results
    node.layer->x_norm = (float *)calloc(sz, sizeof(float));
    // workspace stores the intermediate square results
    node.layer->workspace = (float *)calloc(sz, sizeof(float));
    node.layer->alpha = alpha;
    node.layer->beta = beta;
    node.layer->k = k;
    node.layer->size = local_size;
    BCNN_CHECK_AND_LOG(
        net->log_ctx, local_size < net->tensors[node.src[0]].c,
        BCNN_INVALID_PARAMETER,
        "LRN layer: local size must be inferior to the number of channels");

    bcnn_net_add_node(net, node);

    BCNN_INFO(net->log_ctx,
              "[LocalResponseNormalization] input_shape= %dx%dx%d ouput_shape= "
              "%dx%dx%d",
              net->tensors[node.src[0]].w, net->tensors[node.src[0]].h,
              net->tensors[node.src[0]].c, net->tensors[node.dst[0]].w,
              net->tensors[node.dst[0]].h, net->tensors[node.dst[0]].c);
    return 0;
}

int bcnn_forward_lrn_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                               bcnn_tensor *dst_tensor) {
    int sz = bcnn_tensor_size(src_tensor);
    int sz3d = bcnn_tensor_size3d(src_tensor);
    int sz2d = bcnn_tensor_size2d(src_tensor);
    memset(layer->workspace, 0, sz * sizeof(float));
    for (int b = 0; b < src_tensor->n; ++b) {
        float *p_src = src_tensor->data + b * sz3d;
        float *p_square = layer->workspace + b * sz3d;
        float *p_norm = layer->x_norm + b * sz3d;
        bcnn_vmul(sz3d, p_src, p_src, p_square);
        bcnn_fill_f32(sz2d, layer->k, p_norm);
        for (int c = 0; c < layer->size / 2; ++c) {
            bcnn_axpy(sz2d, layer->alpha / layer->size, p_square + sz2d * c,
                      p_norm);
        }
        for (int c = 1; c < bh_min(1 + (layer->size - 1) / 2,
                                   src_tensor->c - layer->size / 2);
             ++c) {
            bcnn_copy_f32(sz2d, p_norm + sz2d * (c - 1), p_norm + sz2d * c);
            int tail = c + layer->size / 2;
            bcnn_axpy(sz2d, layer->alpha / layer->size, p_square + sz2d * tail,
                      p_norm + sz2d * c);
        }
        for (int c = bh_min(1 + (layer->size - 1) / 2,
                            src_tensor->c - layer->size / 2);
             c < src_tensor->c - layer->size / 2; ++c) {
            bcnn_copy_f32(sz2d, p_norm + sz2d * (c - 1), p_norm + sz2d * c);
            int head = c - (layer->size - 1) / 2 - 1;
            bcnn_axpy(sz2d, -layer->alpha / layer->size, p_square + sz2d * head,
                      p_norm + sz2d * c);
            int tail = c + layer->size / 2;
            bcnn_axpy(sz2d, layer->alpha / layer->size, p_square + sz2d * tail,
                      p_norm + sz2d * c);
        }
        for (int c = bh_max(1, src_tensor->c - layer->size / 2);
             c < src_tensor->c; ++c) {
            bcnn_copy_f32(sz2d, p_norm + sz2d * (c - 1), p_norm + sz2d * c);
            int head = c - (layer->size - 1) / 2 - 1;
            bcnn_axpy(sz2d, -layer->alpha / layer->size, p_square + sz2d * head,
                      p_norm + sz2d * c);
        }
    }
    bcnn_pow(sz, layer->x_norm, -layer->beta, dst_tensor->data);
    bcnn_vmul(sz, src_tensor->data, dst_tensor->data, dst_tensor->data);
    return 0;
}

int bcnn_backward_lrn_layer_cpu(bcnn_layer *layer, bcnn_tensor *src_tensor,
                                bcnn_tensor *dst_tensor) {
    int sz = bcnn_tensor_size(src_tensor);
    int sz3d = bcnn_tensor_size3d(src_tensor);
    int sz2d = bcnn_tensor_size2d(src_tensor);
    bcnn_pow(sz, layer->x_norm, -layer->beta, src_tensor->grad_data);
    bcnn_vmul(sz, dst_tensor->grad_data, src_tensor->grad_data,
              src_tensor->grad_data);
    float *tmp_ratio = (float *)calloc(sz2d, sizeof(float));
    float *tmp_ratio_grad = (float *)calloc(sz2d, sizeof(float));
    float ratio_val = -2.0f * layer->alpha * layer->beta / layer->size;
    for (int b = 0; b < src_tensor->n; ++b) {
        float *p_src = src_tensor->data + b * sz3d;
        float *p_dst = dst_tensor->data + b * sz3d;
        float *p_src_grad = src_tensor->grad_data + b * sz3d;
        float *p_dst_grad = dst_tensor->grad_data + b * sz3d;
        float *p_wrk = layer->workspace + b * sz3d;
        float *p_norm = layer->x_norm + b * sz3d;
        bcnn_vmul(sz3d, p_dst_grad, p_dst, p_wrk);
        bcnn_vdiv(sz3d, p_wrk, p_norm, p_wrk);
        memset(tmp_ratio, 0, sizeof(float) * sz2d);
        for (int c = 0; c < layer->size / 2 - 1; ++c) {
            bcnn_axpy(sz2d, 1.0f, p_wrk + c * sz2d, tmp_ratio);
        }
        for (int c = 0; c < src_tensor->c - layer->size / 2; ++c) {
            bcnn_axpy(sz2d, 1.0f, p_wrk + sz2d * (c + layer->size / 2),
                      tmp_ratio);
            // compute src grad
            bcnn_vmul(sz2d, p_src + c * sz2d, tmp_ratio, tmp_ratio_grad);
            bcnn_axpy(sz2d, ratio_val, tmp_ratio_grad, p_src_grad + c * sz2d);
            bcnn_axpy(sz2d, -1.0f, p_wrk + c * sz2d, tmp_ratio);
        }
        for (int c = src_tensor->c - layer->size / 2; c < src_tensor->c; ++c) {
            // compute src grad
            bcnn_vmul(sz2d, p_src + c * sz2d, tmp_ratio, tmp_ratio_grad);
            bcnn_axpy(sz2d, ratio_val, tmp_ratio_grad, p_src_grad + c * sz2d);
            bcnn_axpy(sz2d, -1.0f, p_wrk + c * sz2d, tmp_ratio);
        }
    }
    free(tmp_ratio);
    free(tmp_ratio_grad);
    return 0;
}

#ifdef BCNN_USE_CUDA

#endif

int bcnn_forward_lrn_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
    // Not implemented
    return 0
// return bcnn_forward_lrn_layer_gpu(node->layer, src, dst);
#else
    return bcnn_forward_lrn_layer_cpu(node->layer, src, dst);
#endif
}

int bcnn_backward_lrn_layer(bcnn_net *net, bcnn_node *node) {
    bcnn_tensor *src = &net->tensors[node->src[0]];
    bcnn_tensor *dst = &net->tensors[node->dst[0]];
#ifdef BCNN_USE_CUDA
// Not implemented
// return bcnn_backward_lrn_layer_gpu(node->layer, src, dst);
#else
    return bcnn_backward_lrn_layer_cpu(node->layer, src, dst);
#endif
    return 0;
}
