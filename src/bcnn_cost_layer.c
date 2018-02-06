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

#include "bcnn_cost_layer.h"

#include <bh/bh.h>
#include <bh/bh_mem.h>

#include "bcnn_mat.h"
#include "bcnn_utils.h"
#include "bh_log.h"

int bcnn_add_cost_layer(bcnn_net *net, bcnn_loss loss,
                        bcnn_loss_metric loss_metric, float scale, char *src_id,
                        char *label_id, char *dst_id) {
    int sz, i;
    bcnn_connection conn = {0};
    bcnn_node dst_node = {0};
    bcnn_node label_node = {0};
    // Create layer
    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = COST;

    bh_check(net->nb_connections >= 1,
             "Cost layer can't be the first layer of the network");
    int is_src_node_found = 0;
    for (i = net->num_nodes - 1; i >= 0; --i) {
        if (strcmp(net->nodes[i].id, src_id) == 0) {
            bcnn_connection_add_src_node(&conn, i);
            is_src_node_found = 1;
            break;
        }
    }
    bh_check(is_src_node_found, "Cost layer: invalid input node name %s",
             src_id);

    conn.layer->scale = scale;
    conn.layer->loss_metric = loss_metric;
    conn.layer->loss = loss;
    // Setup label node
    bcnn_tensor_set_shape(
        &net->nodes[1].tensor, net->nodes[conn.src[0]].tensor.n,
        net->nodes[conn.src[0]].tensor.c, net->nodes[conn.src[0]].tensor.h,
        net->nodes[conn.src[0]].tensor.w, 0);
    bcnn_tensor_allocate(&net->nodes[1].tensor);
    // Add pointer to label node to connection
    bcnn_connection_add_src_node(&conn, 1);

    // Create output node
    bh_strfill(&dst_node.id, dst_id);
    bcnn_tensor_set_shape(&dst_node.tensor, net->nodes[conn.src[0]].tensor.n,
                          net->nodes[conn.src[0]].tensor.c,
                          net->nodes[conn.src[0]].tensor.h,
                          net->nodes[conn.src[0]].tensor.w, 1);
    bcnn_tensor_allocate(&dst_node.tensor);
    // Add node to net
    bcnn_net_add_node(net, dst_node);
    // Add node pointer to connection
    bcnn_connection_add_dst_node(&conn, net->num_nodes - 1);

    bcnn_net_add_connection(net, conn);

    return 0;
}

static void bcnn_huber_loss(int n, float *x, float *label, float *error,
                            float *grad_error, float hdelta) {
    int i;
    float e = 0.0f;

    for (i = 0; i < n; ++i) {
        e = x[i] - label[i];
        if (fabs(e) > hdelta) {
            grad_error[i] = (e > 0) ? 1.0f : -1.0f;
            *error += 2.0f * hdelta * fabs(e) - hdelta * hdelta;
        } else {
            grad_error[i] = e;
            *error += e * e;
        }
    }
    return;
}

static void bcnn_euclidean_loss_forward(bcnn_node *src_node,
                                        bcnn_node *label_node,
                                        bcnn_node *dst_node) {
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    bcnn_tensor label = label_node->tensor;
    int size = bcnn_tensor_get_size(&src);
#ifdef BCNN_USE_CUDA
    bcnn_cuda_copy_f32(size, src.data_gpu, 1, dst.grad_data_gpu, 1);
    bcnn_cuda_axpy(size, -1, label.data_gpu, 1, dst.grad_data_gpu, 1);
#else
    bcnn_copy_f32(size, src.data, dst.grad_data);
    bcnn_axpy(size, -1, label.data, dst.grad_data);
#endif
}

static void bcnn_euclidean_loss_backward(bcnn_node *src_node,
                                         bcnn_node *dst_node,
                                         bcnn_layer *layer) {
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int size = bcnn_tensor_get_size(&src);
#ifdef BCNN_USE_CUDA
    bcnn_cuda_axpy(size, layer->scale, dst.grad_data_gpu, 1, src.grad_data_gpu,
                   1);
#else
    bcnn_axpy(size, layer->scale, dst.grad_data, src.grad_data);
#endif
}

void bcnn_compute_error(bcnn_layer *layer, bcnn_node *src_node,
                        bcnn_node *label_node, bcnn_node *dst_node) {
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    bcnn_tensor label = label_node->tensor;
    int input_size = src.w * src.h * src.c;
    int batch_size = src.n;
    int sz = src.n * input_size;

#ifdef BCNN_USE_CUDA
    bcnn_cuda_memcpy_dev2host(dst.grad_data_gpu, dst.grad_data, sz);
#endif

    switch (layer->loss_metric) {
        case COST_ERROR:
            *(dst.data) = 0.0f;
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(src.data_gpu, src.data, sz);
            bcnn_cuda_memcpy_dev2host(label.data_gpu, label.data, sz);
#endif
            for (int i = 0; i < batch_size; ++i) {
                int offset = i * input_size;
                float p_max = FLT_MIN;
                int j_best = 0;
                for (int j = 0; j < input_size; ++j) {
                    if (src.data[offset + j] > p_max) {
                        p_max = src.data[offset + j];
                        j_best = j;
                    }
                }
                if (label.data[offset + j_best] == 0) {
                    *(dst.data) += 1.0f;
                }
            }
            break;
        case COST_SSE:
            *(dst.data) = bcnn_dot(sz, dst.grad_data, dst.grad_data);
            break;
        case COST_MSE:
            *(dst.data) = bcnn_dot(sz, dst.grad_data, dst.grad_data);
            *(dst.data) /= input_size;
            break;
        case COST_CRPS:
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(src.data_gpu, src.data, sz);
            bcnn_cuda_memcpy_dev2host(label.data_gpu, label.data, sz);
#endif
            *(dst.data) = 0.0f;
            float *input_cpu = (float *)calloc(sz, sizeof(float));
            for (int i = 0; i < batch_size; ++i) {
                int offset = i * input_size;
                for (int j = 1; j < input_size; ++j) {
                    if (src.data[offset + j] < src.data[offset + j - 1]) {
                        input_cpu[offset + j] = src.data[offset + j - 1];
                    }
                }
            }
            *(dst.data) = bcnn_dot(sz, dst.grad_data, dst.grad_data);
            bh_free(input_cpu);
            break;
        case COST_LOGLOSS:
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(src.data_gpu, src.data, sz);
            bcnn_cuda_memcpy_dev2host(label.data_gpu, label.data, sz);
#endif
            *(dst.data) = 0.0f;
            for (int i = 0; i < batch_size; ++i) {
                int offset = i * input_size;
                for (int j = 0; j < input_size; ++j) {
                    if (label.data[offset + j] > 0.0f) {
                        *(dst.data) += (float)-log(bh_clamp(
                            src.data[offset + j], 1e-8f, 1.0f - 1e-8f));
                    }
                }
            }
            break;
        case COST_DICE:
#ifdef BCNN_USE_CUDA
            bcnn_cuda_memcpy_dev2host(src.data_gpu, src.data, sz);
            bcnn_cuda_memcpy_dev2host(label.data_gpu, label.data, sz);
#endif
            *(dst.data) = 0.0f;
            for (int i = 0; i < batch_size; ++i) {
                int offset = i * input_size;
                int n = 0;
                int d = 0;
                for (int j = 0; j < input_size; ++j) {
                    n += (int)(label.data[offset + j] *
                               (src.data[offset + j] > 0.5f));
                    d += (int)(label.data[offset + j] +
                               (src.data[offset + j] > 0.5f));
                }
                *(dst.data) += (float)(2.0f * n + 1.0f) / (d + 1.0f);
            }
            break;
    }
}

int bcnn_forward_cost_layer(bcnn_net *net, bcnn_connection *conn) {
    bcnn_node *src_node = &net->nodes[conn->src[0]];
    bcnn_node *dst_node = &net->nodes[conn->dst[0]];
    bcnn_node *label_node = &net->nodes[1];
    bcnn_layer *layer = conn->layer;
    // If no truth available, do nothing
    if (!label_node->tensor.data) {
        return BCNN_SUCCESS;
    }

    switch (layer->loss) {
        case EUCLIDEAN_LOSS:
            bcnn_euclidean_loss_forward(src_node, label_node, dst_node);
            break;
        case LIFTED_STRUCT_SIMILARITY_SOFTMAX_LOSS:
            bcnn_LiftedStructSimilaritySoftmax_loss_forward(
                layer, src_node, label_node, dst_node);
            break;
    }

    bcnn_compute_error(layer, src_node, label_node, dst_node);

    return BCNN_SUCCESS;
}

int bcnn_backward_cost_layer(bcnn_net *net, bcnn_connection *conn) {
    bcnn_node *src_node = &net->nodes[conn->src[0]];
    bcnn_node *dst_node = &net->nodes[conn->dst[0]];
    bcnn_layer *layer = conn->layer;
    switch (layer->loss) {
        case EUCLIDEAN_LOSS:
            bcnn_euclidean_loss_backward(src_node, dst_node, layer);
            break;
        case LIFTED_STRUCT_SIMILARITY_SOFTMAX_LOSS:
            bcnn_LiftedStructSimilaritySoftmax_loss_backward(layer, src_node,
                                                             dst_node);
            break;
    }

    return BCNN_SUCCESS;
}
