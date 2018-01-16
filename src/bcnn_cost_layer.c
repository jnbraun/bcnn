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

#include <bh/bh.h>
#include <bh/bh_mem.h>

#include "bcnn/bcnn.h"
#include "bh_log.h"

int bcnn_add_cost_layer(bcnn_net *net, bcnn_loss_metric loss_metric, float scale,
    char *src_id, char *label_id, char *dst_id)
{
    int sz, i;
    bcnn_connection conn = { 0 };
    bcnn_node dst_node = { 0 };
    bcnn_node label_node = { 0 };
    // Create layer
    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = COST;

    bh_check(net->nb_connections >= 1,
        "Cost layer can't be the first layer of the network");
    int is_src_node_found = 0;
    for (i = net->num_nodes - 1; i >= 0 ; ++i) {
        if (strcmp(net->nodes[i].id, src_id) == 0) {
            bcnn_connection_add_src_node(&conn, i);
            is_src_node_found = 1;
            break;
        }
    }
    bh_check(is_src_node_found, "Cost layer: invalid input node name %s", src_id);

    conn.layer->scale = scale;
    conn.layer->loss_metric = loss_metric;
    
    // Setup label node
    bcnn_tensor_set_shape(&net->nodes[1].tensor,
        net->nodes[conn.src[0]].tensor.n,
        net->nodes[conn.src[0]].tensor.c,
        net->nodes[conn.src[0]].tensor.h,
        net->nodes[conn.src[0]].tensor.w,
        0);
    bcnn_tensor_allocate(&net->nodes[1].tensor);
    // Add pointer to label node to connection
    bcnn_connection_add_src_node(&conn,1);
    
    // Create output node
    dst_node.id = dst_id;
    bcnn_tensor_set_shape(&dst_node.tensor,
        net->nodes[conn.src[0]].tensor.n,
        net->nodes[conn.src[0]].tensor.c,
        net->nodes[conn.src[0]].tensor.h,
        net->nodes[conn.src[0]].tensor.w,
        1);
    bcnn_tensor_allocate(&dst_node.tensor);
    // Add node to net
    bcnn_net_add_node(net, dst_node);
    // Add node pointer to connection
    bcnn_connection_add_dst_node(&conn, net->num_nodes - 1);

    bcnn_net_add_connection(net, conn);

    return 0;
}

static void _bcnn_l2_loss(int n, float *x, float *label, float *error, float *grad_error)
{
    bcnn_copy_f32(n, x, grad_error);
    bcnn_axpy(n, -1, label, grad_error);
    *error = bcnn_dot(n, grad_error, grad_error);
    return;
}

static void _bcnn_huber_loss(int n, float *x, float *label, float *error, float *grad_error, float hdelta)
{
    int i;
    float e = 0.0f;
    
    for (i = 0; i < n; ++i) {
        e = x[i] - label[i];
        if (fabs(e) > hdelta) {
            grad_error[i] = (e > 0) ? 1.0f : -1.0f;
            *error +=  2.0f * hdelta * fabs(e) - hdelta * hdelta;
        }
        else {
            grad_error[i] = e;
            *error += e * e;
        }
    }
    return;
}


int bcnn_forward_cost_layer_cpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *label_node, bcnn_node *dst_node)
{
    int i, j, offset, j_best, n, d;
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    bcnn_tensor label = label_node->tensor;
    int input_size = src.w * src.h * src.c;
    int batch_size = src.n;
    int sz = src.n * input_size;
    float p_max;
    float *input_cpu = NULL;
    // If no truth available, do nothing
    if (!label.data)
        return BCNN_SUCCESS;

    bcnn_copy_f32(sz, src.data, dst.grad_data);
    bcnn_axpy(sz, -1, label.data, dst.grad_data);

    switch (layer->loss_metric) {
    case COST_ERROR:
        *(dst.data) = 0.0f;
        for (i = 0; i < batch_size; ++i) {
            offset = i * input_size;
            p_max = FLT_MIN;
            j_best = 0;
            for (j = 0; j < input_size; ++j) {
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
        *(dst.data) = 0.0f;
        input_cpu = (float *)calloc(sz, sizeof(float));
        for (i = 0; i < batch_size; ++i) {
            offset = i * input_size;
            for (j = 1; j < input_size; ++j) {
                if (src.data[offset + j] < src.data[offset + j - 1]) {
                    input_cpu[offset + j] = src.data[offset + j - 1];
                }
            }
        }
        *(dst.data) = bcnn_dot(sz, dst.grad_data, dst.grad_data);
        bh_free(input_cpu);
        break;
    case COST_LOGLOSS:
        *(dst.data) = 0.0f;
        for (i = 0; i < batch_size; ++i) {
            offset = i * input_size;
            for (j = 0; j < input_size; ++j) {
                if (label.data[offset + j] > 0.0f) {
                    *(dst.data) += (float)-log(bh_clamp(src.data[offset + j], 1e-8f, 1.0f - 1e-8f));
                }
            }
        }
        break;
    case COST_DICE:
        *(dst.data) = 0.0f;
        for (i = 0; i < batch_size; ++i) {
            offset = i * input_size;
            n = 0;
            d = 0;
            for (j = 0; j < input_size; ++j) {
                n += (int)(label.data[offset + j] * (src.data[offset + j] > 0.5f));
                d += (int)(label.data[offset + j] + (src.data[offset + j] > 0.5f));
            }
            *(dst.data) += (float)(2.0f * n + 1.0f) / (d + 1.0f);

        }
        break;
    }

    return BCNN_SUCCESS;
}


int bcnn_backward_cost_layer_cpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int input_size = src.w * src.h * src.c;
    int sz = src.n * input_size;

    bcnn_axpy(sz, layer->scale, dst.grad_data, src.grad_data);

    return BCNN_SUCCESS;
}

int bcnn_forward_cost_layer(bcnn_net *net, bcnn_connection *conn)
{
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
    bcnn_node *label = &net->nodes[1];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_cost_layer_gpu(conn->layer, src, label, dst);
#else
    return bcnn_forward_cost_layer_cpu(conn->layer, src, label, dst);
#endif
}

int bcnn_backward_cost_layer(bcnn_net *net, bcnn_connection *conn)
{
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_cost_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_backward_cost_layer_cpu(conn->layer, src, dst);
#endif
}