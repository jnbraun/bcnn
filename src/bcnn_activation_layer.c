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

#include "bcnn/bcnn.h"

#include <bh/bh_error.h>
#include <bh/bh_mem.h>
#include <bh/bh_string.h>
#include "bh_log.h"

int bcnn_add_activation_layer(bcnn_net *net, bcnn_activation type, char *src_id)
{
    bcnn_connection conn = { 0 };
    char type_name[256];
    int i;

    bh_check(net->nb_connections >= 1,
        "Activation layer can't be the first layer of the network");
    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = ACTIVATION;

    int is_src_node_found = 0;
    for (i = net->num_nodes - 1; i >= 0 ; --i) {
        if (strcmp(net->nodes[i].id, src_id) == 0) {
            bcnn_connection_add_src_node(&conn, i);
            bcnn_connection_add_dst_node(&conn, i);
            is_src_node_found = 1;
            break;
        }
    }
    bh_check(is_src_node_found, "Activation layer: invalid input node name %s", src_id);
    
    conn.layer->activation = type;
    bcnn_net_add_connection(net, conn);

    switch (type) {
    case TANH:      sprintf(type_name, "Tanh");         break;
    case RELU:      sprintf(type_name, "Relu");         break;
    case RAMP:      sprintf(type_name, "Ramp");         break;
    case SOFTPLUS:  sprintf(type_name, "Softplus");     break;
    case LRELU:     sprintf(type_name, "Leaky-Relu");   break;
    case ABS:       sprintf(type_name, "AbsVal");       break;
    case CLAMP:     sprintf(type_name, "Clamp");        break;
    default:        sprintf(type_name, "None");         break;
    }

    bh_log_info("[Activation] input_shape= %dx%dx%d function= %s output_shape= %dx%dx%d",
        net->nodes[conn.src[0]].tensor.w, net->nodes[conn.src[0]].tensor.h, net->nodes[conn.src[0]].tensor.c,
        type_name,
        net->nodes[conn.dst[0]].tensor.w, net->nodes[conn.dst[0]].tensor.h, net->nodes[conn.dst[0]].tensor.c);

    return BCNN_SUCCESS;
}


int bcnn_forward_activation_cpu(float *x, int sz, bcnn_activation a)
{
    int i;

    switch (a) {
    case TANH:
        for (i = 0; i < sz; ++i) {
            x[i] = (float)(exp(2 * x[i]) - 1) /
            ((float)exp(2 * x[i]) + 1);
        }
        break;
    case RELU:
        for (i = 0; i < sz; ++i) {
            x[i] = x[i] * (x[i] > 0);
        }
        break;
    case LRELU:
        for (i = 0; i < sz; ++i) {
            x[i] = (x[i] > 0 ? x[i] : 0.01f * x[i]);
        }
        break;
    case RAMP:
        for (i = 0; i < sz; ++i) {
            x[i] = x[i] * (x[i] > 0) + 0.1f * x[i];
        }
        break;
    case SOFTPLUS:
        for (i = 0; i < sz; ++i) {
            x[i] = (float)log(1.0f + (float)exp(x[i]));
        }
        break;
    case ABS:
        for (i = 0; i < sz; ++i) {
            x[i] = (float)fabs(x[i]);
        }
        break;
    case CLAMP:
        for (i = 0; i < sz; ++i) {
            x[i] = bh_clamp(x[i], 0, 1);
        }
        break;
    case NONE:
        break;
    default:
        break;
    }
    return BCNN_SUCCESS;
}

int bcnn_forward_activation_layer_cpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int sz = bcnn_tensor_get_size(&dst);

    dst.data = src.data;
    bcnn_forward_activation_cpu(dst.data, sz, layer->activation);

    return BCNN_SUCCESS;
}


int bcnn_backward_activation_cpu(float *x, float *dx, int sz, bcnn_activation a)
{
    int i;

    switch (a) {
    case TANH:
        for (i = 0; i < sz; ++i) {
            dx[i] *= (1 - x[i] * x[i]);
        }
        break;
    case RELU:
        for (i = 0; i < sz; ++i) {
            dx[i] *= ((float)(x[i] > 0));
        }
        break;
    case LRELU:
        for (i = 0; i < sz; ++i) {
            dx[i] *= (x[i] > 0 ? 1.0f : 0.01f);
        }
        break;
    case RAMP:
        for (i = 0; i < sz; ++i) {
            dx[i] *= ((float)(x[i] > 0) + 0.1f);
        }
        break;
    case SOFTPLUS:
        for (i = 0; i < sz; ++i) {
            dx[i] *= 1.0f / (1.0f + (float)exp(-x[i]));
        }
        break;
    case ABS:
        for (i = 0; i < sz; ++i) {
            dx[i] *= (x[i] >= 0 ? 1.0f : -1.0f);
        }
        break;
    case CLAMP:
        for (i = 0; i < sz; ++i) {
            dx[i] *= ((float)(x[i] > 0.0f && x[i] < 1.0f));
        }
        break;
    case NONE:
        break;
    default:
        break;
    }
    return 0;
}

int bcnn_backward_activation_layer_cpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int sz = bcnn_tensor_get_size(&dst);
    
    bcnn_backward_activation_cpu(dst.data, dst.grad_data, sz, layer->activation);
    src.grad_data = dst.grad_data;

    return BCNN_SUCCESS;
}


int bcnn_forward_activation_layer(bcnn_net *net, bcnn_connection *conn)
{
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_activation_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_forward_activation_layer_cpu(conn->layer, src, dst);
#endif
}

int bcnn_backward_activation_layer(bcnn_net *net, bcnn_connection *conn)
{
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_activation_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_backward_activation_layer_cpu(conn->layer, src, dst);
#endif
}