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

int bcnn_add_cost_layer(bcnn_net *net, bcnn_loss_metric loss_metric, float scale)
{
    int nb_connections = net->nb_connections + 1;
    int sz;
    bcnn_connection conn = { 0 };

    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = COST;
    if (nb_connections > 1) {
        conn.src_tensor = net->connections[nb_connections - 2].dst_tensor;
    }
    else {
        conn.src_tensor = net->input_node;
    }
    conn.layer->scale = scale;

    conn.dst_tensor.w = conn.src_tensor.w;
    conn.dst_tensor.h = conn.src_tensor.h;
    conn.dst_tensor.c = conn.src_tensor.c;
    conn.dst_tensor.b = conn.src_tensor.b;
    conn.layer->loss_metric = loss_metric;
    sz =  bcnn_get_tensor_size(&conn.dst_tensor);
    conn.dst_tensor.grad_data = (float *)calloc(sz, sizeof(float));
    conn.dst_tensor.data = (float *)calloc(1, sizeof(float));
#ifdef BCNN_USE_CUDA
    conn.dst_tensor.grad_data_gpu = bcnn_cuda_memcpy_f32(conn.dst_tensor.grad_data, sz);
#endif
    conn.label = (float *)calloc(sz, sizeof(float));
#ifdef BCNN_USE_CUDA
    conn.label_gpu = bcnn_cuda_memcpy_f32(conn.label, sz);
#endif
    net->nb_connections = nb_connections;
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


int bcnn_forward_cost_layer_cpu(bcnn_connection *conn)
{
    int i, j, offset, j_best, n, d;
    bcnn_layer *layer = conn->layer;
    bcnn_tensor src = conn->src_tensor;
    bcnn_tensor dst = conn->dst_tensor;
    int input_size = src.w * src.h * src.c;
    int batch_size = src.b;
    int sz = src.b * input_size;
    float p_max;
    float *input_cpu = NULL;
    // If no truth available, do nothing
    if (!conn->label)
        return BCNN_SUCCESS;

    bcnn_copy_f32(sz, src.data, dst.grad_data);
    bcnn_axpy(sz, -1, conn->label, dst.grad_data);

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
            if (conn->label[offset + j_best] == 0) {
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
                if (conn->label[offset + j] > 0.0f) {
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
                n += (int)(conn->label[offset + j] * (src.data[offset + j] > 0.5f));
                d += (int)(conn->label[offset + j] + (src.data[offset + j] > 0.5f));
            }
            *(dst.data) += (float)(2.0f * n + 1.0f) / (d + 1.0f);

        }
        break;
    }

    return BCNN_SUCCESS;
}


int bcnn_backward_cost_layer_cpu(bcnn_connection *conn)
{
    bcnn_layer *layer = conn->layer;
    bcnn_tensor src = conn->src_tensor;
    bcnn_tensor dst = conn->dst_tensor;
    int input_size = src.w * src.h * src.c;
    int sz = src.b * input_size;

    bcnn_axpy(sz, layer->scale, dst.grad_data, src.grad_data);

    return BCNN_SUCCESS;
}

int bcnn_forward_cost_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
    return bcnn_forward_cost_layer_gpu(conn);
#else
    return bcnn_forward_cost_layer_cpu(conn);
#endif
}

int bcnn_backward_cost_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
    return bcnn_backward_cost_layer_gpu(conn);
#else
    return bcnn_backward_cost_layer_cpu(conn);
#endif
}