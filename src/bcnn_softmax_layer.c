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


#include <bh/bh_mem.h>
#include <bh/bh_string.h>

#include "bcnn/bcnn.h"


int bcnn_add_softmax_layer(bcnn_net *net, char *id)
{
    int nb_connections = net->nb_connections + 1;
    int sz;
    bcnn_connection conn = { 0 };

    if (id != NULL) {
        bh_fill_option(&conn.id, id);
    }
    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = SOFTMAX;
    if (nb_connections > 1) {
        conn.src_tensor = net->connections[nb_connections - 2].dst_tensor;
    }
    else {
        conn.src_tensor = net->input_node;
    }

    conn.dst_tensor.w = conn.src_tensor.w;
    conn.dst_tensor.h = conn.src_tensor.h;
    conn.dst_tensor.c = conn.src_tensor.c;
    conn.dst_tensor.b = conn.src_tensor.b;
    sz = bcnn_get_tensor_size(&conn.dst_tensor);
    conn.dst_tensor.data = (float *)calloc(sz, sizeof(float));
    conn.dst_tensor.grad_data = (float *)calloc(sz, sizeof(float));

#ifdef BCNN_USE_CUDA
    conn.dst_tensor.data_gpu = bcnn_cuda_memcpy_f32(conn.dst_tensor.data, sz);
    conn.dst_tensor.grad_data_gpu = bcnn_cuda_memcpy_f32(conn.dst_tensor.grad_data, sz);
#endif
    net->nb_connections = nb_connections;
    bcnn_net_add_connection(net, conn);

    fprintf(stderr, "[Softmax] input_shape= %dx%dx%d output_shape= %dx%dx%d\n",
        conn.src_tensor.w, conn.src_tensor.h, conn.src_tensor.c,
        conn.dst_tensor.w, conn.dst_tensor.h, conn.dst_tensor.c);

    return BCNN_SUCCESS;
}

int bcnn_forward_softmax_layer_cpu(bcnn_connection *conn)
{
    int b, i, batch_size = conn->dst_tensor.b;
    int src_size = conn->src_tensor.w * conn->src_tensor.h * conn->src_tensor.c;
    float vmax = -FLT_MAX;
    float sum = 0.0f;
    bcnn_tensor src = conn->src_tensor;
    bcnn_tensor dst = conn->dst_tensor;

    for (b = 0; b < batch_size; ++b) {
        vmax = -FLT_MAX;
        sum = 0.0f;
        for (i = 0; i < src_size; ++i) {
            if (src.data[b * src_size + i] > vmax) {
                vmax = src.data[b * src_size + i];
            }
        }
        for (i = 0; i < src_size; ++i){
            sum += (float)exp(src.data[b * src_size + i] - vmax);
        }
        if (sum) {
            sum = vmax + (float)log(sum);
        }
        else
            sum = vmax - 100.0f;
        for (i = 0; i < src_size; ++i){
            dst.data[b * src_size + i] = (float)exp(src.data[b * src_size + i] - sum);
        }
    }
    return BCNN_SUCCESS;
}

int bcnn_backward_softmax_layer_cpu(bcnn_connection *conn)
{
    int i;
    int sz = conn->src_tensor.w * conn->src_tensor.h * conn->src_tensor.c * 
        conn->src_tensor.b;
    bcnn_tensor src = conn->src_tensor;
    bcnn_tensor dst = conn->dst_tensor;

    for (i = 0; i < sz; ++i)
        src.grad_data[i] += dst.grad_data[i];

    return BCNN_SUCCESS;
}


int bcnn_forward_softmax_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
    return bcnn_forward_softmax_layer_gpu(conn);
#else
    return bcnn_forward_softmax_layer_cpu(conn);
#endif
}


int bcnn_backward_softmax_layer(bcnn_connection *conn)
{
#ifdef BCNN_USE_CUDA
    return bcnn_backward_softmax_layer_gpu(conn);
#else
    return bcnn_backward_softmax_layer_cpu(conn);
#endif
}