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
#include <bh/bh_string.h>

#include "bcnn/bcnn.h"
#include "bcnn_utils.h"
#include "bh_log.h"

int bcnn_add_maxpool_layer(bcnn_net *net, int size, int stride, char *src_id, char *dst_id)
{
    int sz, i;
    bcnn_connection conn = { 0 };
    bcnn_node dst_node = { 0 };

    if (net->nb_connections > 0) {
        int is_src_node_found = 0;
        for (i = net->num_nodes - 1; i >= 0 ; --i) {
            if (strcmp(net->nodes[i].id, src_id) == 0) {
                bcnn_connection_add_src_node(&conn, i);
                is_src_node_found = 1;
                break;
            }
        }
        bh_check(is_src_node_found, "Maxpool layer: invalid input node name %s", src_id);
    }
    else {
        bcnn_connection_add_src_node(&conn, 0);
    }

    bh_strfill(&dst_node.id, dst_id);
    bcnn_tensor_set_shape(&dst_node.tensor,
        net->nodes[conn.src[0]].tensor.n,                    // batch size
        net->nodes[conn.src[0]].tensor.c,                    // depth
        (int)(ceil((float)(net->nodes[conn.src[0]].tensor.h - size) / stride)) + 1, // height
        (int)(ceil((float)(net->nodes[conn.src[0]].tensor.w - size) / stride)) + 1, // width
        1);
    bcnn_tensor_allocate(&dst_node.tensor);
    // Add node to net
    bcnn_net_add_node(net, dst_node);
    // Add node pointer to connection
    bcnn_connection_add_dst_node(&conn, net->num_nodes - 1);

    conn.layer = (bcnn_layer *)calloc(1, sizeof(bcnn_layer));
    conn.layer->type = MAXPOOL;
    conn.layer->size = size;
    conn.layer->stride = stride;

    sz = bcnn_tensor_get_size(&net->nodes[conn.dst[0]].tensor);
    conn.layer->indexes = (int *)calloc(sz, sizeof(int));
#ifdef BCNN_USE_CUDA
    conn.layer->indexes_gpu = bcnn_cuda_malloc_i32(sz);
#ifdef BCNN_USE_CUDNN
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&conn.layer->src_tensor_desc));
    bcnn_cudnn_check(cudnnCreateTensorDescriptor(&conn.layer->dst_tensor_desc));
    bcnn_cudnn_check(cudnnCreatePoolingDescriptor(&conn.layer->pooling_desc));
    bcnn_cudnn_check(cudnnSetPooling2dDescriptor(conn.layer->pooling_desc,
                                            CUDNN_POOLING_MAX,
                                            CUDNN_NOT_PROPAGATE_NAN,
                                            conn.layer->size,
                                            conn.layer->size,
                                            0,
                                            0,
                                            conn.layer->stride,
                                            conn.layer->stride));
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(conn.layer->src_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        net->nodes[conn.src[0]].tensor.n, net->nodes[conn.src[0]].tensor.c, net->nodes[conn.src[0]].tensor.h, net->nodes[conn.src[0]].tensor.w)); 
    bcnn_cudnn_check(cudnnSetTensor4dDescriptor(conn.layer->dst_tensor_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        net->nodes[conn.dst[0]].tensor.n, net->nodes[conn.dst[0]].tensor.c, net->nodes[conn.dst[0]].tensor.h, net->nodes[conn.dst[0]].tensor.w)); 
#endif
#endif

    bcnn_net_add_connection(net, conn);

    bh_log_info("[Maxpool] input_shape= %dx%dx%d size= %d stride= %d ouput_shape= %dx%dx%d",
        net->nodes[conn.src[0]].tensor.w, net->nodes[conn.src[0]].tensor.h, net->nodes[conn.src[0]].tensor.c, size, stride,
        net->nodes[conn.dst[0]].tensor.w, net->nodes[conn.dst[0]].tensor.h, net->nodes[conn.dst[0]].tensor.c);
    return 0;
}


int bcnn_forward_maxpool_layer_cpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    int b, i, j, k, m, n, dst_index, valid, src_index, cur_w, cur_h, max_i;
    float max_f = -FLT_MAX, val;
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int batch_size = dst.n;
    int offset_pool = (-layer->size - 1) / 2 + 1;
    int offset0, offset1, offset2;

    for (b = 0; b < batch_size; ++b) { // batch_size
        offset0 = dst.c * b;
        for (k = 0; k < dst.c; ++k) {   // depth
            offset1 = dst.h * (k + offset0);
            for (i = 0; i < dst.h; ++i) {   // height
                offset2 = dst.w * (offset1 + i);
                for (j = 0; j < dst.w; ++j) {   // width
                    dst_index = j + offset2;
                    max_f = -FLT_MAX;
                    max_i = -1;
                    for (n = 0; n < layer->size; ++n) { // pooling window
                        for (m = 0; m < layer->size; ++m) {
                            cur_h = offset_pool + i * layer->stride + n;
                            cur_w = offset_pool + j * layer->stride + m;
                            src_index = cur_w + src.w * (cur_h + src.h * (k + b * src.c));
                            valid = (cur_h >= 0 && cur_h < src.h &&
                                cur_w >= 0 && cur_w < src.w);
                            val = (valid != 0) ? src.data[src_index] : -FLT_MAX;
                            if (val > max_f) {
                                max_f = val;
                                max_i = src_index;
                            }
                        }
                    }
                    dst.data[dst_index] = max_f;
                    layer->indexes[dst_index] = max_i;
                }
            }
        }
    }
    return BCNN_SUCCESS;
}


int bcnn_forward_maxpool_layer(bcnn_net *net, bcnn_connection *conn)
{
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_forward_maxpool_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_forward_maxpool_layer_cpu(conn->layer, src, dst);
#endif
}

int bcnn_backward_maxpool_layer_cpu(bcnn_layer *layer, bcnn_node *src_node, bcnn_node *dst_node)
{
    int i, index;
    bcnn_tensor src = src_node->tensor;
    bcnn_tensor dst = dst_node->tensor;
    int sz = bcnn_tensor_get_size(&dst);

    for (i = 0; i < sz; ++i) {
        index = layer->indexes[i];
        src.grad_data[index] += dst.grad_data[i];
    }

    return BCNN_SUCCESS;
}

int bcnn_backward_maxpool_layer(bcnn_net *net, bcnn_connection *conn)
{
    bcnn_node *src = &net->nodes[conn->src[0]];
    bcnn_node *dst = &net->nodes[conn->dst[0]];
#ifdef BCNN_USE_CUDA
    return bcnn_backward_maxpool_layer_gpu(conn->layer, src, dst);
#else
    return bcnn_backward_maxpool_layer_cpu(conn->layer, src, dst);
#endif
    return 0;
}
